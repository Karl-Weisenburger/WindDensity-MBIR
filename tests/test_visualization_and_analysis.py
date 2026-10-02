import jax.numpy as jnp
import numpy as np
import pytest

import winddensity_mbir.visualization_and_analysis as va


def _disk_mask(n, radius):
    r, c = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
    return (r - (n - 1) / 2) ** 2 + (c - (n - 1) / 2) ** 2 <= radius ** 2


def test_osa_index_conversion_matches_standard_table():
    expected = [(0, 0), (1, -1), (1, 1), (2, -2), (2, 0), (2, 2), (3, -3), (3, -1), (3, 1), (3, 3)]
    assert [va._osa_to_nm(j) for j in range(10)] == expected


def test_zernike_modes_are_orthonormal_on_unit_disk():
    # Midpoint-rule integration over the unit disk.
    n = 400
    x = (np.arange(n) + 0.5) / n * 2 - 1
    X, Y = np.meshgrid(x, x)
    inside = X ** 2 + Y ** 2 <= 1
    rho, theta = np.hypot(X, Y)[inside], np.arctan2(Y, X)[inside]
    dA = (2 / n) ** 2
    modes = [va._osa_to_nm(j) for j in range(15)]
    Z = np.stack([va._zernike(nn, m, rho, theta) for nn, m in modes], axis=1)
    gram = Z.T @ Z * dA
    np.testing.assert_allclose(gram, np.eye(len(modes)), atol=2e-2)


def test_isolate_zernike_mode_range_recovers_modes_in_range():
    roi = _disk_mask(41, 20)
    rows, cols = np.nonzero(roi)
    dy, dx = rows - rows.mean(), cols - cols.mean()
    rho, theta = np.hypot(dx, dy) / np.hypot(dx, dy).max(), np.arctan2(dy, dx)
    signal = (2.0 * va._zernike(1, 1, rho, theta) - 1.0 * va._zernike(2, 0, rho, theta)
              + 0.5 * va._zernike(4, 2, rho, theta))
    image = np.zeros(roi.shape)
    image[rows, cols] = signal

    # Every mode present lies in the fitted range, so the fit must be exact.
    out = va.isolate_zernike_mode_range_for_img(image, roi, 0, 4)
    np.testing.assert_allclose(out[rows, cols], signal, atol=1e-8)
    assert np.all(out[~roi] == 0)


def test_osa_mode_mse_concentrates_in_the_injected_mode():
    roi2d = _disk_mask(41, 20)
    rows, cols = np.nonzero(roi2d)
    dy, dx = rows - rows.mean(), cols - cols.mean()
    rho, theta = np.hypot(dx, dy) / np.hypot(dx, dy).max(), np.arctan2(dy, dx)
    vol = np.zeros((3,) + roi2d.shape)
    vol[:, rows, cols] = 3.0 * va._zernike(2, 2, rho, theta)   # OSA j = 5
    mse = va.compute_osa_mode_mse_for_volume(vol, np.broadcast_to(roi2d, vol.shape), max_j=14)
    assert int(np.argmax(mse)) == 5
    assert mse[5] == pytest.approx(np.mean((3.0 * va._zernike(2, 2, rho, theta)) ** 2), rel=1e-8)
    assert np.sum(np.delete(mse, 5)) < 1e-12 * mse[5]


def test_divide_into_sections_of_opl_constant_volume():
    vol = jnp.full((64, 6, 5), 2.0)
    opl = va.divide_into_sections_of_opl(vol, 4, total_length=0.2)
    assert opl.shape == (4, 6, 5)
    np.testing.assert_allclose(np.array(opl), 2.0 * 0.2 / 4, rtol=1e-6)


@pytest.mark.parametrize('sections', [3, 4, 7])
def test_divide_into_sections_of_opl_preserves_total_path_length(sections):
    rng = np.random.default_rng(0)
    vol = jnp.array(rng.standard_normal((64, 6, 5)), dtype=jnp.float32)
    opl = va.divide_into_sections_of_opl(vol, sections, total_length=0.2)
    total = np.array(vol).sum(axis=0) * 0.2 / vol.shape[0]
    np.testing.assert_allclose(np.array(opl).sum(axis=0), total, rtol=1e-4, atol=1e-6)


def test_divide_into_sections_of_opl_accepts_numpy_input():
    # Newer mbirjax versions return numpy arrays from recon().
    rng = np.random.default_rng(1)
    vol = rng.standard_normal((64, 6, 5)).astype(np.float32)
    np.testing.assert_array_equal(np.array(va.divide_into_sections_of_opl(vol, 4, 0.2)),
                                  np.array(va.divide_into_sections_of_opl(jnp.array(vol), 4, 0.2)))


def test_nrmse_over_roi_basic_values():
    rng = np.random.default_rng(0)
    gt = jnp.array(rng.standard_normal((8, 8)))
    roi = jnp.ones((8, 8), dtype=bool)
    assert float(va.nrmse_over_roi(gt, gt, roi)) == pytest.approx(0.0, abs=1e-7)
    assert float(va.nrmse_over_roi(gt, jnp.zeros_like(gt), roi, option=0)) == pytest.approx(1.0, rel=1e-6)


def test_beam_path_roi_mask_is_a_cylinder_of_the_right_size():
    mask = np.array(va.generate_beam_path_roi_mask((4, 80, 80), beam_pixel_diam=40))
    assert mask.shape == (4, 80, 80)
    assert all((mask[k] == mask[0]).all() for k in range(4))       # same disk in every section
    assert mask[0].sum() == pytest.approx(np.pi * 20 ** 2, rel=0.03)
