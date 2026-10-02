import jax.numpy as jnp
import numpy as np
import pytest

import winddensity_mbir.utilities as utils


def _disk(shape, radius):
    r, c = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    return (r - shape[0] / 2 + 0.5) ** 2 + (c - shape[1] / 2 + 0.5) ** 2 <= radius ** 2


@pytest.fixture
def fov():
    return jnp.array(_disk((24, 32), 10))


def test_remove_tip_tilt_piston_removes_exact_plane(fov):
    r, c = np.meshgrid(np.arange(24), np.arange(32), indexing='ij')
    plane = jnp.array(3.0 + 0.2 * r - 0.7 * c, dtype=jnp.float32)
    out = utils.remove_tip_tilt_piston(plane, FOV=fov)
    # float32 arithmetic (TF32 on some GPUs) limits how exactly the plane cancels.
    assert float(jnp.max(jnp.abs(jnp.where(fov, out, 0)))) < 1e-4 * float(jnp.max(jnp.abs(plane)))


def test_remove_tip_tilt_piston_residual_is_orthogonal_to_plane(fov):
    rng = np.random.default_rng(0)
    data = jnp.array(rng.standard_normal((24, 32)), dtype=jnp.float32)
    out = np.array(utils.remove_tip_tilt_piston(data, FOV=fov))
    m = np.array(fov)
    r, c = np.nonzero(m)
    basis = np.stack([np.ones_like(r), r, c], axis=1).astype(float)
    scale = np.abs(basis).T @ np.abs(out[m])
    assert np.all(np.abs(basis.T @ out[m]) < 1e-4 * scale)


def test_remove_tip_tilt_piston_is_idempotent(fov):
    rng = np.random.default_rng(1)
    data = jnp.array(rng.standard_normal((24, 32)), dtype=jnp.float32)
    once = utils.remove_tip_tilt_piston(data, FOV=fov)
    twice = utils.remove_tip_tilt_piston(once, FOV=fov)
    np.testing.assert_allclose(np.array(once)[np.array(fov)], np.array(twice)[np.array(fov)], atol=1e-5)


def test_estimate_plus_remove_reconstructs_input(fov):
    rng = np.random.default_rng(2)
    data = jnp.array(rng.standard_normal((3, 24, 32)), dtype=jnp.float32)
    fov3 = jnp.broadcast_to(fov, data.shape)
    est = utils.estimate_tip_tilt_piston(data, FOV=fov3)
    rem = utils.remove_tip_tilt_piston(data, FOV=fov3)
    np.testing.assert_allclose(np.array(est + rem)[np.array(fov3)], np.array(data)[np.array(fov3)], atol=1e-5)


def test_remove_piston_gives_zero_mean_in_fov(fov):
    rng = np.random.default_rng(3)
    data = jnp.array(rng.standard_normal((3, 24, 32)) + 5.0, dtype=jnp.float32)
    fov3 = jnp.broadcast_to(fov, data.shape)
    out = utils.remove_piston(data, FOV=fov3)
    for k in range(3):
        assert abs(float(jnp.mean(out[k][fov]))) < 1e-5


def test_circ_block_keeps_disk_area():
    out = utils.circ_block(jnp.ones((101, 101)), diameter=60)
    assert float(out.sum()) == pytest.approx(np.pi * 30 ** 2, rel=0.01)


def test_correct_recon_scaling_recovers_scale(small_model_and_weights, small_volume):
    ct_model, weights = small_model_and_weights
    sinogram = jnp.asarray(ct_model.forward_project(small_volume)) * weights
    scaled = utils.correct_recon_scaling(0.25 * small_volume, ct_model, sinogram, weights)
    np.testing.assert_allclose(np.array(scaled), np.array(small_volume), rtol=1e-3, atol=1e-12)
