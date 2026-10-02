"""
Regression tests: end-to-end simulation + reconstruction of a synthetic phantom.

The golden values below were produced with the pinned dependency versions in
pyproject.toml. If one of these tests fails after a dependency change, the
pipeline no longer produces the same numbers as the published results; do not
update the golden values without understanding why they moved.

The ``slow`` tests rerun the full-size Fig 6 case (needs a GPU to be quick)
and compare against the published reconstruction error. If the cached paper
outputs are present in experiments/, they are also compared directly.
"""
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

import winddensity_mbir.simulation as sim
import winddensity_mbir.utilities as utils
import winddensity_mbir.visualization_and_analysis as va
from conftest import SMALL_RECON_SHAPE, SMALL_TOTAL_LENGTH_M

EXPERIMENTS_DIR = Path(__file__).resolve().parents[1] / 'experiments'

# Golden values for generate_small_volume(seed=0). These pin down the JAX PRNG
# stream and FFT pipeline that generate every ground-truth volume in the paper.
GOLDEN_VOLUME_SUM_SQ = 4.239511082268998e-09
GOLDEN_VOLUME_SAMPLES = {
    (10, 20, 5): 5.091431631853993e-08,
    (50, 3, 12): 2.884705452288472e-07,
    (31, 24, 8): -2.3261226544946112e-07,
}

# Golden OPD_TT 4-plane NRMSE for the small 7v8 geometry (CPU values; the GPU
# gives 0.266345 and 0.197641).
GOLDEN_NRMSE_FBP_SCALED = 0.26634299755096436
GOLDEN_NRMSE_MBIR = 0.197658970952034


def _opd_tt_nrmse(gt, recon, n_sections=4):
    """Same metric as Fig 6 / Table 2: NRMSE of 4 OPD_TT planes within the beam."""
    roi = jnp.array(va.generate_beam_path_roi_mask(
        (n_sections,) + SMALL_RECON_SHAPE[1:], 16))
    g = utils.remove_tip_tilt_piston(va.divide_into_sections_of_opl(gt, n_sections, SMALL_TOTAL_LENGTH_M), FOV=roi)
    r = utils.remove_tip_tilt_piston(va.divide_into_sections_of_opl(recon, n_sections, SMALL_TOTAL_LENGTH_M), FOV=roi)
    return float(va.nrmse_over_roi(g, r, roi, option=2))


def run_reconstructions(ct_model, weights, volume):
    """Simulate OPD_TT data and reconstruct with scale-corrected FBP and MBIR (paper settings)."""
    sinogram = sim.collect_projection_measurement(ct_model, weights, volume, projection_type='OPD_TT')
    fbp = jnp.asarray(ct_model.direct_recon(sinogram))
    fbp_scaled = utils.correct_recon_scaling(fbp, ct_model, sinogram, weights)
    mbir, _ = ct_model.recon(sinogram, weights=weights, max_iterations=20, stop_threshold_change_pct=1)
    return {'fbp_scaled': fbp_scaled, 'mbir': jnp.asarray(mbir)}


@pytest.fixture(scope='module')
def reconstructions(small_model_and_weights, small_volume):
    return run_reconstructions(*small_model_and_weights, small_volume)


def test_volume_generation_matches_golden_values(small_volume):
    v = np.asarray(small_volume, dtype=np.float64)
    assert v.dtype == np.float64 and small_volume.dtype == jnp.float32
    assert np.sum(v ** 2) == pytest.approx(GOLDEN_VOLUME_SUM_SQ, rel=1e-4)
    for idx, value in GOLDEN_VOLUME_SAMPLES.items():
        assert v[idx] == pytest.approx(value, rel=1e-3), idx


def test_mbir_beats_scale_corrected_fbp(small_volume, reconstructions):
    nrmse_mbir = _opd_tt_nrmse(small_volume, reconstructions['mbir'])
    nrmse_fbp = _opd_tt_nrmse(small_volume, reconstructions['fbp_scaled'])
    assert nrmse_mbir < nrmse_fbp


def test_reconstruction_nrmse_matches_golden_values(small_volume, reconstructions):
    # MBIR is iterative, so allow for CPU/GPU floating-point differences.
    assert _opd_tt_nrmse(small_volume, reconstructions['fbp_scaled']) == pytest.approx(GOLDEN_NRMSE_FBP_SCALED, rel=1e-3)
    assert _opd_tt_nrmse(small_volume, reconstructions['mbir']) == pytest.approx(GOLDEN_NRMSE_MBIR, rel=2e-2)


# ---------------------------------------------------------------------------
# Full-size checks: the seed-17 volume and 7v8 geometry used for Fig 6.
# Reference NRMSE values are those of the reconstructions behind the paper
# figure (experiments/fig6_table2/data/, generated 2026-04-06).
# ---------------------------------------------------------------------------
PAPER_SHAPE = (640, 400, 64)
PAPER_CM_PER_PIXEL = 25.0 / 800
PAPER_VOLUME_STD = 3.1847182e-07
PAPER_NRMSE_FBP_SCALED = 0.22071
PAPER_NRMSE_MBIR = 0.15780
CACHED_VOL = EXPERIMENTS_DIR / 'shared_data' / 'vol_seed17.npy'
CACHED_FBP = EXPERIMENTS_DIR / 'fig6_table2' / 'data' / 'recon_seed17_7v8_FBP_scaled.npy'


@pytest.fixture(scope='module')
def paper_case():
    from jax import random
    import winddensity_mbir.configuration_params as config
    vol = sim.generate_random_atmospheric_volume(
        cn2=1e-11, dim=PAPER_SHAPE, delta=0.01 * PAPER_CM_PER_PIXEL, L0=0.02, key=random.PRNGKey(17))
    angles = np.linspace(-4.0, 4.0, 7) * np.pi / 180
    setup = config.define_optical_setup(
        sensor_locations=[(0.0, 0.0)], beam_angles=[list(angles)],
        test_region_dims=tuple(n * PAPER_CM_PER_PIXEL for n in PAPER_SHAPE),
        pixel_pitch=PAPER_CM_PER_PIXEL, beam_fov=2.0)
    ct_model, weights = sim.create_ct_model_and_weights_for_simulation(setup)
    ct_model.max_over_relaxation = 1.25
    return vol, ct_model, weights


def _paper_nrmse(gt, recon):
    """Fig 6 metric (experiments/recon_visualization_prep.py): NRMSE over 4 OPD_TT planes in the 64-pixel beam."""
    num_cols, num_slices, beam = PAPER_SHAPE[1], PAPER_SHAPE[2], 64
    roi = jnp.array(va.generate_beam_path_roi_mask((4, num_cols, num_slices), beam))
    crop = slice(num_cols // 2 - beam // 2, num_cols // 2 + beam // 2)

    def planes(v):
        opl = va.divide_into_sections_of_opl(jnp.array(v), 4, 0.2)
        return np.array(utils.remove_tip_tilt_piston(opl, FOV=roi))[:, crop, :]

    return float(va.nrmse_over_roi(jnp.array(planes(gt)), jnp.array(planes(recon)),
                                   jnp.array(np.array(roi)[:, crop, :]), option=2))


@pytest.mark.slow
def test_paper_volume_regenerates_from_seed(paper_case):
    vol = np.asarray(paper_case[0])
    assert float(vol.std()) == pytest.approx(PAPER_VOLUME_STD, rel=1e-4)
    if CACHED_VOL.exists():
        cached = np.load(CACHED_VOL)
        np.testing.assert_allclose(vol, cached, rtol=0, atol=1e-5 * np.abs(cached).max())


@pytest.mark.slow
def test_paper_reconstructions_reproduce_published_nrmse(paper_case):
    vol, ct_model, weights = paper_case
    recons = run_reconstructions(ct_model, weights, vol)
    assert _paper_nrmse(vol, recons['fbp_scaled']) == pytest.approx(PAPER_NRMSE_FBP_SCALED, rel=1e-3)
    assert _paper_nrmse(vol, recons['mbir']) == pytest.approx(PAPER_NRMSE_MBIR, abs=1e-3)
    if CACHED_FBP.exists():
        cached = np.load(CACHED_FBP)
        rel_l2 = np.linalg.norm(np.asarray(recons['fbp_scaled']) - cached) / np.linalg.norm(cached)
        assert rel_l2 < 1e-4
