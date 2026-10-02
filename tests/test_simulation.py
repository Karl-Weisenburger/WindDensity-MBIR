import jax.numpy as jnp
import numpy as np
import pytest

import winddensity_mbir.simulation as sim
import winddensity_mbir.utilities as utils
from conftest import SMALL_RECON_SHAPE, generate_small_volume


def test_atmospheric_volume_is_deterministic_for_a_seed():
    a = np.array(generate_small_volume(seed=3))
    b = np.array(generate_small_volume(seed=3))
    c = np.array(generate_small_volume(seed=4))
    assert a.shape == SMALL_RECON_SHAPE
    assert np.isrealobj(a) and np.all(np.isfinite(a))
    np.testing.assert_array_equal(a, b)
    assert not np.allclose(a, c)


def test_atmospheric_volume_has_zero_mean():
    # The DC term of the PSD is zeroed, so the volume mean must vanish.
    v = np.array(generate_small_volume(seed=0), dtype=np.float64)
    assert abs(v.mean()) < 1e-6 * v.std()


def test_atmospheric_volume_scales_with_sqrt_cn2():
    kw = dict(dim=(16, 16, 16), delta=1e-4, L0=0.02)
    from jax import random
    a = np.array(sim.generate_random_atmospheric_volume(cn2=1e-11, key=random.PRNGKey(0), **kw))
    b = np.array(sim.generate_random_atmospheric_volume(cn2=4e-11, key=random.PRNGKey(0), **kw))
    np.testing.assert_allclose(b, 2 * a, rtol=1e-4, atol=1e-12)


def test_weights_are_binary_disks(small_setup, small_model_and_weights):
    _, weights = small_model_and_weights
    w = np.array(weights)
    n_views = len(small_setup.beam_angles)
    assert w.shape == (n_views, SMALL_RECON_SHAPE[2], max(SMALL_RECON_SHAPE[:2]))
    assert set(np.unique(w)) <= {0.0, 1.0}
    radius = small_setup.beam_diameter_pixels / 2
    for k in range(n_views):
        assert w[k].sum() == pytest.approx(np.pi * radius ** 2, rel=0.1)


@pytest.mark.parametrize('projection_type', ['OPL', 'OPD', 'OPD_TT'])
def test_measurements_are_zero_outside_fov(small_model_and_weights, small_volume, projection_type):
    ct_model, weights = small_model_and_weights
    sino = np.array(sim.collect_projection_measurement(ct_model, weights, small_volume, projection_type))
    assert np.all(sino[np.array(weights) == 0] == 0)


def test_opd_tt_measurement_has_no_residual_plane(small_model_and_weights, small_volume):
    ct_model, weights = small_model_and_weights
    sino = sim.collect_projection_measurement(ct_model, weights, small_volume, 'OPD_TT')
    fov = weights == 1
    planes = utils.estimate_tip_tilt_piston(sino, FOV=fov)
    assert float(jnp.max(jnp.abs(planes))) < 1e-3 * float(jnp.max(jnp.abs(sino)))


def test_invalid_projection_type_raises(small_model_and_weights, small_volume):
    ct_model, weights = small_model_and_weights
    with pytest.raises(ValueError):
        sim.collect_projection_measurement(ct_model, weights, small_volume, 'bogus')
