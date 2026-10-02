import os

# Must be set before JAX is imported anywhere.
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
os.environ.setdefault('MPLBACKEND', 'Agg')
# Keep BLAS thread pools small so the tests also run on shared login nodes.
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
os.environ.setdefault('OMP_NUM_THREADS', '4')

import numpy as np
import pytest
from jax import random

import winddensity_mbir.configuration_params as config
import winddensity_mbir.simulation as sim

# ---------------------------------------------------------------------------
# A scaled-down version of the paper's 7v8 geometry (7 views over +/-4 deg,
# single sensor, disk-shaped 2 cm beam) that runs in seconds on a CPU.
# ---------------------------------------------------------------------------
SMALL_CM_PER_PIXEL = 0.125
SMALL_RECON_SHAPE = (64, 48, 16)                 # (rows, cols, slices)
SMALL_BEAM_DIAM_CM = 2.0                         # 16 pixels
SMALL_TOTAL_LENGTH_M = SMALL_RECON_SHAPE[0] * SMALL_CM_PER_PIXEL / 100.0
SMALL_CN2 = 1e-11
SMALL_L0 = 0.02
SMALL_DELTA = 0.01 * SMALL_CM_PER_PIXEL


def build_small_setup(half_extent_deg=4.0, n_views=7):
    angles_rad = np.linspace(-half_extent_deg, half_extent_deg, n_views, endpoint=True) * np.pi / 180
    test_region_dims = tuple(n * SMALL_CM_PER_PIXEL for n in SMALL_RECON_SHAPE)
    return config.define_optical_setup(
        sensor_locations=[(0.0, 0.0)],
        beam_angles=[list(angles_rad)],
        test_region_dims=test_region_dims,
        pixel_pitch=SMALL_CM_PER_PIXEL,
        beam_fov=SMALL_BEAM_DIAM_CM,
    )


def generate_small_volume(seed=0):
    return sim.generate_random_atmospheric_volume(
        cn2=SMALL_CN2, dim=SMALL_RECON_SHAPE, delta=SMALL_DELTA, L0=SMALL_L0, key=random.PRNGKey(seed)
    )


@pytest.fixture(scope='session')
def small_setup():
    return build_small_setup()


@pytest.fixture(scope='session')
def small_model_and_weights(small_setup):
    ct_model, weights = sim.create_ct_model_and_weights_for_simulation(small_setup)
    ct_model.max_over_relaxation = 1.25
    return ct_model, weights


@pytest.fixture(scope='session')
def small_volume():
    return generate_small_volume(seed=0)
