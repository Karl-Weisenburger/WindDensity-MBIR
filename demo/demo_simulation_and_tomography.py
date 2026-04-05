# Approved for public release; distribution is unlimited. Public Affairs release approval # 2025-5579
"""
Demo: simulate OPD_TT measurements from a cached atmospheric phase volume
and perform MBIR reconstruction.

Steps
-----
1. Define the viewing geometry and save a schematic.
2. Load (or, first time only, generate) atmospheric phase volume
   ``vol_seed17.npy`` under ``experiments/shared_data/``. This is the same
   volume used by Fig 6 / Fig 12 / Fig 16 in the paper, so running the
   demo first pre-populates the cache used by those figure scripts and
   vice versa.
3. Simulate OPD_TT projection measurements and run MBIR reconstruction
   with ``max_over_relaxation = 1.25`` (matches the paper experiments).
4. Visualise the reconstruction with the same layout used by Figs 6, 12,
   and 16 (``plot_recon_figure``).
"""

import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

import sys
from pathlib import Path

# Make both `demo_utils` and `experiments.*` importable regardless of where
# this script is launched from.
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT  = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

import winddensity_mbir.simulation as sim
import winddensity_mbir.visualization_and_analysis as va
import winddensity_mbir.utilities as utils
import winddensity_mbir.configuration_params as config

from experiments.recon_visualization_prep import (
    load_or_generate_volume,
    prepare_opl_images,
    compute_per_section_nrmse,
    plot_recon_figure,
)

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
OUTPUT_DIR = SCRIPT_DIR / 'output'
OUTPUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Simulation parameters (must match experiments/shared_data/vol_seed17.npy
# so that the cached volume is reusable across the demo and figure scripts)
# ---------------------------------------------------------------------------
SEED             = 17
CM_PER_PIXEL     = 25.0 / 800                      # 0.03125 cm
RECON_SHAPE      = (640, 400, 64)
NUM_ROWS, NUM_COLS, NUM_SLICES = RECON_SHAPE
TEST_REGION_DIMS = (NUM_ROWS * CM_PER_PIXEL,
                    NUM_COLS * CM_PER_PIXEL,
                    NUM_SLICES * CM_PER_PIXEL)     # (20, 12.5, 2) cm
BEAM_DIAM_CM     = 2.0
BEAM_PIXEL_DIAM  = int(round(BEAM_DIAM_CM / CM_PER_PIXEL))  # 64
TOTAL_LENGTH_M   = NUM_ROWS * CM_PER_PIXEL / 100.0          # 0.20 m

# Atmospheric-turbulence parameters (same as the experiments)
CN2   = 1e-11
L0    = 0.02
DELTA = 0.01 * CM_PER_PIXEL

# MBIR reconstruction settings
MAX_OVER_RELAXATION       = 1.25
MAX_ITERATIONS            = 20
STOP_THRESHOLD_CHANGE_PCT = 1

# Visualization settings
SECTIONS        = 4
ZERN_MODE_INDEX = 2   # 2 = OPD_TT (piston + tip + tilt removed)

VOL_PATH = REPO_ROOT / 'experiments' / 'shared_data' / f'vol_seed{SEED}.npy'


# ---------------------------------------------------------------------------
# Step 1 — Viewing geometry
# ---------------------------------------------------------------------------
sensor_locations = [
    jnp.array([26, -2.5]),
    jnp.array([26,  0.0]),
    jnp.array([26,  2.5]),
]
beam_angles = [
    jnp.array([-6.5, -5.5, -4.5]) * jnp.pi / 180,
    jnp.array([-1.0,  0.0,  1.0]) * jnp.pi / 180,
    jnp.array([ 4.5,  5.5,  6.5]) * jnp.pi / 180,
]

optical_params = config.define_optical_setup(
    sensor_locations, beam_angles,
    TEST_REGION_DIMS, CM_PER_PIXEL,
    beam_fov=BEAM_DIAM_CM, windows=True,
)

va.display_viewing_configuration_schematic(
    optical_params,
    roi_thickness_and_num_regions=(optical_params.beam_diameter_cm, 1),
)
plt.savefig(OUTPUT_DIR / 'Viewing_Configuration_Schematic.png',
            bbox_inches='tight', dpi=200)


# ---------------------------------------------------------------------------
# Step 2 — Load (or generate) atmospheric phase volume
# ---------------------------------------------------------------------------
print('\nLoading atmospheric phase volume...')
phase_volume_np = load_or_generate_volume(
    seed=SEED,
    recon_shape=RECON_SHAPE,
    delta=DELTA,
    L0=L0,
    cn2=CN2,
    cache_path=VOL_PATH,
)
phase_volume = jnp.array(phase_volume_np)


# ---------------------------------------------------------------------------
# Step 3 — Simulate OPD_TT measurements and run MBIR reconstruction
# ---------------------------------------------------------------------------
print('\nCreating CT model and FOV mask...')
ct_model, FOV = sim.create_ct_model_and_weights_for_simulation(optical_params)
ct_model.max_over_relaxation = MAX_OVER_RELAXATION #for better convergence stability

print('Simulating OPD_TT measurements...')
OPD_views = sim.collect_projection_measurement(
    ct_model, FOV, phase_volume, projection_type='OPD_TT',
)

print(f'Running MBIR reconstruction...')
recon, _ = ct_model.recon(
    OPD_views, weights=FOV,
    max_iterations=MAX_ITERATIONS,
    stop_threshold_change_pct=STOP_THRESHOLD_CHANGE_PCT,
)
recon_np = np.array(recon)
print('Reconstruction complete.')


# ---------------------------------------------------------------------------
# Step 4 — Visualise reconstruction in the Fig 6 / 12 / 16 style
# ---------------------------------------------------------------------------
gt_images,    roi_beam = prepare_opl_images(
    phase_volume_np, ZERN_MODE_INDEX, SECTIONS, TOTAL_LENGTH_M,
    BEAM_PIXEL_DIAM, NUM_COLS, NUM_SLICES,
)
recon_images, _        = prepare_opl_images(
    recon_np,        ZERN_MODE_INDEX, SECTIONS, TOTAL_LENGTH_M,
    BEAM_PIXEL_DIAM, NUM_COLS, NUM_SLICES,
)

nrmse_per_section = compute_per_section_nrmse(gt_images, recon_images, roi_beam)

fig = plot_recon_figure(
    gt_images              = gt_images,
    recon_images_list      = [recon_images],
    nrmse_per_section_list = [nrmse_per_section],
    recon_labels           = [
        r'WindDensity-MBIR Reconstruction from $\mathrm{OPD}_{\mathrm{TT}}$ measurements',
    ],
    gt_suptitle            = r'$\mathrm{OPD}_{\mathrm{TT}}$ Ground Truth Planes',
    fig_suptitle           = 'Demo: MBIR Reconstruction',
    roi_beam               = roi_beam,
)
fig.savefig(OUTPUT_DIR / 'Recon_Planes.png', bbox_inches='tight', dpi=200)

plt.show()
