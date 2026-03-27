import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))

import numpy as np
import jax.numpy as jnp
from jax import random
from tqdm import trange, tqdm

import winddensity_mbir.configuration_params as config
import winddensity_mbir.simulation as sim
import winddensity_mbir.visualization_and_analysis as va
import winddensity_mbir.utilities as utils

# ============================================================
# PARAMETERS
# ============================================================
cm_per_pixel = 25.0 / 800
recon_shape = (640, 400, 64)
num_rows, num_cols, num_slices = recon_shape
test_region_dims = (num_rows * cm_per_pixel, num_cols * cm_per_pixel, num_slices * cm_per_pixel)
beam_diam_cm = 2.0
beam_pixel_diam = int(round(beam_diam_cm / cm_per_pixel))  # 64
delta = 0.01 * cm_per_pixel
MAX_OVER_RELAXATION = 1.2
MAX_ITERATIONS = 75
STOP_THRESHOLD_PCT = 0   # set >0 for early stopping (e.g. 0.1); 0 = run all MAX_ITERATIONS
N_VOLS = 100

# Grid: full angular extents (degrees total sweep), num_views
# 'Full angular extent' = 2 × half_extent (e.g., 10° total = ±5°)
FULL_EXTENTS_DEG = list(range(9, 17))          # [9, 10, 11, 12, 13, 14, 15, 16]
NUM_VIEWS_LIST   = [3, 5, 7, 9, 11]
RESOLUTIONS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 640]

# Fig7: TTP removed from sino AND recon (OPD_TT projection, compare TTP-removed GT vs recon)


# ============================================================
# BUILD ALL GEOMETRIES UPFRONT
# ============================================================
print('Building geometries...')
geo_models = {}    # (full_ext, n_views) -> (ct_model, weights)
for full_ext in FULL_EXTENTS_DEG:
    half_ext = full_ext / 2.0
    for n_views in NUM_VIEWS_LIST:
        angles_rad = np.linspace(-half_ext, half_ext, n_views, endpoint=True) * np.pi / 180
        optical_setup = config.define_optical_setup(
            sensor_locations=[(0.0, 0.0)],
            beam_angles=[list(angles_rad)],
            test_region_dims=test_region_dims,
            pixel_pitch=cm_per_pixel,
            beam_fov=beam_diam_cm,
        )
        ct_model, weights = sim.create_ct_model_and_weights_for_simulation(optical_setup)
        ct_model.max_over_relaxation = MAX_OVER_RELAXATION
        geo_models[(full_ext, n_views)] = (ct_model, weights)

# Pre-compute per-resolution ROIs
roi_per_res = {}
for s in RESOLUTIONS:
    roi_per_res[s] = np.array(va.generate_beam_path_roi_mask(
        (s, num_cols, num_slices), beam_pixel_diam, location=(0, 0, 0), angle=0
    ))

# ============================================================
# STORAGE
# ============================================================
n_ext  = len(FULL_EXTENTS_DEG)
n_view = len(NUM_VIEWS_LIST)
n_res  = len(RESOLUTIONS)

nrmse_arr = np.zeros((N_VOLS, n_ext, n_view, n_res))

# ============================================================
# MAIN LOOP
# ============================================================
for vol_idx in trange(N_VOLS, desc='Volumes'):
    key = random.PRNGKey(vol_idx)
    vol_gt = sim.generate_random_atmospheric_phase_volume(
        r0=0.05, dim=recon_shape, delta=delta, L0=0.02, key=key
    )

    for ext_idx, full_ext in enumerate(tqdm(FULL_EXTENTS_DEG, desc='Extents', leave=False)):
        for view_idx, n_views in enumerate(NUM_VIEWS_LIST):
            ct_model, weights = geo_models[(full_ext, n_views)]

            sinogram = sim.collect_projection_measurement(
                ct_model, weights, vol_gt, projection_type='OPD_TT'
            )

            recon, _ = ct_model.recon(
                sinogram, weights=weights, init_recon=jnp.zeros(recon_shape),
                max_iterations=MAX_ITERATIONS, stop_threshold_change_pct=STOP_THRESHOLD_PCT,
            )

            for res_idx, s in enumerate(RESOLUTIONS):
                roi_s = roi_per_res[s]
                roi_s_jnp = jnp.array(roi_s)

                gt_s = np.array(va.divide_into_sections_of_opl(vol_gt, s, 0.2))
                recon_s = np.array(va.divide_into_sections_of_opl(recon, s, 0.2))

                # Remove TTP from GT sections for fair comparison (noTTP condition)
                gt_s = np.array(utils.remove_tip_tilt_piston(jnp.array(gt_s), FOV=roi_s_jnp))
                recon_s = np.array(utils.remove_tip_tilt_piston(jnp.array(recon_s), FOV=roi_s_jnp))

                nrmse_arr[vol_idx, ext_idx, view_idx, res_idx] = float(
                    va.nrmse_over_roi(jnp.array(gt_s), jnp.array(recon_s), roi_s_jnp, option=2)
                )

    if (vol_idx + 1) % 10 == 0:
        np.savez(
            'data/fig7_geometry_sweep_partial.npz',
            nrmse=nrmse_arr,
            n_completed=vol_idx + 1,
            full_extents_deg=np.array(FULL_EXTENTS_DEG),
            num_views_list=np.array(NUM_VIEWS_LIST),
            resolutions=np.array(RESOLUTIONS),
            max_over_relaxation=MAX_OVER_RELAXATION,
            max_iterations=MAX_ITERATIONS,
            stop_threshold_pct=STOP_THRESHOLD_PCT,
        )

# ============================================================
# FINAL SAVE
# ============================================================
np.savez(
    'data/fig7_geometry_sweep.npz',
    nrmse=nrmse_arr,
    full_extents_deg=np.array(FULL_EXTENTS_DEG),
    num_views_list=np.array(NUM_VIEWS_LIST),
    resolutions=np.array(RESOLUTIONS),
    max_over_relaxation=MAX_OVER_RELAXATION,
    max_iterations=MAX_ITERATIONS,
)
print('Done. Saved data/fig7_geometry_sweep.npz')
