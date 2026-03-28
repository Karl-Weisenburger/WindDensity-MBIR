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
N_VOLS = 1000

RESOLUTIONS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 640]
N_OSA_MODES    = 45                # OSA/ANSI indices 0–44 (radial degrees 0–8)
ZERN_RESOLUTION = 11               # OPL-section count at which Zernike analysis is done

# Geometries: (label, half_extent_deg, n_views)
GEOMETRIES = [
    ('3v2',   1.0,  3),
    ('11v16', 8.0, 11),
]
TTP_STATES = ['withTTP', 'noTTP']   # OPL, OPD_TT


# ============================================================
# BUILD GEOMETRY
# ============================================================
def build_geometry(half_extent_deg, n_views):
    angles_rad = np.linspace(-half_extent_deg, half_extent_deg, n_views, endpoint=True) * np.pi / 180
    optical_setup = config.define_optical_setup(
        sensor_locations=[(0.0, 0.0)],
        beam_angles=[list(angles_rad)],   # grouped: one sensor group
        test_region_dims=test_region_dims,
        pixel_pitch=cm_per_pixel,
        beam_fov=beam_diam_cm,
    )
    ct_model, weights = sim.create_ct_model_and_weights_for_simulation(optical_setup)
    ct_model.max_over_relaxation = MAX_OVER_RELAXATION
    return ct_model, weights


print('Building geometries...')
geo_models = {}
for label, half_ext, n_views in GEOMETRIES:
    ct_model, weights = build_geometry(half_ext, n_views)
    geo_models[label] = (ct_model, weights)

# Shared beam ROI (same for all geometries since beam and recon shape are the same)
roi_full = np.array(va.generate_beam_path_roi_mask(recon_shape, beam_pixel_diam, location=(0, 0, 0), angle=0))

# Pre-compute per-resolution ROIs (only depend on s, not geometry)
roi_per_res = {}
for s in RESOLUTIONS:
    roi_per_res[s] = np.array(va.generate_beam_path_roi_mask(
        (s, num_cols, num_slices), beam_pixel_diam, location=(0, 0, 0), angle=0
    ))

# ============================================================
# STORAGE
# ============================================================
n_geos = len(GEOMETRIES)
n_ttp = len(TTP_STATES)
n_res = len(RESOLUTIONS)
n_zmodes = N_OSA_MODES + 1   # OSA modes 0–44 individually + total MSE

nrmse_arr     = np.zeros((N_VOLS, n_ttp, n_geos, n_res))
zernike_mse_arr = np.zeros((N_VOLS, n_ttp, n_geos, n_zmodes))

# ============================================================
# MAIN LOOP
# ============================================================
for vol_idx in trange(N_VOLS, desc='Volumes'):
    key = random.PRNGKey(vol_idx)
    vol_gt = sim.generate_random_atmospheric_volume(
        cn2=1e-11, dim=recon_shape, delta=delta, L0=0.02, key=key
    )
    vol_gt_np = np.array(vol_gt)

    for geo_idx, (geo_label, half_ext, n_views) in enumerate(tqdm(GEOMETRIES, desc='Geometries', leave=False)):
        ct_model, weights = geo_models[geo_label]

        for ttp_idx, proj_type in enumerate(tqdm(['OPL', 'OPD_TT'], desc='TTP', leave=False)):
            sinogram = sim.collect_projection_measurement(
                ct_model, weights, vol_gt, projection_type=proj_type
            )

            recon, _ = ct_model.recon(
                sinogram, weights=weights, init_recon=jnp.zeros(recon_shape),
                max_iterations=MAX_ITERATIONS, stop_threshold_change_pct=STOP_THRESHOLD_PCT,
            )
            recon_np = np.array(recon)

            # ---- Multi-resolution NRMSE ----
            for res_idx, s in enumerate(RESOLUTIONS):
                roi_s = roi_per_res[s]
                roi_s_jnp = jnp.array(roi_s)

                gt_s = np.array(va.divide_into_sections_of_opl(vol_gt, s, 0.2))
                recon_s = np.array(va.divide_into_sections_of_opl(recon, s, 0.2))

                if proj_type == 'OPD_TT':
                    # Remove TTP from both ground-truth and recon before comparing
                    gt_s = np.array(utils.remove_tip_tilt_piston(jnp.array(gt_s), FOV=roi_s_jnp))
                    recon_s = np.array(utils.remove_tip_tilt_piston(jnp.array(recon_s), FOV=roi_s_jnp))

                nrmse_arr[vol_idx, ttp_idx, geo_idx, res_idx] = float(
                    va.nrmse_over_roi(jnp.array(gt_s), jnp.array(recon_s), roi_s_jnp, option=2)
                )

                # ---- Zernike MSE (at ZERN_RESOLUTION only) ----
                if s == ZERN_RESOLUTION:
                    error_s = gt_s - recon_s  # already TTP-removed if noTTP

                    # Fit all 45 OSA modes simultaneously (no cross-talk)
                    per_mode_mse = va.compute_osa_mode_mse_for_volume(
                        error_s, roi_s, max_j=N_OSA_MODES - 1
                    )
                    zernike_mse_arr[vol_idx, ttp_idx, geo_idx, :N_OSA_MODES] = per_mode_mse

                    # Total MSE across all pixels in ROI (final index)
                    zernike_mse_arr[vol_idx, ttp_idx, geo_idx, -1] = float(
                        np.mean(error_s[roi_s] ** 2)
                    )

    # Save incrementally every 50 volumes
    if (vol_idx + 1) % 50 == 0:
        np.savez(
            'data/fig10_11_table2_3v2_11v16_partial.npz',
            nrmse=nrmse_arr,
            zernike_mse=zernike_mse_arr,
            n_completed=vol_idx + 1,
            resolutions=np.array(RESOLUTIONS),
            geometry_names=np.array([g[0] for g in GEOMETRIES]),
            ttp_states=np.array(TTP_STATES),
            n_osa_modes=N_OSA_MODES,
            zern_resolution=ZERN_RESOLUTION,
            max_over_relaxation=MAX_OVER_RELAXATION,
            max_iterations=MAX_ITERATIONS,
            stop_threshold_pct=STOP_THRESHOLD_PCT,
        )

# ============================================================
# FINAL SAVE
# ============================================================
np.savez(
    'data/fig10_11_table2_3v2_11v16.npz',
    nrmse=nrmse_arr,
    zernike_mse=zernike_mse_arr,
    resolutions=np.array(RESOLUTIONS),
    geometry_names=np.array([g[0] for g in GEOMETRIES]),
    ttp_states=np.array(TTP_STATES),
    n_osa_modes=N_OSA_MODES,
    zern_resolution=ZERN_RESOLUTION,
    max_over_relaxation=MAX_OVER_RELAXATION,
    max_iterations=MAX_ITERATIONS,
)
print('Done. Saved data/fig10_11_table2_3v2_11v16.npz')
