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

DATA_DIR = Path(__file__).parent / 'data'
DATA_DIR.mkdir(exist_ok=True)

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
MAX_OVER_RELAXATION = 1.25
MAX_ITERATIONS = 20
STOP_THRESHOLD_PCT = 1

N_VOLS = 1000        # total reconstructions (used for figs 14/15 Zernike)
N_NRMSE_VOLS = 100  # first N vols also get full resolution NRMSE (used for figs 13/17)

# Resolution sweep for figs 13/17
RESOLUTIONS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 640]
# Fixed resolution for Zernike analysis (figs 14/15)
ZERN_RESOLUTION = 4
N_OSA_MODES = 45     # OSA/ANSI indices 0–44 (radial degrees 0–8)

# Geometry: 7v8 (7 views, ±4° = 8° total)
GEO_LABEL = '7v8'
HALF_EXTENT_DEG = 4.0
N_VIEWS = 7
# Measurement types (sinogram input): OPL = keep TTP, OPD_TT = remove TTP from sino
MEAS_TYPES = ['OPL', 'OPD_TT']
# Evaluation types (how GT and recon are compared):
#   0 = OPL eval  (no TTP removal from GT/recon)
#   1 = OPD_TT eval (TTP removed from both GT and recon before NRMSE)
EVAL_TYPES = ['OPL', 'OPD_TT']


# ============================================================
# BUILD GEOMETRY
# ============================================================
angles_rad = np.linspace(-HALF_EXTENT_DEG, HALF_EXTENT_DEG, N_VIEWS, endpoint=True) * np.pi / 180
optical_setup = config.define_optical_setup(
    sensor_locations=[(0.0, 0.0)],
    beam_angles=[list(angles_rad)],
    test_region_dims=test_region_dims,
    pixel_pitch=cm_per_pixel,
    beam_fov=beam_diam_cm,
)
ct_model, weights = sim.create_ct_model_and_weights_for_simulation(optical_setup)
ct_model.max_over_relaxation = MAX_OVER_RELAXATION

roi_zern = np.array(va.generate_beam_path_roi_mask(
    (ZERN_RESOLUTION, num_cols, num_slices), beam_pixel_diam, location=(0, 0, 0), angle=0
))

roi_per_res = {}
for s in RESOLUTIONS:
    roi_per_res[s] = np.array(va.generate_beam_path_roi_mask(
        (s, num_cols, num_slices), beam_pixel_diam, location=(0, 0, 0), angle=0
    ))

# ============================================================
# STORAGE
# ============================================================
n_meas   = len(MEAS_TYPES)
n_eval   = len(EVAL_TYPES)
n_res    = len(RESOLUTIONS)
n_zmodes = N_OSA_MODES + 1   # OSA modes 0–44 individually + total MSE

# nrmse_arr[vol, meas, eval, res]  — first N_NRMSE_VOLS volumes only
nrmse_arr = np.zeros((N_NRMSE_VOLS, n_meas, n_eval, n_res))
# zernike_mse_arr[vol, meas, zmode] — all N_VOLS, OPD_TT eval at ZERN_RESOLUTION
zernike_mse_arr = np.zeros((N_VOLS, n_meas, n_zmodes))

# ============================================================
# MAIN LOOP
# ============================================================
for vol_idx in trange(N_VOLS, desc='Volumes'):
    key = random.PRNGKey(vol_idx)
    vol_gt = sim.generate_random_atmospheric_volume(
        cn2=1e-11, dim=recon_shape, delta=delta, L0=0.02, key=key
    )

    for meas_idx, proj_type in enumerate(tqdm(MEAS_TYPES, desc='Meas', leave=False)):
        sinogram = sim.collect_projection_measurement(
            ct_model, weights, vol_gt, projection_type=proj_type
        )

        ct_model.max_over_relaxation = MAX_OVER_RELAXATION
        recon, _ = ct_model.recon(
            sinogram, weights=weights,
            max_iterations=MAX_ITERATIONS, stop_threshold_change_pct=STOP_THRESHOLD_PCT,
        )

        # ---- Zernike at ZERN_RESOLUTION (all vols) ----
        roi_z_jnp  = jnp.array(roi_zern)
        gt_z_raw   = np.array(va.divide_into_sections_of_opl(vol_gt, ZERN_RESOLUTION, 0.2))
        recon_z_raw = np.array(va.divide_into_sections_of_opl(recon, ZERN_RESOLUTION, 0.2))
        gt_z_ttp   = np.array(utils.remove_tip_tilt_piston(jnp.array(gt_z_raw),   FOV=roi_z_jnp))
        recon_z_ttp = np.array(utils.remove_tip_tilt_piston(jnp.array(recon_z_raw), FOV=roi_z_jnp))
        error_z    = gt_z_ttp - recon_z_ttp

        per_mode_mse = va.compute_osa_mode_mse_for_volume(
            error_z, roi_zern, max_j=N_OSA_MODES - 1
        )
        zernike_mse_arr[vol_idx, meas_idx, :N_OSA_MODES] = per_mode_mse
        zernike_mse_arr[vol_idx, meas_idx, -1] = float(np.mean(error_z[roi_zern] ** 2))

        # ---- NRMSE at all resolutions (first N_NRMSE_VOLS vols only) ----
        if vol_idx < N_NRMSE_VOLS:
            for res_idx, s in enumerate(RESOLUTIONS):
                roi_s     = roi_per_res[s]
                roi_s_jnp = jnp.array(roi_s)

                gt_s_raw    = np.array(va.divide_into_sections_of_opl(vol_gt, s, 0.2))
                recon_s_raw = np.array(va.divide_into_sections_of_opl(recon, s, 0.2))
                gt_s_ttp    = np.array(utils.remove_tip_tilt_piston(jnp.array(gt_s_raw),    FOV=roi_s_jnp))
                recon_s_ttp = np.array(utils.remove_tip_tilt_piston(jnp.array(recon_s_raw), FOV=roi_s_jnp))

                # OPL eval (eval_idx=0): NRMSE against OPL GT
                nrmse_arr[vol_idx, meas_idx, 0, res_idx] = float(
                    va.nrmse_over_roi(jnp.array(gt_s_raw), jnp.array(recon_s_raw), roi_s_jnp, option=2)
                )
                # OPD_TT eval (eval_idx=1): NRMSE against TTP-removed GT
                nrmse_arr[vol_idx, meas_idx, 1, res_idx] = float(
                    va.nrmse_over_roi(jnp.array(gt_s_ttp), jnp.array(recon_s_ttp), roi_s_jnp, option=2)
                )

    if (vol_idx + 1) % 50 == 0:
        np.savez(
            DATA_DIR / 'fig13_14_15_17_7v8_partial.npz',
            nrmse=nrmse_arr,
            zernike_mse=zernike_mse_arr,
            n_completed=vol_idx + 1,
            resolutions=np.array(RESOLUTIONS),
            meas_types=np.array(MEAS_TYPES),
            eval_types=np.array(EVAL_TYPES),
            n_nrmse_vols=N_NRMSE_VOLS,
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
    DATA_DIR / 'fig13_14_15_17_7v8.npz',
    nrmse=nrmse_arr,
    zernike_mse=zernike_mse_arr,
    resolutions=np.array(RESOLUTIONS),
    meas_types=np.array(MEAS_TYPES),
    eval_types=np.array(EVAL_TYPES),
    n_nrmse_vols=N_NRMSE_VOLS,
    n_osa_modes=N_OSA_MODES,
    zern_resolution=ZERN_RESOLUTION,
    max_over_relaxation=MAX_OVER_RELAXATION,
    max_iterations=MAX_ITERATIONS,
)
print('Done. Saved data/fig13_14_15_17_7v8.npz')
