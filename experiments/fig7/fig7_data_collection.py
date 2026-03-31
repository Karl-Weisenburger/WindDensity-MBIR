import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

import sys
import time
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
MAX_OVER_RELAXATION = 1.25
MAX_ITERATIONS = 20
STOP_THRESHOLD_PCT = 1
N_VOLS = 100

# Grid: full angular extents (degrees total sweep), num_views
# 'Full angular extent' = 2 × half_extent (e.g., 8° total = ±4°)
FULL_EXTENTS_DEG = [2, 4, 6, 8, 10, 12, 14, 16]
NUM_VIEWS_LIST   = [3, 5, 7, 9, 11]
RESOLUTIONS = [4, 640]   # 4-plane and full resolution only

# Performance tracking: record time, iterations, and final pct change for these geometries
PERF_GEOS = [
    ('3v2',   2,  3),   # 2° total, 3 views
    ('7v8',   8,  7),   # 8° total, 7 views
    ('11v16', 16, 11),  # 16° total, 11 views
]
PERF_GEO_KEYS = {(full_ext, n_views): i for i, (_, full_ext, n_views) in enumerate(PERF_GEOS)}


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
n_perf = len(PERF_GEOS)

nrmse_arr = np.zeros((N_VOLS, n_ext, n_view, n_res))

# Performance arrays for 3v2, 7v8, 11v16
perf_recon_time     = np.full((N_VOLS, n_perf), np.nan)
perf_num_iterations = np.zeros((N_VOLS, n_perf), dtype=int)
perf_final_pct      = np.full((N_VOLS, n_perf), np.nan)

# ============================================================
# MAIN LOOP
# ============================================================
for vol_idx in trange(N_VOLS, desc='Volumes'):
    key = random.PRNGKey(vol_idx)
    vol_gt = sim.generate_random_atmospheric_volume(
        cn2=1e-11, dim=recon_shape, delta=delta, L0=0.02, key=key
    )

    for ext_idx, full_ext in enumerate(tqdm(FULL_EXTENTS_DEG, desc='Extents', leave=False)):
        for view_idx, n_views in enumerate(NUM_VIEWS_LIST):
            ct_model, weights = geo_models[(full_ext, n_views)]

            sinogram = sim.collect_projection_measurement(
                ct_model, weights, vol_gt, projection_type='OPD_TT'
            )

            ct_model.max_over_relaxation = MAX_OVER_RELAXATION
            t0 = time.time()
            recon, recon_dict = ct_model.recon(
                sinogram, weights=weights,
                max_iterations=MAX_ITERATIONS, stop_threshold_change_pct=STOP_THRESHOLD_PCT,
            )
            elapsed = time.time() - t0

            # Record performance for tracked geometries
            geo_key = (full_ext, n_views)
            if geo_key in PERF_GEO_KEYS:
                pi = PERF_GEO_KEYS[geo_key]
                perf_recon_time[vol_idx, pi]     = elapsed
                perf_num_iterations[vol_idx, pi] = recon_dict['recon_params']['num_iterations']
                pct_list = recon_dict['recon_params']['stop_threshold_change_pct']
                perf_final_pct[vol_idx, pi]      = pct_list[-1] if pct_list else np.nan

            for res_idx, s in enumerate(RESOLUTIONS):
                roi_s     = roi_per_res[s]
                roi_s_jnp = jnp.array(roi_s)

                gt_s    = np.array(va.divide_into_sections_of_opl(vol_gt, s, 0.2))
                recon_s = np.array(va.divide_into_sections_of_opl(recon,  s, 0.2))

                # Remove TTP for OPD_TT comparison
                gt_s    = np.array(utils.remove_tip_tilt_piston(jnp.array(gt_s),    FOV=roi_s_jnp))
                recon_s = np.array(utils.remove_tip_tilt_piston(jnp.array(recon_s), FOV=roi_s_jnp))

                nrmse_arr[vol_idx, ext_idx, view_idx, res_idx] = float(
                    va.nrmse_over_roi(jnp.array(gt_s), jnp.array(recon_s), roi_s_jnp, option=2)
                )

    if (vol_idx + 1) % 10 == 0:
        np.savez(
            'data/fig7_geometry_sweep_partial.npz',
            nrmse=nrmse_arr,
            perf_recon_time=perf_recon_time,
            perf_num_iterations=perf_num_iterations,
            perf_final_pct=perf_final_pct,
            n_completed=vol_idx + 1,
            full_extents_deg=np.array(FULL_EXTENTS_DEG),
            num_views_list=np.array(NUM_VIEWS_LIST),
            resolutions=np.array(RESOLUTIONS),
            perf_geo_names=np.array([g[0] for g in PERF_GEOS]),
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
    perf_recon_time=perf_recon_time,
    perf_num_iterations=perf_num_iterations,
    perf_final_pct=perf_final_pct,
    full_extents_deg=np.array(FULL_EXTENTS_DEG),
    num_views_list=np.array(NUM_VIEWS_LIST),
    resolutions=np.array(RESOLUTIONS),
    perf_geo_names=np.array([g[0] for g in PERF_GEOS]),
    max_over_relaxation=MAX_OVER_RELAXATION,
    max_iterations=MAX_ITERATIONS,
)
print('Done. Saved data/fig7_geometry_sweep.npz')

# ============================================================
# TABLE 1 PRINTOUT
# 3×3 table: geometry × metric (recon time, VCD iterations, final pct change)
# OPD_TT measurements only
# ============================================================
col_w = 14
geo_col_w = 8
print('\n')
print('=' * 60)
print('Table 1: Reconstruction Performance (OPD_TT measurements)')
print('         Mean ± std across {:d} volumes'.format(N_VOLS))
print('=' * 60)
header = (f"{'Geometry':<{geo_col_w}}"
          f"{'Recon Time (s)':>{col_w}}"
          f"{'VCD Iterations':>{col_w}}"
          f"{'Final Pct Chg (%)':>{col_w+3}}")
print(header)
print('-' * 60)
for pi, (geo_name, _, _) in enumerate(PERF_GEOS):
    mask = ~np.isnan(perf_recon_time[:, pi])
    if mask.any():
        t_mean  = np.mean(perf_recon_time[mask, pi])
        t_std   = np.std(perf_recon_time[mask, pi], ddof=1)
        i_mean  = np.mean(perf_num_iterations[mask, pi])
        i_std   = np.std(perf_num_iterations[mask, pi].astype(float), ddof=1)
        p_mean  = np.mean(perf_final_pct[mask, pi])
        p_std   = np.std(perf_final_pct[mask, pi], ddof=1)
        row = (f"{geo_name:<{geo_col_w}}"
               f"{f'{t_mean:.1f} ± {t_std:.1f}':>{col_w}}"
               f"{f'{i_mean:.1f} ± {i_std:.1f}':>{col_w}}"
               f"{f'{p_mean:.3f} ± {p_std:.3f}':>{col_w+3}}")
        print(row)
print('=' * 60)
