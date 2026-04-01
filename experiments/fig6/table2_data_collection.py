"""
Table 2 data collection: NRMSE comparison of regular FBP, scale-corrected FBP,
and WindDensity-MBIR for the three paper geometries (3v2, 7v8, 11v16).

Metric: NRMSE of 4 OPD_TT planes (TTP removed from both ground truth and reconstruction).

Output: data/table2_fbp_comparison.npz
  nrmse:          (N_VOLS, n_geos, n_methods)  float
  geometry_names: list of strings ['3v2', '7v8', '11v16']
  method_names:   list of strings ['FBP', 'FBP_scaled', 'MBIR']
  n_completed:    int (for incremental saves)
"""
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
total_length_m = num_rows * cm_per_pixel / 100.0  # 0.20 m

MAX_OVER_RELAXATION = 1.25
MAX_ITERATIONS = 20
STOP_THRESHOLD_PCT = 1
N_VOLS = 100

# OPD_TT planes to evaluate
N_SECTIONS = 4

# Geometries for Table 2 only (not a sweep — fixed 3 points)
GEOMETRIES = [
    ('3v2',   1.0,  3),
    ('7v8',   4.0,  7),
    ('11v16', 8.0, 11),
]
METHOD_NAMES = ['FBP', 'FBP_scaled', 'MBIR']

# ============================================================
# BUILD GEOMETRY MODELS
# ============================================================
def build_geometry(half_extent_deg, n_views):
    angles_rad = np.linspace(-half_extent_deg, half_extent_deg, n_views, endpoint=True) * np.pi / 180
    optical_setup = config.define_optical_setup(
        sensor_locations=[(0.0, 0.0)],
        beam_angles=[list(angles_rad)],
        test_region_dims=test_region_dims,
        pixel_pitch=cm_per_pixel,
        beam_fov=beam_diam_cm,
    )
    ct_model, weights = sim.create_ct_model_and_weights_for_simulation(optical_setup)
    ct_model.max_over_relaxation = MAX_OVER_RELAXATION
    ct_model.set_params(sharpness=2, verbose=0, p=2, q=2)
    return ct_model, weights


print('Building geometry models...')
geo_models = {}
for label, half_ext, n_views in GEOMETRIES:
    geo_models[label] = build_geometry(half_ext, n_views)

# ROI for 4 OPD_TT sections
roi_sections = np.array(va.generate_beam_path_roi_mask(
    (N_SECTIONS, num_cols, num_slices), beam_pixel_diam
))

# ============================================================
# HELPER: scale-corrected FBP
# ============================================================
def fbp_scaled(ct_model, sinogram, weights):
    recon = ct_model.direct_recon(sinogram)
    return utils.correct_recon_scaling(recon, ct_model, sinogram, weights)


# ============================================================
# HELPER: compute 4 OPD_TT sections and NRMSE
# ============================================================
def sections_nrmse(gt_3d, recon_3d, roi_s):
    gt_s = va.divide_into_sections_of_opl(gt_3d, N_SECTIONS, total_length_m)
    recon_s = va.divide_into_sections_of_opl(recon_3d, N_SECTIONS, total_length_m)
    roi_jnp = jnp.array(roi_s)
    gt_s = utils.remove_tip_tilt_piston(gt_s, FOV=roi_jnp)
    recon_s = utils.remove_tip_tilt_piston(recon_s, FOV=roi_jnp)
    return float(va.nrmse_over_roi(gt_s, recon_s, roi_jnp, option=2))


# ============================================================
# STORAGE
# ============================================================
n_geos = len(GEOMETRIES)
n_methods = len(METHOD_NAMES)
nrmse_arr = np.zeros((N_VOLS, n_geos, n_methods))

# ============================================================
# MAIN LOOP
# ============================================================
for vol_idx in trange(N_VOLS, desc='Volumes'):
    key = random.PRNGKey(vol_idx)
    vol_gt = sim.generate_random_atmospheric_volume(
        cn2=1e-11, dim=recon_shape, delta=delta, L0=0.02, key=key
    )

    for geo_idx, (geo_label, half_ext, n_views) in enumerate(tqdm(GEOMETRIES, desc='Geos', leave=False)):
        ct_model, weights = geo_models[geo_label]

        # OPD_TT sinogram (TTP removed) — used by all three methods
        sinogram_opd = sim.collect_projection_measurement(
            ct_model, weights, vol_gt, projection_type='OPD_TT'
        )

        # Method 0: regular FBP (direct_recon, no scale correction)
        recon_fbp = ct_model.direct_recon(sinogram_opd)
        nrmse_arr[vol_idx, geo_idx, 0] = sections_nrmse(vol_gt, recon_fbp, roi_sections)

        # Method 1: scale-corrected FBP
        recon_fbp_scaled = fbp_scaled(ct_model, sinogram_opd, weights)
        nrmse_arr[vol_idx, geo_idx, 1] = sections_nrmse(vol_gt, recon_fbp_scaled, roi_sections)

        # Method 2: MBIR
        ct_model.max_over_relaxation = MAX_OVER_RELAXATION
        recon_mbir, _ = ct_model.recon(
            sinogram_opd, weights=weights,
            max_iterations=MAX_ITERATIONS, stop_threshold_change_pct=STOP_THRESHOLD_PCT,
        )
        nrmse_arr[vol_idx, geo_idx, 2] = sections_nrmse(vol_gt, recon_mbir, roi_sections)

    # Incremental save every 10 volumes
    if (vol_idx + 1) % 10 == 0:
        np.savez(
            DATA_DIR / 'table2_fbp_comparison_partial.npz',
            nrmse=nrmse_arr,
            n_completed=vol_idx + 1,
            geometry_names=np.array([g[0] for g in GEOMETRIES]),
            method_names=np.array(METHOD_NAMES),
            max_over_relaxation=MAX_OVER_RELAXATION,
            max_iterations=MAX_ITERATIONS,
            n_sections=N_SECTIONS,
        )

# ============================================================
# FINAL SAVE
# ============================================================
np.savez(
    DATA_DIR / 'table2_fbp_comparison.npz',
    nrmse=nrmse_arr,
    geometry_names=np.array([g[0] for g in GEOMETRIES]),
    method_names=np.array(METHOD_NAMES),
    max_over_relaxation=MAX_OVER_RELAXATION,
    max_iterations=MAX_ITERATIONS,
    n_sections=N_SECTIONS,
    n_vols=N_VOLS,
)
print('Done. Saved data/table2_fbp_comparison.npz')

# Quick summary printout
print('\n--- Mean NRMSE summary (4 OPD_TT planes) ---')
geo_names = [g[0] for g in GEOMETRIES]
for geo_idx, geo_name in enumerate(geo_names):
    print(f'  {geo_name}:')
    for m_idx, m_name in enumerate(METHOD_NAMES):
        mean_val = np.mean(nrmse_arr[:, geo_idx, m_idx])
        print(f'    {m_name:12s}: {mean_val:.4f}  ({mean_val * 100:.2f}%)')
