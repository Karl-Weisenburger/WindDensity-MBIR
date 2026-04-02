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

N_VOLS = 1000

# Fixed resolution for Zernike analysis (figs 10, 11)
ZERN_RESOLUTION = 4
N_OSA_MODES = 45     # OSA/ANSI indices 0–44 (radial degrees 0–8)

# Geometries: (label, half_extent_deg, n_views)
GEOMETRIES = [
    ('3v2',   1.0,  3),
    ('11v16', 8.0, 11),
]


# ============================================================
# BUILD GEOMETRIES
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
    return ct_model, weights


print('Building geometries...')
geo_models = {}
for label, half_ext, n_views in GEOMETRIES:
    geo_models[label] = build_geometry(half_ext, n_views)

roi_zern = np.array(va.generate_beam_path_roi_mask(
    (ZERN_RESOLUTION, num_cols, num_slices), beam_pixel_diam, location=(0, 0, 0), angle=0
))

# ============================================================
# STORAGE
# ============================================================
n_geos   = len(GEOMETRIES)
n_zmodes = N_OSA_MODES + 1   # OSA modes 0–44 individually + total MSE

# zernike_mse_arr[vol, geo, zmode]
zernike_mse_arr = np.zeros((N_VOLS, n_geos, n_zmodes))

# ============================================================
# MAIN LOOP
# ============================================================
for vol_idx in trange(N_VOLS, desc='Volumes'):
    key = random.PRNGKey(vol_idx)
    vol_gt = sim.generate_random_atmospheric_volume(
        cn2=1e-11, dim=recon_shape, delta=delta, L0=0.02, key=key
    )

    for geo_idx, (geo_label, half_ext, n_views) in enumerate(tqdm(GEOMETRIES, desc='Geometries', leave=False)):
        ct_model, weights = geo_models[geo_label]

        sinogram = sim.collect_projection_measurement(
            ct_model, weights, vol_gt, projection_type='OPD_TT'
        )

        ct_model.max_over_relaxation = MAX_OVER_RELAXATION
        recon, _ = ct_model.recon(
            sinogram, weights=weights,
            max_iterations=MAX_ITERATIONS, stop_threshold_change_pct=STOP_THRESHOLD_PCT,
        )

        # Zernike error: raw sections, no TTP removal — all modes including tip/tilt/piston
        gt_z    = np.array(va.divide_into_sections_of_opl(vol_gt, ZERN_RESOLUTION, 0.2))
        recon_z = np.array(va.divide_into_sections_of_opl(recon,  ZERN_RESOLUTION, 0.2))
        error_z = gt_z - recon_z

        per_mode_mse = va.compute_osa_mode_mse_for_volume(
            error_z, roi_zern, max_j=N_OSA_MODES - 1
        )
        zernike_mse_arr[vol_idx, geo_idx, :N_OSA_MODES] = per_mode_mse
        zernike_mse_arr[vol_idx, geo_idx, -1] = float(np.mean(error_z[roi_zern] ** 2))

    if (vol_idx + 1) % 50 == 0:
        np.savez(
            DATA_DIR / 'fig10_11_zernike_partial.npz',
            zernike_mse=zernike_mse_arr,
            n_completed=vol_idx + 1,
            geometry_names=np.array([g[0] for g in GEOMETRIES]),
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
    DATA_DIR / 'fig10_11_zernike.npz',
    zernike_mse=zernike_mse_arr,
    geometry_names=np.array([g[0] for g in GEOMETRIES]),
    n_osa_modes=N_OSA_MODES,
    zern_resolution=ZERN_RESOLUTION,
    max_over_relaxation=MAX_OVER_RELAXATION,
    max_iterations=MAX_ITERATIONS,
)
print('Done. Saved data/fig10_11_zernike.npz')
