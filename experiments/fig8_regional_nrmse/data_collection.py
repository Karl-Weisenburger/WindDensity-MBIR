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
N_VOLS = 3000

ROW_PAD     = 10          # skip this many rows at each end
N_SECTIONS  = 8           # divide valid rows into this many sections

# Corner-case geometries: (label, half_extent_deg, n_views)
GEOMETRIES = [
    ('3v2',  1.0, 3),    # 3 views, ±1° — narrow
    ('3v16', 8.0, 3),    # 3 views, ±8° — wide
]
TTP_STATES = ['withTTP', 'noTTP']   # OPL, OPD_TT


# ============================================================
# BUILD GEOMETRIES
# ============================================================
print('Building geometries...')
geo_models = {}
for label, half_ext, n_views in GEOMETRIES:
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
    geo_models[label] = (ct_model, weights)

# Full-volume beam ROI
roi_full = np.array(va.generate_beam_path_roi_mask(recon_shape, beam_pixel_diam, location=(0, 0, 0), angle=0))
roi_full_jnp = jnp.array(roi_full)

# Row section boundaries (excluding padded rows)
valid_row_start = ROW_PAD
valid_row_end   = num_rows - ROW_PAD   # 630
valid_rows      = np.arange(valid_row_start, valid_row_end)
section_indices = np.array_split(valid_rows, N_SECTIONS)
section_bounds  = [(rows[0], rows[-1] + 1) for rows in section_indices]  # (start, end) inclusive end

# ============================================================
# STORAGE
# ============================================================
n_geos     = len(GEOMETRIES)
n_ttp      = len(TTP_STATES)

nrmse_regional = np.zeros((N_VOLS, n_ttp, n_geos, N_SECTIONS))

# ============================================================
# MAIN LOOP
# ============================================================
for vol_idx in trange(N_VOLS, desc='Volumes'):
    key = random.PRNGKey(vol_idx)
    vol_gt = sim.generate_random_atmospheric_phase_volume(
        r0=0.05, dim=recon_shape, delta=delta, L0=0.02, key=key
    )
    vol_gt_np = np.array(vol_gt)

    for geo_idx, (geo_label, half_ext, n_views) in enumerate(tqdm(GEOMETRIES, desc='Geometries', leave=False)):
        ct_model, weights = geo_models[geo_label]

        for ttp_idx, proj_type in enumerate(['OPL', 'OPD_TT']):
            sinogram = sim.collect_projection_measurement(
                ct_model, weights, vol_gt, projection_type=proj_type
            )

            recon, _ = ct_model.recon(
                sinogram, weights=weights, init_recon=jnp.zeros(recon_shape),
                max_iterations=MAX_ITERATIONS, stop_threshold_change_pct=STOP_THRESHOLD_PCT,
            )
            recon_np = np.array(recon)

            # For noTTP: remove TTP from full volumes before sectioning
            if proj_type == 'OPD_TT':
                gt_cmp = np.array(utils.remove_tip_tilt_piston(vol_gt, FOV=roi_full_jnp))
                recon_cmp = np.array(utils.remove_tip_tilt_piston(recon, FOV=roi_full_jnp))
            else:
                gt_cmp = vol_gt_np
                recon_cmp = recon_np

            # Per-section NRMSE along the row axis
            for sec_idx, (r_start, r_end) in enumerate(section_bounds):
                gt_sec    = gt_cmp[r_start:r_end]
                recon_sec = recon_cmp[r_start:r_end]
                roi_sec   = roi_full[r_start:r_end]

                nrmse_regional[vol_idx, ttp_idx, geo_idx, sec_idx] = float(
                    va.nrmse_over_roi(
                        jnp.array(gt_sec), jnp.array(recon_sec), jnp.array(roi_sec), option=2
                    )
                )

    if (vol_idx + 1) % 100 == 0:
        np.savez(
            'data/fig8_regional_nrmse_partial.npz',
            nrmse_regional=nrmse_regional,
            n_completed=vol_idx + 1,
            geometry_names=np.array([g[0] for g in GEOMETRIES]),
            ttp_states=np.array(TTP_STATES),
            section_bounds=np.array(section_bounds),
            row_pad=ROW_PAD,
            n_sections=N_SECTIONS,
            max_over_relaxation=MAX_OVER_RELAXATION,
            max_iterations=MAX_ITERATIONS,
            stop_threshold_pct=STOP_THRESHOLD_PCT,
        )

# ============================================================
# FINAL SAVE
# ============================================================
np.savez(
    'data/fig8_regional_nrmse.npz',
    nrmse_regional=nrmse_regional,
    geometry_names=np.array([g[0] for g in GEOMETRIES]),
    ttp_states=np.array(TTP_STATES),
    section_bounds=np.array(section_bounds),
    row_pad=ROW_PAD,
    n_sections=N_SECTIONS,
    max_over_relaxation=MAX_OVER_RELAXATION,
    max_iterations=MAX_ITERATIONS,
)
print('Done. Saved data/fig8_regional_nrmse.npz')
