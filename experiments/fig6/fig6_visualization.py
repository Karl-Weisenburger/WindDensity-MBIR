"""
Fig 6: MBIR vs scale-corrected FBP — 4 OPD_TT planes, 7v8 geometry.

Rows (top to bottom): GT, MBIR, Scale-Corrected FBP
Columns: 4 depth sections
Per-subplot title: NRMSE (%) for that section

Volume and reconstructions are cached to data/ on the first run.
"""

import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

import winddensity_mbir.configuration_params as config
import winddensity_mbir.simulation as sim
import winddensity_mbir.utilities as utils
from experiments.recon_visualization_prep import (
    load_or_generate_volume,
    prepare_opl_images,
    compute_per_section_nrmse,
    compute_overall_nrmse,
    plot_recon_figure,
)

# ============================================================
# Parameters
# ============================================================
SEED            = 17
CM_PER_PIXEL    = 25.0 / 800
RECON_SHAPE     = (640, 400, 64)
NUM_ROWS, NUM_COLS, NUM_SLICES = RECON_SHAPE
TEST_REGION_DIMS = (NUM_ROWS * CM_PER_PIXEL, NUM_COLS * CM_PER_PIXEL, NUM_SLICES * CM_PER_PIXEL)
TOTAL_LENGTH_M  = NUM_ROWS * CM_PER_PIXEL / 100.0   # 0.20 m
BEAM_DIAM_CM    = 2.0
BEAM_PIXEL_DIAM = int(round(BEAM_DIAM_CM / CM_PER_PIXEL))   # 64
DELTA           = 0.01 * CM_PER_PIXEL
CN2             = 1e-11
L0              = 0.02
SECTIONS                  = 4
ZERN_MODE_INDEX           = 2    # OPD_TT (piston + tip + tilt removed)
MAX_OVER_RELAXATION       = 1.25
MAX_ITERATIONS            = 20
STOP_THRESHOLD_CHANGE_PCT = 1

# 7v8 geometry
ANGLE_EXTENT_DEG = 4.0   # half-extent
N_VIEWS          = 7

OUT_DIR  = Path(__file__).parent / 'data'
VOL_PATH = Path(__file__).parents[1] / 'shared_data' / f'vol_seed{SEED}.npy'

# ============================================================
# Geometry setup
# ============================================================
angles_rad = np.linspace(-ANGLE_EXTENT_DEG, ANGLE_EXTENT_DEG, N_VIEWS, endpoint=True) * np.pi / 180
optical_setup = config.define_optical_setup(
    sensor_locations=[(0.0, 0.0)],
    beam_angles=[list(angles_rad)],
    test_region_dims=TEST_REGION_DIMS,
    pixel_pitch=CM_PER_PIXEL,
    beam_fov=BEAM_DIAM_CM,
)
ct_model, weights = sim.create_ct_model_and_weights_for_simulation(optical_setup)
ct_model.max_over_relaxation = MAX_OVER_RELAXATION


def _load_or_run_recon(cache_path, compute_fn):
    cache_path = Path(cache_path)
    if cache_path.exists():
        print(f'Loading cached recon from {cache_path}')
        return np.load(cache_path)
    print(f'Running reconstruction → {cache_path.name}…')
    recon = compute_fn()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, np.array(recon))
    print(f'Saved to {cache_path}')
    return np.array(recon)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Volume ---
    vol_gt = load_or_generate_volume(
        SEED, RECON_SHAPE, DELTA, L0, CN2, VOL_PATH
    )
    vol_jnp = jnp.array(vol_gt)

    # --- Sinogram (OPD_TT) ---
    sinogram = sim.collect_projection_measurement(ct_model, weights, vol_jnp, projection_type='OPD_TT')

    # --- MBIR ---
    def run_mbir():
        recon, _ = ct_model.recon(
            sinogram, weights=weights,
            max_iterations=MAX_ITERATIONS,
            stop_threshold_change_pct=STOP_THRESHOLD_CHANGE_PCT,
        )
        return recon

    recon_mbir = _load_or_run_recon(OUT_DIR / f'recon_seed{SEED}_7v8_MBIR.npy', run_mbir)

    # --- Scale-corrected FBP ---
    def run_fbp_scaled():
        recon_fbp = ct_model.direct_recon(sinogram)
        return utils.correct_recon_scaling(recon_fbp, ct_model, sinogram, weights)

    recon_fbp = _load_or_run_recon(OUT_DIR / f'recon_seed{SEED}_7v8_FBP_scaled.npy', run_fbp_scaled)

    # --- OPL image preparation ---
    gt_images,    roi_beam = prepare_opl_images(
        vol_gt, ZERN_MODE_INDEX, SECTIONS, TOTAL_LENGTH_M,
        BEAM_PIXEL_DIAM, NUM_COLS, NUM_SLICES,
    )
    mbir_images,  _        = prepare_opl_images(
        recon_mbir, ZERN_MODE_INDEX, SECTIONS, TOTAL_LENGTH_M,
        BEAM_PIXEL_DIAM, NUM_COLS, NUM_SLICES,
    )
    fbp_images,   _        = prepare_opl_images(
        recon_fbp, ZERN_MODE_INDEX, SECTIONS, TOTAL_LENGTH_M,
        BEAM_PIXEL_DIAM, NUM_COLS, NUM_SLICES,
    )

    # --- Per-section NRMSE ---
    nrmse_mbir = compute_per_section_nrmse(gt_images, mbir_images, roi_beam)
    nrmse_fbp  = compute_per_section_nrmse(gt_images, fbp_images,  roi_beam)

    # --- Full-resolution NRMSE ---
    gt_full,   roi_full = prepare_opl_images(
        vol_gt,     ZERN_MODE_INDEX, NUM_ROWS, TOTAL_LENGTH_M, BEAM_PIXEL_DIAM, NUM_COLS, NUM_SLICES,
    )
    mbir_full, _        = prepare_opl_images(
        recon_mbir, ZERN_MODE_INDEX, NUM_ROWS, TOTAL_LENGTH_M, BEAM_PIXEL_DIAM, NUM_COLS, NUM_SLICES,
    )
    fbp_full,  _        = prepare_opl_images(
        recon_fbp,  ZERN_MODE_INDEX, NUM_ROWS, TOTAL_LENGTH_M, BEAM_PIXEL_DIAM, NUM_COLS, NUM_SLICES,
    )

    nrmse_mbir_full     = compute_overall_nrmse(gt_full,   mbir_full, roi_full)
    nrmse_fbp_full      = compute_overall_nrmse(gt_full,   fbp_full,  roi_full)
    nrmse_mbir_sections = compute_overall_nrmse(gt_images, mbir_images, roi_beam)
    nrmse_fbp_sections  = compute_overall_nrmse(gt_images, fbp_images,  roi_beam)

    w = 30
    print(f'\n--- Fig 6 NRMSE Summary ---')
    print(f'{"":>{w}}  Full-res NRMSE  {SECTIONS}-section NRMSE')
    print(f'{"MBIR (7v8):":<{w}}  {nrmse_mbir_full:.4f}          {nrmse_mbir_sections:.4f}')
    print(f'{"Scale-Corrected FBP (7v8):":<{w}}  {nrmse_fbp_full:.4f}          {nrmse_fbp_sections:.4f}')

    # --- Plot ---
    fig = plot_recon_figure(
        gt_images              = gt_images,
        recon_images_list      = [mbir_images, fbp_images],
        nrmse_per_section_list = [nrmse_mbir, nrmse_fbp],
        recon_labels           = [
            r'WindDensity-MBIR Reconstruction with 7v,8$^\circ$-geometry',
            r'Scale-Corrected FBP Reconstruction with 7v,8$^\circ$-geometry',
        ],
        gt_suptitle            = '$OPD_{TT}$ Ground Truth Planes',
        fig_suptitle           = 'WindDensity-MBIR vs Scale-Corrected FBP',
        roi_beam               = roi_beam,
    )

    for ext in ('pdf', 'png'):
        out = OUT_DIR / f'fig6_mbir_vs_fbp.{ext}'
        fig.savefig(out, bbox_inches='tight', dpi=200)
        print(f'Saved {out}')
    plt.show()


if __name__ == '__main__':
    main()
