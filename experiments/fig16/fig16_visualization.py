"""
Figs 16a and 16b: MBIR — OPL vs OPD_TT measurement comparison, 7v8 geometry.

Fig 16a: GT + OPL-measurement recon + OPD_TT-measurement recon — OPL planes   (zern_mode_index=0)
Fig 16b: GT + OPL-measurement recon + OPD_TT-measurement recon — OPD_TT planes (zern_mode_index=2)

Both use the same volume (seed=17) and 7v8 geometry. The two panels are
placed side-by-side in a single combined figure with (a)/(b) labels
underneath, matching the Fig 7 layout.

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
SECTIONS        = 4
MAX_OVER_RELAXATION       = 1.5
MAX_ITERATIONS            = 15
STOP_THRESHOLD_CHANGE_PCT = 0.2

# 7v8 geometry
ANGLE_EXTENT_DEG = 4.0
N_VIEWS          = 7

RECON_CACHE_DIR = Path(__file__).parent / 'data'
OUT_DIR         = Path(__file__).parent / 'figures'
VOL_PATH        = Path(__file__).parents[1] / 'shared_data' / f'vol_seed{SEED}.npy'

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


# ============================================================
# Per-panel setup: prepare images, NRMSE, and labels
# ============================================================
def _panel_inputs(vol_gt, recon_opl, recon_opdtt, zern_mode_index, plane_name, fig_id):
    gt_images, roi_beam = prepare_opl_images(
        vol_gt, zern_mode_index, SECTIONS, TOTAL_LENGTH_M,
        BEAM_PIXEL_DIAM, NUM_COLS, NUM_SLICES,
    )

    m1_type = ['OPL', 'OPD', 'OPD_{TT}']
    m2_type = ['OPL', 'OPD', r'$\text{OPD}_{\text{TT}}$']

    recon_images_list = []
    nrmse_list        = []
    recon_labels      = []
    for recon_3d, meas_str in [(recon_opl, 'OPL'), (recon_opdtt, r'$OPD_{TT}$')]:
        r_imgs, _ = prepare_opl_images(
            recon_3d, zern_mode_index, SECTIONS, TOTAL_LENGTH_M,
            BEAM_PIXEL_DIAM, NUM_COLS, NUM_SLICES,
        )
        recon_images_list.append(r_imgs)
        nrmse_list.append(compute_per_section_nrmse(gt_images, r_imgs, roi_beam))
        recon_labels.append(f'${m1_type[zern_mode_index]}$ with {meas_str} Measurements')

    # Full-resolution NRMSE printout
    gt_full, roi_full = prepare_opl_images(
        vol_gt, zern_mode_index, NUM_ROWS, TOTAL_LENGTH_M, BEAM_PIXEL_DIAM, NUM_COLS, NUM_SLICES,
    )
    w = 10
    print(f'\n--- Fig {fig_id} NRMSE Summary ({plane_name} planes) ---')
    print(f'{"Measurement":<{w}}  Full-res NRMSE  {SECTIONS}-section NRMSE')
    for (recon_3d, meas_str), r_imgs in zip(
        [(recon_opl, 'OPL'), (recon_opdtt, 'OPD_TT')], recon_images_list
    ):
        recon_full, _ = prepare_opl_images(
            recon_3d, zern_mode_index, NUM_ROWS, TOTAL_LENGTH_M, BEAM_PIXEL_DIAM, NUM_COLS, NUM_SLICES,
        )
        nrmse_full     = compute_overall_nrmse(gt_full,   recon_full, roi_full)
        nrmse_sections = compute_overall_nrmse(gt_images, r_imgs,     roi_beam)
        print(f'{meas_str:<{w}}  {nrmse_full:.4f}          {nrmse_sections:.4f}')

    return dict(
        gt_images              = gt_images,
        recon_images_list      = recon_images_list,
        nrmse_per_section_list = nrmse_list,
        recon_labels           = recon_labels,
        gt_suptitle            = f'${m1_type[zern_mode_index]}$ Ground Truth Planes',
        fig_suptitle           = f'{m2_type[zern_mode_index]} Reconstruction and Model Mismatch',
        roi_beam               = roi_beam,
    )


def main():
    RECON_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Volume ---
    vol_gt  = load_or_generate_volume(SEED, RECON_SHAPE, DELTA, L0, CN2, VOL_PATH)
    vol_jnp = jnp.array(vol_gt)

    # --- Two sinograms (OPL and OPD_TT) ---
    sino_opl   = sim.collect_projection_measurement(ct_model, weights, vol_jnp, projection_type='OPL')
    sino_opdtt = sim.collect_projection_measurement(ct_model, weights, vol_jnp, projection_type='OPD_TT')

    # --- MBIR reconstructions ---
    def run_opl_recon():
        recon, _ = ct_model.recon(
            sino_opl, weights=weights,
            max_iterations=MAX_ITERATIONS,
            stop_threshold_change_pct=STOP_THRESHOLD_CHANGE_PCT,
        )
        return recon

    def run_opdtt_recon():
        recon, _ = ct_model.recon(
            sino_opdtt, weights=weights,
            max_iterations=MAX_ITERATIONS,
            stop_threshold_change_pct=STOP_THRESHOLD_CHANGE_PCT,
        )
        return recon

    recon_opl   = _load_or_run_recon(RECON_CACHE_DIR / f'recon_seed{SEED}_7v8_OPL.npy',    run_opl_recon)
    recon_opdtt = _load_or_run_recon(RECON_CACHE_DIR / f'recon_seed{SEED}_7v8_OPD_TT.npy', run_opdtt_recon)

    # --- Panel inputs ---
    panel_a = _panel_inputs(vol_gt, recon_opl, recon_opdtt, zern_mode_index=0, plane_name='OPL',    fig_id='16a')
    panel_b = _panel_inputs(vol_gt, recon_opl, recon_opdtt, zern_mode_index=2, plane_name='OPD_TT', fig_id='16b')

    # --- Combined figure: two columns side by side, each the size of a
    #     stand-alone panel, with (a)/(b) labels underneath. ---
    sections   = panel_a['gt_images'].shape[0]
    n_recons   = len(panel_a['recon_images_list'])
    panel_w    = sections * 4.3
    panel_h    = (n_recons + 1) * 4.9
    combined_figsize = (panel_w * 2, panel_h)

    fig = plt.figure(figsize=combined_figsize, constrained_layout=True)
    column_subfigs = fig.subfigures(nrows=1, ncols=2)

    for col_subfig, panel, label in zip(column_subfigs, (panel_a, panel_b), ('(a)', '(b)')):
        plot_recon_figure(parent=col_subfig, **panel)
        col_subfig.text(0.5, -0.02, label, ha='center', va='top', fontsize=41)

    for ext in ('pdf', 'png'):
        out = OUT_DIR / f'fig16_measurement_comparison.{ext}'
        fig.savefig(out, bbox_inches='tight', dpi=200)
        print(f'Saved {out}')
    plt.show()


if __name__ == '__main__':
    main()
