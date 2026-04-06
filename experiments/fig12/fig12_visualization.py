"""
Figs 12a and 12b: MBIR reconstructions — 3v2 vs 11v16 geometry comparison.

Fig 12a: GT + 11v16 recon + 3v2 recon — OPL planes  (zern_mode_index=0)
Fig 12b: GT + 11v16 recon + 3v2 recon — OPD_TT planes (zern_mode_index=2)

Both use the same volume (seed=17) and MBIR reconstructions with OPD_TT
sinograms. The two panels are placed side-by-side in a single combined
figure with (a)/(b) labels underneath, matching the Fig 7 layout.

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

# Two geometries: (label, half_extent_deg, n_views)
GEOMETRIES = [
    ('3v2',   1.0,  3),
    ('11v16', 8.0, 11),
]

RECON_CACHE_DIR = Path(__file__).parent / 'data'
OUT_DIR         = Path(__file__).parent / 'figures'
VOL_PATH        = Path(__file__).parents[1] / 'shared_data' / f'vol_seed{SEED}.npy'

# ============================================================
# Build geometry models
# ============================================================

def _build_geometry(half_extent_deg, n_views):
    angles_rad = np.linspace(-half_extent_deg, half_extent_deg, n_views, endpoint=True) * np.pi / 180
    optical_setup = config.define_optical_setup(
        sensor_locations=[(0.0, 0.0)],
        beam_angles=[list(angles_rad)],
        test_region_dims=TEST_REGION_DIMS,
        pixel_pitch=CM_PER_PIXEL,
        beam_fov=BEAM_DIAM_CM,
    )
    ct_model, weights = sim.create_ct_model_and_weights_for_simulation(optical_setup)
    ct_model.max_over_relaxation = MAX_OVER_RELAXATION
    return ct_model, weights


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
def _panel_inputs(vol_gt, recons, zern_mode_index, plane_name, fig_id):
    """Build every input plot_recon_figure needs for one panel (a or b)."""
    gt_images, roi_beam = prepare_opl_images(
        vol_gt, zern_mode_index, SECTIONS, TOTAL_LENGTH_M,
        BEAM_PIXEL_DIAM, NUM_COLS, NUM_SLICES,
    )

    # Order: 11v16 first, then 3v2
    recon_images_list = []
    nrmse_list        = []
    recon_labels = [
        r'11v,$16^\circ$-geometry with WindDensity-MBIR and $OPD_{TT}$ Measurements',
        r'3v,$2^\circ$-geometry with WindDensity-MBIR and $OPD_{TT}$ Measurements',
    ]
    for geo_label in ['11v16', '3v2']:
        r_imgs, _ = prepare_opl_images(
            recons[geo_label], zern_mode_index, SECTIONS, TOTAL_LENGTH_M,
            BEAM_PIXEL_DIAM, NUM_COLS, NUM_SLICES,
        )
        recon_images_list.append(r_imgs)
        nrmse_list.append(compute_per_section_nrmse(gt_images, r_imgs, roi_beam))

    # Full-resolution NRMSE printout
    gt_full, roi_full = prepare_opl_images(
        vol_gt, zern_mode_index, NUM_ROWS, TOTAL_LENGTH_M, BEAM_PIXEL_DIAM, NUM_COLS, NUM_SLICES,
    )
    w = 12
    print(f'\n--- Fig {fig_id} NRMSE Summary ({plane_name} planes) ---')
    print(f'{"Geometry":<{w}}  Full-res NRMSE  {SECTIONS}-section NRMSE')
    for geo_label, r_imgs in zip(['11v16', '3v2'], recon_images_list):
        recon_full, _ = prepare_opl_images(
            recons[geo_label], zern_mode_index, NUM_ROWS, TOTAL_LENGTH_M, BEAM_PIXEL_DIAM, NUM_COLS, NUM_SLICES,
        )
        nrmse_full     = compute_overall_nrmse(gt_full,   recon_full, roi_full)
        nrmse_sections = compute_overall_nrmse(gt_images, r_imgs,     roi_beam)
        print(f'{geo_label:<{w}}  {nrmse_full:.4f}          {nrmse_sections:.4f}')

    m1_type = ['OPL', 'OPD', 'OPD_{TT}']
    m2_type = ['OPL', 'OPD', r'$\text{OPD}_{\text{TT}}$']
    return dict(
        gt_images              = gt_images,
        recon_images_list      = recon_images_list,
        nrmse_per_section_list = nrmse_list,
        recon_labels           = recon_labels,
        gt_suptitle            = f'${m1_type[zern_mode_index]}$ Ground Truth Planes',
        fig_suptitle           = f'Effect of Geometry: 4 {m2_type[zern_mode_index]} Planes',
        roi_beam               = roi_beam,
    )


def main():
    RECON_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Volume ---
    vol_gt  = load_or_generate_volume(SEED, RECON_SHAPE, DELTA, L0, CN2, VOL_PATH)
    vol_jnp = jnp.array(vol_gt)

    # --- Reconstructions for each geometry ---
    recons = {}
    for geo_label, half_ext, n_views in GEOMETRIES:
        ct_model, weights = _build_geometry(half_ext, n_views)
        sinogram = sim.collect_projection_measurement(
            ct_model, weights, vol_jnp, projection_type='OPD_TT'
        )

        def run_mbir(ct=ct_model, sino=sinogram, w=weights):
            recon, _ = ct.recon(
                sino, weights=w,
                max_iterations=MAX_ITERATIONS,
                stop_threshold_change_pct=STOP_THRESHOLD_CHANGE_PCT,
            )
            return recon

        recons[geo_label] = _load_or_run_recon(
            RECON_CACHE_DIR / f'recon_seed{SEED}_{geo_label}_MBIR.npy', run_mbir
        )

    # --- Panel inputs ---
    panel_a = _panel_inputs(vol_gt, recons, zern_mode_index=0, plane_name='OPL',    fig_id='12a')
    panel_b = _panel_inputs(vol_gt, recons, zern_mode_index=2, plane_name='OPD_TT', fig_id='12b')

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
        out = OUT_DIR / f'fig12_geometry_comparison.{ext}'
        fig.savefig(out, bbox_inches='tight', dpi=200)
        print(f'Saved {out}')
    plt.show()


if __name__ == '__main__':
    main()
