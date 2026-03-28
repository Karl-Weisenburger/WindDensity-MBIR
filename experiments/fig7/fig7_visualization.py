"""
Figs 7a and 7b: NRMSE vs total angular extent.

Fig 7a: Full-resolution OPD_TT NRMSE vs total angular extent (one line per num_views)
Fig 7b: 4-plane OPD_TT NRMSE vs total angular extent (one line per num_views)

Data: data/fig7_geometry_sweep.npz
  nrmse:            (N_VOLS, n_ext, n_view, n_res)
  full_extents_deg: 1-D array of total angular extents (degrees)
  num_views_list:   1-D array of view counts
  resolutions:      1-D array of OPL-section counts (last entry = full resolution)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))

import numpy as np
import matplotlib.pyplot as plt

# ---- Paths ----------------------------------------------------------------
DATA_FILE = Path(__file__).parent / 'data' / 'fig7_geometry_sweep.npz'
OUT_DIR   = Path(__file__).parent / 'data'

# ---- Style ----------------------------------------------------------------
COLORS  = ['#0072B2', '#D55E00', '#CC79A7', '#009E73', '#F0E442']
MARKERS = ['o', 's', '^', 'D', 'x']
HIGHLIGHT_COLOR = 'red'
HIGHLIGHT_SEP_DEG = 2.0   # mark geometries with exactly this view separation


def _plot_panel(ax, nrmse, full_extents_deg, num_views_list, res_idx, title):
    """Draw a single NRMSE-vs-angle panel."""
    n_vols = nrmse.shape[0]

    for vi, n_views in enumerate(num_views_list):
        y_pct = nrmse[:, :, vi, res_idx] * 100          # (N_VOLS, n_ext)
        mean_y = y_pct.mean(axis=0)
        err    = 2 * y_pct.std(axis=0) / np.sqrt(n_vols)

        ax.errorbar(
            full_extents_deg, mean_y, yerr=err,
            label=f'{n_views} views',
            color=COLORS[vi % len(COLORS)],
            marker=MARKERS[vi % len(MARKERS)],
            capsize=4, linewidth=1.5, markersize=6,
        )

        # Highlight 2°-separation points with red open circles
        if n_views > 1:
            for ei, ext in enumerate(full_extents_deg):
                if abs(ext / (n_views - 1) - HIGHLIGHT_SEP_DEG) < 1e-9:
                    ax.plot(
                        ext, mean_y[ei], 'o',
                        color=HIGHLIGHT_COLOR, markersize=14,
                        fillstyle='none', markeredgewidth=2, zorder=5,
                    )

    ax.set_xlabel('Total Angular Extent (degrees)', fontsize=14)
    ax.set_ylabel('NRMSE (%)', fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True)


def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f'Data not found: {DATA_FILE}\n'
            'Run data_collection.py first.'
        )

    data = np.load(DATA_FILE)
    nrmse            = data['nrmse']               # (N_VOLS, n_ext, n_view, n_res)
    full_extents_deg = data['full_extents_deg']
    num_views_list   = data['num_views_list']
    resolutions      = data['resolutions']

    res_full_idx = len(resolutions) - 1            # last entry = full resolution
    res_4_candidates = np.where(resolutions == 4)[0]
    if len(res_4_candidates) == 0:
        raise ValueError('Resolution 4 not found in saved resolutions array.')
    res_4_idx = int(res_4_candidates[0])

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        r'NRMSE vs Total Angular Extent (OPD$_{TT}$ condition)',
        fontsize=18,
    )

    _plot_panel(
        axes[0], nrmse, full_extents_deg, num_views_list, res_full_idx,
        title=f'Fig 7a: Full-Resolution NRMSE ({int(resolutions[res_full_idx])} planes)',
    )
    _plot_panel(
        axes[1], nrmse, full_extents_deg, num_views_list, res_4_idx,
        title=f'Fig 7b: {int(resolutions[res_4_idx])}-Plane NRMSE',
    )

    plt.tight_layout()

    for ext in ('pdf', 'png'):
        out = OUT_DIR / f'fig7_nrmse_vs_angle.{ext}'
        plt.savefig(out, bbox_inches='tight', dpi=200)
        print(f'Saved {out}')

    plt.show()


if __name__ == '__main__':
    main()
