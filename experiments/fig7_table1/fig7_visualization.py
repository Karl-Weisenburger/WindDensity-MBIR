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
from matplotlib.lines import Line2D

# ---- Paths ----------------------------------------------------------------
DATA_FILE = Path(__file__).parent / 'data' / 'fig7_geometry_sweep.npz'
OUT_DIR   = Path(__file__).parent / 'figures'

# ---- Style ----------------------------------------------------------------
COLORS  = ['#0072B2', '#D55E00', '#CC79A7', '#009E73', '#F0E442']
MARKERS = ['o', 's', '^', 'D', 'x']
HIGHLIGHT_COLOR = 'red'
HIGHLIGHT_SEP_DEG = 2.0   # mark geometries with exactly this view separation


def _plot_panel(ax, nrmse, full_extents_deg, num_views_list, res_idx, title, panel_label):
    """Draw a single NRMSE-vs-angle panel."""
    n_vols = nrmse.shape[0]

    highlight_x, highlight_y = [], []

    for vi, n_views in enumerate(num_views_list):
        y = nrmse[:, :, vi, res_idx]                     # (N_VOLS, n_ext), raw NRMSE
        mean_y = y.mean(axis=0)
        err    = 2 * y.std(axis=0, ddof=1) / np.sqrt(n_vols)

        ax.errorbar(
            full_extents_deg, mean_y, yerr=err,
            label=f'{n_views} total views',
            color=COLORS[vi % len(COLORS)],
            marker=MARKERS[vi % len(MARKERS)],
            capsize=5, linewidth=1.5, markersize=6,
        )

        # Collect 2°-separation highlight points
        if n_views > 1:
            for ei, ext in enumerate(full_extents_deg):
                if abs(ext / (n_views - 1) - HIGHLIGHT_SEP_DEG) < 1e-9:
                    highlight_x.append(ext)
                    highlight_y.append(mean_y[ei])

    # Draw highlight circles
    if highlight_x:
        ax.scatter(
            highlight_x, highlight_y,
            marker='o', facecolor='none', edgecolor=HIGHLIGHT_COLOR,
            linewidths=2, zorder=10, s=300,
        )

    # Legend: view lines first, highlight entry last
    handles, labels = ax.get_legend_handles_labels()
    highlight_handle = Line2D(
        [0], [0], marker='o', color='w',
        markerfacecolor='none', markeredgecolor=HIGHLIGHT_COLOR,
        markeredgewidth=2, markersize=12,
        label=f'Geometries w/ ${HIGHLIGHT_SEP_DEG:.0f}^\\circ$ between adjacent views',
    )
    handles.append(highlight_handle)
    labels.append(highlight_handle.get_label())
    ax.legend(handles=handles, labels=labels, fontsize=12)

    ax.set_xlabel('total angular extent', fontsize=16)
    ax.set_ylabel('NRMSE', fontsize=16)
    ax.set_title(title, fontsize=18)
    ax.grid(True)
    ax.set_axisbelow(True)
    ax.text(0.5, -0.08, panel_label, transform=ax.transAxes,
            ha='center', va='top', fontsize=24)


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

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    res_full_idx = len(resolutions) - 1            # last entry = full resolution
    res_4_candidates = np.where(resolutions == 4)[0]
    if len(res_4_candidates) == 0:
        raise ValueError('Resolution 4 not found in saved resolutions array.')
    res_4_idx = int(res_4_candidates[0])

    addendum = '\n *using $\\text{OPD}_{\\text{TT}}$ measurements*'
    title_full = 'Average NRMSE for Full Resolution Reconstruction' + addendum
    title_4    = f'Average NRMSE for Reconstructing {int(resolutions[res_4_idx])} OPL Planes' + addendum

    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    fig.subplots_adjust(bottom=0.15)

    _plot_panel(axes[0], nrmse, full_extents_deg, num_views_list, res_full_idx,
                title=title_full, panel_label='(a)')
    _plot_panel(axes[1], nrmse, full_extents_deg, num_views_list, res_4_idx,
                title=title_4, panel_label='(b)')

    plt.tight_layout(rect=[0, 0.05, 1, 1])

    for ext in ('pdf', 'png'):
        out = OUT_DIR / f'fig7_nrmse_vs_angle.{ext}'
        plt.savefig(out, bbox_inches='tight', dpi=200)
        print(f'Saved {out}')

    plt.show()


if __name__ == '__main__':
    main()
