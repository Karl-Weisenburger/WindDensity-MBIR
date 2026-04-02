"""
Figs 13, 14, 15: 7v8 geometry (7 views, 8° total extent).

Fig 13: OPD_TT NRMSE vs resolution — OPL vs OPD_TT measurement types
Fig 14: Zernike RMSE per OSA mode (all 45) — OPL vs OPD_TT measurement types
Fig 15: Same as Fig 14, zoomed to low-order modes (OSA 0–14)

Data: data/fig13_14_15_17_7v8.npz
  nrmse:         (N_NRMSE_VOLS, n_meas, n_eval, n_res)
    meas dim 0 = OPL measurement,  1 = OPD_TT measurement
    eval dim 0 = OPL eval,         1 = OPD_TT eval
  zernike_mse:   (N_VOLS, n_meas, n_zmodes)  — OPD_TT eval at ZERN_RESOLUTION
  resolutions:   [2, 3, ..., 11, 640]
  meas_types:    ['OPL', 'OPD_TT']
  eval_types:    ['OPL', 'OPD_TT']
  n_osa_modes:   45
  zern_resolution: 4
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))

import numpy as np
import matplotlib.pyplot as plt

# ---- Paths ----------------------------------------------------------------
DATA_FILE = Path(__file__).parent / 'data' / 'fig13_14_15_17_7v8.npz'
OUT_DIR   = Path(__file__).parent / 'data'

# ---- Style ----------------------------------------------------------------
MEAS_STYLE = {
    'OPL':    dict(color='#0072B2', linestyle='-',  marker='o',
                   hatch=None, label='OPL measurement'),
    'OPD_TT': dict(color='#D55E00', linestyle='--', marker='s',
                   hatch='//', label=r'OPD$_{TT}$ measurement'),
}
BAR_WIDTH  = 0.35
LABEL_SIZE = 13
TITLE_SIZE = 16


def _decode(arr):
    return [s.decode() if isinstance(s, bytes) else s for s in arr]


# ============================================================
# Fig 13: OPD_TT NRMSE vs resolution
# ============================================================

def plot_fig13(data):
    nrmse       = data['nrmse']          # (N_NRMSE_VOLS, n_meas, n_eval, n_res)
    meas_types  = _decode(data['meas_types'])
    resolutions = data['resolutions']
    n_vols      = nrmse.shape[0]
    eval_idx    = 1                      # OPD_TT eval

    x_labels = [str(int(r)) for r in resolutions]
    x_labels[-1] = 'Full'
    x = np.arange(len(resolutions))

    fig, ax = plt.subplots(figsize=(10, 6))

    for meas_idx, meas_label in enumerate(meas_types):
        sty = MEAS_STYLE.get(meas_label, dict(color='gray', linestyle='-', marker='o', label=meas_label))

        y_pct  = nrmse[:, meas_idx, eval_idx, :] * 100
        mean_y = y_pct.mean(axis=0)
        err    = 2 * y_pct.std(axis=0) / np.sqrt(n_vols)

        ax.errorbar(
            x, mean_y, yerr=err,
            color=sty['color'], linestyle=sty['linestyle'], marker=sty['marker'],
            label=sty['label'], capsize=4, linewidth=1.5, markersize=6,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=12, rotation=45)
    ax.set_xlabel('Number of OPL Planes', fontsize=LABEL_SIZE)
    ax.set_ylabel(r'OPD$_{TT}$ NRMSE (%)', fontsize=LABEL_SIZE)
    ax.set_title(r'Fig 13: OPD$_{TT}$ NRMSE vs Resolution — 7v8 Geometry', fontsize=TITLE_SIZE)
    ax.legend(fontsize=LABEL_SIZE)
    ax.grid(True)
    plt.tight_layout()

    for ext in ('pdf', 'png'):
        out = OUT_DIR / f'fig13_nrmse_vs_resolution_7v8.{ext}'
        fig.savefig(out, bbox_inches='tight', dpi=200)
        print(f'Saved {out}')
    plt.show()


# ============================================================
# Figs 14 and 15: Zernike RMSE
# ============================================================

def _osa_x_labels(n_osa_modes):
    return [str(j) for j in range(n_osa_modes)] + ['Total']


def _plot_zernike_comparison(ax, zernike_mse, meas_types, n_osa_modes, title):
    """
    Grouped bar chart: one group per OSA mode, one bar per measurement type.

    zernike_mse: (N_VOLS, n_meas, n_zmodes)  where n_zmodes = n_osa_modes + 1
    """
    n_vols, n_meas, n_zmodes = zernike_mse.shape
    x = np.arange(n_zmodes)
    x_labels = _osa_x_labels(n_osa_modes)
    offsets = np.linspace(-(n_meas - 1) * BAR_WIDTH / 2,
                          (n_meas - 1) * BAR_WIDTH / 2, n_meas)

    for meas_idx, meas_label in enumerate(meas_types):
        sty = MEAS_STYLE.get(meas_label, dict(color='gray', hatch=None, label=meas_label))

        rmse_per_vol = np.sqrt(zernike_mse[:, meas_idx, :])
        mean_rmse = rmse_per_vol.mean(axis=0)
        err       = 2 * rmse_per_vol.std(axis=0) / np.sqrt(n_vols)

        ax.bar(
            x + offsets[meas_idx], mean_rmse, BAR_WIDTH,
            color=sty['color'], hatch=sty.get('hatch'),
            label=sty['label'], alpha=0.85, edgecolor='black', linewidth=0.7,
        )
        ax.errorbar(
            x + offsets[meas_idx], mean_rmse, yerr=err,
            fmt='none', color='black', capsize=4, linewidth=1.2,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=max(6, LABEL_SIZE - 2), rotation=90)
    ax.set_xlabel('OSA/ANSI Zernike Index', fontsize=LABEL_SIZE)
    ax.set_ylabel('RMSE (OPL units)', fontsize=LABEL_SIZE)
    ax.set_title(title, fontsize=TITLE_SIZE)
    ax.legend(fontsize=LABEL_SIZE)
    ax.grid(True, axis='y')


def plot_figs_14_15(data):
    zernike_mse = data['zernike_mse']    # (N_VOLS, n_meas, n_zmodes)
    meas_types  = _decode(data['meas_types'])
    n_osa_modes = int(data['n_osa_modes'])

    fig, axes = plt.subplots(1, 2, figsize=(22, 7))
    fig.suptitle(
        r'Zernike Error Analysis: OPL vs OPD$_{TT}$ Measurement — 7v8 Geometry',
        fontsize=18,
    )

    # Fig 14: all 45 OSA modes
    _plot_zernike_comparison(
        axes[0], zernike_mse, meas_types, n_osa_modes,
        title='Fig 14: Per-OSA-Mode Zernike RMSE (all 45 modes)',
    )
    # Fig 15: zoom to low-order modes (OSA 0–14, radial degrees 0–4)
    zoom = min(15, n_osa_modes)
    _plot_zernike_comparison(
        axes[1], zernike_mse[:, :, :zoom + 1], meas_types, zoom,
        title='Fig 15: Zernike RMSE — Low-Order Modes (OSA 0–14)',
    )

    plt.tight_layout()
    for ext in ('pdf', 'png'):
        out = OUT_DIR / f'fig14_15_zernike_7v8.{ext}'
        fig.savefig(out, bbox_inches='tight', dpi=200)
        print(f'Saved {out}')
    plt.show()


# ============================================================
# Main
# ============================================================

def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f'Data not found: {DATA_FILE}\n'
            'Run fig13_14_15_data_collection.py first.'
        )

    data = np.load(DATA_FILE, allow_pickle=True)
    plot_fig13(data)
    plot_figs_14_15(data)


if __name__ == '__main__':
    main()
