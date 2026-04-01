"""
Fig 17: NRMSE as a function of depth-axis resolution — 3v2 and 11v16 geometries.

2×2 grid of subplots:
  Rows:    (a) OPL planes (TTP included in recon),  (b) OPD_TT planes (TTP removed)
  Columns: 3v2 geometry,  11v16 geometry

Each cell shows two lines: OPL measurement vs OPD_TT measurement.

Data: data/fig10_11_17_3v2_11v16.npz
  nrmse:       (N_NRMSE_VOLS, n_meas, n_eval, n_geos, n_res)
    meas dim:  0 = OPL measurement,  1 = OPD_TT measurement
    eval dim:  0 = OPL eval,         1 = OPD_TT eval
    geo dim:   0 = 3v2,              1 = 11v16
  meas_types:  ['OPL', 'OPD_TT']
  eval_types:  ['OPL', 'OPD_TT']
  geometry_names: ['3v2', '11v16']
  resolutions: [2, 3, ..., 11, 640]
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))

import numpy as np
import matplotlib.pyplot as plt

# ---- Paths ----------------------------------------------------------------
DATA_FILE = Path(__file__).parent / 'data' / 'fig10_11_17_3v2_11v16.npz'
OUT_DIR   = Path(__file__).parent

# ---- Style ----------------------------------------------------------------
MEAS_STYLE = {
    'OPL':    dict(color='#0072B2', linestyle='-',  marker='o', label='OPL measurement'),
    'OPD_TT': dict(color='#D55E00', linestyle='--', marker='s', label=r'OPD$_{TT}$ measurement'),
}
LABEL_SIZE = 12
TITLE_SIZE = 13


def _decode(arr):
    return [s.decode() if isinstance(s, bytes) else s for s in arr]


def _plot_panel(ax, nrmse, meas_types, resolutions, eval_idx, geo_idx, title, ylabel):
    """Draw one NRMSE-vs-resolution panel for given eval type and geometry."""
    n_vols = nrmse.shape[0]
    x_labels = [str(int(r)) for r in resolutions]
    x_labels[-1] = 'Full'
    x = np.arange(len(resolutions))

    for meas_idx, meas_label in enumerate(meas_types):
        sty = MEAS_STYLE.get(meas_label, dict(color='gray', linestyle='-', marker='o', label=meas_label))

        # nrmse shape: (n_vols, n_meas, n_eval, n_geos, n_res)
        y_pct  = nrmse[:, meas_idx, eval_idx, geo_idx, :] * 100
        mean_y = y_pct.mean(axis=0)
        err    = 2 * y_pct.std(axis=0) / np.sqrt(n_vols)

        ax.errorbar(
            x, mean_y, yerr=err,
            color=sty['color'], linestyle=sty['linestyle'], marker=sty['marker'],
            label=sty['label'], capsize=4, linewidth=1.5, markersize=6,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=10, rotation=45)
    ax.set_xlabel('Number of Planes Reconstructed', fontsize=LABEL_SIZE)
    ax.set_ylabel(ylabel, fontsize=LABEL_SIZE)
    ax.set_title(title, fontsize=TITLE_SIZE)
    ax.legend(fontsize=LABEL_SIZE)
    ax.grid(True)


def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f'Data not found: {DATA_FILE}\n'
            'Run fig10_11_17_data_collection.py first.'
        )

    data        = np.load(DATA_FILE, allow_pickle=True)
    nrmse       = data['nrmse']           # (N_NRMSE_VOLS, n_meas, n_eval, n_geos, n_res)
    meas_types  = _decode(data['meas_types'])
    eval_types  = _decode(data['eval_types'])
    geo_names   = _decode(data['geometry_names'])
    resolutions = data['resolutions']

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Fig 17: NRMSE vs Resolution — 3v2 and 11v16 Geometries', fontsize=16)

    for eval_idx, eval_label in enumerate(eval_types):
        if eval_label == 'OPL':
            ylabel = 'OPL NRMSE (%)'
            row_label = '(a) OPL Planes'
        else:
            ylabel = r'OPD$_{TT}$ NRMSE (%)'
            row_label = r'(b) OPD$_{TT}$ Planes'

        for geo_idx, geo_name in enumerate(geo_names):
            ax = axes[eval_idx, geo_idx]
            title = f'{row_label} — {geo_name}'
            _plot_panel(ax, nrmse, meas_types, resolutions, eval_idx, geo_idx, title, ylabel)

    plt.tight_layout()
    for ext in ('pdf', 'png'):
        out = OUT_DIR / f'fig17_nrmse_vs_resolution_3v2_11v16.{ext}'
        fig.savefig(out, bbox_inches='tight', dpi=200)
        print(f'Saved {out}')
    plt.show()


if __name__ == '__main__':
    main()
