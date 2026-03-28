"""
Fig 17: NRMSE as a function of depth-axis resolution — 7v8 geometry.

(a) NRMSE for reconstructing OPL planes
(b) NRMSE for reconstructing OPD_TT planes

Each panel shows two lines: OPL measurements vs OPD_TT measurements.
Key result: for OPD_TT planes, using OPD_TT measurements is only marginally
worse than using OPL measurements.

Data: ../fig13_14_15/data/fig13_14_15_17_7v8.npz
  nrmse:       (N_NRMSE_VOLS, n_meas, n_eval, n_res)
    meas dim:  0 = OPL measurement, 1 = OPD_TT measurement
    eval dim:  0 = OPL eval,        1 = OPD_TT eval
  meas_types:  ['OPL', 'OPD_TT']
  eval_types:  ['OPL', 'OPD_TT']
  resolutions: [2, 3, ..., 11, 640]
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))

import numpy as np
import matplotlib.pyplot as plt

# ---- Paths ----------------------------------------------------------------
DATA_FILE = Path(__file__).parents[1] / 'fig13_14_15' / 'data' / 'fig13_14_15_17_7v8.npz'
OUT_DIR   = Path(__file__).parent

# ---- Style ----------------------------------------------------------------
MEAS_STYLE = {
    'OPL':    dict(color='#0072B2', linestyle='-',  marker='o', label='OPL measurement'),
    'OPD_TT': dict(color='#D55E00', linestyle='--', marker='s', label=r'OPD$_{TT}$ measurement'),
}
LABEL_SIZE = 13
TITLE_SIZE = 16


def _decode(arr):
    return [s.decode() if isinstance(s, bytes) else s for s in arr]


def _plot_panel(ax, nrmse, meas_types, resolutions, eval_idx, title, ylabel):
    """Draw one NRMSE-vs-resolution panel for a given eval type."""
    n_vols = nrmse.shape[0]
    x_labels = [str(int(r)) for r in resolutions]
    x_labels[-1] = 'Full'
    x = np.arange(len(resolutions))

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
    ax.set_ylabel(ylabel, fontsize=LABEL_SIZE)
    ax.set_title(title, fontsize=TITLE_SIZE)
    ax.legend(fontsize=LABEL_SIZE)
    ax.grid(True)


def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f'Data not found: {DATA_FILE}\n'
            'Run fig13_14_15/fig13_14_15_data_collection.py first.'
        )

    data        = np.load(DATA_FILE, allow_pickle=True)
    nrmse       = data['nrmse']           # (N_NRMSE_VOLS, n_meas, n_eval, n_res)
    meas_types  = _decode(data['meas_types'])
    resolutions = data['resolutions']

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Fig 17: NRMSE vs Resolution — 7v8 Geometry', fontsize=18)

    _plot_panel(
        axes[0], nrmse, meas_types, resolutions,
        eval_idx=0,
        title='(a) OPL Planes',
        ylabel='OPL NRMSE (%)',
    )
    _plot_panel(
        axes[1], nrmse, meas_types, resolutions,
        eval_idx=1,
        title=r'(b) OPD$_{TT}$ Planes',
        ylabel=r'OPD$_{TT}$ NRMSE (%)',
    )

    plt.tight_layout()
    for ext in ('pdf', 'png'):
        out = OUT_DIR / f'fig17_nrmse_vs_resolution_opl_vs_opdtt.{ext}'
        fig.savefig(out, bbox_inches='tight', dpi=200)
        print(f'Saved {out}')
    plt.show()


if __name__ == '__main__':
    main()
