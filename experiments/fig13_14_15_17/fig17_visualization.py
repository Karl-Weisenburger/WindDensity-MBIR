"""
Fig 17: NRMSE vs depth-axis resolution — 7v8 geometry.

Single figure with two side-by-side panels:
  (a) reconstructing OPL planes    (eval = OPL)
  (b) reconstructing OPD_TT planes (eval = OPD_TT)

Each panel compares OPL vs OPD_TT measurements and shows the full-resolution
(640-plane) baseline for each as a horizontal dashed line. The panels keep
their individual (7, 4) sizing; the combined figure is (14, 4).

Data: data/fig13_14_15_17_7v8.npz
  nrmse: (N_NRMSE_VOLS, n_meas, n_eval, n_res)
    meas 0 = OPL measurement,  1 = OPD_TT measurement
    eval 0 = OPL eval,         1 = OPD_TT eval
  resolutions: [2, 3, ..., 11, 640]
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))

import numpy as np
import matplotlib.pyplot as plt

# ---- Paths ----------------------------------------------------------------
DATA_FILE = Path(__file__).parent / 'data' / 'fig13_14_15_17_7v8.npz'
OUT_DIR   = Path(__file__).parent / 'figures'

# ---- Style ----------------------------------------------------------------
CUSTOM_MARKERS = ['o', 's', '^', 'D', 'x', '+']
CUSTOM_COLORS  = ['#0072B2', '#D55E00', '#CC79A7', '#009E73', '#F0E442', '#56B4E9']


def _plot_panel(ax, nrmse, resolutions, eval_idx, planetype, panel_label):
    """Draw one NRMSE-vs-resolution panel onto `ax`."""
    slc = nrmse[:, :, eval_idx, :]                  # (n_vols, n_meas, n_res)
    NRMSE = slc.mean(axis=0).T                       # (n_res, n_meas)
    SD    = (2 * slc.std(axis=0, ddof=1) / np.sqrt(slc.shape[0])).T

    ax.set_prop_cycle(marker=CUSTOM_MARKERS, color=CUSTOM_COLORS)

    for i in range(SD.shape[1]):
        ax.errorbar(
            resolutions[:-1], NRMSE[:-1, i], yerr=SD[:-1, i], capsize=5,
        )

    ax.set_xticks(resolutions[:-1])
    ax.axhline(y=NRMSE[-1, 0], color='#0072B2', linestyle='--')
    ax.axhline(y=NRMSE[-1, 1], color='#D55E00', linestyle='-.')

    ax.legend([
        f'Reconst. 640 {planetype} planes using OPL measurements',
        f'Reconst. 640 {planetype} planes using ' r'$\text{OPD}_{\text{TT}}$' ' measurements',
        'Using OPL measurements',
        r'Using $\text{OPD}_{\text{TT}}$ measurements',
    ])
    ax.set_xlabel(f'Number of {planetype} planes reconstructed')
    ax.set_ylabel('NRMSE')
    if eval_idx == 1:
        ax.set_title(
            'NRMSE Relative to Resolution\n'
            r'*Reconstructing $\text{OPD}_{\text{TT}}$ Planes*'
        )
    else:
        ax.set_title(
            'NRMSE Relative to Resolution \n'
            ' *Reconstructing OPL Planes*'
        )
    ax.grid(True)
    ax.text(0.5, -0.22, panel_label, transform=ax.transAxes,
            ha='center', va='top', fontsize=24)


def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f'Data not found: {DATA_FILE}\n'
            'Run fig13_14_15_17_data_collection.py first.'
        )
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    data        = np.load(DATA_FILE, allow_pickle=True)
    nrmse       = data['nrmse']           # (vols, meas, eval, res)
    resolutions = data['resolutions']

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    fig.subplots_adjust(bottom=0.22)

    _plot_panel(axes[0], nrmse, resolutions, eval_idx=0,
                planetype='OPL', panel_label='(a)')
    _plot_panel(axes[1], nrmse, resolutions, eval_idx=1,
                planetype=r'$\text{OPD}_{\text{TT}}$', panel_label='(b)')

    plt.tight_layout(rect=[0, 0.05, 1, 1])

    for ext in ('pdf', 'png'):
        out = OUT_DIR / f'fig17_nrmse_vs_resolution_7v8.{ext}'
        fig.savefig(out, bbox_inches='tight', dpi=300)
        print(f'Saved {out}')
    plt.show()


if __name__ == '__main__':
    main()
