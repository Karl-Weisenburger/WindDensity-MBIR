"""
Fig 17: NRMSE vs depth-axis resolution — 7v8 geometry.

Two separate figures:
  17a — reconstructing OPL planes    (eval = OPL)
  17b — reconstructing OPD_TT planes (eval = OPD_TT)

Each figure compares the two measurement types (OPL vs OPD_TT) and shows
the full-resolution (640-plane) baseline for each as a horizontal line.

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
OUT_DIR   = Path(__file__).parent


def _plot_panel(nrmse, resolutions, eval_idx, planetype, out_name):
    """
    Single-panel NRMSE-vs-resolution plot for a fixed eval type.

    nrmse slice: (n_vols, n_meas, n_res)
    """
    slc = nrmse[:, :, eval_idx, :]            # (n_vols, n_meas, n_res)
    NRMSE = slc.mean(axis=0)                   # (n_meas, n_res)
    SD    = 2 * slc.std(axis=0, ddof=1) / np.sqrt(slc.shape[0])
    # Transpose to match original layout (n_res, n_meas) so that SD[:, i]
    # indexes by measurement type.
    NRMSE = NRMSE.T
    SD    = SD.T

    plt.figure(figsize=(7, 4))
    custom_markers = ['o', 's', '^', 'D', 'x', '+']
    custom_colors  = ['#0072B2', '#D55E00', '#CC79A7', '#009E73', '#F0E442', '#56B4E9']
    plt.rcParams['axes.prop_cycle'] = plt.cycler(marker=custom_markers, color=custom_colors)

    for i in range(SD.shape[1]):
        plt.errorbar(
            resolutions[:-1], NRMSE[:-1, i], yerr=SD[:-1, i], capsize=5,
        )

    plt.xticks(resolutions[:-1])
    plt.axhline(y=NRMSE[-1, 0], color='#0072B2', linestyle='--')
    plt.axhline(y=NRMSE[-1, 1], color='#D55E00', linestyle='-.')

    plt.legend([
        f'Reconst. 640 {planetype} planes using OPL measurements',
        f'Reconst. 640 {planetype} planes using ' r'$\text{OPD}_{\text{TT}}$' ' measurements',
        'Using OPL measurements',
        r'Using $\text{OPD}_{\text{TT}}$ measurements',
    ])
    plt.xlabel(f'Number of {planetype} planes reconstructed')
    plt.ylabel('NRMSE')
    if eval_idx == 1:
        plt.title(
            'NRMSE Relative to Resolution\n'
            r'*Reconstructing $\text{OPD}_{\text{TT}}$ Planes*'
        )
    else:
        plt.title(
            'NRMSE Relative to Resolution \n'
            ' *Reconstructing OPL Planes*'
        )
    plt.grid(True)
    plt.tight_layout()

    for ext in ('pdf', 'png'):
        out = OUT_DIR / f'{out_name}.{ext}'
        plt.savefig(out, bbox_inches='tight', dpi=300)
        print(f'Saved {out}')


def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f'Data not found: {DATA_FILE}\n'
            'Run fig13_14_15_17_data_collection.py first.'
        )

    data        = np.load(DATA_FILE, allow_pickle=True)
    nrmse       = data['nrmse']           # (vols, meas, eval, res)
    resolutions = data['resolutions']

    # Fig 17a — reconstructing OPL planes
    _plot_panel(
        nrmse, resolutions,
        eval_idx=0,
        planetype='OPL',
        out_name='fig17a_nrmse_vs_resolution_opl_planes_7v8',
    )
    # Fig 17b — reconstructing OPD_TT planes
    _plot_panel(
        nrmse, resolutions,
        eval_idx=1,
        planetype=r'$\text{OPD}_{\text{TT}}$',
        out_name='fig17b_nrmse_vs_resolution_opdtt_planes_7v8',
    )
    plt.show()


if __name__ == '__main__':
    main()
