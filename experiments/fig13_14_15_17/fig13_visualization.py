"""
Fig 13: NRMSE vs depth-axis resolution — 7v8 geometry, OPD_TT measurement.

Compares two evaluation targets (held the measurement type fixed at OPD_TT):
  line 1 — OPL planes (eval OPL,    blue circles)
  line 2 — OPD_TT planes (eval OPD_TT, red triangles)

Horizontal dashed lines mark the full-resolution baseline for each target.

Data: data/fig13_14_15_17_7v8.npz
  nrmse: (N_NRMSE_VOLS, n_meas, n_eval, n_res)
    meas 1 = OPD_TT measurement
    eval 0 = OPL eval,   eval 1 = OPD_TT eval
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
    n_vols      = nrmse.shape[0]

    # Fixed: OPD_TT measurement (meas index 1). Vary: eval type.
    # NRMSE1 = OPL eval, NRMSE2 = OPD_TT eval
    NRMSE1 = nrmse[:, 1, 0, :].mean(axis=0)
    NRMSE2 = nrmse[:, 1, 1, :].mean(axis=0)
    SD1    = 2 * nrmse[:, 1, 0, :].std(axis=0, ddof=1) / np.sqrt(n_vols)
    SD2    = 2 * nrmse[:, 1, 1, :].std(axis=0, ddof=1) / np.sqrt(n_vols)

    plt.figure(figsize=(10, 5))
    custom_markers = ['o', '^', 's', 'D', 'x', '+']
    custom_colors  = ['#0072B2', '#D55E00', '#CC79A7', '#009E73', '#F0E442', '#56B4E9']
    plt.rcParams['axes.prop_cycle'] = plt.cycler(marker=custom_markers, color=custom_colors)

    plt.errorbar(resolutions[:-1], NRMSE1[:-1], yerr=SD1[:-1], capsize=5)
    plt.errorbar(resolutions[:-1], NRMSE2[:-1], yerr=SD2[:-1], capsize=5)

    plt.xticks(resolutions[:-1])
    plt.axhline(y=NRMSE1[-1], color='#0072B2', linestyle='--')
    plt.axhline(y=NRMSE2[-1], color='#D55E00', linestyle='-.')

    plt.legend([
        'Reconstructing 640 OPL planes',
        r'Reconstructing 640 $\text{OPD}_{\text{TT}}$ planes',
        'Reconstructing OPL planes',
        r'Reconstructing $\text{OPD}_{\text{TT}}$ planes',
    ])
    plt.xlabel('Number of planes reconstructed')
    plt.ylabel('NRMSE')
    plt.title(
        'NRMSE Relative to Resolution \n'
        r' *Reconstructing with $\text{OPD}_{\text{TT}}$ measurements using the 7-view $8^\circ$ geometry*'
    )
    plt.grid(True)
    plt.tight_layout()

    for ext in ('pdf', 'png'):
        out = OUT_DIR / f'fig13_nrmse_vs_resolution_7v8.{ext}'
        plt.savefig(out, bbox_inches='tight', dpi=300)
        print(f'Saved {out}')
    plt.show()


if __name__ == '__main__':
    main()
