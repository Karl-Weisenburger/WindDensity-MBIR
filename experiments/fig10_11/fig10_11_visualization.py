"""
Figs 10 and 11: Zernike error analysis comparing 3v2 vs 11v16 geometry.

Fig 10: Per-radial-degree Zernike RMSE — OPL (withTTP) condition
Fig 11: Per-radial-degree Zernike RMSE — OPD_TT (noTTP) condition

Data: data/fig10_11_17_3v2_11v16.npz
  nrmse:       (N_NRMSE_VOLS, n_meas, n_eval, n_geos, n_res)
  zernike_mse: (N_VOLS, n_meas, n_geos, n_zmodes)
    n_zmodes = N_OSA_MODES + 1 (OSA 0–44 individually, then total)
  geometry_names: ['3v2', '11v16']
  meas_types:     ['OPL', 'OPD_TT']
  eval_types:     ['OPL', 'OPD_TT']
  resolutions:    [2, 3, ..., 11, 640]
  zern_resolution: int (section count at which Zernike analysis was performed)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))

import numpy as np
import matplotlib.pyplot as plt

# ---- Paths ----------------------------------------------------------------
DATA_FILE = Path(__file__).parent / 'data' / 'fig10_11_17_3v2_11v16.npz'
OUT_DIR   = Path(__file__).parent / 'data'

# ---- Style ----------------------------------------------------------------
GEO_STYLE = {
    '3v2':   dict(color='#0072B2', hatch=None,  label='3v2 (narrow)'),
    '11v16': dict(color='#D55E00', hatch='//',  label='11v16 (wide)'),
}
BAR_WIDTH   = 0.35
LABEL_SIZE  = 13
TITLE_SIZE  = 16


def _decode(arr):
    return [s.decode() if isinstance(s, bytes) else s for s in arr]


def _osa_x_labels(n_osa_modes):
    """OSA index labels: '0', '1', ..., '44', 'Total'."""
    return [str(j) for j in range(n_osa_modes)] + ['Total']


def _plot_zernike_panel(ax, zernike_mse, geo_names, n_osa_modes, ttp_idx, title):
    """
    Bar chart of per-mode Zernike RMSE for the given TTP condition.

    zernike_mse: (N_VOLS, n_ttp, n_geos, n_zmodes)  where n_zmodes = n_osa_modes + 1
    """
    n_vols, _, n_geos, n_zmodes = zernike_mse.shape
    x = np.arange(n_zmodes)
    x_labels = _osa_x_labels(n_osa_modes)
    offsets = np.linspace(-(n_geos - 1) * BAR_WIDTH / 2,
                          (n_geos - 1) * BAR_WIDTH / 2, n_geos)

    for gi, geo_name in enumerate(geo_names):
        mse_per_vol = zernike_mse[:, ttp_idx, gi, :]     # (N_VOLS, n_zmodes)
        rmse_per_vol = np.sqrt(mse_per_vol)               # (N_VOLS, n_zmodes)
        mean_rmse = rmse_per_vol.mean(axis=0)
        err       = 2 * rmse_per_vol.std(axis=0) / np.sqrt(n_vols)

        sty = GEO_STYLE.get(geo_name, dict(color='gray', hatch=None, label=geo_name))
        ax.bar(
            x + offsets[gi], mean_rmse, BAR_WIDTH,
            color=sty['color'], hatch=sty['hatch'],
            label=sty['label'], alpha=0.85, edgecolor='black', linewidth=0.7,
        )
        ax.errorbar(
            x + offsets[gi], mean_rmse, yerr=err,
            fmt='none', color='black', capsize=4, linewidth=1.2,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=max(6, LABEL_SIZE - 2), rotation=90)
    ax.set_xlabel('OSA/ANSI Zernike Index', fontsize=LABEL_SIZE)
    ax.set_ylabel('RMSE (OPL units)', fontsize=LABEL_SIZE)
    ax.set_title(title, fontsize=TITLE_SIZE)
    ax.legend(fontsize=LABEL_SIZE)
    ax.grid(True, axis='y')


def plot_figs_10_11(data):
    zernike_mse = data['zernike_mse']    # (N_VOLS, n_meas, n_geos, n_zmodes)
    geo_names   = _decode(data['geometry_names'])
    meas_types  = _decode(data['meas_types'])
    n_osa_modes = int(data['n_osa_modes'])

    meas_idx_opl = meas_types.index('OPL')
    meas_idx_opd = meas_types.index('OPD_TT')

    fig, axes = plt.subplots(1, 2, figsize=(22, 7))
    fig.suptitle('Zernike Error Analysis: 3v2 vs 11v16 Geometry', fontsize=18)

    _plot_zernike_panel(
        axes[0], zernike_mse, geo_names, n_osa_modes, meas_idx_opl,
        title='Fig 10: OPL Measurement',
    )
    _plot_zernike_panel(
        axes[1], zernike_mse, geo_names, n_osa_modes, meas_idx_opd,
        title=r'Fig 11: OPD$_{TT}$ Measurement',
    )

    plt.tight_layout()
    for ext in ('pdf', 'png'):
        out = OUT_DIR / f'fig10_11_zernike_geometry.{ext}'
        fig.savefig(out, bbox_inches='tight', dpi=200)
        print(f'Saved {out}')
    plt.show()


def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f'Data not found: {DATA_FILE}\n'
            'Run fig10_11_17_data_collection.py first.'
        )

    data = np.load(DATA_FILE, allow_pickle=True)
    plot_figs_10_11(data)


if __name__ == '__main__':
    main()
