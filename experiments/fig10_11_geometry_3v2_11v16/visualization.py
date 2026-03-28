"""
Figs 10 and 11: Zernike error analysis comparing 3v2 vs 11v16 geometry.

Fig 10: Per-radial-degree Zernike RMSE — OPL (withTTP) condition
Fig 11: Per-radial-degree Zernike RMSE — OPD_TT (noTTP) condition

Also produces Fig 12a/12b NRMSE-vs-resolution curves (same data).

Data: data/fig10_11_table2_3v2_11v16.npz
  nrmse:       (N_VOLS, n_ttp, n_geos, n_res)
  zernike_mse: (N_VOLS, n_ttp, n_geos, n_zmodes)
    n_zmodes = 10 — Zernike radial degrees 0..8 individually, then all combined
  geometry_names: ['3v2', '11v16']
  ttp_states:     ['withTTP', 'noTTP']
  resolutions:    [2, 3, ..., 11, 640]
  zernike_degrees: [0, 1, ..., 8]
  zern_resolution: int (section count at which Zernike analysis was performed)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))

import numpy as np
import matplotlib.pyplot as plt

# ---- Paths ----------------------------------------------------------------
DATA_FILE = Path(__file__).parent / 'data' / 'fig10_11_table2_3v2_11v16.npz'
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
    nrmse       = data['nrmse']          # (N_VOLS, n_ttp, n_geos, n_res)
    zernike_mse = data['zernike_mse']    # (N_VOLS, n_ttp, n_geos, n_zmodes)
    geo_names   = _decode(data['geometry_names'])
    ttp_states  = _decode(data['ttp_states'])
    n_osa_modes = int(data['n_osa_modes'])

    # -- Figs 10 and 11: Zernike RMSE panels --
    ttp_idx_opl   = ttp_states.index('withTTP')    # OPL condition
    ttp_idx_opd   = ttp_states.index('noTTP')      # OPD_TT condition

    fig, axes = plt.subplots(1, 2, figsize=(22, 7))
    fig.suptitle('Zernike Error Analysis: 3v2 vs 11v16 Geometry', fontsize=18)

    _plot_zernike_panel(
        axes[0], zernike_mse, geo_names, n_osa_modes, ttp_idx_opl,
        title='Fig 10: OPL Measurement (withTTP)',
    )
    _plot_zernike_panel(
        axes[1], zernike_mse, geo_names, n_osa_modes, ttp_idx_opd,
        title=r'Fig 11: OPD$_{TT}$ Measurement (noTTP)',
    )

    plt.tight_layout()
    for ext in ('pdf', 'png'):
        out = OUT_DIR / f'fig10_11_zernike_geometry.{ext}'
        fig.savefig(out, bbox_inches='tight', dpi=200)
        print(f'Saved {out}')
    plt.show()


def plot_nrmse_vs_resolution(data):
    """Supplemental: NRMSE vs resolution for 3v2 and 11v16 (withTTP and noTTP)."""
    nrmse      = data['nrmse']           # (N_VOLS, n_ttp, n_geos, n_res)
    geo_names  = _decode(data['geometry_names'])
    ttp_states = _decode(data['ttp_states'])
    resolutions = data['resolutions']

    n_vols = nrmse.shape[0]

    # Build x-tick labels — replace last entry (full res) with 'Full'
    x_labels = [str(int(r)) for r in resolutions]
    x_labels[-1] = 'Full'
    x = np.arange(len(resolutions))

    COLORS  = ['#0072B2', '#D55E00']
    LS      = ['-', '--']
    MARKERS = ['o', 's']

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=False)
    fig.suptitle('NRMSE vs Resolution: 3v2 vs 11v16', fontsize=18)

    for ai, (ttp_label, ttp_name) in enumerate(zip(ttp_states, ['OPL', r'OPD$_{TT}$'])):
        ax = axes[ai]
        ttp_idx = ttp_states.index(ttp_label)
        for gi, geo_name in enumerate(geo_names):
            y_pct  = nrmse[:, ttp_idx, gi, :] * 100
            mean_y = y_pct.mean(axis=0)
            err    = 2 * y_pct.std(axis=0) / np.sqrt(n_vols)
            ax.errorbar(
                x, mean_y, yerr=err,
                label=geo_name,
                color=COLORS[gi], linestyle=LS[gi], marker=MARKERS[gi],
                capsize=4, linewidth=1.5, markersize=6,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=12, rotation=45)
        ax.set_xlabel('Number of OPL Planes', fontsize=LABEL_SIZE)
        ax.set_ylabel('NRMSE (%)', fontsize=LABEL_SIZE)
        ax.set_title(f'{ttp_name} Measurement', fontsize=TITLE_SIZE)
        ax.legend(fontsize=LABEL_SIZE)
        ax.grid(True)

    plt.tight_layout()
    for ext in ('pdf', 'png'):
        out = OUT_DIR / f'fig10_11_nrmse_vs_resolution.{ext}'
        fig.savefig(out, bbox_inches='tight', dpi=200)
        print(f'Saved {out}')
    plt.show()


def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f'Data not found: {DATA_FILE}\n'
            'Run data_collection.py first.'
        )

    data = np.load(DATA_FILE, allow_pickle=True)
    plot_figs_10_11(data)
    plot_nrmse_vs_resolution(data)


if __name__ == '__main__':
    main()
