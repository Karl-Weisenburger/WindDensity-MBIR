"""
Figs 10 and 11: Zernike error analysis comparing 3v2 vs 11v16 geometry.
OPD_TT measurements only.

Fig 10: Zernike RMSE grouped by radial degree (0–8)
Fig 11: Zernike RMSE per individual OSA/ANSI mode (0–44) + total

Data: data/fig10_11_zernike.npz
  zernike_mse: (N_VOLS, n_geos, n_zmodes)
    n_zmodes = 45 individual OSA modes (0–44) + 1 total = 46
  geometry_names: ['3v2', '11v16']
  n_osa_modes:    45
  zern_resolution: 4
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))

import numpy as np
import matplotlib.pyplot as plt

DATA_FILE = Path(__file__).parent / 'data' / 'fig10_11_zernike.npz'
OUT_DIR   = Path(__file__).parent

GEO_STYLE = {
    '3v2':   dict(color='#0072B2', hatch=None, label='3v2 (narrow)'),
    '11v16': dict(color='#D55E00', hatch='//',  label='11v16 (wide)'),
}
BAR_WIDTH  = 0.35
LABEL_SIZE = 13
TITLE_SIZE = 16


def _decode(arr):
    return [s.decode() if isinstance(s, bytes) else s for s in arr]


def _radial_degree(j):
    """Return radial degree n for OSA/ANSI index j."""
    n = 0
    while (n + 1) * (n + 2) // 2 <= j:
        n += 1
    return n


def _modes_by_radial_degree(n_osa_modes):
    """Return dict mapping radial degree -> list of OSA indices."""
    groups = {}
    for j in range(n_osa_modes):
        n = _radial_degree(j)
        groups.setdefault(n, []).append(j)
    return groups


def _bar_panel(ax, x, means, errs, geo_names, x_labels, xlabel, title):
    n_geos = len(geo_names)
    offsets = np.linspace(-(n_geos - 1) * BAR_WIDTH / 2,
                           (n_geos - 1) * BAR_WIDTH / 2, n_geos)
    for gi, geo_name in enumerate(geo_names):
        sty = GEO_STYLE.get(geo_name, dict(color='gray', hatch=None, label=geo_name))
        ax.bar(x + offsets[gi], means[gi], BAR_WIDTH,
               color=sty['color'], hatch=sty['hatch'],
               label=sty['label'], alpha=0.85, edgecolor='black', linewidth=0.7)
        ax.errorbar(x + offsets[gi], means[gi], yerr=errs[gi],
                    fmt='none', color='black', capsize=4, linewidth=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=max(6, LABEL_SIZE - 2), rotation=45)
    ax.set_xlabel(xlabel, fontsize=LABEL_SIZE)
    ax.set_ylabel('RMSE (OPL units)', fontsize=LABEL_SIZE)
    ax.set_title(title, fontsize=TITLE_SIZE)
    ax.legend(fontsize=LABEL_SIZE)
    ax.grid(True, axis='y')


def plot_fig10(zernike_mse, geo_names, n_osa_modes):
    """Fig 10: RMSE grouped by radial degree."""
    groups = _modes_by_radial_degree(n_osa_modes)
    radial_degrees = sorted(groups.keys())

    means, errs = [], []
    for gi in range(len(geo_names)):
        m, e = [], []
        for n in radial_degrees:
            mse_n = zernike_mse[:, gi, groups[n]].sum(axis=-1)  # total MSE for degree n
            rmse_n = np.sqrt(mse_n)
            m.append(rmse_n.mean())
            e.append(2 * rmse_n.std(ddof=1) / np.sqrt(len(rmse_n)))
        means.append(np.array(m))
        errs.append(np.array(e))

    fig, ax = plt.subplots(figsize=(10, 6))
    _bar_panel(ax, np.arange(len(radial_degrees)), means, errs, geo_names,
               x_labels=[str(n) for n in radial_degrees],
               xlabel='Radial Degree',
               title='Fig 10: Zernike RMSE by Radial Degree\n(OPD$_{TT}$ measurements)')
    plt.tight_layout()
    for ext in ('pdf', 'png'):
        out = OUT_DIR / f'fig10_zernike_radial.{ext}'
        fig.savefig(out, bbox_inches='tight', dpi=200)
        print(f'Saved {out}')
    plt.show()


def plot_fig11(zernike_mse, geo_names, n_osa_modes):
    """Fig 11: RMSE per individual OSA mode + total."""
    n_zmodes = n_osa_modes + 1
    x_labels = [str(j) for j in range(n_osa_modes)] + ['Total']

    means, errs = [], []
    for gi in range(len(geo_names)):
        rmse = np.sqrt(zernike_mse[:, gi, :])   # (N_VOLS, n_zmodes)
        means.append(rmse.mean(axis=0))
        errs.append(2 * rmse.std(axis=0, ddof=1) / np.sqrt(rmse.shape[0]))

    fig, ax = plt.subplots(figsize=(22, 7))
    _bar_panel(ax, np.arange(n_zmodes), means, errs, geo_names,
               x_labels=x_labels,
               xlabel='OSA/ANSI Zernike Index',
               title='Fig 11: Zernike RMSE per OSA Mode\n(OPD$_{TT}$ measurements)')
    plt.tight_layout()
    for ext in ('pdf', 'png'):
        out = OUT_DIR / f'fig11_zernike_osa_modes.{ext}'
        fig.savefig(out, bbox_inches='tight', dpi=200)
        print(f'Saved {out}')
    plt.show()


def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f'Data not found: {DATA_FILE}\n'
            'Run fig10_11_data_collection.py first.'
        )
    data        = np.load(DATA_FILE, allow_pickle=True)
    zernike_mse = data['zernike_mse']          # (N_VOLS, n_geos, n_zmodes)
    geo_names   = _decode(data['geometry_names'])
    n_osa_modes = int(data['n_osa_modes'])

    plot_fig10(zernike_mse, geo_names, n_osa_modes)
    plot_fig11(zernike_mse, geo_names, n_osa_modes)


if __name__ == '__main__':
    main()
