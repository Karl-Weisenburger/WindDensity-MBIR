"""
Figs 10 and 11: Zernike error analysis — 3v2 vs 11v16 geometries.

Both figures use OPD_TT measurements and compute the error at the
ZERN_RESOLUTION (= 4 OPL planes) fixed in the data collection script.
Error is shown as normalized MSE (MSE / GT-mean-square).

Fig 10: Normalized MSE projected onto Zernike *radial degree* subspaces
        (degrees 0 – 8) plus a 'Higher Modes' bucket for leftover error.

Fig 11: Normalized MSE for individual OSA/ANSI indices 0 – 20 plus a
        'Higher Modes' bucket. Zernike basis thumbnails and a radial-degree
        grouping bar are drawn under the x-axis.

Data: data/fig10_11_zernike.npz
  zernike_mse: (N_VOLS, n_geos, n_zmodes)   n_zmodes = 45 OSA + 1 total
  gt_mean_sq : (N_VOLS,)                    mean-square of GT at ZERN_RESOLUTION
  geometry_names: ['3v2', '11v16']
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))

import math
from functools import lru_cache

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import seaborn as sns

# ---- Paths ----------------------------------------------------------------
DATA_FILE = Path(__file__).parent / 'data' / 'fig10_11_zernike.npz'
OUT_DIR   = Path(__file__).parent / 'figures'

# ---- Style ----------------------------------------------------------------
sns.set_style('whitegrid')
Y_NAME  = '(Mean Squared Error) / (Mean Squared Ground Truth)'
Y_TITLE = 'Normalized Mean Squared Error'

GEO_LABELS_DISPLAY = [r'3-View $2^\circ$ Geometry', r'11-View $16^\circ$ Geometry']


# ==========================================================================
# Zernike / OSA helpers
# ==========================================================================
@lru_cache(maxsize=None)
def _fact(k):
    return math.factorial(k) if k >= 0 else 0


def _radial_zernike(n, abs_m, rho):
    rad = np.zeros_like(rho)
    for s in range((n - abs_m) // 2 + 1):
        sign = (-1) ** s
        coeff = _fact(n - s) / (
            _fact(s) * _fact((n + abs_m) // 2 - s) * _fact((n - abs_m) // 2 - s)
        )
        rad += sign * coeff * rho ** (n - 2 * s)
    return rad


def _zernike(n, m, rho, theta):
    abs_m = abs(m)
    kron = 1 if m == 0 else 0
    norm = np.sqrt((2 * (n + 1)) / np.pi / (1 + kron))
    radial = _radial_zernike(n, abs_m, rho)
    if m > 0:
        ang = np.cos(m * theta)
    elif m < 0:
        ang = np.sin(abs_m * theta)
    else:
        ang = np.ones_like(theta)
    return norm * radial * ang


def _osa_to_nm(j):
    if j == 0:
        return 0, 0
    n = int(np.ceil((-3 + np.sqrt(9 + 8 * j)) / 2))
    m = 2 * j - n * (n + 2)
    return n, m


def _get_osa_modes(max_j):
    return [_osa_to_nm(j) for j in range(max_j + 1)]


def _generate_basis_thumbnails(modes, size=50):
    thumbnails = []
    y, x = np.mgrid[-1:1:size * 1j, -1:1:size * 1j]
    rho = np.sqrt(x ** 2 + y ** 2)
    theta = np.arctan2(y, x)
    mask = rho <= 1
    for n, m in modes:
        Z = np.zeros_like(rho)
        Z[mask] = _zernike(n, m, rho[mask], theta[mask])
        Z[~mask] = np.nan
        abs_max = np.nanmax(np.abs(Z))
        norm = plt.Normalize(
            vmin=-abs_max if abs_max != 0 else -1,
            vmax= abs_max if abs_max != 0 else  1,
        )
        rgba = cm.jet(norm(Z))
        rgba[np.isnan(Z), 3] = 0.0
        thumbnails.append(rgba)
    return thumbnails


def _aggregate_osa_to_radial(zern_mse, max_n=8):
    """
    zern_mse: (vols, geos, 46)  — 45 OSA modes + total at index -1
    Returns:  (vols, geos, max_n + 2) — degrees 0..max_n + total at index -1
    """
    vols, geos, _ = zern_mse.shape
    agg = np.zeros((vols, geos, max_n + 2))
    for j in range(45):
        n, _ = _osa_to_nm(j)
        if n <= max_n:
            agg[..., n] += zern_mse[..., j]
    agg[..., -1] = zern_mse[..., -1]
    return agg


def _decode(arr):
    return [s.decode() if isinstance(s, bytes) else s for s in arr]


# ==========================================================================
# Fig 10 — radial degree
# ==========================================================================
def plot_fig10(zern_mse, gt_mean_sq):
    zern_rad = _aggregate_osa_to_radial(zern_mse, max_n=8)  # (vols, geos, 10)
    num_samples, n_geos, _ = zern_rad.shape

    # Normalize and compute leftover ('Higher Modes')
    error_array = zern_rad.copy() / gt_mean_sq[:, None, None]
    error_array[:, :, -1] = (
        zern_rad[:, :, -1] - np.sum(zern_rad[:, :, :-1], axis=-1)
    ) / gt_mean_sq[:, None]

    zernike_deg_vec = list(range(9))
    zernike_legend = [f'degree {i}' for i in zernike_deg_vec]
    zernike_legend[0] = 'degree 0\n(piston)'
    zernike_legend[1] = 'degree 1\n(tip-tilt)'
    zernike_legend.append('Higher Modes')

    multi_index = pd.MultiIndex.from_product(
        [list(range(num_samples)), GEO_LABELS_DISPLAY, zernike_legend],
        names=['Volume Index', 'Geometry', 'Zernike Radial Degree'],
    )
    df = pd.DataFrame(error_array.reshape(-1), index=multi_index).reset_index()
    df.rename(columns={0: Y_NAME}, inplace=True)

    g = sns.catplot(
        data=df, kind='bar',
        x='Zernike Radial Degree', y=Y_NAME, hue='Geometry',
        errorbar=('se', 2),
        alpha=0.6, height=4.5, capsize=0.5,
    )
    g.despine(left=True)
    g.set_axis_labels('Zernike Radial Degree', Y_NAME)
    g.legend.set_title('Geometry Used')
    g.fig.set_figwidth(18)
    sns.move_legend(g, bbox_to_anchor=(0.65, 0.82), loc='upper right')
    plt.title(
        f'{Y_TITLE} projected onto Zernike radial degree subspaces \n'
        f' *Computed for 4 OPL planes along depth axis*'
    )
    g.tight_layout()

    for ext in ('pdf', 'png'):
        out = OUT_DIR / f'fig10_zernike_radial.{ext}'
        g.savefig(out, dpi=300, bbox_inches='tight')
        print(f'Saved {out}')


# ==========================================================================
# Fig 11 — OSA modes 0..20 (with thumbnails)
# ==========================================================================
def plot_fig11(zern_mse, gt_mean_sq):
    num_samples, n_geos, _ = zern_mse.shape
    max_j = 20
    num_modes = max_j + 1

    # Build truncated array: first 21 OSA modes + total in last slot
    zern_trunc = np.zeros((num_samples, n_geos, num_modes + 1))
    zern_trunc[..., :-1] = zern_mse[..., :num_modes]
    zern_trunc[..., -1]  = zern_mse[..., -1]

    error_array = zern_trunc.copy() / gt_mean_sq[:, None, None]
    error_array[:, :, -1] = (
        zern_trunc[:, :, -1] - np.sum(zern_trunc[:, :, :-1], axis=-1)
    ) / gt_mean_sq[:, None]

    zernike_legend = [str(j) for j in range(num_modes)] + ['Higher\nModes']

    multi_index = pd.MultiIndex.from_product(
        [list(range(num_samples)), GEO_LABELS_DISPLAY, zernike_legend],
        names=['Volume Index', 'Geometry', 'OSA/ANSI Index (j)'],
    )
    df = pd.DataFrame(error_array.reshape(-1), index=multi_index).reset_index()
    df.rename(columns={0: Y_NAME}, inplace=True)

    g = sns.catplot(
        data=df, kind='bar',
        x='OSA/ANSI Index (j)', y=Y_NAME, hue='Geometry',
        errorbar=('se', 2),
        alpha=0.6, height=4.5, aspect=12 / 4.5, capsize=0.3,
    )
    g.despine(left=True)
    g.legend.set_title('Geometry Used')
    sns.move_legend(g, bbox_to_anchor=(0.85, 0.82), loc='upper right')
    g.ax.set_title(
        f'{Y_TITLE} projected onto Zernike modes \n'
        f' *Computed for 4 OPL planes along depth axis*'
    )
    g.ax.set_xticks(np.arange(len(zernike_legend)))
    g.ax.set_xticklabels(zernike_legend)

    # Zernike basis thumbnails under each tick
    thumbnails = _generate_basis_thumbnails(_get_osa_modes(max_j), size=50)
    for j, img in enumerate(thumbnails):
        ab = AnnotationBbox(
            OffsetImage(img, zoom=0.30),
            (j, 0),
            xybox=(0, -28),
            xycoords=('data', 'axes fraction'),
            boxcoords='offset points',
            pad=0, frameon=False,
        )
        g.ax.add_artist(ab)

    # Radial-degree grouping bar
    n_groups = {}
    for j in range(max_j + 1):
        n, _m = _osa_to_nm(j)
        if n not in n_groups:
            n_groups[n] = [j, j]
        else:
            n_groups[n][1] = j

    line_y = -0.21
    text_y = -0.23
    for n, (start, end) in n_groups.items():
        g.ax.plot(
            [start - 0.4, end + 0.4], [line_y, line_y],
            color='black', lw=1.2, clip_on=False,
            transform=g.ax.get_xaxis_transform(),
        )
        g.ax.text(
            (start + end) / 2, text_y, f'n={n}',
            ha='center', va='top', fontsize=10, fontweight='bold',
            transform=g.ax.get_xaxis_transform(),
        )

    g.ax.set_xlabel(
        r'OSA/ANSI Index  |  $\mathbf{Radial\ Degree\ (n)}$',
        labelpad=30, fontsize=12,
    )
    g.fig.subplots_adjust(top=0.85, bottom=0.28)

    for ext in ('pdf', 'png'):
        out = OUT_DIR / f'fig11_zernike_osa_modes.{ext}'
        g.savefig(out, dpi=300, bbox_inches='tight')
        print(f'Saved {out}')


# ==========================================================================
def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f'Data not found: {DATA_FILE}\nRun fig10_11_data_collection.py first.'
        )
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    data       = np.load(DATA_FILE, allow_pickle=True)
    zern_mse   = data['zernike_mse']              # (N_VOLS, n_geos, n_zmodes)
    gt_mean_sq = data['gt_mean_sq']               # (N_VOLS,)
    geo_names  = _decode(data['geometry_names'])

    assert geo_names == ['3v2', '11v16'], f'Unexpected geometries: {geo_names}'

    plot_fig10(zern_mse, gt_mean_sq)
    plot_fig11(zern_mse, gt_mean_sq)
    plt.show()


if __name__ == '__main__':
    main()
