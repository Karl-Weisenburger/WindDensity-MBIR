"""
Fig 9: Regional NRMSE along flow axis — 3v2 vs 3v16, OPD_TT measurements.

Data: data/fig9_regional_nrmse.npz
  nrmse_regional: (N_VOLS, n_geos, N_SECTIONS)
    geo 0 = '3v2'  (3 views, 2° total)
    geo 1 = '3v16' (3 views, 16° total)
  geometry_names: ['3v2', '3v16']
  section_bounds: (N_SECTIONS, 2) row-index start/end for each section
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))

import numpy as np
import matplotlib.pyplot as plt

# ---- Paths ----------------------------------------------------------------
DATA_FILE = Path(__file__).parent / 'data' / 'fig9_regional_nrmse.npz'
OUT_DIR   = Path(__file__).parent / 'figures'


def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f'Data not found: {DATA_FILE}\n'
            'Run fig9_data_collection.py first.'
        )
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    data           = np.load(DATA_FILE, allow_pickle=True)
    nrmse_regional = data['nrmse_regional']   # (N_VOLS, n_geos, N_SECTIONS)
    n_vols, n_geos, n_sections = nrmse_regional.shape

    avg_vals = np.mean(nrmse_regional, axis=0)                                          # (n_geos, N_SECTIONS)
    SD_3v2   = 2 * np.std(nrmse_regional[:, 0, :], axis=0, ddof=1) / np.sqrt(n_vols)  # (N_SECTIONS,)
    SD_3v16  = 2 * np.std(nrmse_regional[:, 1, :], axis=0, ddof=1) / np.sqrt(n_vols)  # (N_SECTIONS,)

    region_ind = [i + 1 for i in range(n_sections)]

    custom_markers = ['o', 's', '^', 'D', 'x', '+']
    custom_colors  = ['#0072B2', '#D55E00', '#CC79A7', '#009E73', '#F0E442', '#56B4E9']
    plt.rcParams['axes.prop_cycle'] = plt.cycler(marker=custom_markers, color=custom_colors)

    plt.figure(figsize=(10, 4))
    plt.errorbar(region_ind, avg_vals[0, :], fmt='^-', yerr=SD_3v2,  capsize=5)
    plt.errorbar(region_ind, avg_vals[1, :], fmt='o-', yerr=SD_3v16, capsize=5)

    plt.legend(['3 views 2 degrees', '3 views 16 degrees'])
    plt.xticks(region_ind)
    plt.xlabel('Region in Wind Tunnel')
    plt.ylabel('NRMSE')
    plt.title('Average NRMSE of OPL Planes Relative to Location Along the Depth Axis')
    plt.grid(True)
    plt.tight_layout()

    for ext in ('pdf', 'png'):
        out = OUT_DIR / f'fig9_regional_nrmse.{ext}'
        plt.savefig(out, bbox_inches='tight', dpi=200)
        print(f'Saved {out}')
    plt.show()


if __name__ == '__main__':
    main()
