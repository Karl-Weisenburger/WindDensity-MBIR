"""
Fig 9: Regional NRMSE along flow axis — 3v2 vs 3v16, withTTP vs noTTP.

Data: data/fig9_regional_nrmse.npz
  nrmse_regional: (N_VOLS, n_ttp, n_geos, N_SECTIONS)
  geometry_names: e.g. ['3v2', '3v16']
  ttp_states:     ['withTTP', 'noTTP']
  section_bounds: (N_SECTIONS, 2) row-index start/end for each section
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))

import numpy as np
import matplotlib.pyplot as plt

# ---- Paths ----------------------------------------------------------------
DATA_FILE = Path(__file__).parent / 'data' / 'fig9_regional_nrmse.npz'
OUT_DIR   = Path(__file__).parent / 'data'

# ---- Shared geometry parameters -------------------------------------------
CM_PER_PIXEL = 25.0 / 800

# ---- Style ----------------------------------------------------------------
STYLE = {
    ('3v2',  'withTTP'): dict(color='#0072B2', linestyle='-',  marker='o', label='3v2, OPL'),
    ('3v2',  'noTTP'):   dict(color='#0072B2', linestyle='--', marker='s', label='3v2, OPD$_{TT}$'),
    ('3v16', 'withTTP'): dict(color='#D55E00', linestyle='-',  marker='^', label='3v16, OPL'),
    ('3v16', 'noTTP'):   dict(color='#D55E00', linestyle='--', marker='D', label='3v16, OPD$_{TT}$'),
}


def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f'Data not found: {DATA_FILE}\n'
            'Run fig9_data_collection.py first.'
        )

    data = np.load(DATA_FILE, allow_pickle=True)
    nrmse_regional = data['nrmse_regional']    # (N_VOLS, n_ttp, n_geos, N_SECTIONS)
    geo_names      = [s.decode() if isinstance(s, bytes) else s for s in data['geometry_names']]
    ttp_states     = [s.decode() if isinstance(s, bytes) else s for s in data['ttp_states']]
    section_bounds = data['section_bounds']    # (N_SECTIONS, 2)

    n_vols, n_ttp, n_geos, n_sections = nrmse_regional.shape
    section_centers_cm = (section_bounds[:, 0] + section_bounds[:, 1]) / 2 * CM_PER_PIXEL

    fig, ax = plt.subplots(figsize=(10, 6))

    for geo_idx, geo_name in enumerate(geo_names):
        for ttp_idx, ttp_state in enumerate(ttp_states):
            style_key = (geo_name, ttp_state)
            if style_key not in STYLE:
                continue
            sty = STYLE[style_key]

            y_pct = nrmse_regional[:, ttp_idx, geo_idx, :] * 100
            mean_y = y_pct.mean(axis=0)
            err    = 2 * y_pct.std(axis=0) / np.sqrt(n_vols)

            ax.errorbar(
                section_centers_cm, mean_y, yerr=err,
                color=sty['color'], linestyle=sty['linestyle'],
                marker=sty['marker'], label=sty['label'],
                capsize=4, linewidth=1.5, markersize=6,
            )

    ax.set_xlabel('Position along Flow Axis (cm)', fontsize=14)
    ax.set_ylabel('NRMSE (%)', fontsize=14)
    ax.set_title('Fig 9: Regional NRMSE along Flow Axis', fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True)
    plt.tight_layout()

    for ext in ('pdf', 'png'):
        out = OUT_DIR / f'fig9_regional_nrmse.{ext}'
        fig.savefig(out, bbox_inches='tight', dpi=200)
        print(f'Saved {out}')
    plt.show()


if __name__ == '__main__':
    main()
