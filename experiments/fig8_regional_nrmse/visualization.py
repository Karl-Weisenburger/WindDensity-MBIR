"""
Fig 8: Geometry overhead schematics for 3v2 and 3v16 (no data file needed).
Fig 9: Regional NRMSE along flow axis — 3v2 vs 3v16, withTTP vs noTTP.

Data: data/fig8_regional_nrmse.npz
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
import winddensity_mbir.visualization_and_analysis as va
import winddensity_mbir.configuration_params as config

# ---- Paths ----------------------------------------------------------------
DATA_FILE = Path(__file__).parent / 'data' / 'fig8_regional_nrmse.npz'
OUT_DIR   = Path(__file__).parent / 'data'

# ---- Shared geometry parameters -------------------------------------------
CM_PER_PIXEL   = 25.0 / 800
RECON_SHAPE    = (640, 400, 64)
NUM_ROWS, NUM_COLS, NUM_SLICES = RECON_SHAPE
TEST_REGION_DIMS = (
    NUM_ROWS  * CM_PER_PIXEL,   # 20 cm — flow direction
    NUM_COLS  * CM_PER_PIXEL,   # 12.5 cm — depth direction
    NUM_SLICES * CM_PER_PIXEL,  # 2 cm — vertical
)
BEAM_DIAM_CM = 2.0

# ---- Style ----------------------------------------------------------------
# Lines: (geo_label, ttp_label) -> color, linestyle, marker
STYLE = {
    ('3v2',  'withTTP'): dict(color='#0072B2', linestyle='-',  marker='o', label='3v2, OPL'),
    ('3v2',  'noTTP'):   dict(color='#0072B2', linestyle='--', marker='s', label='3v2, OPD$_{TT}$'),
    ('3v16', 'withTTP'): dict(color='#D55E00', linestyle='-',  marker='^', label='3v16, OPL'),
    ('3v16', 'noTTP'):   dict(color='#D55E00', linestyle='--', marker='D', label='3v16, OPD$_{TT}$'),
}


# ============================================================
# Fig 8: geometry schematics
# ============================================================

def _build_optical_setup(half_extent_deg, n_views):
    angles_rad = np.linspace(-half_extent_deg, half_extent_deg, n_views, endpoint=True) * np.pi / 180
    return config.define_optical_setup(
        sensor_locations=[(0.0, 0.0)],
        beam_angles=[list(angles_rad)],
        test_region_dims=TEST_REGION_DIMS,
        pixel_pitch=CM_PER_PIXEL,
        beam_fov=BEAM_DIAM_CM,
    )


def plot_fig8():
    """Plot Fig 8a (3v2) and Fig 8b (3v16) geometry overhead schematics."""
    setups = [
        ('Fig 8a: 3 views, 2° total (3v2)',  1.0, 3),
        ('Fig 8b: 3 views, 16° total (3v16)', 8.0, 3),
    ]

    for title, half_ext, n_views in setups:
        optical_setup = _build_optical_setup(half_ext, n_views)
        fig, ax = va.display_viewing_configuration_schematic(
            optical_setup=optical_setup,
            show_beam_diameter=True,
            scale=1.0,
            title=title,
            plane='transverse',
            outer_buffer=(1.0, 2.0),
            roi_thickness_and_num_regions=(BEAM_DIAM_CM, 5),
            legend_scale=1.2,
        )
        label = '3v2' if n_views == 3 and half_ext == 1.0 else '3v16'
        for ext in ('pdf', 'png'):
            out = OUT_DIR / f'fig8_{label}_schematic.{ext}'
            fig.savefig(out, bbox_inches='tight', dpi=200)
            print(f'Saved {out}')
        plt.close(fig)


# ============================================================
# Fig 9: regional NRMSE
# ============================================================

def plot_fig9(data):
    nrmse_regional = data['nrmse_regional']    # (N_VOLS, n_ttp, n_geos, N_SECTIONS)
    geo_names      = [s.decode() if isinstance(s, bytes) else s for s in data['geometry_names']]
    ttp_states     = [s.decode() if isinstance(s, bytes) else s for s in data['ttp_states']]
    section_bounds = data['section_bounds']     # (N_SECTIONS, 2)

    n_vols, n_ttp, n_geos, n_sections = nrmse_regional.shape

    # x-axis: center row of each section, converted to cm
    section_centers_cm = (section_bounds[:, 0] + section_bounds[:, 1]) / 2 * CM_PER_PIXEL

    fig, ax = plt.subplots(figsize=(10, 6))

    for geo_idx, geo_name in enumerate(geo_names):
        for ttp_idx, ttp_state in enumerate(ttp_states):
            style_key = (geo_name, ttp_state)
            if style_key not in STYLE:
                continue
            sty = STYLE[style_key]

            y_pct = nrmse_regional[:, ttp_idx, geo_idx, :] * 100   # (N_VOLS, N_SECTIONS)
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


# ============================================================
# Main
# ============================================================

def main():
    # Fig 8 (no data needed)
    plot_fig8()

    # Fig 9 (needs data)
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f'Data not found: {DATA_FILE}\n'
            'Run data_collection.py first.'
        )
    data = np.load(DATA_FILE, allow_pickle=True)
    plot_fig9(data)


if __name__ == '__main__':
    main()
