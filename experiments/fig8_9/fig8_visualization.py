"""
Fig 8: Geometry overhead schematics for 3v2 and 3v16 (no data file needed).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))

import numpy as np
import matplotlib.pyplot as plt
import winddensity_mbir.visualization_and_analysis as va
import winddensity_mbir.configuration_params as config

# ---- Paths ----------------------------------------------------------------
OUT_DIR = Path(__file__).parent / 'data'

# ---- Shared geometry parameters -------------------------------------------
CM_PER_PIXEL   = 25.0 / 800
RECON_SHAPE    = (640, 400, 64)
NUM_ROWS, NUM_COLS, NUM_SLICES = RECON_SHAPE
TEST_REGION_DIMS = (
    NUM_ROWS  * CM_PER_PIXEL,
    NUM_COLS  * CM_PER_PIXEL,
    NUM_SLICES * CM_PER_PIXEL,
)
BEAM_DIAM_CM = 2.0


def _build_optical_setup(half_extent_deg, n_views):
    angles_rad = np.linspace(-half_extent_deg, half_extent_deg, n_views, endpoint=True) * np.pi / 180
    return config.define_optical_setup(
        sensor_locations=[(0.0, 0.0)],
        beam_angles=[list(angles_rad)],
        test_region_dims=TEST_REGION_DIMS,
        pixel_pitch=CM_PER_PIXEL,
        beam_fov=BEAM_DIAM_CM,
    )


def main():
    """Plot Fig 8a (3v2) and Fig 8b (3v16) geometry overhead schematics."""
    setups = [
        ('Fig 8a: 3 views, 2° total (3v2)',   1.0, 3),
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
        label = '3v2' if half_ext == 1.0 else '3v16'
        for ext in ('pdf', 'png'):
            out = OUT_DIR / f'fig8_{label}_schematic.{ext}'
            fig.savefig(out, bbox_inches='tight', dpi=200)
            print(f'Saved {out}')
        plt.close(fig)


if __name__ == '__main__':
    main()
