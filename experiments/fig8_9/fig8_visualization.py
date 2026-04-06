"""
Fig 8: Geometry overhead schematics for 3v2 (a) and 3v16 (b), side by side.

No data file needed — geometry schematics are drawn directly from parameters.
"""

import io
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import winddensity_mbir.visualization_and_analysis as va

# ---- Paths ----------------------------------------------------------------
OUT_DIR = Path(__file__).parent / 'figures'

# ---- Panel definitions (matching original code) --------------------------
# Each panel: (extent_deg, title)
#   angles = [-extent, 0, extent] in degrees → 3 views, 2*extent total
PANELS = [
    (1, r'3 views and $2^\circ$ of overall angular extent'),
    (8, r'3 views and $16^\circ$ of overall angular extent'),
]

# Shared schematic parameters (from original code)
SCHEMATIC_KWARGS = dict(
    sensor_locations=[(0, 0)],
    beam_diameter=2.0,
    show_beam_diameter=True,
    scale=1,
    overlap_threshold=0,
    dims=(20, 12, 20),
    plane='transverse',
    outer_buffer=(1, 2),
    roi_thickness_and_num_regions=(2, 5),
    legend_scale=1.2,
)

# ---- Label sizing (consistent with fig 7 reference: figwidth=20, fontsize=24)
COMBINED_FIG_W = 20
LABEL_FONTSIZE = 30
LABEL_Y        = -0.03
RENDER_DPI     = 300


def _render_panel(extent_deg, title):
    """Render one schematic panel to an RGBA image array."""
    angles = [np.array([-extent_deg, 0, extent_deg]) * np.pi / 180]
    fig, _ax = va.display_viewing_configuration_schematic(
        angles=angles, title=title, **SCHEMATIC_KWARGS,
    )
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=RENDER_DPI)
    plt.close(fig)
    buf.seek(0)
    return mpimg.imread(buf)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    imgs = [_render_panel(ext, title) for ext, title in PANELS]

    # Pad the shorter image so both have the same height
    max_h = max(im.shape[0] for im in imgs)
    padded = []
    for im in imgs:
        if im.shape[0] < max_h:
            pad = np.ones((max_h - im.shape[0], im.shape[1], im.shape[2]),
                          dtype=im.dtype)
            im = np.vstack([im, pad])
        padded.append(im)

    # Determine combined figure height to preserve aspect ratio
    total_w_px = sum(im.shape[1] for im in padded)
    aspect = max_h / total_w_px
    combined_fig_h = COMBINED_FIG_W * aspect

    fig, axes = plt.subplots(1, 2, figsize=(COMBINED_FIG_W, combined_fig_h))
    fig.subplots_adjust(wspace=0.02, left=0, right=1, top=1, bottom=0.08)

    for ax, im, label in zip(axes, padded, ('(a)', '(b)')):
        ax.imshow(im)
        ax.axis('off')
        ax.text(0.5, LABEL_Y, label, transform=ax.transAxes,
                ha='center', va='top', fontsize=LABEL_FONTSIZE)

    for ext in ('pdf', 'png'):
        out = OUT_DIR / f'fig8_geometry_schematics.{ext}'
        fig.savefig(out, bbox_inches='tight', dpi=RENDER_DPI)
        print(f'Saved {out}')
    plt.show()


if __name__ == '__main__':
    main()
