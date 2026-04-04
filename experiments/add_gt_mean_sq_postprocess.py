"""
TEMPORARY post-processing script.

Adds `gt_mean_sq` field to existing data files that were generated before
the data collection scripts were updated to include it. Regenerates GT
volumes using the same PRNG seeds as the original data collection, computes
the mean-square of the GT OPL sections at ZERN_RESOLUTION inside the Zernike
ROI, and re-saves the npz with the new field added. Existing fields are
preserved unchanged.

Once the data collection scripts have been re-run end-to-end, this script
can be deleted — the updated collection scripts write `gt_mean_sq` directly.

Target files:
  experiments/fig10_11/data/fig10_11_zernike.npz
  experiments/fig13_14_15_17/data/fig13_14_15_17_7v8.npz
"""
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

import numpy as np
import jax.numpy as jnp
from jax import random
from tqdm import trange

import winddensity_mbir.simulation as sim
import winddensity_mbir.visualization_and_analysis as va

# ---- Parameters must match data collection scripts ----
CM_PER_PIXEL   = 25.0 / 800
RECON_SHAPE    = (640, 400, 64)
BEAM_DIAM_CM   = 2.0
BEAM_PIXEL_DIAM = int(round(BEAM_DIAM_CM / CM_PER_PIXEL))
DELTA          = 0.01 * CM_PER_PIXEL
ZERN_RESOLUTION = 4

NUM_ROWS, NUM_COLS, NUM_SLICES = RECON_SHAPE

TARGETS = [
    Path(__file__).parent / 'fig10_11' / 'data' / 'fig10_11_zernike.npz',
    Path(__file__).parent / 'fig13_14_15_17' / 'data' / 'fig13_14_15_17_7v8.npz',
]


def compute_gt_mean_sq(n_vols, roi_zern):
    arr = np.zeros(n_vols)
    for vol_idx in trange(n_vols, desc='GT mean square'):
        key = random.PRNGKey(vol_idx)
        vol_gt = sim.generate_random_atmospheric_volume(
            cn2=1e-11, dim=RECON_SHAPE, delta=DELTA, L0=0.02, key=key
        )
        _gt_z = np.array(va.divide_into_sections_of_opl(vol_gt, ZERN_RESOLUTION, 0.2))
        arr[vol_idx] = float(np.mean(_gt_z[roi_zern] ** 2))
    return arr


def patch_file(path: Path, roi_zern: np.ndarray):
    if not path.exists():
        print(f'[skip] {path} — file not found')
        return

    data = dict(np.load(path, allow_pickle=True))
    if 'gt_mean_sq' in data:
        print(f'[skip] {path} — gt_mean_sq already present')
        return

    # Determine N_VOLS from zernike_mse (first axis is the volume axis)
    n_vols = int(data['zernike_mse'].shape[0])
    print(f'[patch] {path} — computing gt_mean_sq for {n_vols} volumes')

    gt_mean_sq = compute_gt_mean_sq(n_vols, roi_zern)
    data['gt_mean_sq'] = gt_mean_sq

    np.savez(path, **data)
    print(f'[done]  {path} — saved with gt_mean_sq')


def main():
    roi_zern = np.array(va.generate_beam_path_roi_mask(
        (ZERN_RESOLUTION, NUM_COLS, NUM_SLICES), BEAM_PIXEL_DIAM,
        location=(0, 0, 0), angle=0,
    ))
    for path in TARGETS:
        patch_file(path, roi_zern)


if __name__ == '__main__':
    main()
