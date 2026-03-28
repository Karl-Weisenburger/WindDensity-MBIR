"""
Shared prep utilities for single-volume reconstruction visualization (Figs 6, 12, 16).

Provides:
  load_or_generate_volume   — load cached volume or generate and save
  prepare_opl_images        — OPL planes → beam cross-section images + ROI
  compute_per_section_nrmse — per-section NRMSE between GT and recon beam images
  plot_recon_figure         — standard subfigure layout used across all three figures
"""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

import numpy as np
import jax.numpy as jnp
from jax import random
import matplotlib.pyplot as plt

import winddensity_mbir.simulation as sim
import winddensity_mbir.visualization_and_analysis as va
import winddensity_mbir.utilities as utils


# ---------------------------------------------------------------------------
# Volume loading / generation
# ---------------------------------------------------------------------------

def load_or_generate_volume(seed, recon_shape, delta, L0, cn2, cache_path):
    """
    Return a cached 3-D atmospheric volume (float32 ndarray), generating and
    saving it on the first call.

    Args:
        seed (int): JAX PRNGKey seed (also used as the volume identifier).
        recon_shape (tuple): (num_rows, num_cols, num_slices)
        delta (float): Turbulence inner-scale parameter (m).
        L0 (float): Outer scale (m).
        cn2 (float): Cn² structure constant.
        cache_path (str or Path): Where to save/load the .npy file.

    Returns:
        np.ndarray: Volume of shape recon_shape.
    """
    cache_path = Path(cache_path)
    if cache_path.exists():
        print(f'Loading cached volume from {cache_path}')
        return np.load(cache_path)

    print(f'Generating volume (seed={seed})…')
    key = random.PRNGKey(seed)
    vol = sim.generate_random_atmospheric_volume(cn2=cn2, dim=recon_shape, delta=delta, L0=L0, key=key)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, np.array(vol))
    print(f'Saved to {cache_path}')
    return np.array(vol)


# ---------------------------------------------------------------------------
# OPL image preparation
# ---------------------------------------------------------------------------

def prepare_opl_images(vol_3d, zern_mode_index, sections, total_length_m,
                        beam_pixel_diam, num_cols, num_slices):
    """
    Compute sectioned OPL planes from a 3-D volume, apply Zernike processing,
    and crop to the beam cross-section.

    zern_mode_index:
      0 — raw OPL (no Zernike removal)
      1 — OPD: piston removed
      2 — OPD_TT: piston + tip + tilt removed

    Returns:
        beam_images : np.ndarray, shape (sections, beam_pixel_diam, num_slices)
        roi_beam    : np.ndarray bool, shape (sections, beam_pixel_diam, num_slices)
    """
    vol_jnp = jnp.array(vol_3d)
    opl = va.divide_into_sections_of_opl(vol_jnp, sections, total_length_m)
    # opl: (sections, num_cols, num_slices)

    roi_full = jnp.array(
        va.generate_beam_path_roi_mask((sections, num_cols, num_slices), beam_pixel_diam)
    )

    if zern_mode_index == 0:
        processed = opl
    elif zern_mode_index == 1:
        processed = utils.remove_piston(opl, FOV=roi_full)
    elif zern_mode_index == 2:
        processed = utils.remove_tip_tilt_piston(opl, FOV=roi_full)
    else:
        raise ValueError(f'Unsupported zern_mode_index={zern_mode_index}. Use 0, 1, or 2.')

    # Crop columns to beam path (centered on the depth axis)
    ind1 = num_cols // 2 - beam_pixel_diam // 2
    ind2 = num_cols // 2 + beam_pixel_diam // 2
    beam_images = np.array(processed)[:, ind1:ind2, :]      # (sections, beam_pixel_diam, num_slices)
    roi_beam    = np.array(roi_full)[:, ind1:ind2, :]        # same shape, bool

    return beam_images, roi_beam


# ---------------------------------------------------------------------------
# Per-section NRMSE
# ---------------------------------------------------------------------------

def compute_per_section_nrmse(gt_beam, recon_beam, roi_beam):
    """
    Compute NRMSE for each section between GT and reconstruction beam images.

    Args:
        gt_beam, recon_beam : np.ndarray (sections, beam_pixel_diam, num_slices)
        roi_beam            : np.ndarray bool, same shape

    Returns:
        list of float — NRMSE per section (option=2: interpercentile range normalisation)
    """
    sections = gt_beam.shape[0]
    nrmse_list = []
    for s in range(sections):
        nrmse_s = float(va.nrmse_over_roi(
            jnp.array(gt_beam[s]),
            jnp.array(recon_beam[s]),
            jnp.array(roi_beam[s]),
            option=2,
        ))
        nrmse_list.append(nrmse_s)
    return nrmse_list


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_recon_figure(gt_images, recon_images_list, nrmse_per_section_list,
                      recon_labels, gt_suptitle, fig_suptitle):
    """
    Standard reconstruction visualisation layout (Figs 6, 12, 16).

    Layout: one subfigure per row (GT first, then one per reconstruction).
    Each row has one subplot per section (4 subplots).
    vmin/vmax is shared between GT and the first reconstruction only.

    Args:
        gt_images              : np.ndarray (sections, H, W) — GT beam images
        recon_images_list      : list of (sections, H, W) arrays
        nrmse_per_section_list : list of lists — nrmse_per_section_list[r][s] = NRMSE
        recon_labels           : list of str — subfigure suptitles for each recon row
        gt_suptitle            : str — subfigure suptitle for GT row
        fig_suptitle           : str — overall figure suptitle

    Returns:
        matplotlib.figure.Figure
    """
    sections  = gt_images.shape[0]
    n_recons  = len(recon_images_list)
    figsize   = (sections * 4.3, (n_recons + 1) * 4.9)

    fig = plt.figure(figsize=figsize)
    fig.suptitle(fig_suptitle, fontsize=40, y=1.06)

    subfigs = fig.subfigures(nrows=n_recons + 1, ncols=1)

    # vmin/vmax: GT and first recon share the same range
    shared_vals = np.concatenate([gt_images.ravel(), recon_images_list[0].ravel()])
    vmin_shared = float(shared_vals.min())
    vmax_shared = float(shared_vals.max())

    # ---------- GT row ----------
    _fill_row(
        subfig       = subfigs[0],
        images       = gt_images,
        subtitles    = [f'Region {i + 1}' for i in range(sections)],
        title_kwargs = dict(fontsize=30, y=1.25, fontweight='bold'),
        vmin         = vmin_shared,
        vmax         = vmax_shared,
        row_suptitle = gt_suptitle,
        suptitle_kwargs = dict(fontsize=30, x=0.02, y=0.85, ha='left'),
    )

    # ---------- Reconstruction rows ----------
    for ri, (recon_imgs, nrmse_list, label) in enumerate(
        zip(recon_images_list, nrmse_per_section_list, recon_labels)
    ):
        if ri == 0:
            vmin_r, vmax_r = vmin_shared, vmax_shared
        else:
            vmin_r = float(recon_imgs.min())
            vmax_r = float(recon_imgs.max())

        _fill_row(
            subfig       = subfigs[ri + 1],
            images       = recon_imgs,
            subtitles    = [f'NRMSE={v * 100:.2f}%' for v in nrmse_list],
            title_kwargs = dict(fontsize=20),
            vmin         = vmin_r,
            vmax         = vmax_r,
            row_suptitle = label,
            suptitle_kwargs = dict(fontsize=30, x=0.02, y=1.0, ha='left'),
        )

    return fig


def _fill_row(subfig, images, subtitles, title_kwargs, vmin, vmax,
              row_suptitle, suptitle_kwargs):
    """Helper: populate one subfigure row with imshow subplots."""
    sections = images.shape[0]
    subfig.suptitle(row_suptitle, **suptitle_kwargs)
    axes = subfig.subplots(1, sections)
    if sections == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        H, W = images[i].shape
        im_ratio = W / H

        im = ax.imshow(
            images[i].T, cmap='jet', vmin=vmin, vmax=vmax,
            extent=(0, 0.02, 0, 0.02),
        )
        ax.set_title(subtitles[i], **title_kwargs)
        ax.set_xticks([])
        ax.set_yticks([])

        cb = plt.colorbar(im, ax=ax, fraction=0.048 * im_ratio)
        cb.ax.tick_params(labelsize=15)
        cb.ax.yaxis.offsetText.set_fontsize(15)
        cb.ax.yaxis.offsetText.set_position((1.0, 0))
