import jax
import jax.numpy as jnp
from wind_density_tomo.visualization_and_analysis import nrmse_over_roi
import matplotlib.pyplot as plt

def display_raw_data_and_processed_data(raw_data,processed_data,weight_matrix):
    """"""
    N = raw_data.shape[0]
    # Set values outside the roi to NaN for better visualization
    # find indices for roi in each plane
    vmin = min(jnp.nanmin(jnp.array(raw_data)),jnp.nanmin(jnp.array(processed_data)))
    vmax = max(jnp.nanmax(jnp.array(raw_data)),jnp.nanmax(jnp.array(processed_data)))
    scale = 0.75
    fig = plt.figure(figsize=(6 * N * scale, 12 * scale), constrained_layout=True)
    fig.suptitle('Comparing Raw Data to Processed Data', fontsize=18)
    subfigs = fig.subfigures(3, 1)
    subfigs[0].suptitle(f'Raw Data', fontsize=16, fontstyle='italic')
    subfigs[1].suptitle(f'Processed Data', fontsize=16, fontstyle='italic',y=0.85)
    subfigs[2].suptitle(f'CT Weight Matrix (i.e., FOV)', fontsize=16, fontstyle='italic',y=0.85)
    axes_raw = subfigs[0].subplots(1, N)
    axes_proc = subfigs[1].subplots(1, N)
    axes_weight = subfigs[2].subplots(1, N)
    for i in range(N):
        axes_raw[i].imshow(raw_data[i], cmap='jet', origin='lower', vmin=vmin, vmax=vmax)  # , extent=extent)
        axes_raw[i].set_title(f'View {i+1}', fontsize=12)
        axes_raw[i].set_xticks([])
        axes_raw[i].set_yticks([])
        axes_raw[i].grid(False)

        axes_proc[i].imshow(processed_data[i], cmap='jet', origin='lower', vmin=vmin,
                                   vmax=vmax)  # , extent=extent)
        axes_proc[i].set_title(f'View {i+1}', fontsize=12)
        axes_proc[i].set_xticks([])
        axes_proc[i].set_yticks([])
        axes_proc[i].grid(False)

        axes_weight[i].imshow(weight_matrix[i], cmap='jet', origin='lower')
        axes_weight[i].set_title(f'View {i + 1}', fontsize=12)
        axes_weight[i].set_xticks([])
        axes_weight[i].set_yticks([])
        axes_weight[i].grid(False)


def display_planes_from_recon_and_ground_truth(recon_planes, gt_planes, roi_planes, title=f'Reconstruction of planes'):
    """
    Display N planes from the reconstruction and ground truth. Includes options to remove the TT from the reconstructed planes

    Args:
        recon_planes (jax.numpy.ndarray): Reconstructed planes with shape (N, H, W).
        gt_planes (jax.numpy.ndarray): Ground truth planes with shape (N, H, W).
        roi_planes (jax.numpy.ndarray): Boolean mask defining the region of interest with shape (N, H, W).
        title (str): Title for display

    Returns:
    ￼Display  planes from the reconstruction and ground truth.
    """
    N=recon_planes.shape[0]
    #Set values outside the roi to NaN for better visualization
    recon_planes_processed=recon_planes.at[~roi_planes].set(jnp.nan)
    gt_planes_processed=gt_planes.at[~roi_planes].set(jnp.nan)
    #find indices for roi in each plane
    gt_images=[]
    recon_images=[]
    for i in range(N):
        roi_indices=jnp.where(roi_planes[i])
        min_row=jnp.min(roi_indices[0])
        max_row=jnp.max(roi_indices[0])
        min_col=jnp.min(roi_indices[1])
        max_col=jnp.max(roi_indices[1])
        gt_images.append(gt_planes_processed[i,min_row:max_row+1,min_col:max_col+1])
        recon_images.append(recon_planes_processed[i,min_row:max_row+1,min_col:max_col+1])

    vmin=jnp.nanmin(jnp.array(gt_images+recon_images))
    vmax=jnp.nanmax(jnp.array(gt_images+recon_images))
    scale=0.75
    fig=plt.figure(figsize=(4 * N*scale, 8*scale),constrained_layout=True)
    fig.suptitle(title, fontsize=18)
    subfigs = fig.subfigures(2, 1)
    subfigs[0].suptitle(f'Ground Truth Planes', fontsize=16,fontstyle='italic')
    subfigs[1].suptitle(f'Reconstruction Planes', fontsize=16,fontstyle='italic')
    axes_gt = subfigs[0].subplots(1, N)
    axes_recon = subfigs[1].subplots(1, N)
    for i in range(N):
        # extent = (0, gt_images[i].shape[1]*pixel_size, 0, gt_images[i].shape[0]*pixel_size)

        im1 = axes_gt[i].imshow(gt_images[i].T, cmap='jet', origin='lower', vmin=vmin, vmax=vmax) #, extent=extent)
        im_ratio = gt_images[i].shape[0] / gt_images[i].shape[1]
        axes_gt[i].set_title(f'Region {i+1}', fontsize=14,fontweight='bold')
        axes_gt[i].set_xlabel('streamwise-axis (cm)', fontsize=12)
        axes_gt[i].set_ylabel('vertical-axis (cm)', fontsize=12)
        axes_gt[i].grid(False)
        fig.colorbar(im1, ax=axes_gt[i],fraction=0.05*im_ratio)

        im2 = axes_recon[i].imshow(recon_images[i].T, cmap='jet', origin='lower', vmin=vmin, vmax=vmax) #, extent=extent)
        im_ratio = recon_images[i].shape[0] / recon_images[i].shape[1]
        axes_recon[i].set_title(f'NRMSE: {100 * nrmse_over_roi(recon_planes_processed[i], gt_planes_processed[i], roi_planes[i], option=2):.1f}%', fontsize=14)
        axes_recon[i].set_xlabel('streamwise-axis (cm)', fontsize=12)
        axes_recon[i].set_ylabel('vertical-axis (cm)', fontsize=12)
        axes_recon[i].grid(False)
        fig.colorbar(im2, ax=axes_recon[i],fraction=0.05*im_ratio)