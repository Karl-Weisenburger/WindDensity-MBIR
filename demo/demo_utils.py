import jax
import jax.numpy as jnp
from wind_tomo.utilities import remove_tip_tilt_jax, jax_nrmse_ROI, divide_into_sections_of_OPL
import matplotlib.pyplot as plt
def display_N_planes_from_recon_and_ground_truth(recon,ground_truth,ROI,depth_axis_length,N=4,plane_type='OPD_TT'):
    """
    Display N planes from the reconstruction and ground truth. Includes options to remove the TT from the reconstructed planes

    Args:
        recon (jax.numpy.ndarray): Reconstructed refractive index volume with shape (N, H, W).
        ground_truth (jax.numpy.ndarray): Ground truth refractive index volume with shape (N, H, W).
        ROI (jax.numpy.ndarray): Boolean mask defining the region of interest with shape (N, H, W).
        depth_axis_length (float): Physical length of the depth axis in cm. Depth axis is the first axis in this case
        N (int): Number of planes to display.
        plane_type (str): Type of plane to display. Options are 'OPL' or 'OPD_TT'. Default is 'OPD_TT'.

    Returns:
    ￼Display N planes from the reconstruction and ground truth.
    """
    if plane_type not in ['OPL','OPD_TT']:
        raise ValueError("plane_type must be either 'OPL' or 'OPD_TT'")

    pixel_size=depth_axis_length/recon.shape[0]
    recon_planes = divide_into_sections_of_OPL(recon, N, depth_axis_length)
    gt_planes = divide_into_sections_of_OPL(ground_truth, N, depth_axis_length)
    ROI_planes = ROI[:N]
    if plane_type=='OPD_TT':
        recon_planes = remove_tip_tilt_jax(recon_planes,ROI_planes)
        gt_planes = remove_tip_tilt_jax(gt_planes,ROI_planes)

    #Set values outside the ROI to NaN for better visualization
    recon_planes=recon_planes.at[~ROI_planes].set(jnp.nan)
    gt_planes=gt_planes.at[~ROI_planes].set(jnp.nan)
    #find indices for ROI in each plane
    gt_images=[]
    recon_images=[]
    for i in range(N):
        roi_indices=jnp.where(ROI_planes[i])
        min_row=jnp.min(roi_indices[0])
        max_row=jnp.max(roi_indices[0])
        min_col=jnp.min(roi_indices[1])
        max_col=jnp.max(roi_indices[1])
        gt_images.append(gt_planes[i,min_row:max_row+1,min_col:max_col+1])
        recon_images.append(recon_planes[i,min_row:max_row+1,min_col:max_col+1])

    vmin=jnp.nanmin(jnp.array(gt_images+recon_images))
    vmax=jnp.nanmax(jnp.array(gt_images+recon_images))

    fig=plt.figure(figsize=(4 * N, 8),constrained_layout=True)
    fig.suptitle(f'Reconstruction of {N} {plane_type} planes', fontsize=18)
    subfigs = fig.subfigures(2, 1)
    subfigs[0].suptitle(f'Ground Truth {plane_type} Planes', fontsize=16,fontstyle='italic')
    subfigs[1].suptitle(f'Reconstruction {plane_type} Planes', fontsize=16,fontstyle='italic')
    axes_gt = subfigs[0].subplots(1, N)
    axes_recon = subfigs[1].subplots(1, N)
    for i in range(N):
        extent = (0, gt_images[i].shape[1]*pixel_size, 0, gt_images[i].shape[0]*pixel_size)

        im1 = axes_gt[i].imshow(gt_images[i].T, cmap='jet', origin='lower', vmin=vmin, vmax=vmax, extent=extent)
        im_ratio = gt_images[i].shape[0] / gt_images[i].shape[1]
        axes_gt[i].set_title(f'Region {i+1}', fontsize=14,fontweight='bold')
        axes_gt[i].set_xlabel('streamwise-axis (cm)', fontsize=12)
        axes_gt[i].set_ylabel('vertical-axis (cm)', fontsize=12)
        axes_gt[i].grid(False)
        fig.colorbar(im1, ax=axes_gt[i],fraction=0.05*im_ratio)

        im2 = axes_recon[i].imshow(recon_images[i].T, cmap='jet', origin='lower', vmin=vmin, vmax=vmax, extent=extent)
        im_ratio = recon_images[i].shape[0] / recon_images[i].shape[1]
        axes_recon[i].set_title(f'NRMSE: {100*jax_nrmse_ROI(recon_planes[i],gt_planes[i],ROI_planes[i],option=2):.1f}%', fontsize=14)
        axes_recon[i].set_xlabel('streamwise-axis (cm)', fontsize=12)
        axes_recon[i].set_ylabel('vertical-axis (cm)', fontsize=12)
        axes_recon[i].grid(False)
        fig.colorbar(im2, ax=axes_recon[i],fraction=0.05*im_ratio)