# utilities.py
import jax.numpy as jnp
from jax import jit, vmap
import numpy as np
from scipy.ndimage import gaussian_filter
import jax
from zernike import RZern
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.patches import Patch
from matplotlib.lines import Line2D


def divide_into_sections_of_OPL(recon, sections, total_length):
    """
    Divide a refractive index volume into sections and compute the optical path length (OPL) for each section.
    Args:
        recon (jax.numpy.ndarray): 3D refractive index volume with shape (N, H, W).
        sections (int): Number of sections to divide the volume into.
        total_length(float): Total physical length of the volume along the first axis.

    Returns:

    """
    N = recon.shape[0]
    if sections == N:
        return recon
    S = sections
    L = N * 1.0 / S
    H, W = recon.shape[1:]
    zeros = jnp.zeros((1, H, W), dtype=recon.dtype)
    cumsum = jnp.concatenate([zeros, jnp.cumsum(recon, axis=0)], axis=0)

    def integral_to(x):
        floor_x = jnp.floor(x).astype(jnp.int32)
        frac_x = x - floor_x
        recon_idx = jnp.minimum(floor_x, N - 1)
        return cumsum[floor_x] + frac_x * recon[recon_idx]

    ks = jnp.arange(S)
    starts = ks * L
    ends = (ks + 1) * L
    integrals_starts = jax.vmap(integral_to)(starts)
    integrals_ends = jax.vmap(integral_to)(ends)
    weighted_sums = integrals_ends - integrals_starts
    OPLs = weighted_sums * total_length / (L * S)
    return OPLs


def gaussian_kernel2d(nsig, truncate=4):
    """
    Generate a 2D Gaussian kernel using scipy's gaussian_filter.

    Args:
        nsig (float): Standard deviation of the Gaussian.
        truncate (float): Truncate the filter at this many standard deviations.

    Returns:
        np.ndarray: 2D Gaussian kernel.
    """

    kernlen = 1 + 2 * np.ceil(truncate * nsig).astype(int)
    inp = np.zeros((kernlen, kernlen))
    inp[kernlen // 2, kernlen // 2] = 1
    return gaussian_filter(inp, nsig)


def _convolve2D(x, kern):
    """Perform 2D convolution using JAX."""
    return jax.scipy.signal.convolve2d(x, kern, "same")


@jit
def stacked_convolve2D(x_stack, kern):
    """Perform 2D convolution on a stack of 2D arrays using JAX.
    Args:
        x_stack (jnp.ndarray): Stack of 2D arrays (shape: [N, H, W]).
        kern (jnp.ndarray): 2D convolution kernel.
    Returns:
        jnp.ndarray: Stack of convolved 2D arrays (shape: [N, H, W]).
    """
    return vmap(_convolve2D, in_axes=(0, None), out_axes=0)(x_stack, kern)


def circ_block(view, diameter, center_offset=(0, 0)):
    """
    Set everything outside a disk equal to zero using JAX.

    Args:
        view (jnp.ndarray): 2D array to be modified.
        diameter (float): Diameter of the disk in pixels.
        center_offset (tuple): Pixel center of the disk relative to the center of the array. (row, col)
                               (0, 0) corresponds to the center of the array.

    Returns:
        jnp.ndarray: Modified 2D array.
    """
    center = (
        view.shape[0] / 2 - 0.5 + center_offset[0],
        view.shape[1] / 2 - 0.5 + center_offset[1],
    )
    height, width = view.shape
    y, x = jnp.meshgrid(jnp.arange(height), jnp.arange(width), indexing="ij")
    dist_from_center = jnp.sqrt((x - center[1]) ** 2 + (y - center[0]) ** 2)
    mask = dist_from_center <= (diameter / 2)
    return view * mask


def align_fov_with_optical_axis(fov_mask, axis_center, theta, num_slices, num_channels):
    """
    Align the FOV center with the optical axis center.

    Args:
        fov_mask (jnp.ndarray): Boolean mask of the FOV.
        axis_center (tuple): (x, y) coordinates of the sensor location in pixels.
        theta (float): Beam angle in radians.
        num_slices (int): Number of slices in the test region.
        num_channels (int): Number of channels in the test region.

    Returns:
        tuple: (row_offset, col_offset) to align the FOV center with the optical axis center.
    """
    # Calculate the optical axis center
    optical_axis_center = jnp.array([
        num_slices // 2,
        num_channels // 2 + axis_center[0] * jnp.sin(-theta) + axis_center[1]
    ])

    # Calculate the true center of the FOV
    fov_indices = jnp.argwhere(fov_mask)
    fov_center = fov_indices.mean(axis=0)

    # Calculate offsets to align the centers
    row_offset, col_offset = jnp.round(optical_axis_center - fov_center).astype(int)

    return row_offset, col_offset


def ift3_jax(X, scale=1):
    """
    3D Inverse Fast Fourier Transform using JAX.

    Args:
        X (jax.numpy.ndarray): Input array in Fourier space.
        scale (float, optional): Scaling factor. Defaults to 1.

    Returns:
        jax.numpy.ndarray: 3D Inverse Fourier transform of the input array.
    """
    return jnp.fft.ifftshift(jnp.fft.ifftn(jnp.fft.fftshift(X))) * scale


def ft3_jax(X, scale=1):
    """
    3D Inverse Fast Fourier Transform using JAX.

    Args:
        X (jax.numpy.ndarray): Input array in Fourier space.
        scale (float, optional): Scaling factor. Defaults to 1.

    Returns:
        jax.numpy.ndarray: 3D Inverse Fourier transform of the input array.
    """
    return jnp.fft.fftshift(jnp.fft.fftn(jnp.fft.fftshift(X))) * scale


def compute_normalized_coords(mask):
    """
    Compute normalized coordinates for a 2D mask.

    Args:
        mask (jax.numpy.ndarray): 2D boolean mask array.

    Returns:
        tuple: (X_normalized, Y_normalized) flattened arrays.
    """
    m, n = mask.shape
    Y, X = jnp.meshgrid(-jnp.arange(m), jnp.arange(n), indexing="ij")
    X_flat = X.ravel()
    Y_flat = Y.ravel()
    mask_flat = mask.ravel()

    sum_mask = jnp.sum(mask_flat)
    avg_X = jnp.sum(X_flat * mask_flat) / sum_mask
    avg_Y = jnp.sum(Y_flat * mask_flat) / sum_mask
    X_centered = X_flat - avg_X
    Y_centered = Y_flat - avg_Y
    max_X = jnp.max(jnp.where(mask_flat, X_centered, -jnp.inf))
    max_Y = jnp.max(jnp.where(mask_flat, Y_centered, -jnp.inf))
    X_normalized = X_centered / max_X
    Y_normalized = Y_centered / max_Y
    return X_normalized, Y_normalized, X, Y


def fit_plane_2d_coeff(data, mask):
    """
    Fit a 2D plane to data using normalized coordinates.
    Args:
        data (jax.numpy.ndarray): 2D array of data.
        mask (jax.numpy.ndarray): 2D boolean mask array.

    Returns:
        jax.numpy.ndarray: Coefficients [piston, tip, tilt].
    """
    X_normalized, Y_normalized, _, _ = compute_normalized_coords(mask)
    data_flat = data.ravel()
    mask_flat = mask.ravel()

    A = jnp.c_[jnp.ones_like(X_normalized), Y_normalized, X_normalized]
    ATA = jnp.einsum("i,ij,ik->jk", mask_flat, A, A)
    ATb = jnp.einsum("i,ij,i->j", mask_flat, A, data_flat)
    C = jnp.linalg.solve(ATA, ATb)
    return C


def recon_plane_2d(C, mask):
    """
    Reconstruct a 2D plane from coefficients with masked normalization.

    Args:
        C (jax.numpy.ndarray): Coefficients [piston, tip, tilt].
        mask (jax.numpy.ndarray): 2D boolean mask array.

    Returns:
        jax.numpy.ndarray: Reconstructed 2D plane.
    """
    m, n = mask.shape
    X_normalized, Y_normalized, _, _ = compute_normalized_coords(mask)
    X_normalized = X_normalized.reshape(m, n)
    Y_normalized = Y_normalized.reshape(m, n)
    plane = (
            C[0] * jnp.ones_like(X_normalized) + C[1] * Y_normalized + C[2] * X_normalized
    )
    return jnp.where(mask, plane, 0)


def fit_plane_2d_estimate(data, mask):
    """
    Fit and estimate a 2D plane for data using normalized coordinates.

    Args:
        data (jax.numpy.ndarray): 2D array of data.
        mask (jax.numpy.ndarray): 2D boolean mask array.

    Returns:
        jax.numpy.ndarray: Fitted plane, masked by FOV.
    """
    C = fit_plane_2d_coeff(data, mask)
    return recon_plane_2d(C, mask)


def fit_plane_2d_remove(data, mask):
    """
    Fit and remove a 2D plane from data using normalized coordinates.

    Args:
        data (jax.numpy.ndarray): 2D array of data.
        mask (jax.numpy.ndarray): 2D boolean mask array.

    Returns:
        jax.numpy.ndarray: Data with fitted plane subtracted.
    """
    return data - fit_plane_2d_estimate(data, mask)


# Main function to remove tip-tilt
def remove_tip_tilt_jax(arr, FOV=None):
    """
    Remove tip-tilt from a 2D or 3D JAX array and ensure zero mean.

    Args:
        arr (jax.numpy.ndarray): Input 2D or 3D array.
        FOV (jax.numpy.ndarray, optional): Mask array, same shape as arr.

    Returns:
        jax.numpy.ndarray: Array with linear trends removed.
    """
    if FOV is None:
        FOV = jnp.ones(arr.shape, dtype=bool)
    if arr.ndim == 2:
        return fit_plane_2d_remove(arr, FOV)
    elif arr.ndim == 3:
        return vmap(fit_plane_2d_remove, in_axes=(0, 0))(arr, FOV)
    else:
        raise ValueError("Input array must be 2D or 3D.")


# Main function to estimate tip-tilt
def estimate_tip_tilt_jax(arr, FOV=None):
    """
    Estimate tip-tilt for a 2D or 3D JAX array.

    Args:
        arr (jax.numpy.ndarray): Input 2D or 3D array.
        FOV (jax.numpy.ndarray, optional): Mask array, same shape as arr.

    Returns:
        jax.numpy.ndarray: Fitted tip-tilt planes.
    """
    if FOV is None:
        FOV = jnp.ones(arr.shape, dtype=bool)
    if arr.ndim == 2:
        return fit_plane_2d_estimate(arr, FOV)
    elif arr.ndim == 3:
        return vmap(fit_plane_2d_estimate, in_axes=(0, 0))(arr, FOV)
    else:
        raise ValueError("Input array must be 2D or 3D.")


# Function to estimate coefficients
def estimate_piston_tip_tilt_coeff_jax(arr, mask, remove_mean_piston=True):
    """
    Estimate piston, tip, and tilt coefficients from a 2D or 3D JAX array.

    Args:
        arr (jax.numpy.ndarray): Input 2D or 3D array.
        mask (jax.numpy.ndarray): Mask array, same shape as arr.

    Returns:
        jax.numpy.ndarray: Coefficients array [piston_0, tip_0, tilt_0, ..., piston_N, tip_N, tilt_N].
    """
    if arr.ndim == 2:
        return fit_plane_2d_coeff(arr, mask)
    elif arr.ndim == 3:
        coeffs = vmap(fit_plane_2d_coeff, in_axes=(0, 0))(arr, mask)
        if remove_mean_piston:
            pistons = coeffs[:, 0]
            mean_piston = jnp.mean(pistons)
            coeffs = coeffs.at[:, 0].set(coeffs[:, 0] - mean_piston)
        return coeffs.ravel()
    else:
        raise ValueError("Input array must be 2D or 3D.")


# Function to construct planes from coefficients
def construct_piston_tip_tilt_from_coeff_jax(coeff, mask):
    """
    Construct tip-tilt planes from coefficients for a 2D or 3D mask.

    Args:
        coeff (jax.numpy.ndarray): Coefficients array (1D or 2D).
        mask (jax.numpy.ndarray): 2D or 3D mask array.

    Returns:
        jax.numpy.ndarray: Constructed planes.
    """
    if mask.ndim == 2:
        if coeff.ndim != 1 or len(coeff) != 3:
            raise ValueError("For 2D mask, coeff must be 1D with length 3.")
        return recon_plane_2d(coeff, mask)
    elif mask.ndim == 3:
        N = mask.shape[0]
        if coeff.ndim == 1:
            if len(coeff) != 3 * N:
                raise ValueError(
                    f"For 3D mask with {N} planes, coeff must have length {3 * N}."
                )
            coeff_reshaped = coeff.reshape(N, 3)
        elif coeff.ndim == 2:
            if coeff.shape != (N, 3):
                raise ValueError(
                    f"For 3D mask with {N} planes, coeff must be shape ({N}, 3)."
                )
            coeff_reshaped = coeff
        else:
            raise ValueError("coeff must be 1D or 2D.")
        return vmap(recon_plane_2d, in_axes=(0, 0))(coeff_reshaped, mask)
    else:
        raise ValueError("mask must be 2D or 3D.")


# # Main functions
def remove_piston_jax(arr, FOV=None):
    """Remove piston from a 2D or 3D JAX array and ensure zero mean.
    Args:
        arr (jax.numpy.ndarray): Input 2D or 3D array.
        FOV (jax.numpy.ndarray, optional): Mask array.

    Returns:
        jax.numpy.ndarray: Array with piston removed.
    """
    if FOV is None:
        FOV = jnp.ones(arr.shape, dtype=bool)
    if arr.ndim == 2:
        result = arr - jnp.mean(arr[FOV])
    elif arr.ndim == 3:
        nan_arr = arr.at[~FOV].set(jnp.nan)
        result = arr - jnp.nanmean(nan_arr, axis=(1, 2), keepdims=True)
    else:
        raise ValueError("Input array must be 2D or 3D.")
    return result


def fit_zernike_coefficients_to_image(image, max_radial_degree, pixel_diameter=None, return_class=False):
    """
    Fit Zernike polynomials to a 2D image over a disk.

    Args:
        image (2D array): The input image.
        max_radial_degree (int): The maximum radial degree of Zernike polynomials.
        pixel_diameter (int, optional): The pixel diameter of the disk. Defaults to the minimum side length of the image.

    Returns:
        np.ndarray: Fitted Zernike coefficients organized by radial degree.
    """
    # Get the dimensions of the image
    height, width = image.shape

    # Set pixel_diameter to the minimum side length if not provided
    if pixel_diameter is None:
        pixel_diameter = min(height, width)

    mask = circ_block(np.ones(image.shape), pixel_diameter) == 0
    masked_image = np.ma.masked_where(mask, image)

    zern = RZern(max_radial_degree)

    # Create a grid
    x = np.linspace(-1, 1, height)
    y = np.linspace(-1, 1, width)
    xv, yv = np.meshgrid(x, y)
    zern.make_cart_grid(xv, yv)

    # Fit Zernike polynomials to the masked array
    coefficients = zern.fit_cart_grid(masked_image.filled(0))[0]
    if return_class:
        return coefficients, zern
    else:
        return coefficients


def reconstruct_image_from_zernike_coefficients(
        full_coefficients, min_rad_order, max_rad_order, pixel_diameter, zern=None
):
    """
    Reconstruct a 2D image from Zernike coefficients.

    Args:
        zernike_coeffs (list of tuples): Zernike coefficients organized by radial degree.
        pixel_diameter (int): The pixel diameter of the disk.

    Returns:
        np.ma.MaskedArray: Reconstructed 2D image with the region outside the inner disc masked.
    """
    if zern is None:
        zern = RZern(max_rad_order)

        # Create a grid
        x = np.linspace(-1, 1, pixel_diameter)
        y = np.linspace(-1, 1, pixel_diameter)
        xv, yv = np.meshgrid(x, y)
        zern.make_cart_grid(xv, yv)

    subset_coefficients = np.zeros_like(full_coefficients)
    N_min = sum(i + 1 for i in range(min_rad_order - 1))
    N_max = sum(i + 1 for i in range(max_rad_order + 1))

    subset_coefficients[N_min:N_max] = full_coefficients[N_min:N_max]
    # Determine the radius of the disk
    reconstructed_image = zern.eval_grid(subset_coefficients, matrix=True)
    masked_recon = np.ma.masked_invalid(reconstructed_image)

    return masked_recon


def generate_beam_path_ROI_mask(recon_dim, beam_pixel_diam, location=(0, 0, 0), angle=0):
    """
    Generate a 3D mask for the beam path ROI.

    Args:
        recon_dim (tuple): Dimensions of the reconstruction volume (rows, cols, slices).
        beam_pixel_diam (float): Beam diameter in pixels.
        location (tuple): (x, y, z) location of the beam center in pixels.
        angle (float): Beam angle from row axis in radians.

    Returns:
        jnp.ndarray: 3D boolean mask of the beam path ROI.
    """

    rows, cols, slices = recon_dim
    if len(location) == 2:
        a, b = location
        c = 0
    else:
        a, b, c = location
    radius = beam_pixel_diam / 2
    center_r = rows / 2
    center_c = cols / 2
    center_s = slices / 2
    i = jnp.arange(rows)
    j = jnp.arange(cols)
    k = jnp.arange(slices)
    I, J, K = jnp.meshgrid(i, j, k, indexing="ij")
    x = I - center_r
    y = -(J - center_c)
    z = -(K - center_s)
    sin_theta = jnp.sin(angle)
    cos_theta = jnp.cos(angle)
    perp_dist_xy = sin_theta * (x - a) + cos_theta * (y - b)
    dist_z = z - c
    dist = jnp.sqrt(perp_dist_xy ** 2 + dist_z ** 2)
    mask = dist <= radius
    return mask


def jax_nrmse_ROI_flat(GT_flat, recon_flat, indices, option=0):
    """
    Compute NRMSE using flattened arrays and integer indices.

    Args:
        GT_flat: Flattened ground truth array.
        recon_flat: Flattened reconstructed array.
        indices: Integer indices where ROI is True.
        option: Denominator option (0: RMS of GT, 1: range, 2: interpercentile range).

    Returns:
        float: Normalized RMSE.
    """
    selected_GT = GT_flat[indices]
    selected_recon = recon_flat[indices]
    rmse = jnp.sqrt(jnp.mean((selected_recon - selected_GT) ** 2))

    if option == 0:
        denominator = jnp.sqrt(jnp.mean(selected_GT ** 2))
    elif option == 1:
        denominator = jnp.max(selected_GT) - jnp.min(selected_GT)
    else:
        q1, q2 = jnp.percentile(selected_GT, jnp.array([5, 95]))
        denominator = q2 - q1

    return rmse / denominator


# JIT-compile the function
jax_nrmse_ROI_flat_jit = jax.jit(jax_nrmse_ROI_flat, static_argnums=3)


def jax_nrmse_ROI(GT, recon, ROI, option=0):
    """
    Compute NRMSE over a region of interest (ROI) using JAX.

    Args:
        GT (jax.numpy.ndarray): Ground truth array.
        recon (jax.numpy.ndarray): Reconstructed array.
        ROI (jax.numpy.ndarray): Boolean mask defining the region of interest.
        option (int): Denominator option (0: RMS of GT, 1: range, 2: interpercentile range).

    Returns:
        float: Normalized RMSE over the ROI.
    """
    GT_flat = GT.flatten()
    recon_flat = recon.flatten()
    ROI_flat = ROI.flatten()
    indices = jnp.where(ROI_flat)[0]
    return jax_nrmse_ROI_flat_jit(GT_flat, recon_flat, indices, option)

def display_viewing_configuration_schematic(locations, angles, diameter=0, scale=1, threshold=None,title=None, plane='transverse', dims=(20,25,20),outer_buffer=(2,2),roi_thickness_and_num_regions=None,legend_scale=1):
    """
    Display a schematic of the viewing configuration for wind tunnel tomography.

    Args:
        locations (list of tuples): List of (x, y) locations for each beam in pixels.
        angles (list of lists): List of lists containing angles (in radians) for each beam.
        diameter (float): Diameter of the beam path ROI in pixels. Default is 0 (no ROI).
        scale (float): Scaling factor for the figure size. Default is 1.
        threshold (int): If provided, display mark all pixels that are seen by this many beams. Default is None.
        title (str): Title of the plot. Default is None.
        plane (str): Plane of view ('transverse' or 'sagittal'). Default is 'transverse'.
        dims (tuple): Dimensions of the test region (rows, columns, slices). Default is (20,25,20).
        outer_buffer (tuple): Outer buffer around the test region (width_buffer, height_buffer). Default is (2,2).
        roi_thickness_and_num_regions (tuple): If provided, a tuple containing the thickness of the ROI and number of regions to divide it into.
        legend_scale (float): Scaling factor for the legend size. Default is 1.

    Returns:
        None: Displays the schematic plot.
    """

    if plane=='transverse':
        height= dims[1] # columns
        width=dims[0] #rows
    if plane=='sagittal':
        height=dims[2] #slices
        width=dims[0] #rows
    else:
        ValueError("Plane options are only transverse or sagittal")

    # Create a figure and axis with the specified dimensions
    fig, ax = plt.subplots(figsize=((width+outer_buffer[0]*2) * scale, (height+outer_buffer[1]*2) * scale))
    h_bounds = height / 2 + outer_buffer[1]
    w_bounds = width / 2 + outer_buffer[0]

    # Set the limits of the plot
    ax.set_xlim(-w_bounds, w_bounds)
    ax.set_ylim(-h_bounds, h_bounds)

    # Set the overall background color to white
    ax.set_facecolor('white')

    # Draw the rectangle centered at (0, 0) with width 20 and height 25

    # rectangle = plt.Rectangle((-width/2, -height/2), width, height, edgecolor='black', facecolor='lightskyblue', zorder=0)
    # ax.add_patch(rectangle)
    if plane == 'transverse':
        background_rect=plt.Rectangle((-width / 2, -h_bounds*1.2), width, h_bounds*2*1.2, edgecolor='k', facecolor='0.8',
                                  zorder=0,linewidth=4 * scale)
        ax.add_patch(background_rect)

    rectangle = plt.Rectangle((-width / 2, -height / 2), width, height, edgecolor=None, facecolor='#87CEFA',
                              zorder=1)
    ax.add_patch(rectangle)

    custom_legend = [
        Patch(facecolor="#87CEFA", label="Wind Tunnel Test Region"),
    ]

    ax.plot([-width / 2, -width / 2], [-height / 2, height / 2], '#87CEFA', linewidth=4 * scale, zorder=1)
    ax.plot([width / 2, width / 2], [-height / 2, height / 2], '#87CEFA', linewidth=4 * scale, zorder=1)
    # if roi_thickness_and_num_regions is not None:
    #     roi_thickness=roi_thickness_and_num_regions[0]
    #     num_regions=roi_thickness_and_num_regions[1]
    #     inc=width/num_regions
    #     for i in range(num_regions):
    #         rectangle = plt.Rectangle((-width / 2+i*inc, -roi_thickness / 2), inc, roi_thickness, edgecolor=None, facecolor='lightgreen',
    #                                   zorder=1)
    #         ax.add_patch(rectangle)
    #     custom_legend.append(Patch(facecolor="lightgreen",edgecolor='black', label="ROI"))

    if roi_thickness_and_num_regions is not None:
        roi_thickness=roi_thickness_and_num_regions[0]
        rectangle = plt.Rectangle((-width / 2, -roi_thickness / 2), width, roi_thickness, edgecolor='#D0F0C0',
                                  facecolor='#D0F0C0',
                                  zorder=1)
        ax.add_patch(rectangle)
        custom_legend.append(Patch(facecolor="#D0F0C0", edgecolor='#D0F0C0', label="ROI"))

        num_regions=roi_thickness_and_num_regions[1]
        inc=width/num_regions
        for i in range(1,num_regions):
            x=-width / 2+inc*i
            y_top=roi_thickness / 2
            y_bot=-roi_thickness / 2
            ax.plot([x,x], [y_bot, y_top], 'k', linewidth=2 * scale, zorder=1.5)

    # Store bands for overlap calculation
    bands = []

    # Plot lines from each location at specified angles
    for i, location in enumerate(locations):
        x, y = location[0], location[1]
        for angle in angles[i]:
            # Convert angle to radians relative to the negative x-axis
            # angle_rad = np.deg2rad(angle + 180)
            angle_rad = angle + np.pi
            # Calculate the end point of the line to the outer edge of the plot
            if np.cos(angle_rad) != 0:
                x_end1 = w_bounds if np.cos(angle_rad) > 0 else -w_bounds
                x_end2 = -w_bounds if np.cos(angle_rad) > 0 else w_bounds
                y_end1 = y + (x_end1 - x) * np.tan(angle_rad)
                y_end2 = y - (x_end2 - x) * np.tan(angle_rad)

            else:
                y_end1 = h_bounds if np.sin(angle_rad) > 0 else -h_bounds
                y_end2 = -h_bounds if np.sin(angle_rad) > 0 else h_bounds
                x_end1 = x + (y_end1 - y) / np.tan(angle_rad)
                x_end2 = x - (y_end2 - y) / np.tan(angle_rad)
                # x_end = x + (y_end - y) / np.tan(angle_rad)


            # Plot the line with increased thickness
            # ax.plot([x, x_end], [y, y_end], 'r--', linewidth=3 * scale, zorder=2)
            ax.plot([-x_end2, x_end1], [y_end2, y_end1], 'r--', linewidth=3 * scale, zorder=2)

            # If diameter is positive, add a translucent red band around the line
            if diameter is not None and diameter > 0:
                # length = np.sqrt((x_end - x)**2 + (y_end - y)**2)
                length = np.sqrt((2*x_end) ** 2 + (2*y_end) ** 2)
                angle_deg = np.rad2deg(angle_rad)
                band = Rectangle((-x_end + diameter * np.sin(angle_rad) / 2, -y_end - diameter * np.cos(angle_rad) / 2), length, diameter,
                                 angle=angle_deg, color='red', alpha=0.2, zorder=1.5, edgecolor=None)
                bands.append(band)
    if diameter is not None and diameter > 0:
        custom_legend.append(Patch(facecolor="red", alpha=0.2, edgecolor=None, label="Beam Path"))

    if threshold is not None and diameter is not None and diameter > 0:
        # Create a grid to check overlaps
        grid_sizex = 18*(width+4)
        grid_sizey= 18*(height+4)
        x_grid = np.linspace(-w_bounds, w_bounds, grid_sizex)
        y_grid = np.linspace(-h_bounds, h_bounds, grid_sizey)
        overlap_grid = np.zeros((grid_sizex, grid_sizey))

        for band in bands:
            band_coords = band.get_corners()

            for i in range(grid_sizex):
                for j in range(grid_sizey):
                    if _is_point_in_rectangle(band_coords, (x_grid[i],y_grid[j])):
                        overlap_grid[i,j] += 1

        for i in range(grid_sizex):
            for j in range(grid_sizey):
                if overlap_grid[i,j] >= threshold:
                    rect = Rectangle((x_grid[i], y_grid[j]), (x_grid[1] - x_grid[0]), (y_grid[1] - y_grid[0]), color='red', alpha=0.3)
                    ax.add_patch(rect)

    else:
        for band in bands:
            ax.add_patch(band)
    if title is not None:
        plt.title(title,fontsize=50*scale)
    space=' '
    axes_ftsize=int(30*scale)
    if plane=="transverse":
        plt.ylabel('<-- Flow Direction (x-axis) <--',fontsize=axes_ftsize)
        plt.xlabel(f'Laser Side{space*int(scale*w_bounds*80/axes_ftsize)}Depth Axis (y-axis){space*int(scale*w_bounds*80/axes_ftsize)}Camera Side',fontsize=axes_ftsize)
        # plt.xlabel(f'Laser Side{space*int(width*4)}Depth Axis{space*int(width*4)}Camera Side',fontsize=ftsize)
    elif plane=='sagittal':
        plt.ylabel('DOWN <--    Vertical Axis (z-axis)     --> UP',fontsize=axes_ftsize)
        plt.xlabel(f'Laser Side{space*int(scale*w_bounds*80/axes_ftsize)}Depth Axis{space*int(scale*w_bounds*80/axes_ftsize)}Camera Side',fontsize=axes_ftsize)
    else:
        ValueError("Plane options are only transverse or sagittal")
    custom_legend.append(Line2D([0], [0], color='red', linestyle='--', label='Beam Optical Axis'))
    plt.legend(handles=custom_legend,loc='upper right', bbox_to_anchor=(1-2/(w_bounds*2),1-2/(h_bounds*2)),fontsize=int(25*scale*legend_scale))
    # Show the plot
    plt.grid(False)
    return bands

def _is_point_in_rectangle(recty, point):
    """
    Check if a point is inside a rectangle.

    Args:
        rect (list): List of four tuples representing the corners of the rectangle in clockwise or counterclockwise order.
        point (tuple): Tuple representing the coordinates of the point (x, y).

    Returns:
        bool: True if the point is inside the rectangle, False otherwise.
    """
    def cross_product(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    def is_point_in_triangle(p, a, b, c):
        # Check if point p is inside the triangle formed by points a, b, and c
        return (cross_product(p, a, b) >= 0 and
                cross_product(p, b, c) >= 0 and
                cross_product(p, c, a) >= 0)

    # Split the rectangle into two triangles
    triangle1 = [recty[0], recty[1], recty[2]]
    triangle2 = [recty[2], recty[3], recty[0]]

    # Check if the point is inside either of the triangles
    return is_point_in_triangle(point, *triangle1) or is_point_in_triangle(point, *triangle2)
