#Approved for public release; distribution is unlimited. Public Affairs release approval # 2025-5579

# utilities.py
import warnings
import jax.numpy as jnp
from jax import vmap
import jax

def correct_recon_scaling(recon,ct_model, sinogram, weights):
    """
    Correct the scaling of the reconstruction.
    Args:
        recon (jnp.ndarray): Reconstructed volume.
        ct_model (mbirjax.TomographyModel): CT model used for reconstruction.
        sinogram (jnp.ndarray): Measured sinogram data.
        weights (jnp.ndarray): Weight matrix used in reconstruction.
    Returns:
        jnp.ndarray: Scaled reconstruction.
    """
    error_sinogram = ct_model.forward_project(recon)
    weighted_error_sinogram = weights * error_sinogram  # Note that fm_constant will be included below

    wtd_err_sino_norm = jnp.sum(weighted_error_sinogram * error_sinogram)

    alpha = jnp.sum(weighted_error_sinogram * sinogram) / wtd_err_sino_norm
    alpha = alpha.item()
    return alpha*recon

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


def _compute_normalized_coords(mask):
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


def _fit_plane_2d_coeff(data, mask):
    """
    Fit a 2D plane to data using normalized coordinates.
    Args:
        data (jax.numpy.ndarray): 2D array of data.
        mask (jax.numpy.ndarray): 2D boolean mask array.

    Returns:
        jax.numpy.ndarray: Coefficients [piston, tip, tilt].
    """
    x_normalized, y_normalized, _, _ = _compute_normalized_coords(mask)
    data_flat = data.ravel()
    mask_flat = mask.ravel()

    A = jnp.c_[jnp.ones_like(x_normalized), y_normalized, x_normalized]
    ATA = jnp.einsum("i,ij,ik->jk", mask_flat, A, A)
    ATb = jnp.einsum("i,ij,i->j", mask_flat, A, data_flat)
    C = jnp.linalg.solve(ATA, ATb)
    return C


def _recon_plane_2d(C, mask):
    """
    Reconstruct a 2D plane from coefficients with masked normalization.

    Args:
        C (jax.numpy.ndarray): Coefficients [piston, tip, tilt].
        mask (jax.numpy.ndarray): 2D boolean mask array.

    Returns:
        jax.numpy.ndarray: Reconstructed 2D plane.
    """
    m, n = mask.shape
    x_normalized, y_normalized, _, _ = _compute_normalized_coords(mask)
    x_normalized = x_normalized.reshape(m, n)
    y_normalized = y_normalized.reshape(m, n)
    plane = (
            C[0] * jnp.ones_like(x_normalized) + C[1] * y_normalized + C[2] * x_normalized
    )
    return jnp.where(mask, plane, 0)


def _fit_plane_2d_estimate(data, mask):
    """
    Fit and estimate a 2D plane for data using normalized coordinates.

    Args:
        data (jax.numpy.ndarray): 2D array of data.
        mask (jax.numpy.ndarray): 2D boolean mask array.

    Returns:
        jax.numpy.ndarray: Fitted plane, masked by FOV.
    """
    C = _fit_plane_2d_coeff(data, mask)
    return _recon_plane_2d(C, mask)


def _fit_plane_2d_remove(data, mask):
    """
    Fit and remove a 2D plane from data using normalized coordinates.

    Args:
        data (jax.numpy.ndarray): 2D array of data.
        mask (jax.numpy.ndarray): 2D boolean mask array.

    Returns:
        jax.numpy.ndarray: Data with fitted plane subtracted.
    """
    return data - _fit_plane_2d_estimate(data, mask)


# Main function to remove tip-tilt
def remove_tip_tilt_piston(arr, FOV=None):
    """
    Remove tip-tilt piston from a 2D or 3D JAX array and ensure zero mean.

    Args:
        arr (jax.numpy.ndarray): Input 2D or 3D array.
        FOV (jax.numpy.ndarray, optional): Mask array, same shape as arr.

    Returns:
        jax.numpy.ndarray: Array with linear trends removed.
    """
    if FOV is None:
        FOV = jnp.ones(arr.shape, dtype=bool)
    if arr.ndim == 2:
        return _fit_plane_2d_remove(arr, FOV)
    elif arr.ndim == 3:
        return vmap(_fit_plane_2d_remove, in_axes=(0, 0))(arr, FOV)
    else:
        raise ValueError("Input array must be 2D or 3D.")


# Main function to estimate tip-tilt
def estimate_tip_tilt_piston(arr, FOV=None):
    """
    Estimate tip-tilt piston for a 2D or 3D JAX array.

    Args:
        arr (jax.numpy.ndarray): Input 2D or 3D array.
        FOV (jax.numpy.ndarray, optional): Mask array, same shape as arr.

    Returns:
        jax.numpy.ndarray: Fitted tip-tilt planes.
    """
    if FOV is None:
        FOV = jnp.ones(arr.shape, dtype=bool)
    if arr.ndim == 2:
        return _fit_plane_2d_estimate(arr, FOV)
    elif arr.ndim == 3:
        return vmap(_fit_plane_2d_estimate, in_axes=(0, 0))(arr, FOV)
    else:
        raise ValueError("Input array must be 2D or 3D.")


def remove_piston(arr, FOV=None):
    """
    Remove piston from a 2D or 3D JAX array and ensure zero mean.

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
