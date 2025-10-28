# simulation.py
import jax.numpy as jnp
import numpy as np
from jax import random
import mbirjax
from wind_tomo.utilities import circ_block, ift3_jax,remove_tip_tilt_jax,remove_piston_jax,align_fov_with_optical_axis
import warnings

def create_ct_model_and_weights(optical_setup, beam_fov,return_weights_only=False):
    """
    Create the mbirjax CT model and weight matrix based on the optical setup.

    Args:
        optical_setup (dict): Optical setup parameters from `define_optical_setup`.
        beam_fov (float, jnp.ndarray, or np.ndarray): Beam FOV. Can be:
            - Scalar (beam diameter in cm, assumes disk-shaped FOV).
            - 2D boolean array (shared FOV shape for all beams).
            - 3D boolean array (unique FOV shape for each beam).
        return_weights_only (bool): If True, only returns the weight matrix without creating the CT model.

    Returns:
        tuple: (ct_model, weight_matrix)
    """
    # Extract parameters from optical_setup
    sensor_locations = optical_setup["sensor_locations_pixels"]
    beam_angles = optical_setup["beam_angles"]
    test_region_pixel_dims = optical_setup["test_region_pixel_dims"]
    pixel_pitch = optical_setup["pixel_pitch"]
    windows = optical_setup["windows"]

    num_rows, num_cols, num_slices = test_region_pixel_dims
    num_channels = max(num_rows, num_cols)
    num_views = len(beam_angles)

    # Initialize weight matrix
    weights = jnp.zeros((num_views, num_slices, num_channels))

    for i, (theta, axis_center) in enumerate(zip(beam_angles, sensor_locations)):

        if jnp.isscalar(beam_fov):  # Case 1: Scalar (disk-shaped FOV)
            beam_pixel_diam = beam_fov / pixel_pitch
            weights = weights.at[i, :, :].set(1) # start with all ones
            weights = weights.at[i, :, :].set(
                circ_block(
                    weights[i, :, :],
                    beam_pixel_diam,
                    (0, axis_center[0] * jnp.sin(-theta) + axis_center[1]),
                )
            )
        else:
            # Handle 2D and 3D FOV cases
            if beam_fov.ndim == 2:  # Case 2: Shared 2D boolean array
                fov_valid = beam_fov
            elif beam_fov.ndim == 3:  # Case 3: Unique 3D boolean array
                if beam_fov.shape[0] != len(beam_angles):
                    raise ValueError("Invalid beam_fov format. The first axis length must match the number of beams.")
                fov_valid = beam_fov[i]
            else:
                raise ValueError("Invalid beam_fov format. Must be scalar, 2D, or 3D array.")

            row_offset, col_offset = align_fov_with_optical_axis(
                fov_valid, axis_center, theta, num_slices, num_channels
            )
            weights = weights.at[
                      i, row_offset:row_offset + fov_valid.shape[0], col_offset:col_offset + fov_valid.shape[1]
                      ].set(fov_valid.astype(jnp.float32))

        if windows:
            # Set lower and upper sections to zero
            center = num_channels / 2
            d = num_cols * jnp.cos(theta) - num_rows * jnp.sin(jnp.abs(theta))
            low_ind = max(0, int(jnp.round(center - d / 2)))
            high_ind = min(int(jnp.round(center + d / 2)) + 1, num_channels)
            # Save the current state of weights for comparison
            original_weights = weights[i, :, :].copy()

            weights = weights.at[i, :, 0:low_ind].set(0)
            weights = weights.at[i, :, high_ind:].set(0)

            # Check if weights were modified
            if not jnp.array_equal(original_weights, weights[i, :, :]):
                warnings.warn(
                    "According to the input optical_setup, the FOV of at least one beam is being cropped by the window."
                    "Consider adjusting the dimensions or configuration."
                )

    # Create CT model
    ct_model = mbirjax.ParallelBeamModel((num_views, num_slices, num_channels), beam_angles)
    ct_model.set_params(recon_shape=(num_rows, num_cols, num_slices),verbose=0, sharpness=2,p=2,q=2)
    if return_weights_only:
        return weights
    else:
        return ct_model, weights


def collect_projection_measurement(ct_model, weights,volume,projection_type='OPD_TT'):
    """
    Collect sinogram measurements from a 3D volume using the CT model and weight matrix.

    Args:
        ct_model (mbirjax.TomographyModel): Instance of TomographyModel class from mbirjax.
        weights (jnp.ndarray): Weight matrix for the sinogram.
        volume (jnp.ndarray): 3D volume (y, x, z) of simulated refractive index.
        projection_type (str): Type of projection to simulate. Options are:
            - 'OPD_TT': Remove tip and tilt.
            - 'OPD': Remove piston only.
            - 'OPL': No removal. Ideal projection.

    Returns:
        jnp.ndarray: Sinogram measurements.
    """
    sinogram = ct_model.forward_project(volume)
    if projection_type=='OPD_TT':
        FOV = weights == 1
        sinogram = remove_tip_tilt_jax(sinogram, FOV=FOV)
    elif projection_type=='OPD':
        FOV = weights == 1
        sinogram = remove_piston_jax(sinogram, FOV=FOV)
    elif projection_type!='OPL':
        raise ValueError("Invalid projection_type. Choose from 'OPD_TT', 'OPD', or 'OPL'.")

    return sinogram * weights

def generate_random_atmospheric_phase_volume(r0, dim, delta, L0=np.inf, l0=0.0, key=None):
    """
    Generate phase volume directly with NxMxP dimensions using von Kármán PSD.

    Args:
        r0 (float): Fried's coherence length [m].
        dim (tuple): Tuple containing the dimensions (N, M, P).
        delta (float): Grid sampling interval in [m].
        L0 (float, optional): [inf] von Karman PSD, one over outer scale frequency [m].
        l0 (float, optional): [0] von Karman PSD, one over inner scale frequency [m].
        key (jax.random.PRNGKey): JAX random key for reproducibility.

    Returns:
        jax.numpy.ndarray (float32): NxMxP phase volume.

    References:
        - J. D. Schmidt and S. of Photo-optical Instrumentation Engineers., Numerical simulation of
          optical wave propagation with examples in MATLAB, 149–184. Press monograph ; 199,
          SPIE, Bellingham, Wash (2010)
    """
    if key is None:
        key = random.PRNGKey(0)
    N, M, P = dim

    # Define frequency grids with correct spacing
    del_f_x = 1 / (N * delta)
    del_f_y = 1 / (M * delta)
    del_f_z = 1 / (P * delta)
    fx = jnp.arange(-N // 2, N // 2) * del_f_x
    fy = jnp.arange(-M // 2, M // 2) * del_f_y
    fz = jnp.arange(-P // 2, P // 2) * del_f_z
    fx, fy, fz = jnp.meshgrid(fx, fy, fz, indexing="ij")

    f = jnp.sqrt(fx**2 + fy**2 + fz**2)

    # Calculate inner and outer scale frequencies
    # fm = 5.92 / l0 / (2 * np.pi)             # inner scale frequency [1/m]
    oneover_fm = (
        l0 * (2.0 * np.pi) / 5.92
    )  # one over inner scale frequency [1/m]^-1 (avoid div by zero)
    f0 = 1 / L0  # outer scale frequency [1/m]

    with np.errstate(divide="ignore"):
        PSD_phi = (
            0.023
            * r0 ** (-5 / 3)
            * jnp.divide(jnp.exp(-((f * oneover_fm) ** 2)), (f**2 + f0**2) ** (11 / 6))
        )
    PSD_phi = PSD_phi.at[N // 2, M // 2, P // 2].set(0)

    key, subkey1, subkey2 = random.split(key, 3)
    cn = (
        (random.normal(subkey1, shape=dim) + 1j * random.normal(subkey2, shape=dim))
        * jnp.sqrt(PSD_phi)
        * jnp.sqrt(del_f_x * del_f_y * del_f_z)
    )
    phz = jnp.real(ift3_jax(cn, scale=(N * M * P)))

    return phz

