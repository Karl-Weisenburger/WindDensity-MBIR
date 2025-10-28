# tomography.py
import jax.numpy as jnp
import numpy as np
import warnings
import mbirjax
from wind_tomo.utilities import align_fov_with_optical_axis

def define_optical_setup(sensor_locations, beam_angles, test_region_dims, pixel_pitch, windows=False):
    """
    Define the optical setup for the simulation pipeline, converting dimensions to pixel units.

    Args:
        sensor_locations (list of tuples): List of (x, y) coordinates for sensor locations in cm.
        beam_angles (list of lists or list): List of beam angles for each sensor (grouped) or a single list of angles (flattened).
        test_region_dims (tuple): Dimensions of the test region (rows, cols, slices) in cm
        pixel_pitch (float): Pixel pitch in cm. Determined by sensor pixel size.
        windows (bool): Whether the test region has windows on the sides (default: True).

    Returns:
        dict: Dictionary containing the processed optical setup parameters in pixel units.
    """
    # Convert test region dimensions to pixel units
    test_region_dims_pixels = tuple(int(dim / pixel_pitch) for dim in test_region_dims)
    # if beam_diameter is not None:
    #     if test_region_dims[0] < beam_diameter or test_region_dims[1] < beam_diameter:
    #         warnings.warn(
    #             "The test region dimensions are smaller than the beam diameter. "
    #             "We haven't implemented an automatic increase in volume size, so this will cause problems."
    #         )
    #     # Convert beam diameter to pixel units
    #     beam_diameter_pixels = beam_diameter / pixel_pitch
    # else:
    #     beam_diameter_pixels = None

    # Determine input format
    if isinstance(beam_angles, (list, np.ndarray,jnp.ndarray)) and all(
            isinstance(angles, (list, np.ndarray,jnp.ndarray)) for angles in beam_angles):
        # Grouped format: sensor_locations is a list of tuples, beam_angles is a list of lists or arrays
        if len(sensor_locations) != len(beam_angles):
            warnings.warn("Number of sensor locations does not match the number of angle groups.")
        # Flatten the grouped format
        angles = jnp.array([angle for sublist in beam_angles for angle in sublist])
        axis_centers = [loc for loc, sublist in zip(sensor_locations, beam_angles) for _ in sublist]
    elif isinstance(beam_angles, (list, np.ndarray,jnp.ndarray)) and len(sensor_locations) == len(beam_angles):
        # Flattened format: sensor_locations and beam_angles are both lists or arrays of the same length
        angles = jnp.array(beam_angles) #if isinstance(beam_angles, (list, np.ndarray)) else beam_angles
        axis_centers = sensor_locations
    else:
        # Invalid input format
        warnings.warn(
            "Invalid input format for sensor_locations and beam_angles. Ensure they follow the expected structure.")
        return None

    # Convert sensor locations to pixel units
    axis_centers_pixels = [(x / pixel_pitch, y / pixel_pitch) for x, y in axis_centers]

    return {
        "sensor_locations": sensor_locations,
        "sensor_locations_pixels": axis_centers_pixels,
        "beam_angles": angles,
        "test_region_dims": test_region_dims,
        "test_region_pixel_dims": test_region_dims_pixels,
        "pixel_pitch": pixel_pitch,
        "windows": windows,
    }

def generate_ct_model_sinogram_weights_from_experimental_data(optical_setup, experimental_data):
    """
    Generate a CT model, sinogram, and weight matrix based on the optical setup and experimental data.

    Args:
        optical_setup (dict): Optical setup parameters from `define_optical_setup`.
        experimental_data (jnp.ndarray): Experimental data array of shape (len(beam_angles), H, W). Please set values outside beam's FOV to NaN.

    Returns:
        tuple: (ct_model, sinogram, weight_matrix)
    """
    # Extract parameters from the optical setup
    sensor_locations = optical_setup["sensor_locations_pixels"]
    beam_angles = optical_setup["beam_angles"]
    test_region_pixel_dims = optical_setup["test_region_pixel_dims"]

    num_views = len(beam_angles)
    num_slices = test_region_pixel_dims[-1]
    num_channels = max(test_region_pixel_dims[:2])

    # Initialize sinogram and weight matrix
    sinogram = jnp.full((num_views, num_slices, num_channels), jnp.nan)
    num_data_slices,num_data_channels=experimental_data.shape[1:3]

    if num_data_slices>num_slices or num_data_channels>num_channels:
        warnings.warn("The experimental data dimensions exceed the test region dimensions. Data will be cropped.")
        bottom_left_corner=(num_data_slices//2-num_slices,num_data_channels//2-num_channels)
        experimental_data=experimental_data[:,bottom_left_corner[0]:bottom_left_corner[0]+num_slices,bottom_left_corner[1]:bottom_left_corner[1]+num_channels]
        num_data_slices,num_data_channels=experimental_data.shape[1:3]

    # Process each view in experimental_data
    for i, (view, theta, axis_center) in enumerate(zip(experimental_data, beam_angles, sensor_locations)):
        # Determine the FOV center in the experimental data
        fov_mask = ~jnp.isnan(view)
        row_offset, col_offset = align_fov_with_optical_axis(
            fov_mask, axis_center, theta, num_slices, num_channels
        )
        sinogram = sinogram.at[
                   i, row_offset:row_offset + num_data_slices, col_offset:col_offset + num_data_channels
                   ].set(view)

    weight_matrix=jnp.where(jnp.isnan(sinogram), 0, 1)
    sinogram=jnp.nan_to_num(sinogram,nan=0.0)

    # Create the CT model
    ct_model = mbirjax.ParallelBeamModel((num_views, num_slices, num_channels), beam_angles)
    ct_model.set_params(recon_shape=test_region_pixel_dims, verbose=0, sharpness=2,p=2,q=2)

    return ct_model, sinogram, weight_matrix
