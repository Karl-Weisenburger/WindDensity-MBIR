import warnings
import jax.numpy as jnp
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Union
@dataclass()
class OpticalSetup:
    """
    Data class for defining tomography optical configuration.
    """
    sensor_locations: List[Tuple[float, float]]
    sensor_locations_pixels: List[Tuple[float, float]]
    beam_angles: jnp.ndarray
    test_region_dims: Tuple[float, float, float]
    test_region_pixel_dims: Tuple[int, int, int]
    pixel_pitch: float
    beam_fov: Optional[Union[float, jnp.ndarray, np.ndarray]] = None
    beam_diameter_cm: Optional[float] = None
    beam_diameter_pixels: Optional[float] = None
    windows: Optional[bool] = False

def define_optical_setup(sensor_locations, beam_angles, test_region_dims, pixel_pitch, beam_fov=None, windows=False) -> OpticalSetup:
    """
    Define the optical setup for the simulation or tomography pipeline, includes converting cm dimensions to pixel units.

    Args:
        sensor_locations (list of tuples): List of (x, y) coordinates for sensor locations in cm.
        beam_angles (list of lists or list): List of beam angles for each sensor (grouped) or a single list of angles (flattened).
        test_region_dims (tuple[float, float, float]): Dimensions of the test region (rows, cols, slices) in cm
        pixel_pitch (float): Pixel pitch in cm. Determined by sensor pixel size.
        beam_fov (float, jnp.ndarray, or np.ndarray, optional): Beam FOV. Can be:
            - Scalar (beam diameter in cm, assumes disk-shaped FOV).
            - 2D boolean array (shared FOV shape for all beams).
            - 3D boolean array (unique FOV shape for each beam).
        windows (bool): Whether the test region has windows on the sides (default: False).

    Returns:
        OpticalSetup: Instance containing the processed optical setup parameters
    """
    # Convert to Python scalars to prevent JAX arrays from propagating
    pixel_pitch = float(pixel_pitch)
    sensor_locations = [(float(x), float(y)) for x, y in sensor_locations]
    test_region_dims = tuple(float(dim) for dim in test_region_dims)

    # Convert test region dimensions to pixel units. ensure correct type
    test_region_pixel_dims = tuple(int(float(dim / pixel_pitch)) for dim in test_region_dims)

    # Determine input format
    if isinstance(beam_angles, (list, np.ndarray, jnp.ndarray)) and all(
            isinstance(angles, (list, np.ndarray, jnp.ndarray)) for angles in beam_angles):
        # Grouped format: sensor_locations is a list of tuples, beam_angles is a list of lists or arrays
        if len(sensor_locations) != len(beam_angles):
            raise ValueError("Number of sensor locations does not match the number of angle groups.")
        # Flatten the grouped format
        angles = jnp.array([angle for sublist in beam_angles for angle in sublist])
        axis_centers = [loc for loc, sublist in zip(sensor_locations, beam_angles) for _ in sublist]
    elif isinstance(beam_angles, (list, np.ndarray, jnp.ndarray)) and len(sensor_locations) == len(beam_angles):
        # Flattened format: sensor_locations and beam_angles are both lists or arrays of the same length
        angles = jnp.array(beam_angles)  # if isinstance(beam_angles, (list, np.ndarray)) else beam_angles
        axis_centers = sensor_locations
    else:
        # Invalid input format
        raise ValueError('Invalid input format for sensor_locations and beam_angles. Ensure they follow the expected structure.')
    if beam_fov is None:
        warnings.warn("beam_fov is not defined. It will be inferred from NaN values in experimental data.")
        beam_diameter_cm = None
    elif jnp.isscalar(beam_fov):
        beam_diameter_cm = beam_fov
    else:
        print('beam_fov is an array, inferring beam_diameter_cm from maximum aperture axial extent.')
        # set beam_diameter_cm to the largest length between true values on either the rows or columns of beam_fov
        beam_fov_array = jnp.array(beam_fov)
        if beam_fov_array.ndim == 2:
            # find maximum and minimum indices of true values along rows and columns
            beam_diameter_cm = _find_max_true_extent(beam_fov_array) * pixel_pitch
        elif beam_fov_array.ndim == 3:
            if beam_fov_array.shape[0] != len(beam_angles):
                raise ValueError("Invalid beam_fov format. The first axis length must match the number of beams.")
            # find maximum extent across all beams
            max_extents = [_find_max_true_extent(beam_fov_array[i]) for i in range(beam_fov_array.shape[0])]
            beam_diameter_cm = max(max_extents) * pixel_pitch
        else:
            raise ValueError("Invalid beam_fov format. Must be None, scalar, 2D, or 3D array.")

    # Convert sensor locations to pixel units, forcing Python floats
    axis_centers_pixels = [(float(x / pixel_pitch), float(y / pixel_pitch)) for x, y in axis_centers]

    return OpticalSetup(
        sensor_locations=sensor_locations,
        sensor_locations_pixels=axis_centers_pixels,
        beam_angles=angles,
        test_region_dims=test_region_dims,
        test_region_pixel_dims=test_region_pixel_dims,
        beam_fov=beam_fov,
        beam_diameter_cm=beam_diameter_cm,
        beam_diameter_pixels=beam_diameter_cm / pixel_pitch if beam_diameter_cm is not None else None,
        pixel_pitch=pixel_pitch,
        windows=windows,
    )

def _find_max_true_extent(fov_array: jnp.ndarray) -> float:
    # find maximum and minimum indices of true values along rows and columns
    row_indices, col_indices = jnp.where(fov_array)
    row_extent = jnp.max(row_indices) - jnp.min(row_indices) + 1
    col_extent = jnp.max(col_indices) - jnp.min(col_indices) + 1
    return max(row_extent, col_extent)