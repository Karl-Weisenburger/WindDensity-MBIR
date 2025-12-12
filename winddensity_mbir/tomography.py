#Approved for public release; distribution is unlimited. Public Affairs release approval # 2025-5579

# tomography.py
import warnings
import mbirjax
import jax.numpy as jnp
import winddensity_mbir.utilities as utils
import winddensity_mbir.configuration_params as config

def generate_ct_model_sinogram_weights_from_experimental_data(optical_setup: config.OpticalSetup | None = None, experimental_data: jnp.ndarray | None = None, **kwargs):
    """
    Generate a CT model, sinogram, and weight matrix based on the optical setup and experimental data.
    The sinogram is constructed by aligning the experimental data with the optical axes defined in the optical setup.
    After running this function, a reconstruction using the GGMRF prior can be generated with the output by running "ct_model.recon(sinogram, weights=weight_matrix)"

    Args:
        optical_setup (config.OpticalSetup, optional): Optical setup parameters from `config.define_optical_setup`.
            Can be supplied manually as a config.OpticalSetup instance. If None, will be built using **kwargs**
            passed to `config.define_optical_setup`.
        experimental_data (jax.numpy.ndarray): Experimental data array with shape (num_views, num_slices, num_channels).
        **kwargs: Parameters to define the optical setup if optical_setup is not provided. Required parameters are:

            - **sensor_locations_pixels** (List[Tuple[float, float]]): List of (x, y) sensor locations in pixels.

            - **beam_angles** (List[float]): List of beam angles in radians.

            - **test_region_pixel_dims** (Tuple[int, int, int]): Dimensions of the test region in pixels (rows, cols, slices).

            - **pixel_pitch** (float): Pixel pitch in meters.

            - **windows** (bool): Whether to apply windowing to the beams.

            - **beam_fov** (Union[float, jnp.ndarray]): Field of view for each beam. Can be a scalar (disk-shaped FOV), a 2D boolean array (shared FOV), or a 3D boolean array (unique FOV per beam).

    Returns:
        tuple: (**ct_model**, **sinogram**, **weight_matrix**)

            - **ct_model** (mbirjax.TomographyModel): The generated MBIRJAX CT model.

            - **sinogram** (jax.numpy.ndarray): The sinogram constructed from experimental data.

            - **weight_matrix** (jax.numpy.ndarray): The weight matrix indicating each beam's FOV.
    """
    if experimental_data is None:
        raise ValueError("experimental_data is required")

    if optical_setup is None:
        if kwargs:
            optical_setup = config.define_optical_setup(**kwargs)
        else:
            raise ValueError("Must provide optical_setup or individual parameters via **kwargs")

    # Extract parameters from the optical setup

    sensor_locations = optical_setup.sensor_locations_pixels
    beam_angles = optical_setup.beam_angles
    test_region_pixel_dims = optical_setup.test_region_pixel_dims
    if optical_setup.beam_fov is None:
        warnings.warn("beam_fov is not defined in the optical_setup. It will be inferred from NaN values in experimental_data.")
    else:
        beam_fov = optical_setup.beam_fov
        # create fov_mask based beam_fov
        if jnp.isscalar(beam_fov):
            beam_pixel_diam = optical_setup.beam_diameter_pixels if optical_setup.beam_diameter_cm is not None else beam_fov / optical_setup.pixel_pitch
            beam_radius = beam_pixel_diam / 2
            beam_center = (experimental_data.shape[1] // 2, experimental_data.shape[2] // 2)
            Y, X = jnp.ogrid[:experimental_data.shape[1], :experimental_data.shape[2]]
            dist_from_center = jnp.sqrt((X - beam_center[1]) ** 2 + (Y - beam_center[0]) ** 2)
            fov_mask = dist_from_center <= beam_radius
        else:
            beam_fov_array = jnp.array(beam_fov)
            if beam_fov_array.ndim == 2:
                fov_mask = beam_fov_array
            elif beam_fov_array.ndim == 3:
                if beam_fov_array.shape[0] != len(beam_angles):
                    raise ValueError("Invalid beam_fov format. The first axis length must match the number of beams.")
                fov_mask = beam_fov_array
            else:
                raise ValueError("Invalid beam_fov format. Must be scalar, 2D, or 3D array.")

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
        if optical_setup.beam_fov is None:
            fov_mask = ~jnp.isnan(view)
        else:
            if beam_fov_array.ndim == 3:
                fov_mask = beam_fov_array[i]
            else:
                fov_mask = fov_mask

        row_offset, col_offset = utils.align_fov_with_optical_axis(
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
