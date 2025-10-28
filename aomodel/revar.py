import aomodel
import numpy as np
import aomodel._utils as utils
import aomodel.pca as pca
import aomodel.long_range_var as vector_ar
import time
from datetime import timedelta


class ReVAR:
    """The ReVAR algorithm uses Principal Component Analysis (PCA) and Long-Range Vector Auto-Regression (LRVAR) to
    generate time-series of images with the same spatial and temporal statistics as an input dataset.

    Args:
        mask (ndarray, optional): [Default=None] numpy 2-D boolean array of shape (image height, image width)
            indicating which 2-D data indices correspond to valid pixel values.

            - If set to None, the function infers the mask based on which data values are "nan."

        time_lags (Union[int, list, ndarray], optional): [Default=1] either an integer, list, or numpy 1-D array
            indicating the time lags to use for the model.

            - If time_lags is an integer, the uses the previous time-steps up to this integer value.

            - If time_lags is a list or numpy 1-D array, the model uses these time lags.

        prediction_subspace_dimension (int, optional): [Default=None] number of (top) vector components to use in
            for linear time prediction. If set to True, the function uses this subspace as the prediction window.
            A prediction subspace and a prediction window distance cannot both be used.

            - If set to None, the function does not use a prediction subspace.

        num_moving_averages (int, optional): [Default=0] the number of moving averages to use in the linear time
            predictive model.

        moving_average_coefficients (ndarray, optional): [Default=None] a 1-D array of length (num_moving_averages,)
            containing the coefficients of each moving average.

        load_file (str, optional) [Default=None] directory to a file from which the model's instance variables can
            be loaded.
    """

    def __init__(self, mask, time_lags=1, prediction_subspace_dimension=None, num_moving_averages=0,
                 moving_average_coefficients=None, load_file=None):
        assert (mask.dtype == bool)
        assert (len(mask.shape) == 2)

        self.mask = mask
        num_pixels = mask.sum()

        # Initializes remaining instance variables:
        self.principal_components = None
        self.singular_values = None
        self.standard_deviation_vector = None
        self.mean_vector = None

        # Uses the load_file (if input by the user) to load necessary instance variables:
        if load_file is None:
            # Uses an instance from LongRangeVAR class to store relevant information and run the (vector) AR model:
            self.LRVAR = vector_ar.LongRangeVAR(vector_dimensionality=num_pixels, time_lags=time_lags,
                                                prediction_subspace_dimension=prediction_subspace_dimension,
                                                num_moving_averages=num_moving_averages,
                                                moving_average_coefficients=moving_average_coefficients)

            # Initializes instance variables used for PCA information:
            self.principal_components = np.identity(num_pixels)
            self.singular_values = np.ones((num_pixels,))
            self.standard_deviation_vector = np.ones((num_pixels,))
            self.mean_vector = np.zeros((num_pixels,))
        else:
            assert (type(load_file) == str)
            self.load(load_file=load_file)

    def lasso(self, data, tuning, tol=0.01, save_file=None, num_batches=1):
        """Computes LASSO calculation for the given data to determine which prediction coefficients to use for the
        regression calculation.

        Args:
            data (ndarray): numpy 3-D array containing the data to fit; must be a sequence of 2-D data samples (or
            "images") of shape (number of images, image height, image width).
            tuning (float): the tuning parameter for LASSO
            tol (float, optional): [Default=0.01] the tolerance to use for convergence of the LASSO calculation (used
                as an argument to the lasso function)
            save_file (str, optional): [Default=None] directory of the file to which we would like to save the boolean
                matrix of nonzero prediction coefficient indices (as returned by LASSO)

                - If set to None, the indices are not saved.

            num_batches (int, optional): [Default=1] the number of data batches to use for LASSO calculation
        """
        # Ensure that the data can be fit with this model (i.e., it has a valid mask):
        assert ((data.shape[1], data.shape[2]) == self.mask.shape)
        assert (not np.isnan(utils.img_to_vec(image_data=data, mask=self.mask)).any())

        start_time = time.time()

        # Uses the PCA class to analyze the data:
        data_pca = pca.PCA(data=data, mask=self.mask)

        # Copies over important PCA information for the input data:
        self.principal_components = data_pca.principal_components
        self.singular_values = data_pca.singular_values
        self.standard_deviation_vector = data_pca.standard_deviation_vector
        self.mean_vector = data_pca.mean_vector

        # Calls the lasso() function to compute LASSO prediction coefficients:
        self.LRVAR.lasso(data=data_pca.data_coefficients, tuning=tuning, tol=tol, save_file=save_file,
                         num_batches=num_batches, show_runtime=False)

        runtime_in_seconds = time.time() - start_time
        elapsed_time = str(timedelta(seconds=runtime_in_seconds))
        print("LASSO Calculation Completed in {} (hr:min:sec)\n".format(elapsed_time))

    def fit(self, data, percent_variance=None, cutoff_frequency=None, psd_block_size=None):
        """Calculates the model prediction coefficients A and error noise modulation B using least squares regression of
        the input data values.

        Args:
            data (ndarray): numpy 3-D array containing the data to fit; must be a sequence of 2-D data samples (or
                "images") of shape (number of images, image height, image width).

            percent_variance (float, optional): [Default=None] percent variance of the subspace to use for linear time
                prediction in the LongRangeVAR model.

                - If set to None, the prediction window indices are not updated.

            cutoff_frequency (float, optional): [Default=None] cutoff frequency (in units of cycles/sample) to use for
                estimation of the moving average parameter alpha.

                - If set to None, self.num_moving_averages is positive, and self.moving_average_coefficients is None,
                  then this attribute estimates the cutoff frequency as the frequency at which the temporal PSD of
                  the temporal coefficients (self.PCA.data_coefficients) reaches its maximum value.

            psd_block_size (int, optional): [Default=None] time-block size for calculating the temporal PSD of the
                temporal coefficients.

                - If set to None and the moving average coefficients need to be estimated, it uses a default block size
                  of data.shape[0] / 10 (i.e., one-tenth the length of the time-series).
        """
        # Ensure that the data can be fit with this model (i.e., it has a valid mask):
        assert ((data.shape[1], data.shape[2]) == self.mask.shape)
        assert (not np.isnan(utils.img_to_vec(image_data=data, mask=self.mask)).any())

        print("\nReVAR Parameter Estimation\n"
              "==========================")
        start_time = time.time()

        # Uses the PCA class to analyze the data:
        data_pca = pca.PCA(data=data, mask=self.mask)

        # Copies over important PCA information for the input data:
        self.principal_components = data_pca.principal_components
        self.singular_values = data_pca.singular_values
        self.standard_deviation_vector = data_pca.standard_deviation_vector
        self.mean_vector = data_pca.mean_vector

        # Calculates the number of components to include in the subspace:
        if percent_variance is not None:
            assert ((percent_variance > 0) and (percent_variance <= 1.0))
            num_coefficients = data_pca.find_top_coefficients(percent_variance)

            # Re-sets the prediction window indices of the linear time prediction model:
            self.set_indices(prediction_subspace_dimension=num_coefficients)

        # Has the Vector AR model fit the data coefficients (with respect to the principal components):
        self.LRVAR.fit(data=data_pca.data_coefficients, cutoff_frequency=cutoff_frequency,
                       psd_block_size=psd_block_size, show_runtime=False)

        runtime_in_seconds = time.time() - start_time
        elapsed_time = str(timedelta(seconds=runtime_in_seconds))
        print("ReVAR Parameter Estimation Completed in {} (hr:min:sec)\n".format(elapsed_time))

    def run(self, num_images):
        """Runs the forward autoregressive (AR) model using the prediction array A and the error noise modulation B, as
        set by either the fit() or load() instance methods. This model generate samples from the input data's
        distribution (with temporal dependence).

        Args:
            num_images (int): the number of time samples to generate.

        Returns:
            **samples** (*ndarray*) -- numpy 3-D array of shape (n, image height, image width) containing the model's
            output (which we call the generated samples).
        """
        print("\nReVAR Synthesis\n"
              "===============")
        start_time = time.time()

        # Copies over PCA information from the PCA instance:
        principal_components = self.principal_components
        singular_values = self.singular_values

        # Uses the PCA generative algorithm to create the initial images (i.e., in the coefficient space):
        time_lags = self.LRVAR.time_lags
        white_noise = np.random.normal(size=(principal_components.shape[0], max(time_lags)))
        standard_deviation_matrix = np.diag(np.sqrt(singular_values))
        initial_images = np.dot(standard_deviation_matrix, white_noise)

        # Generates samples using the Vector AR model:
        samples = self.LRVAR.run(initial_vectors=initial_images, num_vectors=num_images, show_runtime=False)

        # Projects the output of the AR model back onto the standard coordinate space:
        samples = np.dot(principal_components.T, samples)

        # Puts the data in the correct units:
        samples = np.multiply(samples, self.standard_deviation_vector[:, np.newaxis]) + self.mean_vector[:, np.newaxis]

        # Switch from vector to raster form (2-D images):
        samples = utils.vec_to_img(data_vec=samples.T, mask=self.mask)

        runtime_in_seconds = time.time() - start_time
        elapsed_time = str(timedelta(seconds=runtime_in_seconds))
        print("ReVAR Synthesis Completed in {} (hr:min:sec)\n".format(elapsed_time))

        return samples

    def save(self, save_file):
        """Saves all necessary information to re-construct the trained ReVAR model with a new instance:

        Args:
            save_file (str): directory of the file to which the data will be saved
        """
        assert (type(save_file) == str)
        prediction_coefficients = \
            self.LRVAR.prediction_coefficients[self.LRVAR.valid_2d_indices_of_prediction_coefficients_array]

        save_arrays = {'indices': self.LRVAR.indices, 'noise_modulation': self.LRVAR.noise_modulation,
                       'residuals_mean': self.LRVAR.residuals_mean,
                       'prediction_coefficients': prediction_coefficients, 'time_lags': self.LRVAR.time_lags,
                       'principal_components': self.principal_components, 'singular_values': self.singular_values,
                       'standard_deviation_vector': self.standard_deviation_vector, 'mean_vector': self.mean_vector,
                       'mask': self.mask}

        if self.LRVAR.prediction_subspace_dimension is not None:
            save_arrays['prediction_subspace_dimension'] = self.LRVAR.prediction_subspace_dimension

        if self.LRVAR.moving_average_coefficients is not None:
            save_arrays['moving_average_coefficients'] = self.LRVAR.moving_average_coefficients

        np.savez(save_file, **save_arrays)

    def load(self, load_file):
        """Loads the ReVARAR model information as saved by the save() method and re-constructs the model.

        Args:
            load_file (str): directory of the file from which the data will be loaded
        """
        assert (type(load_file) == str)

        # Load data containing instance variables:
        data = np.load(file=load_file, allow_pickle=True)

        # Ensure that the loaded model is compatible with the current model:
        data_mask = data['mask']
        assert ((self.mask == data_mask).all())
        num_pixels = np.sum(data_mask)

        # Creates a new LongRangeVAR instance with the loaded data:
        self.LRVAR = vector_ar.LongRangeVAR(vector_dimensionality=num_pixels, load_file=load_file)

        # Saves PCA instance variables:
        self.principal_components = data['principal_components']
        self.singular_values = data['singular_values']
        self.standard_deviation_vector = data['standard_deviation_vector']
        self.mean_vector = data['mean_vector']

    def set_indices(self, indices=None, predicted_comps=None, time_lags=None, prediction_subspace_dimension=None,
                    num_moving_averages=None):
        """Modifies the LongRangeVAR instance to store the correct indices.

        Args:
            indices (ndarray, optional): [Default=None] numpy boolean 2-D array of shape (num_pixels, num_pixels *
                num_lags) indicating which prediction coefficients to use for regression.

                - If set to None, the function computes the indices from the remaining parameters.

            predicted_comps (ndarray, optional): [Default=None] numpy 1-D array of indices of principal components
                for which we compute prediction coefficients.
            time_lags (Union[int, list, ndarray], optional): [Default=None] either an integer, list, or numpy 1-D array
                indicating the time lags to use for the model.

                - If set to None, the function uses the instance variable self.time_lags for the value of time_lags.

            prediction_subspace_dimension (int, optional): [Default=None] the number of (top) vector components to use
                in the linear time predictive model.

                - If set to None, the function uses all vector components in the predictive model.

            num_moving_averages (int, optional): [Default=0] the number of moving averages to use in the linear time
                predictive model.
        """
        self.LRVAR.set_indices(indices=indices, predicted_comps=predicted_comps, time_lags=time_lags,
                               prediction_subspace_dimension=prediction_subspace_dimension,
                               num_moving_averages=num_moving_averages)


def slopes_psd(data_values, locations=None, axis=2, time_block_size=1024, sampling_frequency=None, remove_mean=True,
               use_overlapping_blocks=True):
    """Approximates the slopes (gradient) of the data values (using a second order finite difference method) with
    respect to a user-specified axis and uses temporal_psd() to estimate the temporal power spectral density (PSD) of
    these slopes.

    Args:
        data_values (ndarray): numpy 3-D array of shape (number of images, image height, image width) containing the
            data values.
        locations (ndarray, optional): [Default=None] numpy 1-D array containing the coordinates of the axis to which
            the gradient is to be calculated. Sent as an argument to np.gradient.

            - If set to None, this argument is not included in the call to np.gradient. The indices are instead used.

        axis (int, optional): [Default=2] the axis of data values to take the gradient with respect to.

            - If axis = 2, the gradient is calculated with respect to the x-axis.

            - If axis = 1, the gradient is calculated with respect to the y-axis.

            - If axis = 0, the gradient is calculated with respect to time.

        time_block_size (int, optional): [Default=1000] the size of each time block to use for the PSD approximation.
            The full time-series is broken up into distinct "time-blocks" of the indicated size. For each time-block,
            the PSD is calculated independently. The final PSD calculation is then the average over each time-block.
            This value must be a positive even integer and can be at most the number of time-steps in data_values. The
            parameter is sent as an argument to the function temporal_psd().

        sampling_frequency (float, optional): [Default=None] the sampling frequency of the discrete-time signal
            "data_values." This input should be included if the desired PSD units are energy per unit time instead of
            energy per unit sample. In this case, the frequency bins are in units of cycles per unit time instead of
            cycles per unit sample. This value is sent as an argument to the function temporal_psd().

            - If set to None, the PSD units are energy/ample and the frequency units are cycles/sample.

        remove_mean (bool, optional): [Default=True] choice of removing the temporal mean of each vector component
            before computing the PSD. It is recommended to set this to True in most cases. If set to False, the lowest
            frequencies may be offset.

        use_overlapping_blocks (bool, optional): [Default=True] whether to use overlapping time-blocks when calculating
            the PSD. If set to True, then the time-blocks will have a 50% overlap. This method allows one to maintain
            the same block size while also reducing noise in the PSD calculation. If set to False, then the time-blocks
            will have no overlap.

    Returns:
        - **frequencies** (*ndarray*) -- A numpy 1-D array containing the frequency bins of the PSD calculation.
        - **psd_estimate** (*ndarray*) -- A numpy 1-D array containing the PSD estimates for each frequency bin.
    """
    assert (axis in [0, 1, 2])

    if locations is None:
        data_slopes = np.gradient(data_values, axis=axis)
    else:
        data_slopes = np.gradient(data_values, locations, axis=axis)

    # Sets the mask to be the intersection of valid data values for all images in the time-series:
    mask = (np.average(1 - np.uint8(np.isnan(data_slopes)), axis=0) == 1)
    return temporal_psd(data_values=data_slopes, time_block_size=time_block_size, mask=mask,
                        sampling_frequency=sampling_frequency, remove_mean=remove_mean,
                        use_overlapping_blocks=use_overlapping_blocks)


def temporal_psd(data_values, time_block_size=1024, mask=None, sampling_frequency=None, remove_mean=True,
                 use_overlapping_blocks=True):
    """Uses the vector_temporal_psd function to estimate the temporal power spectral density (PSD) of an input
    time-series of 2-D arrays (i.e., images).

    Args:
        data_values (ndarray): numpy 3-D array of shape (number of images, image height, image width) containing the
            data values
        time_block_size (int, optional): [Default=1024] the size of each time block to use for the PSD estimation.
            The full time-series is broken up into "time-blocks" of the indicated size. For each time-block, the PSD is
            calculated independently. The final PSD calculation is then the average over each time-block. This value
            must be a positive integer and can be at most the number of time-steps in data_values.
        mask (ndarray, optional): [Default=None] numpy 2-D boolean array of shape (image height, image width) indicating
            which 2-D data indices correspond to valid pixel values.

            - If set to None, the function infers the mask based on which data values are "nan."

        sampling_frequency (float, optional): [Default=None] the sampling frequency of the discrete-time signal
            "data_values." This input should be included if the desired PSD units are energy per unit time instead of
            energy per unit sample. In this case, the frequency bins are in units of cycles per unit time instead of
            cycles per unit sample.

            - If set to None, the PSD units are energy/sample and the frequency units are cycles/sample.

        remove_mean (bool, optional): [Default=True] choice of removing the temporal mean of each vector component
            before computing the PSD. It is recommended to set this to True in most cases. If set to False, the lowest
            frequencies may be offset.

        use_overlapping_blocks (bool, optional): [Default=True] whether to use overlapping time-blocks when calculating
            the PSD. If set to True, then the time-blocks will have a 50% overlap. This method allows one to maintain
            the same block size while also reducing noise in the PSD calculation. If set to False, then the time-blocks
            will have no overlap.

    Returns:
        - **frequencies** (*ndarray*) -- A numpy 1-D array containing the frequency bins of the PSD calculation.
        - **psd_estimate** (*ndarray*) -- A numpy 1-D array containing the PSD estimates for each frequency bin.
    """
    assert (len(data_values.shape) == 3)

    if mask is None:
        # Sets the mask to be the intersection of valid data values for all images in the time-series:
        mask = (np.average(1 - np.uint8(np.isnan(data_values)), axis=0) == 1)
    else:
        # Checks the mask on all images in the data set:
        assert (mask.dtype == bool)
        assert (mask.shape == (data_values.shape[1], data_values.shape[2]))
        assert (not np.isnan(utils.img_to_vec(image_data=data_values, mask=mask)).any())

    # Converts the input OPD images to vectors:
    data_vector = utils.img_to_vec(image_data=data_values, mask=mask)

    return vector_ar.vector_temporal_psd(data_values=data_vector, time_block_size=time_block_size,
                                         sampling_frequency=sampling_frequency, remove_mean=remove_mean,
                                         use_overlapping_blocks=use_overlapping_blocks)


def anisotropic_structure_function(data, mask=None, compute_square_root=False):
    """Estimate a generalized quasi-homogeneous Kolmogorov spatial structure function of an input time-series of 2-D
    data. The spatial structure function values are the (estimated) mean-squared differences of the input array "data"
    values at pairs of 2-D spatial locations. While the standard structure function computes these values as a function
    just the of relative separation (or the pixel distance between two spatial locations), the anisotrpic structure
    function depends on two variables: the relative separation and the angle of the difference between the two spatial
    locations.

    Args:
        data (ndarray): numpy 3-D array of shape (number of images, image height, image width) containing the data from
            which we would like to calculate turbulence structure function values.
        mask (ndarray, optional): [Default=None] numpy 2-D boolean array of shape (image height, image width) indicating
            which 2-D data indices correspond to valid pixel values.

            - If set to None, the function infers the mask based on which data values are "nan."

        compute_square_root (bool, optional): [Default=False] choice of whether to compute (and return) the square root
            of the structure function values (instead of the structure function values themselves)

    Returns:
        - **structure_function_inputs** (*ndarray*) -- A numpy 2-D array of shape (number of inputs, 2) containing each
          (relative separation, angle) input to the structure function.
        - **structure_function** (*ndarray*) -- A numpy 1-D array of shape (number of inputs,) containing the estimated
          structure function values (in the same order as the first output).
    """
    if mask is None:
        # Sets the mask to be the intersection of valid data values for all images in the time-series:
        mask = (np.average(1 - np.uint8(np.isnan(data)), axis=0) == 1)

    # Normalize the data's statistics by removing the moving and dividing by the standard deviation
    # (along each pixel):
    data_flat = aomodel.utils.img_to_vec(image_data=data, mask=mask).T
    mean = np.average(data_flat, axis=1)
    data_mean_removed = data_flat - mean[:, np.newaxis]
    standard_deviation_vector = np.sqrt(np.sum(data_mean_removed ** 2, axis=1) / data.shape[0])
    data_normalized = data_mean_removed / standard_deviation_vector[:, np.newaxis]

    # Calculate the list of relative separations between pixels (units: number of pixels)
    spatial_indices = np.argwhere(mask)
    differences = spatial_indices[:, np.newaxis, :] - spatial_indices[np.newaxis, :, :]
    relative_separations = np.linalg.norm(differences, axis=2)
    # Extract unique relative separations
    relative_separations = relative_separations[np.triu_indices(n=relative_separations.shape[0], k=1)]

    # Compute the quasi-homogeneous spatial structure function using the normalized data
    spatial_covariance = (1.0 / data.shape[0]) * (data_normalized @ data_normalized.T)
    # Extract the spatial covariance values that we need for structure function calculations
    covariance_values = spatial_covariance[np.triu_indices(n=spatial_covariance.shape[0], k=1)]
    structure_function_values = 2 * (1 - covariance_values)

    # Take the square root of all structure function values (if prompted to by the user):
    if compute_square_root:
        structure_function_values = np.sqrt(structure_function_values)

    # Sort the relative separation values in ascending order
    sort_indices = np.argsort(relative_separations)
    relative_separations = relative_separations[sort_indices]

    # Sort the structure function array accordining to the same indices
    structure_function_values = structure_function_values[sort_indices]

    # Compute and sort the angle of each difference
    angles = np.arctan2(differences[:, :, 0], differences[:, :, 1])
    angles = angles[np.triu_indices(n=angles.shape[0], k=1)]
    angles = np.mod(angles[sort_indices], np.pi)

    # Average the structure function values of each (relative separation, angle) pair
    unique_relative_separations = np.unique(relative_separations)
    structure_function_inputs = []
    structure_function = []
    for index, relative_separation in enumerate(unique_relative_separations):
        relative_separation_indices = np.squeeze(np.argwhere(relative_separations == relative_separation))
        associated_angles = angles[relative_separation_indices]
        unique_associated_angles = np.sort(np.unique(associated_angles))
        for angle in unique_associated_angles:
            angle_indices = np.squeeze(np.argwhere(angles == angle))
            intersect_indices = np.intersect1d(relative_separation_indices, angle_indices)
            structure_function_inputs.append([relative_separation, angle])
            structure_function.append(np.average(structure_function_values[intersect_indices]))

    structure_function_inputs = np.array(structure_function_inputs)
    structure_function = np.array(structure_function)

    return structure_function_inputs, structure_function