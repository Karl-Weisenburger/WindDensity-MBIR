import jax.numpy as jnp
import numpy as np
import time
from datetime import timedelta
import aomodel.pca as pca


class LongRangeVAR:
    """Implements a Long-Range Vector Auto-Regressive (LRVAR) model for generating synthetic data (i.e., a time-series
    of vectors) with the same spatial and temporal statistics as an input dataset. LRVAR uses two mechanisms to fit the
    temporal statistics of the input data: (1) a VAR component to fit high-frequency statistics and (2) a moving average
    (MA) component to fit low-frequency statistics. LRVAR then uses a Re-whitening step to fit the spatial statistics of
    the input data.

    Args:
        vector_dimensionality (int) dimensionality of the data vector(s) to model.
        time_lags (Union[int, list, ndarray], optional): [Default=1] either an integer, list, or numpy 1-D array
            indicating the time lags to use for the model.

            - If time_lags is an integer, the uses the previous time-steps up to this integer value.

            - If time_lags is a list or numpy 1-D array, the model uses these time lags.

        prediction_window_distance (int, optional): [Default=None] the prediction window distance (within the
            vector components).

            - If set to None, all vector components are included in each prediction window.

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

    def __init__(self, vector_dimensionality, time_lags=1, prediction_window_distance=None,
                 prediction_subspace_dimension=None, num_moving_averages=0, moving_average_coefficients=None,
                 load_file=None):
        assert (vector_dimensionality > 0)

        time_lags = np.array(time_lags)
        assert (len(time_lags.shape) <= 1)
        if len(time_lags.shape) == 0:
            assert (time_lags > 0)
            time_lags = np.arange(1, time_lags + 1)
        else:
            assert (time_lags > 0).all()
        if prediction_window_distance is not None:
            assert ((prediction_window_distance > 0) and (prediction_subspace_dimension is None))
        if prediction_subspace_dimension is not None:
            assert ((prediction_subspace_dimension > 0) and (prediction_subspace_dimension <= vector_dimensionality))
        assert (num_moving_averages >= 0)
        if moving_average_coefficients is not None:
            moving_average_coefficients = np.array(moving_average_coefficients)
            if len(moving_average_coefficients.shape) == 0:
                moving_average_coefficients = np.array([moving_average_coefficients])
            assert (moving_average_coefficients > 0).all() & (moving_average_coefficients < 1).all()
            num_moving_averages = len(moving_average_coefficients)

        self.time_lags = time_lags
        self.vector_dimensionality = vector_dimensionality
        self.prediction_window_distance = prediction_window_distance
        self.prediction_subspace_dimension = prediction_subspace_dimension
        self.num_moving_averages = num_moving_averages
        self.moving_average_coefficients = moving_average_coefficients

        # Objects used to perform PCA on the prediction error (or residuals):
        self.residuals_mean = np.zeros((vector_dimensionality,))
        self.noise_modulation = np.zeros((vector_dimensionality, vector_dimensionality))

        # Initializing instance attributes (to be filled in using either self.set_indices or self.load):
        self.prediction_coefficients = None
        self.predicted_comps = None
        self.indices = None
        self.valid_2d_indices_of_prediction_coefficients_array = None
        self.compressed_prediction_window_indexing_map = None

        if load_file is None:
            self.set_indices(time_lags=time_lags, prediction_window_distance=prediction_window_distance,
                             num_moving_averages=num_moving_averages)
        else:
            assert (type(load_file) == str)
            self.load(load_file=load_file)

    def lasso(self, data, tuning, tol=0.01, save_file=None, num_batches=1, show_runtime=True):
        """Computes LASSO calculation for the given data to determine which prediction coefficients to use for the
        regression calculation.

        Args:
            data (ndarray): numpy 2-D array of shape (data vector dimensionality, number of time samples) containing the
                data to fit.
            tuning (float): the tuning parameter for LASSO.
            tol (float, optional): [Default=0.01] the tolerance to use for convergence of the LASSO calculation (used
                as an argument to the lasso function).
            save_file (str, optional): [Default=None] directory of the file to which we would like to save the boolean
                array indicating the non-zero prediction coefficient indices (as returned by LASSO).

                - If set to None, the indices are not saved.

            num_batches (int, optional): [Default=1] the number of data batches to use for LASSO calculation

            show_runtime (bool, optional): [Default=True] indicates whether to compute (and print) the runtime of the
                calculation.
        """
        assert (len(data.shape) == 2)

        # Updates the data vector dimensionality for this model:
        self.vector_dimensionality = data.shape[0]

        print("\nLRVAR Model Pre-Fitting: LASSO\n"
              "==================================================\n"
              "{:<35}".format("Tuning Parameter") + "|%10.2e" % tuning + " |\n"
              "{:<35}".format("Number of Time-Steps in Input Data") + "|%10d" % data.shape[1] + " |\n"
              "================================================\n")

        if show_runtime:
            start_time = time.time()

        # Calls the lasso() function to compute LASSO prediction coefficient indices:
        indices = lasso(tuning=tuning, num_lags=1, data=data, tol=tol, predicted_comps=self.predicted_comps,
                        num_batches=num_batches)

        if show_runtime:
            elapsed_time = str(timedelta(seconds=time.time() - start_time))
            print("LASSO Calculation Completed in {} (hr:min:sec)\n".format(elapsed_time))

        # Print out the results of the LASSO calculation:
        nonzero_coefficients = len(indices.nonzero()[0])
        print("Number of Non-Zero Prediction Coefficients: %d\n" % nonzero_coefficients)

        # Concatenates prediction coefficient indices to include all time lags:
        indices = np.tile(indices, (1, len(self.time_lags)))

        # Sets new indices for regression:
        self.set_indices(indices=indices)

        # Saves the indices array (if save_file is included as input):
        if save_file is not None:
            assert (type(save_file) == str)
            np.savez(file=save_file, indices=indices)

    def fit(self, data, cutoff_frequency=None, psd_block_size=None, show_runtime=True):
        """Calculates the model prediction coefficients and prediction error noise modulation matrix using least squares
        regression from the input data values.

        Args:
            data (ndarray): numpy 2-D array of shape (data vector dimensionality, number of time samples) containing the
                data to fit.

            cutoff_frequency (float, optional): [Default=None] cutoff frequency (in units of cycles/sample) to use for
                estimation of the moving average parameter alpha.

                - If set to None, self.num_moving_averages is positive, and self.moving_average_coefficients is None,
                  then this attribute estimates the cutoff frequency as the frequency at which the temporal PSD of
                  the temporal coefficients (self.PCA.data_coefficients) reaches its maximum value.

            psd_block_size (int, optional): [Default=None] time-block size for calculating the temporal PSD of the
                temporal coefficients.

                - If set to None and the moving average coefficients need to be estimated, it uses a default block size
                  of data.shape[0] / 10 (i.e., one-tenth the length of the time-series).

            show_runtime (bool, optional): [Default=True] indicates whether to compute (and print) the runtime of the
                calculation.
        """
        if (self.num_moving_averages > 0) and (self.moving_average_coefficients is not None):
            assert (len(self.moving_average_coefficients) == self.num_moving_averages)
        assert (len(data.shape) == 2)

        # Updates the data vector dimensionality for this model:
        self.vector_dimensionality = data.shape[0]
        num_samples = data.shape[1]
        time_lags = self.time_lags

        # Print out model parameters:
        print("\nLong-Range Vector Auto-Regression (LRVAR) Model Training\n"
              "========================================================\n"
              "Number of Time-Steps in Input Data:  %d" % num_samples + '\n')

        if show_runtime:
            start_time = time.time()

        # If not already specified, estimates the moving average coefficients using a cutoff frequency:
        if (self.num_moving_averages > 0) and (self.moving_average_coefficients is None):
            self.moving_average_coefficients = np.zeros(self.num_moving_averages)
            # If not provided, estimates the cutoff frequency as the frequency at which the temporal PSD of the
            # temporal coefficients reaches its maximum value:
            if cutoff_frequency is None:
                # Ensures the temporal PSD computation has at least 10 dB:
                if psd_block_size is None:
                    psd_block_size = int(num_samples // 50.5)

                # Computes the PSD of only the vector components which are included in the prediction window:
                if self.prediction_subspace_dimension is not None:
                    num_components = self.prediction_subspace_dimension
                else:
                    num_components = len(self.predicted_comps)
                truncated_data = data[:num_components]

                frequencies, training_data_psd = vector_temporal_psd(data_values=truncated_data.T,
                                                                     time_block_size=psd_block_size)
                cutoff_frequency = find_maximal_frequency(frequencies, training_data_psd)

            # Uses the cutoff frequency to estimate the first moving average coefficient:
            time_constant = 1 / (2 * np.pi * cutoff_frequency)
            alpha = 1 / (10.0 * time_constant)
            self.moving_average_coefficients[0] = alpha

            # Sets additional moving average coefficients to an order of magnitude below the first coefficients:
            for moving_average_index in range(1, self.num_moving_averages):
                self.moving_average_coefficients[moving_average_index] = (
                    self.moving_average_coefficients[moving_average_index - 1] / 10.0)

        predicted_comps = self.predicted_comps
        prediction_array_indices = self.compressed_prediction_window_indexing_map

        # Uses least squares regression to calculate the prediction coefficients for each vector component:
        prediction_coefficients, residuals = (
            least_squares_solution(time_lags=time_lags, data=data,
                                   moving_average_coefficients=self.moving_average_coefficients,
                                   prediction_window_indices=prediction_array_indices[1],
                                   predicted_comps=predicted_comps))

        # Calculates and removes the mean of the residuals for PCA:
        self.residuals_mean = np.average(residuals, axis=1)
        residuals_mean_removed = residuals - self.residuals_mean[:, np.newaxis]

        # Calculates the noise modulation matrix (encoding the covariance of the prediction error):
        residuals_principal_components, residuals_singular_values = (
            pca.compute_principal_components(residuals_mean_removed))
        noise_modulation = np.dot(residuals_principal_components, np.diag(np.sqrt(residuals_singular_values)))
        self.noise_modulation = noise_modulation

        # Saves the least-squares solution values as prediction coefficients at the corresponding indices of the 2-D
        # prediction coefficients array (i.e., at indices that should contain non-zero values):
        self.prediction_coefficients[self.valid_2d_indices_of_prediction_coefficients_array] = \
            prediction_coefficients.flatten()

        if show_runtime:
            elapsed_time = str(timedelta(seconds=time.time() - start_time))
            print("LRVAR Model Fitting Completed in {} (hr:min:sec)\n".format(elapsed_time))

    def run(self, initial_vectors, num_vectors, show_runtime=True):
        """Runs the LRVAR model to generate synthetic data using the prediction coefficients and prediction error noise
        modulation matrix, as set by either the fit() or load() instance methods.

        Args:
            initial_vectors (ndarray): numpy 2-D array of shape (data vector dimensionality, number of time lags)
                containing the initial data vectors used by the generative algorithm.
            num_vectors (int): the number of time samples to generate.
            show_runtime (bool, optional): [Default=True] indicates whether to compute (and print) the runtime of the
                calculation.

        Returns:
            **samples** (*ndarray*) -- a 2-D numpy array of shape (data vector dimensionality, number of samples)
            containing the model's output samples.
        """
        assert (num_vectors > 0)
        assert (initial_vectors.shape == (self.vector_dimensionality, max(self.time_lags)))

        print("\nLRVAR Model Generation\n"
              "======================\n"
              "Number of Time-Samples to Generate:  %d" % num_vectors + '\n')

        if show_runtime:
            start_time = time.time()

        time_lags = self.time_lags
        num_lags = len(time_lags)
        vector_dimensionality = self.vector_dimensionality
        prediction_coefficients = self.prediction_coefficients
        noise_modulation = self.noise_modulation
        residuals_mean = self.residuals_mean
        predicted_comps = self.predicted_comps

        # Finds the indices of vector components for which there are no prediction coefficients:
        remaining_comps = np.setdiff1d(np.arange(0, vector_dimensionality), predicted_comps)

        # Computes the time index offsets for generation. If moving averages are used, adds space in the array for the
        # averages:
        if self.moving_average_coefficients is not None:
            num_moving_averages = len(self.moving_average_coefficients)
            num_time_steps = num_lags + num_moving_averages

            num_time_indices = np.arange(num_time_steps * vector_dimensionality)
            time_offset_vals = (num_time_indices // vector_dimensionality) - num_moving_averages

            time_index_offsets = np.zeros_like(num_time_indices)
            moving_average_indices = (time_offset_vals < 0)
            time_index_offsets[moving_average_indices] = time_offset_vals[moving_average_indices] * (-1)
            time_index_offsets[~moving_average_indices] = time_lags[time_offset_vals[~moving_average_indices]] * (-1)
        else:
            num_time_steps = num_lags
            time_index_offsets = time_lags[np.arange(num_lags * vector_dimensionality) // vector_dimensionality] * (-1)

        # Indices for the time step offsets and vector components to use for prediction:
        time_step_offsets_for_prediction_window = \
            np.repeat(time_index_offsets[np.newaxis, :],
                      vector_dimensionality, 0)[self.compressed_prediction_window_indexing_map]
        vector_component_indices_for_prediction_window = \
            np.tile(np.arange(vector_dimensionality),
                    (vector_dimensionality, num_time_steps))[self.compressed_prediction_window_indexing_map]

        # Initializes the output 'samples' array and fills in the initial vectors:
        samples = np.zeros((vector_dimensionality, num_vectors + max(time_lags)))
        samples[:, :max(time_lags)] = initial_vectors

        # Uses PCA to generate synthetic prediction error (or residuals) for VAR generation:
        white_noise = np.random.normal(size=(vector_dimensionality, num_vectors))
        error_noise = np.dot(noise_modulation, white_noise) + residuals_mean[:, np.newaxis]

        # Initializes arrays to store the moving averages:
        if self.moving_average_coefficients is not None:
            num_ma_rows = np.sum(time_step_offsets_for_prediction_window[0] > 0)

            # Defines the row indices and arrays for each moving average:
            moving_average_rows = []
            moving_averages = []
            for moving_average_index in range(num_moving_averages):
                current_moving_average_columns = \
                    (time_step_offsets_for_prediction_window[0] == num_moving_averages - moving_average_index)
                current_moving_average_row_indices = (
                    vector_component_indices_for_prediction_window[:, current_moving_average_columns])
                moving_average_rows.append(current_moving_average_row_indices)
                moving_averages.append(np.zeros_like(samples[current_moving_average_row_indices, 0]))

        # Applies the LRVAR model recursively for the vector components that have been assigned (non-zero) prediction
        # coefficients:
        for j in range(max(time_lags), num_vectors + max(time_lags)):
            # Pulls the previous values (to use for linear time prediction) into an array:
            if self.moving_average_coefficients is not None:
                # Previous data values from the prediction window:
                previous_data_vector_values = (
                    samples[vector_component_indices_for_prediction_window[:, num_ma_rows:],
                    j + time_step_offsets_for_prediction_window[:, num_ma_rows:]])

                # Combines the previous data values and moving averages in the array:
                moving_average_array = np.concatenate(moving_averages, axis=1)
                previous_values = np.concatenate((moving_average_array, previous_data_vector_values), axis=1)
            else:
                previous_values = \
                    samples[vector_component_indices_for_prediction_window, j + time_step_offsets_for_prediction_window]

            # Computes the linear time prediction:
            samples[predicted_comps, j] = \
                np.einsum('ij,ij->i', previous_values, prediction_coefficients, optimize='greedy') + \
                error_noise[(predicted_comps, j - max(time_lags))]

            # Updates the moving averages:
            if self.moving_average_coefficients is not None:
                for moving_average_index in range(num_moving_averages):
                    alpha = self.moving_average_coefficients[moving_average_index]
                    moving_averages[moving_average_index] = (
                            alpha * samples[moving_average_rows[moving_average_index], j] +
                            (1 - alpha) * moving_averages[moving_average_index])

        # Fill in the noise for the remaining (non-predicted) components:
        samples[remaining_comps, max(time_lags):] = error_noise[remaining_comps]

        if show_runtime:
            elapsed_time = str(timedelta(seconds=time.time() - start_time))
            print("LRVAR Model Generation Completed in {} (hr:min:sec)\n".format(elapsed_time))
        return samples[:, max(time_lags):]

    def save(self, save_file):
        """Saves all necessary LRVAR model information to re-construct our model with a new instance.

        Args:
            save_file (str): directory of the file to which the data will be saved.
        """
        assert (type(save_file) == str)
        prediction_coefficients = self.prediction_coefficients[self.valid_2d_indices_of_prediction_coefficients_array]

        save_arrays = {'noise_modulation': self.noise_modulation, 'residuals_mean': self.residuals_mean,
                       'indices': self.indices, 'time_lags': self.time_lags,
                       'prediction_coefficients': prediction_coefficients}

        if self.prediction_window_distance is not None:
            save_arrays['prediction_window_distance'] = self.prediction_window_distance

        if self.prediction_subspace_dimension is not None:
            save_arrays['prediction_subspace_dimension'] = self.prediction_subspace_dimension

        if self.moving_average_coefficients is not None:
            save_arrays['moving_average_coefficients'] = self.moving_average_coefficients

        np.savez(save_file, **save_arrays)

    def load(self, load_file):
        """Loads the LRVAR model information as saved by the save() method and re-constructs the model.

        Args:
            load_file (str): directory of the file from which the data will be loaded.
        """
        assert (type(load_file) == str)

        print("Loading LongRangeVAR Instance Variables From {}...".format(load_file))
        data = np.load(file=load_file, allow_pickle=True)

        time_lags = data['time_lags']
        indices = data['indices']
        prediction_coefficients = data['prediction_coefficients']

        if 'prediction_window_distance' in data:
            prediction_window_distance = data['prediction_window_distance']
            self.prediction_window_distance = prediction_window_distance
        else:
            prediction_window_distance = None

        if 'prediction_subspace_dimension' in data:
            prediction_subspace_dimension = data['prediction_subspace_dimension']
            self.prediction_subspace_dimension = prediction_subspace_dimension
        else:
            prediction_subspace_dimension = None

        if 'moving_average_coefficients' in data:
            moving_average_coefficients = data['moving_average_coefficients']
            assert (moving_average_coefficients > 0).all() & (moving_average_coefficients < 1).all()
            self.moving_average_coefficients = moving_average_coefficients
            self.num_moving_averages = len(self.moving_average_coefficients)
        else:
            self.num_moving_averages = 0

        self.vector_dimensionality = indices.shape[0]
        self.noise_modulation = data['noise_modulation']
        self.residuals_mean = data['residuals_mean']

        print("Finished Loading LongRangeVAR Instance Variables From {}\n".format(load_file))

        self.set_indices(indices=indices, time_lags=time_lags, prediction_window_distance=prediction_window_distance,
                         prediction_subspace_dimension=prediction_subspace_dimension)

        self.prediction_coefficients[self.valid_2d_indices_of_prediction_coefficients_array] = prediction_coefficients

    def set_indices(self, indices=None, predicted_comps=None, time_lags=None, prediction_window_distance=None,
                    prediction_subspace_dimension=None, num_moving_averages=None):
        """Defines the instance variables relevant to the prediction coefficient indices (i.e., the nonzero prediction
        coefficients calculated by the fit() method and used in the forward model).

        Args:
            indices (ndarray, optional): [Default=None] numpy boolean 2-D array of shape (data vector dimensionality,
                data vector dimensionality * number of time lags) indicating which prediction coefficients to solve for
                in the model fitting.

                - If set to None, the function infers the indices from the remaining parameters.

            predicted_comps (ndarray, optional): [Default=None] numpy 1-D integer array containing the indices of
                the data vector's components for which the model will compute prediction coefficients. The remaining
                components will only be determined by the prediction error noise.
            time_lags (Union[int, list, ndarray], optional): [Default=None] either an integer, list, or numpy 1-D array
                indicating the time lags to use for the model.

                - If set to None, the function uses the instance variable self.time_lags for the value of time_lags.

            prediction_window_distance (int, optional): [Default=None] value to use for the prediction window distance
                in the model fitting.

                - If set to None, the function does not use a uniform prediction window.

            prediction_subspace_dimension (int, optional): [Default=None] number of (top) vector components to use in
                for linear time prediction. If set to True, the function uses this subspace as the prediction window.
                A prediction subspace and a prediction window distance cannot both be used.

                - If set to None, the function does not use a prediction subspace.

            num_moving_averages (int, optional): [Default=0] the number of moving averages to use in the linear time
                predictive model.
        """
        if time_lags is None:
            time_lags = self.time_lags
        else:
            time_lags = np.array(time_lags)
            assert (len(time_lags.shape) <= 1)
            if len(time_lags.shape) == 0:
                assert (time_lags > 0)
                time_lags = np.arange(1, time_lags + 1)
            else:
                assert (time_lags > 0).all()
            self.time_lags = time_lags

        if num_moving_averages is not None:
            assert (num_moving_averages >= 0)
            if num_moving_averages != self.num_moving_averages:
                self.moving_average_coefficients = None
            self.num_moving_averages = num_moving_averages
        else:
            num_moving_averages = self.num_moving_averages

        num_time_lags = len(time_lags)

        total_lags = num_time_lags + num_moving_averages

        vector_dimensionality = self.vector_dimensionality

        # Form the boolean 'indices' array:
        if indices is None:
            if predicted_comps is None:
                predicted_comps = np.arange(vector_dimensionality)
            else:
                assert (len(predicted_comps.shape) == 1)
                assert (predicted_comps.dtype == int)
                assert ((0 <= predicted_comps).all() and (predicted_comps < vector_dimensionality).all())

            if prediction_window_distance is None:
                if prediction_subspace_dimension is not None:
                    assert ((prediction_subspace_dimension > 0) and
                            (prediction_subspace_dimension <= vector_dimensionality))
                    self.prediction_subspace_dimension = prediction_subspace_dimension
                    self.prediction_window_distance = None
                    predicted_comps = np.arange(prediction_subspace_dimension)
                    indices = np.full(shape=(vector_dimensionality, vector_dimensionality),
                                      fill_value=False, dtype=bool)
                    indices[:prediction_subspace_dimension, :prediction_subspace_dimension] = True
                    indices = np.tile(indices, (1, total_lags))
                else:
                    indices = np.full(shape=(vector_dimensionality, total_lags * vector_dimensionality),
                                      fill_value=True, dtype=bool)
            else:
                distance_upper_bound = (vector_dimensionality // 2) - 1 + (vector_dimensionality % 2)
                assert ((prediction_window_distance > 0) and (prediction_window_distance <= distance_upper_bound))
                assert (prediction_subspace_dimension is None)
                self.prediction_window_distance = prediction_window_distance
                self.prediction_subspace_dimension = None
                indices = np.full(shape=(vector_dimensionality, vector_dimensionality), fill_value=False, dtype=bool)
                component_indices = np.arange(vector_dimensionality)[np.newaxis, :]
                prediction_window = np.arange(-prediction_window_distance,
                                              prediction_window_distance + 1)[:, np.newaxis]

                # Uniform prediction windows created for each vector component:
                prediction_indices = (np.tile(np.arange(vector_dimensionality)[:, np.newaxis],
                                              (1, 2 * prediction_window_distance + 1)),
                                      ((component_indices + prediction_window) % vector_dimensionality).T)

                indices[prediction_indices] = True
                indices = np.tile(indices, (1, total_lags))
        else:
            assert (indices.dtype == bool)
            assert (indices.shape == (vector_dimensionality, total_lags * vector_dimensionality))

            # If moving averages are used, ensures that the indices array is compatible (i.e., each lag has the same
            # vector components):
            if num_moving_averages > 0:
                first_lag_indices = indices[:, :vector_dimensionality]
                for lag_index in range(1, total_lags):
                    start_index = lag_index * vector_dimensionality
                    end_index = start_index + vector_dimensionality
                    current_lag_indices = indices[:, start_index:end_index]
                    assert (current_lag_indices == first_lag_indices).all()

            # Infers the "predicted components" (i.e., components which are assigned non-zero prediction coefficients)
            # from the entries of the "indices" boolean array:
            predicted_comps = np.sort(np.unique(indices.nonzero()[0]))

        # Removes all non-zero prediction coefficient entries of "indices" that are assigned by the "remaining
        # components" (i.e., the vector components that should not be assigned prediction coefficients, as indicated by
        # the "predicted_comps" input):
        remaining_comps = np.setdiff1d(np.arange(0, vector_dimensionality), predicted_comps)
        indices[remaining_comps] = False

        # Creates compressed indexing arrays which efficiently capture the prediction window (used for model fitting
        # calculations) and the prediction coefficients array
        num_coefficients = np.sum(indices, axis=1)
        max_num_coefficients = np.max(num_coefficients)
        num_predicted_comps = len(predicted_comps)

        # Indexing map for valid entries of the prediction coefficients 2-D array:
        valid_row_indices_of_prediction_coefficients_array = np.repeat(np.arange(num_predicted_comps),
                                                                       num_coefficients[num_coefficients.nonzero()])
        valid_col_indices_of_prediction_coefficients_array = np.concatenate([np.arange(val) for val in
                                                                             num_coefficients])
        valid_2d_indices_of_prediction_coefficients_array = (valid_row_indices_of_prediction_coefficients_array,
                                                             valid_col_indices_of_prediction_coefficients_array)

        # Index mapping from compressed prediction window array to full prediction window array:
        valid_prediction_window_indices = indices.nonzero()
        compressed_prediction_window_indexing_map = (np.full((num_predicted_comps, max_num_coefficients), -1,
                                                             dtype=np.int_),
                                                     np.full((num_predicted_comps, max_num_coefficients), -1,
                                                             dtype=np.int_))
        compressed_prediction_window_indexing_map[0][valid_2d_indices_of_prediction_coefficients_array] = \
            valid_prediction_window_indices[0]
        compressed_prediction_window_indexing_map[1][valid_2d_indices_of_prediction_coefficients_array] = \
            valid_prediction_window_indices[1]

        self.indices = indices
        self.valid_2d_indices_of_prediction_coefficients_array = valid_2d_indices_of_prediction_coefficients_array
        self.compressed_prediction_window_indexing_map = compressed_prediction_window_indexing_map
        self.predicted_comps = predicted_comps
        self.prediction_coefficients = np.zeros(self.compressed_prediction_window_indexing_map[0].shape)

        num_prediction_coefficients = indices.sum()

        # Print out model parameters
        print("\nLong-Range VAR (LRVAR) Model Parameters\n"
              "=========================================\n"
              "{:<31}".format("Vector Dimensionality") + "|%7s" % vector_dimensionality + " |\n"
              "{:<31}".format("Number of Time Lags") + "|%7s" % num_time_lags + " |")

        if self.num_moving_averages > 0:
            print("{:<31}".format("Number of Moving Averages") + "|%7s" % self.num_moving_averages + " |")

        if self.prediction_window_distance is not None:
            print("{:<31}".format("Prediction Window Distance") + "|%7s" % self.prediction_window_distance + " |")

        if self.prediction_subspace_dimension is not None:
            print("{:<31}".format("Prediction Subspace Dimension") + "|%7s" % self.prediction_subspace_dimension + " |")

        print("{:<31}".format("Number of Predicted Components") + "|%7s" % num_predicted_comps + " |\n"
              "{:<31}".format("Number of Prediction Weights") + "|%7s" % num_prediction_coefficients + " |\n"
              "=========================================\n")


def find_maximal_frequency(frequencies, temporal_psd_values):
    """Finds the frequency at which a given temporal PSD reaches its maximum value using parabolic interpolation.

    Args:
        frequencies (ndarray): numpy 1-D array containing the frequency bins of the temporal PSD.
        temporal_psd_values (ndarray): numpy 1-D array containing the temporal PSD values at each frequency bin.

    Returns:
        **max_frequency** (*float*) -- the frequency at which the input temporal PSD reaches its maximum value.
    """
    assert (frequencies.shape == temporal_psd_values.shape)

    maximal_index = np.squeeze(np.argmax(temporal_psd_values))

    # Fits a parabola to the three values centered at the maximum:
    neighboring_frequencies = frequencies[maximal_index - 1:maximal_index + 2]
    neighboring_psd_values = temporal_psd_values[maximal_index - 1:maximal_index + 2]
    psd_parabolic_interpolation = np.polynomial.polynomial.Polynomial.fit(neighboring_frequencies,
                                                                          neighboring_psd_values, deg=2)

    # Finds the maximum value of the parabola:
    interpolated_frequencies = np.linspace(neighboring_frequencies[0], neighboring_frequencies[-1], num=100)
    interpolated_psd = psd_parabolic_interpolation(interpolated_frequencies)

    interpolated_maximal_index = np.squeeze(np.argmax(interpolated_psd))
    max_frequency = interpolated_frequencies[interpolated_maximal_index]

    return max_frequency


def vector_temporal_psd(data_values, time_block_size=1024, sampling_frequency=None, remove_mean=True,
                        use_overlapping_blocks=True):
    """Estimates the temporal Power Spectral Density (PSD) of the input data values by averaging the 1-D PSD
    estimates for each vector component (i.e., for each time-series of data values at a single vector component). Each
    1-D PSD is estimated using Welch's method, in which the time-series is broken up into independent "blocks" of length
    "time_block_size." A Hamming window is applied to each block and the PSD is estimated using an FFT and a scaling
    factor which lowers the variance of the estimate.

    Args:
        data_values (ndarray): numpy 2-D array of shape (number of time-steps, vector dimensionality) containing the
            data values

        time_block_size (int, optional): [Default=1024] the size of each time block to use for the PSD estimation.
            The full time-series is broken up into "time-blocks" of the indicated size. For each time-block, the PSD is
            calculated independently. The final PSD calculation is then the average over each time-block. This value
            must be a positive integer and can be at most the number of time-steps in data_values.

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
    assert (len(data_values.shape) == 2)
    assert (0 < time_block_size <= data_values.shape[0])

    # Calculates the number of time-blocks we average over:
    if use_overlapping_blocks:
        num_blocks = 2 * (data_values.shape[0] // time_block_size) - 1
    else:
        num_blocks = data_values.shape[0] // time_block_size

    # Initializes a Hamming window and computes relevant quantities:
    hamming_window = np.hamming(time_block_size)
    welch_scaling_factor = np.sum(hamming_window ** 2)

    # Frequencies along the x-axis:
    frequencies = np.fft.rfftfreq(n=time_block_size)

    # Removes the temporal mean from each vector component:
    if remove_mean:
        temporal_mean = np.mean(data_values, axis=0)
        data_values = data_values - temporal_mean[np.newaxis, :]

    # Iterates over each time-block and averages the temporal PSD:
    psd_estimates = np.zeros((len(frequencies), data_values.shape[1]))
    for block_idx in range(num_blocks):
        # Extracts the time-series for the current block:
        if use_overlapping_blocks:
            block_data = data_values[int((block_idx / 2) * time_block_size):
                                     int(((block_idx / 2) + 1) * time_block_size)]
        else:
            block_data = data_values[block_idx * time_block_size: (block_idx + 1) * time_block_size]

        # If mean-removal is not selected, takes the FFT of the windowed block data (with its mean):
        windowed_data = block_data * hamming_window[:, np.newaxis]

        # Uses an FFT to approximate the DFT of the current section:
        block_dft = np.fft.rfft(windowed_data, axis=0)

        # Uses the FFT to approximate the energy spectrum of the current time-block:
        block_energy_spectrum = np.abs(block_dft) ** 2

        # Applies the Welch's method scaling factor to compute the PSD estimate for this block
        psd_estimates += block_energy_spectrum / welch_scaling_factor

    # Averages the PSD estimates across each row (i.e., across the estimates for each pixel) and divides by the number
    # of sections used:
    psd_estimate = np.average(psd_estimates, axis=1) / num_blocks

    # If a sampling frequency is provided, divides the PSD estimate by the sampling frequency to ensure correct unit
    # conversion:
    if sampling_frequency:
        assert (sampling_frequency > 0)
        frequencies = sampling_frequency * frequencies
        psd_estimate = psd_estimate / sampling_frequency

    return frequencies, psd_estimate


def least_squares_solution(time_lags, data, prediction_window_indices, predicted_comps,
                           moving_average_coefficients=None):
    """Uses least-squares regression to approximate the Minimum Mean Squared Error (MMSE) coefficients for a vector
    autoregressive (AR) model which uses a finite number of time lags (num_lags). The function calculates the optimal
    prediction coefficient values independently for each component of the random vector that is listed in the
    "predicted_comps" input. The prediction coefficients are output in a 1-D array, in the order of the vector
    components to which they are associated. A 1-D array is chosen in place of a 2-D array in the case that the number
    of prediction coefficients is not uniform across all components.

    Args:
        time_lags (ndarray, optional): [Default=None] numpy 1-D array of time lags to use for the model.
        data (ndarray): numpy 2-D array of shape (data vector dimensionality, number of samples) containing the
            ground-truth data values for the regression calculation.
        prediction_window_indices (ndarray): numpy 2-D array of shape (number of predicted vector components, maximum
            prediction window size) containing the (integer) indices for which to calculate prediction coefficients. The
            first axis may not be of the same length as the vector dimensionality, in the case that prediction
            coefficient calculation is not carried out for some vector components. The "prediction window size" is how
            many vector components from previous time-steps are used for the prediction (i.e., how many vector
            components are multiplied by non-zero prediction coefficients at each previous time lag). In the case that
            one vector component has a smaller prediction window size than others, the remaining spaces in the row
            associated to the component should be filled in with -1.
        predicted_comps (ndarray): numpy 1-D array containing the (integer) indices of the vector components for which
            prediction coefficients will be calculated.

        moving_average_coefficients (ndarray, optional): [Default=None] a 1-D array of length (num_moving_averages,)
            containing the coefficients of each moving average.

    Returns:
        - **prediction_coefficients_array** (*ndarray*) -- A numpy 1-D array containing the (concatenated) prediction
          coefficients for each vector component.
        - **residuals** (*ndarray*) -- A numpy 2-D array of shape (vector dimensionality, num_samples - max(time_lags))
          containing the residuals.
    """
    assert (len(time_lags.shape) == 1)
    assert (time_lags > 0).all()

    assert (prediction_window_indices.dtype == int)
    assert (len(data.shape) == len(prediction_window_indices.shape) == 2)

    assert (len(predicted_comps.shape) == 1)
    assert (predicted_comps.dtype == int)
    assert ((0 <= predicted_comps).all() and (predicted_comps < data.shape[1]).all())

    vector_dimensionality = data.shape[0]
    num_samples = data.shape[1]

    # Array that stores the linear prediction (from the previous values multiplied by the prediction coefficients):
    prediction = np.zeros((vector_dimensionality, num_samples - max(time_lags)))

    # Creates indexing arrays for the compressed prediction window:
    valid_prediction_window_indices = (prediction_window_indices != -1)
    data_time_step_indices_for_regression = np.arange(num_samples - max(time_lags))[:, np.newaxis]
    vector_component_indices_for_prediction_window = prediction_window_indices % vector_dimensionality

    # If moving averages are used, adds space in the array of time-step offsets for the moving averages:
    if moving_average_coefficients is not None:
        num_moving_averages = len(moving_average_coefficients)
        time_step_offsets_for_prediction_window = np.zeros_like(prediction_window_indices)
        time_lag_indices = (prediction_window_indices // vector_dimensionality) - num_moving_averages
        moving_average_indices = (time_lag_indices < 0)
        time_step_offsets_for_prediction_window[moving_average_indices] = time_lag_indices[moving_average_indices] * (-1)
        time_step_offsets_for_prediction_window[~moving_average_indices] = time_lags[time_lag_indices[~moving_average_indices]] * (-1)
    else:
        time_step_offsets_for_prediction_window = time_lags[prediction_window_indices // vector_dimensionality] * (-1)

    # Uses a 1-D array to store the prediction coefficients:
    prediction_coefficients_array = np.array([])

    # Use least squares regression to calculate prediction coefficients independently for each vector component:
    for i, comp in enumerate(predicted_comps):
        # Fill in prediction array for current vector component "comp":
        vector_component_indices = \
            np.tile(vector_component_indices_for_prediction_window[i, valid_prediction_window_indices[i]],
                    (num_samples - max(time_lags), 1))
        time_step_indices = np.tile(max(time_lags) +
                                    time_step_offsets_for_prediction_window[i, valid_prediction_window_indices[i]],
                                    (num_samples - max(time_lags), 1)) + data_time_step_indices_for_regression

        # Defines the prediction array containing the previous (known) data values:
        if moving_average_coefficients is not None:
            all_moving_average_rows = vector_component_indices[0, (time_step_indices[0] > max(time_lags))]
            num_total_ma_rows = len(all_moving_average_rows)

            # Array containing the previous data vectors:
            data_vector_arrays = (
                data[vector_component_indices[:, num_total_ma_rows:], time_step_indices[:, num_total_ma_rows:]])

            # Computes the moving averages:
            moving_average_arrays = []
            for moving_average_index in range(num_moving_averages):
                alpha = moving_average_coefficients[moving_average_index]

                # Extracts the rows associated to the current moving average:
                current_moving_average_column_indices = \
                    (time_step_indices[0] == num_moving_averages - moving_average_index + max(time_lags))
                moving_average_rows = vector_component_indices[0, current_moving_average_column_indices]
                num_rows = len(moving_average_rows)

                moving_average = np.zeros((num_samples, num_rows))
                moving_average[1] = alpha * data[moving_average_rows, 0]
                for time_index in range(2, num_samples):
                    moving_average[time_index] = (alpha * data[moving_average_rows, time_index - 1] +
                                                  (1 - alpha) * moving_average[time_index - 1])

                moving_average_arrays.append(moving_average[max(time_lags):])

            # Array containing the moving averages:
            moving_average_arrays = np.concatenate(moving_average_arrays, axis=1)

            # Full prediction array:
            prediction_array = np.concatenate((moving_average_arrays, data_vector_arrays), axis=1)
        else:
            prediction_array = data[vector_component_indices, time_step_indices]

        # Ground-truth data values:
        component_data = data[comp, max(time_lags):num_samples, np.newaxis]

        # Estimates the MMSE prediction coefficients:
        scaling = np.dot(prediction_array.T, prediction_array)

        # Calculate least squares solution:
        if np.linalg.matrix_rank(scaling) == scaling.shape[0]:
            prediction_coefficients = np.linalg.solve(scaling, np.dot(prediction_array.T, component_data))
        else:
            # If the scaling matrix is not full rank, uses the lstsq function from numpy.linalg:
            prediction_coefficients = np.linalg.lstsq(a=prediction_array, b=component_data, rcond=None)[0]

        # Append new prediction coefficients to the 1-D array:
        prediction_coefficients_array = np.concatenate((prediction_coefficients_array,
                                                        np.squeeze(prediction_coefficients, axis=1)))

        # Calculates linear prediction using the new prediction coefficients:
        prediction[comp] = np.squeeze(np.dot(prediction_array, prediction_coefficients))

    # Compute the residuals of the linear prediction:
    residuals = data[:, max(time_lags):num_samples] - prediction

    return prediction_coefficients_array, residuals


def least_squares_loss(prediction_coefficients, data):
    """Uses the module jax.numpy to compute the mean-squared error (MSE) between the ground-truth data and the linear
    prediction, using the input "prediction_coefficients." This function is used by the lasso() function for the least
    squares calculation of an autoregressive (AR) model.

    Args:
        prediction_coefficients (ndarray): jax.numpy 1-D array containing the prediction coefficients for the AR model.
        data (tuple): the tuple (prediction_array, ground_truth):

            - prediction_array (ndarray):
                jax.numpy 2-D array of shape (number of time steps, prediction window size) containing the prediction
                window data values.

            - ground_truth (ndarray):
                jax.numpy 2-D array of shape (number of time steps, 1) containing the ground-truth data values.

    Returns:
        **residuals** (*float*) -- the MSE between the linear prediction and the ground-truth data values.
    """
    prediction_array, ground_truth = data
    prediction = jnp.dot(prediction_array, prediction_coefficients)
    residuals = jnp.mean((ground_truth - prediction) ** 2)
    return residuals


def lasso(tuning, data, num_lags=1, tol=0.01, predicted_comps=None, num_batches=1):
    """Implements LASSO to find the optimal indices of (non-zero) prediction coefficients for a vector AR model. The
    LASSO computation is done by the class ProximalGradient from the jaxopt package. This allows the computation to be
    run on a GPU for faster convergence.

    Args:
        tuning (float): the tuning parameter used in the input for ProximalGradient.run().
        data (ndarray): numpy 2-D array of shape (number of vectors, vector dimensionality) containing the ground-truth
            data values.
        num_lags (int, optional): [Default=1] the number of time lags to use for the model.
        tol (float, optional): [Default=0.01] the tolerance to use for ProximalGradient convergence.
        predicted_comps (ndarray, optional): [Default=None] numpy 1-D array containing the indices of the vector
            components for which to calculate prediction coefficients.

            - If set to None, all vector components are included in the calculation.

        num_batches (int, optional): [Default=1] the number of data batches to use for LASSO optimization.

    Returns:
        **nonzero_indices** (*ndarray*) -- 2-D boolean numpy array of shape (data vector dimensionality, data vector
        dimensionality * num_lags) indicating which prediction coefficients to use in the regression calculation.
    """
    assert (tuning > 0)
    assert (num_lags > 0)
    assert (tol > 0)
    assert (len(data.shape) == 2)

    from jaxopt import ProximalGradient
    from jaxopt.prox import prox_lasso

    data_copy = jnp.array(data)

    vector_dimensionality = data_copy.shape[0]
    num_vectors = data_copy.shape[1]
    batch_size = (num_vectors - num_lags) // num_batches

    if predicted_comps is None:
        predicted_comps = jnp.arange(vector_dimensionality)
    else:
        assert (len(predicted_comps.shape) == 1)
        assert (predicted_comps.dtype == int)
        assert ((0 <= predicted_comps).all() and (predicted_comps < data.shape[1]).all())

    # Creates a 2-D array to store the LASSO solution for prediction coefficients associated to each vector component:
    prediction_coefficients = jnp.zeros((vector_dimensionality, vector_dimensionality * num_lags))

    # Normalizes the tuning parameter to find more accurate values:
    tuning_normalized = tuning / (vector_dimensionality * num_lags)

    # Sets up index arrays for computing the LASSO solution:
    times = jnp.arange(num_vectors - num_lags)[:, jnp.newaxis]
    offsets = jnp.arange(num_lags - 1, -1, -1)
    time_indices = jnp.tile(times, (1, num_lags * vector_dimensionality)) + \
                   jnp.tile(jnp.repeat(offsets, vector_dimensionality), (num_vectors - num_lags, 1))
    prediction_coefficient_indices = jnp.tile(jnp.arange(vector_dimensionality), (num_vectors - num_lags, num_lags))

    # Array of previous data values used in the LASSO calculation:
    prediction_array = data_copy[prediction_coefficient_indices, time_indices]

    # Calculating prediction coefficients independently for each vector component:
    pg = ProximalGradient(fun=least_squares_loss, prox=prox_lasso, tol=tol, jit=True)
    for comp in predicted_comps:
        # Ground-truth data values:
        component_data = data_copy[comp, num_lags:num_vectors, jnp.newaxis]

        # Running LASSO:
        # Initial coefficients:
        coefficients = jnp.zeros(num_lags * vector_dimensionality)
        for batch_idx in range(num_batches):
            batch_data = component_data[batch_idx * batch_size: (batch_idx + 1) * batch_size]
            coefficients_object = pg.run(coefficients, hyperparams_prox=tuning_normalized,
                                         data=(prediction_array, batch_data))
            coefficients = coefficients_object.params
        prediction_coefficients = prediction_coefficients.at[comp].set(coefficients)

    # Determines the indices corresponding to non-zero prediction coefficients, as returned by LASSO:
    nonzero_indices = (np.array(prediction_coefficients) != 0)

    return nonzero_indices
