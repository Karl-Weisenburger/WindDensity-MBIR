import numpy as np
import matplotlib.pyplot as plt
import aomodel._utils as utils


class PCA:
    """Uses Principal Component Analysis (PCA) to analyze an input 3-D data set containing a time-series of 2-D arrays
    (which are interpreted as images). Models the data as temporally independent and computes the principal components
    of the 2-D data.

    Args:
        data (ndarray): numpy 3-D array of data values to analyze, with shape (number of images, image height, image
            width).
        mask (ndarray, optional): [Default=None] numpy 2-D boolean array of shape (image height, image width)
            indicating which 2-D data indices correspond to valid pixel values.

            - If set to None, the function infers the mask based on which data values are "nan."

        sample (int, optional): [Default=None] sampling period for the input data values

            - If set to None, the data is not sub-sampled (all time-steps are used).
    """

    def __init__(self, data, mask=None, sample=None):
        # Makes sure that the input data has three axes:
        assert (len(data.shape) == 3)

        # Saves the mask as an instance variable. If a mask is not provided (as input), includes all data values which
        # are not nan:
        if mask is None:
            # Sets the mask to be the intersection of valid data values for all images in the time-series:
            mask = (np.average(1 - np.uint8(np.isnan(data)), axis=0) == 1)
        else:
            # Ensures that this is a valid mask for the data:
            assert (mask.dtype == bool)
            assert (mask.shape == (data.shape[1], data.shape[2]))
            assert (not np.isnan(utils.img_to_vec(image_data=data, mask=mask)).any())
        self.mask = mask

        # Sub-samples the data if prompted to:
        if sample is not None:
            assert (sample > 0)
            data = data[::sample]

        # Finds the number of images (or frames) we have in this data set:
        self.num_images = data.shape[0]

        # Flattens the data values at each image index (in raster order) and takes the transpose. The new shape of the
        # array is (number of pixels, self.num_images):
        data_matrix = utils.img_to_vec(image_data=data, mask=mask).T

        # Computes the mean vector of this data:
        self.mean_vector = np.average(data_matrix, axis=1)

        # Removes the mean from the data:
        data_mean_removed = data_matrix - self.mean_vector[:, np.newaxis]

        # Normalizes the data (i.e., divides the time-series of each pixel by its standard deviation)
        self.standard_deviation_vector = np.sqrt(np.sum(data_mean_removed ** 2, axis=1) / self.num_images)
        data_normalized = data_mean_removed / self.standard_deviation_vector[:, np.newaxis]

        # Computes the principal components and singular values using Singular Value Decomposition (SVD):
        print("\nPrincipal Component Analysis (PCA) of Input Data\n"
              "================================================\n"
              "Computing Singular Value Decomposition (SVD)...")
        principal_components, singular_values = compute_principal_components(data_normalized)
        print("SVD Computation Completed.\n")

        # Saves the singular values of the covariance estimate. This gives the variance for each principal component
        # (ordered from largest to smallest). This variable has length equal to the number of pixels:
        self.singular_values = singular_values

        # Saves the principal components (i.e., the eigenvectors of the covariance estimate). This matrix is positioned
        # so that self.principal_components[i] gives the i-th principal component. It has shape (number of pixels,
        # number of pixels):
        self.principal_components = principal_components.T

        # Finds the principal coefficients for each data image. This variable has shape (number of pixels, number of
        # images):
        self.data_coefficients = np.dot(self.principal_components, data_normalized)

    def generate_samples(self, num_samples=None):
        """Samples from the 2-D distribution of the input data set using the generative PCA algorithm. The output images
        have the same distribution as the input 2-D data set.

        Args:
            num_samples (int, optional): [Default=None] the number of output images (i.e. samples) to produce.

                -   If set to None, then the method uses the value stored in 'self.num_images' (i.e. the number of data
                    samples we are given).

        Returns:
            **output_samples** (*ndarray*) -- 3-D numpy array of shape (num_samples, image height, image width)
            containing the generated data samples.
        """
        if num_samples is None:
            num_samples = self.num_images

        # Use the generative PCA algorithm to generate samples from the (normalized) spatial distribution of our data:
        samples_normalized = generative_pca_algorithm(num_samples=num_samples,
                                                      principal_components=self.principal_components.T,
                                                      singular_values=self.singular_values)

        # Multiplies by the standard deviation estimators and adds the mean vector. Flattens the samples to a 2-D array
        # of shape (num_samples, number of pixels):
        flattened_samples = (samples_normalized * self.standard_deviation_vector[:, np.newaxis] + self.mean_vector[:, np.newaxis]).T

        # Reshapes the array so that it has shape (num_samples, image height, image width):
        output_samples = utils.vec_to_img(data_vec=flattened_samples, mask=self.mask)

        return output_samples

    def find_top_coefficients(self, percent_variance):
        """Finds the number of vector components of each coefficient vector (i.e., each column of
        self.data_coefficients) containing the given percentage of the total variance.

        Args:
            percent_variance (float): percentage of the total variance to look for.

        Returns:
            **num_vector_components** (*int*) -- number of (top) vector components containing at least the given percent
            variance.
        """
        assert ((percent_variance > 0) and (percent_variance <= 1.0))
        vector_dimensionality = len(self.singular_values)

        # If we need 100% of the variance, we use all vector components:
        if percent_variance == 1.0:
            num_vector_components = vector_dimensionality
        else:
            # If less than 100% of the variance is desired, search for the fewest number of vector components containing
            # this amount of variance:
            total_variance = np.sum(self.singular_values)

            # Vector containing the cumulative percent variance of the first i vector components:
            cumulative_percent_variance = np.zeros(vector_dimensionality)
            for component_index in range(vector_dimensionality):
                cumulative_variance = np.sum(self.singular_values[:component_index])
                cumulative_percent_variance[component_index] = cumulative_variance / total_variance

            # Fewest components needed to capture the given percent variance:
            num_vector_components = np.argwhere(cumulative_percent_variance >= percent_variance)[0, 0]

        return num_vector_components


def compute_principal_components(data):
    """Computes the principal components and the associated standard deviation values of an input array containing
    samples of a (zero-mean) multivariate Gaussian distribution N(0, R). Uses Singular Value Decomposition (SVD) of the
    covariance matrix.

    Args:
        data (ndarray): numpy 2-D array of shape (vector dimensionality, number of samples) containing samples
            of the multivariable Gaussian distribution.

    Returns:
        - **principal_components** (*ndarray*) -- 2-D numpy array of shape (vector dimensionality, vector
          dimensionality) containing the principal components.
        - **singular_values** (*ndarray*) -- 2-D array of shape (vector dimensionality, vector dimensionality)
          containing the singular values (i.e., the standard deviation of each principal component).
    """
    assert (len(data.shape) == 2)

    # Estimates the covariance matrix of the distribution:
    covariance_estimate = np.dot(data, data.T) / data.shape[1]

    # Compute SVD to find principal components and singular values:
    principal_components, singular_values = np.linalg.svd(covariance_estimate)[:2]

    # modulation_matrix = np.dot(principal_components, np.diag(np.sqrt(std_estimates)))
    return principal_components, singular_values


def generative_pca_algorithm(num_samples, principal_components, singular_values):
    """Generates samples from a zero-mean multivariate Gaussian distribution N(0, R). This uses the PCA generative
    algorithm, which generates white noise vectors and then (1) multiplies them by matrices to set the covariance matrix
    and (2) adds the mean vectors.

    Args:
        num_samples (int): number of samples to generate.
        principal_components (ndarray): numpy 2-D array of shape (random vector dimensionality, random vector
            dimensionality) containing the principal components of the distribution (eigenvectors of the covariance
            matrix R) as columns.
        singular_values (ndarray): numpy 1-D array of shape (random vector dimensionality) containing the singular
            values of the covariance matrix R. The values are ordered so that the i-th entry gives the variance
            estimate of the i-th column of the previous input.

    Returns:
        **samples** (*ndarray*) -- 2-D array of shape (random vector dimensionality, num_samples) whose columns contain
        the samples from the desired distribution N(0, R).
    """
    assert (num_samples > 0)
    assert (len(principal_components.shape) == 2)
    assert (len(singular_values.shape) == 1)

    # Gaussian i.i.d random variables (mean 0, variance 1):
    white_noise = np.random.normal(size=(principal_components.shape[0], num_samples))

    # The diagram matrix containing the standard deviation of each component - i.e., the square root of singular values:
    standard_deviation_matrix = np.diag(np.sqrt(singular_values))

    # Samples from the distribution
    samples = np.dot(principal_components, np.dot(standard_deviation_matrix, white_noise))

    return samples


def spatial_psd(data_values, block_length=None, mask=None, sampling_frequency=None, remove_mean=True):
    """Estimates the spatial Power Spectral Density (PSD) of the input time-series of 2-D data. The function
    estimates the 2-D spatial PSD at each time-step and then averages over the entire time-series. Each 2-D PSD is
    estimated using Welch's method (extended to two-dimensional power spectra) with FFTs.

    Args:
        data_values (ndarray): numpy 3-D array of shape (number of images, image height, image width) containing the
            data from which to compute the spatial PSD.
        block_length (int, optional): [Default=None] the length of each square block to average PSD estimates over.

            - If set to None, a single block is used, with the largest possible length.

        mask (ndarray, optional): [Default=None] numpy 2-D boolean array of shape (image height, image width)
            indicating which 2-D data indices correspond to valid pixel values.

            - If set to None, the function infers the mask based on which data values are "nan."

        sampling_frequency (float, optional): [Default=None] the (spatial) sampling frequency of the data set. If
            included, the frequencies are updated to units of cycles per spatial unit and the PSD values are updated to
            units of energy per spatial unit.

            - If set to None, the frequencies are in units of cycles per sample and PSD values are in units of energy
              per sample.

        remove_mean (bool, optional): [Default=True] choice of removing the spatial mean of the data before computing
            the spatial PSD. It is recommended to set this to True in most cases. If set to False, the lowest
            frequencies may be offset.

    Returns:
        - **frequencies** (*ndarray*) -- A numpy 1-D array containing the frequency bins of the PSD calculation.
        - **psd_estimate** (*ndarray*) -- A numpy 2-D array containing the spatial PSD estimates for each frequency bin.
    """
    # Makes sure that the input data has three axes:
    assert (len(data_values.shape) == 3)

    if mask is None:
        # Sets the mask to be the intersection of valid data values for all images in the time-series:
        mask = (np.average(1 - np.uint8(np.isnan(data_values)), axis=0) == 1)
    else:
        # Ensures that this is a valid mask for the data
        assert (mask.dtype == bool)
        assert (mask.shape == (data_values.shape[1], data_values.shape[2]))
        assert (not np.isnan(utils.img_to_vec(image_data=data_values, mask=mask)).any())

    # Finds the center of the images:
    center = [data_values.shape[1] // 2, data_values.shape[2] // 2]

    # Finds the largest square from the center of the images containing valid data values:
    dist_from_center = min(center) - 1
    while (mask[center[0] - dist_from_center:center[0] + (dist_from_center + 1),
           center[1] - dist_from_center:center[1] + (dist_from_center + 1)] == 0).any():
        dist_from_center -= 1

    # Length of the largest such square:
    square_length = 2 * dist_from_center + 1
    if block_length is None:
        block_length = square_length
    else:
        assert (0 < block_length <= square_length)
    print("\nSpatial Power Spectral Density (PSD) Calculation\n"
          "================================================\n"
          "The Length of the Square Window is {} Pixels.\n".format(square_length))

    # Extracts the largest possible square from the data:
    data_square = data_values[:, center[0] - dist_from_center:center[0] + dist_from_center + 1,
                              center[1] - dist_from_center:center[1] + dist_from_center + 1]

    # Finds the number of (spatial) blocks to use:
    num_blocks = square_length // block_length

    # Creates a 2-D Hamming window and computes the Welch scaling factor:
    hamming_window = np.outer(np.hamming(block_length), np.hamming(block_length))[np.newaxis, :]
    welch_scaling_factor = np.sum(hamming_window ** 2)

    # Frequencies along the last axis:
    frequencies = np.fft.fftshift(np.fft.fftfreq(n=block_length))

    # Removes the spatial mean from each time-step:
    if remove_mean:
        spatial_mean = np.average(data_square, axis=(1, 2))
        data_square = data_square - spatial_mean[:, np.newaxis, np.newaxis]

    # Averages the PSD estimates over each block:
    psd_estimates = np.zeros((data_values.shape[0], len(frequencies), len(frequencies)))
    for block_idx_1 in range(num_blocks):
        for block_idx_2 in range(num_blocks):
            # Extracts the data for the current block:
            block_data = data_square[:, block_length * block_idx_1:block_length * (block_idx_1 + 1),
                                     block_length * block_idx_2:block_length * (block_idx_2 + 1)]

            # If mean-removal is not selected, takes the FFT of the windowed block data (with its mean):
            windowed_values = hamming_window * block_data

            # Calculates the 2-D FFT of the current block:
            block_dft = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(windowed_values, axes=(1, 2)), axes=(1, 2)),
                                        axes=(1, 2))

            # Calculates the energy spectrum of the current block:
            block_energy_spectrum = np.abs(block_dft) ** 2

            # Calculates the power spectrum of the current block:
            psd_estimates += block_energy_spectrum / welch_scaling_factor

    # Averages over all images in the data set to get a more accurate PSD:
    psd_estimate = np.average(psd_estimates, axis=0) / (num_blocks ** 2)

    # If a sampling frequency is included, incorporates units:
    if sampling_frequency is not None:
        frequencies = sampling_frequency * frequencies
        psd_estimate = psd_estimate / (sampling_frequency ** 2)

    return frequencies, psd_estimate
