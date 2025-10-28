import numpy as np


def vec_to_img(data_vec, mask):
    """Converts from a row vector to a 2-D array (an "image") in raster order, using a boolean array called a "mask"
    that indicates pixels in the image which should not take a value. The data values of the output array at pixels
    which the mask specifies not to include are set to float("nan"). The function can convert either a vector to a
    single image or a 2-D array (containing a sequence of vectors) to a sequence of images (in a 3-D array).

    Args:
        data_vec (ndarray): numpy 1-D or 2-D array with the flattened image pixel values.

            - If data_vec is 1-D, a single vector is converted to a single image.

            - If data_vec is 2-D, it has shape (number of images, image dimensionality) and contains multiple vectors.

       mask (ndarray): numpy 2-D boolean array of shape (image height, image width) indicating which 2-D data indices
        correspond to valid pixel values. The number of values in the mask that are set to "True" must equal the
        dimensionality of the input data vector(s).

    Returns:
        numpy ndarray: A 2-D or 3-D array containing the image pixel values. If "data_vec" is 1-D, only a single 2-D
        image array of shape (image height, image width) is returned. If "data_vec" is 2-D, a 3-D image array of shape
        (number of images, image height, image width) is returned.
    """
    assert (mask.sum() == data_vec.shape[-1])
    assert (1 <= len(data_vec.shape) <= 2)

    # Determines if we are converting to a single image or a sequence of images:
    if len(data_vec.shape) == 1:  # converting to a single image
        output_shape = [1, mask.shape[0], mask.shape[1]]
    else:  # converting to a sequence of images
        output_shape = [data_vec.shape[0], mask.shape[0], mask.shape[1]]

    # Output image array:
    output_img = np.full(output_shape, float("nan"))

    # Sets the valid indices to the corresponding data value:
    output_img[:, mask] = data_vec

    return output_img.squeeze()


def img_to_vec(image_data, mask):
    """Converts from a 2-D array (an "image") to a row vector in raster order, using a boolean array called a "mask"
    that indicates the pixels in the image which should be included in the vector. The function can convert either a
    single image to a single vector or a 3-D array (containing a sequence of images) to a sequence of rows vectors
    (i.e., to the rows of a 2-D array).

    Args:
        image_data (ndarray): numpy 2-D or 3-D array containing the image pixel values.

            - If image_data is 2-D, a single image is converted to a single vector. In this case, the input must have
                shape (image height, image width).
            - If image_data is 3-D, a sequence of images is converted to a sequence of row vectors. In this case, the
                input must have shape (number of images, image height, image width).

        mask (ndarray): numpy 2-D boolean array of shape (image height, image width) indicating which 2-D data indices
            to include in the output vector(s).

    Returns:
        numpy ndarray: a 1-D or 2-D array with the (flattened) image pixel values. If "image_data" is 2-D, only a single
        1-D array is returned. If "image_data" is 3-D, a 2-D array of shape (number of images, image dimensionality) is
        returned.
    """
    assert (2 <= len(image_data.shape) <= 3)
    # Determines if we are converting a single image or a sequence of images:
    if len(image_data.shape) == 2:  # converting a single image
        assert (image_data.shape == mask.shape)
        output_vec = image_data[mask]
    else:
        assert (image_data.shape[1:] == mask.shape)
        # Converting a sequence of images
        output_vec = image_data[:, mask]

    return output_vec
