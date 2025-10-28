import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani
import aomodel
import scipy


def surface_plot(data, x_axis, y_axis, title='', x_label='', y_label=''):
    """Creates a 3-D surface plot 2-D data using matplotlib.pyplot.

    Args:
        data (ndarray): numpy 2-D array containing the data to plot.
        x_axis (ndarray): numpy 1-D array containing data to plot along the x-axis (must have the same length as the
            second axis of data).
        y_axis (ndarray): numpy 1-D array containing data to plot along the y-axis (must have the same length as the
            first axis of data).
        title (str, optional): [Default=''] title of the plot.
        x_label (str, optional): [Default=''] label to put on the x-axis of the plot.
        y_label (str, optional): [Default='] label to put on the y-axis of the plot.

    Returns:
        matplotlib.figure: The matplotlib.pyplot figure object containing the surface plot.
    """
    assert (len(data.shape) == 2)
    assert (len(x_axis) == data.shape[1])
    assert (len(y_axis) == data.shape[0])

    # Creates the meshgrid:
    x, y = np.meshgrid(x_axis, y_axis)

    fig = plt.figure()

    # Creates a 3D color plot for the data:
    ax = fig.add_subplot(111, projection='3d')

    # Plots the surface:
    surf = ax.plot_surface(x, y, data, cmap=plt.cm.coolwarm)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    fig.colorbar(surf, shrink=0.5, aspect=5)

    return fig


def image_figure(data, title='', xlabel='', ylabel='', xticks=None, xticklabels=None, yticks=None, yticklabels=None,
                 data2=None, mask=None, title2=''):
    """Creates an image from 2-D data using the imshow() method from matplotlib.pyplot and includes a colorbar.

    Args:
        data (ndarray): numpy 2-D array of shape (image height, image width) containing the data to create images of.
        title (str, optional): [Default=''] title to include above the image.
        data2 (ndarray, optional): [Default=None] numpy 2-D array with the same shape as "data" containing additional
            data to create an image of. If included, the two images are shown side-by-side.

            - If set to None, the figure will only show a single image.

        xlabel (str, optional): [Default=''] matplotlib.pyplot parameter "xlabel"
        ylabel (str, optional): [Default=''] matplotlib.pyplot parameter "ylabel"
        xticks (str, optional): [Default=None] matplotlib.pyplot parameter "xticks"
        xticklabels (str, optional): [Default=None] matplotlib.pyplot parameter "xticklabels"
        yticks (str, optional): [Default=None] matplotlib.pyplot parameter "yticks"
        yticklabels (str, optional): [Default=None] matplotlib.pyplot parameter "yticklabels"

        mask (ndarray, optional): [Default=None] numpy 2-D boolean array of shape (image height, image width) indicating
            which 2-D data indices correspond to valid pixel values. Only used if "data2" is included as an input
            argument.

            - If set to None, the function infers the mask based on which data values are "nan."

        title2 (str, optional): [Default=''] title to include above the image created from "data2." Only used by the
            function if "data2" is not None.

    Returns:
        matplotlib.figure: The matplotlib.pyplot figure object containing the image (or pair of images).
    """
    assert (len(data.shape) == 2)

    # Used if there are two images to create side-by-side:
    if data2 is not None:
        assert (data.shape == data2.shape)
        if mask is None:
            # Intersection of valid data values for both data and data2:
            mask = ((1 - np.uint8(np.isnan(data))) * (1 - np.uint8(np.isnan(data2)))).astype(bool)
        else:
            # Ensures the given mask is valid:
            assert (mask.dtype == bool)
            assert (mask.shape == (data.shape[0], data.shape[1]))
            assert (not np.isnan(aomodel.utils.img_to_vec(image_data=data, mask=mask)).any())
            assert (not np.isnan(aomodel.utils.img_to_vec(image_data=data2, mask=mask)).any())

        fig, axs = plt.subplots(nrows=1, ncols=2)

        # Finds the minimum and maximum values across the two images:
        datavals = aomodel.utils.img_to_vec(image_data=data, mask=mask)
        data2vals = aomodel.utils.img_to_vec(image_data=data2, mask=mask)
        vmin = np.array([datavals.min(), data2vals.min()]).min()
        vmax = np.array([datavals.max(), data2vals.max()]).max()

        # Creates both images according to the same quantization:
        im = axs[0].imshow(data, interpolation='none', vmin=vmin, vmax=vmax)
        axs[0].set_title(title)
        axs[0].set_xlabel(xlabel)
        axs[0].set_ylabel(ylabel)
        if xticks is not None:
            axs[0].set_xticks(xticks)
        if xticklabels is not None:
            axs[0].set_xticklabels(xticklabels)
        if yticks is not None:
            axs[0].set_yticks(yticks)
        if yticklabels is not None:
            axs[0].set_yticklabels(yticklabels)
        im = axs[1].imshow(data2, interpolation='none', vmin=vmin, vmax=vmax)
        axs[1].set_title(title2)
        axs[1].set_xlabel(xlabel)
        axs[1].set_ylabel(ylabel)
        if xticks is not None:
            axs[1].set_xticks(xticks)
        if xticklabels is not None:
            axs[1].set_xticklabels(xticklabels)
        if yticks is not None:
            axs[1].set_yticks(yticks)
        if yticklabels is not None:
            axs[1].set_yticklabels([])
        fig.colorbar(im, ax=axs, fraction=0.046, pad=0.04, shrink=0.55)
    # Used if there is only one image to display:
    else:
        fig = plt.figure()
        ax = plt.axes()
        im = ax.imshow(data, interpolation='none')
        plt.title(title, loc='center')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if xticks is not None:
            ax.set_xticks(xticks)
        if xticklabels is not None:
            ax.set_xticklabels(xticklabels)
        if yticks is not None:
            ax.set_yticks(yticks)
        if yticklabels is not None:
            ax.set_yticklabels(yticklabels)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, shrink=1)
        plt.tight_layout()

    return fig


def create_video(data, title='', mask=None, data2=None, title2=''):
    """Creates a video from a sequence of images using matplotlib.pyplot and matplotlib.animation. Shows each image on
    the same scale and includes a colorbar. Each frame of the video can either have a single image or two images
    side-by-side.

    Args:
        data (ndarray): numpy 3-D array of shape (number of frames, image height, image width) containing the data
            values to create a video of.
        title (str, optional): [Default=''] title to place above the video.
        mask (ndarray, optional): [Default=None] numpy 2-D boolean array of shape (image height, image width) indicating
            which 2-D data indices correspond to valid pixel values.

            - If set to None, the function infers the mask based on which data values are "nan."

        data2 (ndarray, optional): [Default=None] numpy 3-D array with the same shape as "data" and which contains
            additional data to create a video from. If included, each frame of the video will include two images
            side-by-side.

            - If set to None, only one image is included in each frame.

        title2 (str, optional): [Default=''] title above the images from "data2." Only used by the function if "data2"
            is included as an input argument.

    Returns:
        matplotlib.animation.ArtistAnimation: The ArtistAnimation figure containing the video.
    """
    assert (len(data.shape) == 3)

    if mask is None:
        # Sets the mask to be the intersection of valid data values for all images in the sequence:
        mask = (np.average(1 - np.uint8(np.isnan(data)), axis=0) == 1)
    else:
        # Checks the mask on all images in the data set:
        assert (mask.dtype == bool)
        assert (mask.shape == (data.shape[1], data.shape[2]))
        assert (not np.isnan(aomodel.utils.img_to_vec(image_data=data, mask=mask)).any())

    num_frames = data.shape[0]
    video_frames = []

    if data2 is not None:
        # Ensures that the mask is valid for data2:
        assert (not np.isnan(aomodel.utils.img_to_vec(image_data=data2, mask=mask)).any())

        fig, axs = plt.subplots(1, 2)

        # Finds the minimum and maximum values of the data for the purpose of colorbar creation:
        datavals = aomodel.utils.img_to_vec(image_data=data, mask=mask)
        data2vals = aomodel.utils.img_to_vec(image_data=data2, mask=mask)
        vmin = np.array([datavals.min(), data2vals.min()]).min()
        vmax = np.array([datavals.max(), data2vals.max()]).max()

        # Adding each frame to the list:
        for idx in range(num_frames):
            im0 = axs[0].imshow(data[idx], interpolation='none', vmin=vmin, vmax=vmax, animated=True)
            im = axs[1].imshow(data2[idx], interpolation='none', vmin=vmin, vmax=vmax, animated=True)
            video_frames.append([im0, axs[0].set_title(title, loc='center'), im,
                                 axs[1].set_title(title2, loc='center')])

        # Add the colorbar:
        fig.colorbar(im, ax=axs, fraction=0.046, pad=0.04, shrink=0.55)

    else:
        fig = plt.figure()
        axs = plt.axes()
        datavals = aomodel.utils.img_to_vec(image_data=data, mask=mask)
        vmin = datavals.min()
        vmax = datavals.max()

        # Adding each frame to the list:
        for idx in range(num_frames):
            im = plt.imshow(data[idx], interpolation='none', vmin=vmin, vmax=vmax, animated=True)
            video_frames.append([im, plt.title(title, loc='center')])

        fig.colorbar(im, ax=axs, fraction=0.046, pad=0.04, shrink=1)

    video = ani.ArtistAnimation(fig, video_frames)

    return video


def structure_function_image(structure_function_inputs, structure_function, interpolation_scale=1,
                             interpolate_using_zero=True, x_label='', y_label='', colorbar_label='', image_title='',
                             colorbar_scale=1, savefile=None, show=True):
    """Uses matplotlib.pyplot to create an image of the 2-D structure function. This function plots the (relative
    separation, angle) pairs as polar coordiantes in the image and wraps around to 2*pi. It uses the inputs
    "x_pixel_length," "y_pixel_length," "num_x_pixels," and "num_y_pixels" to determine where the centers of each pixel
    occur. Then, the function interpolates at the center of each pixel using bi-linear interpolation.

    Args:
        structure_function_inputs (ndarray): numpy 2-D array of shape (number of inputs, 2) containing the pairs
            (relative separation, angle) used as inputs to the structure function
        structure_function (ndarray): numpy 1-D array of shape (number of inputs,) containing the structure function
            values associated to each input pair
        interpolation_scale (int, optional): [Default=1] the (integer) value used to scale the number of pixels along
            each axis
        interpolate_using_zero (bool, optional): [Default=True] whether to use a value of zero at the origin to
            interpolate at the center of each pixel
        x_label (str, optional): [Default=''] the label to display along the x-axis
        y_label (str, optional): [Default=''] the label to display along the y-axis
        colorbar_label (str, optional): [Default=''] the label to put next to the color-bar in the image
        image_title (str, optional): [Default=''] the title of the image
        colorbar_scale (float, optional): [Default=1] the scale of the colorbar in relation to the image
        savefile (str, optional): [Default=None] the filename to save the figure to

            - If set to None, the figure is not saved.

        show (bool, optional): [Default=True] whether to display the image

    Returns:
        matplotlib.figure: The matplotlib.pyplot figure object containing the image.
    """

    # Extend angles from the range [0, pi) to [0, 2*pi):
    additive_pi = np.tile(np.array([0, np.pi]), (structure_function_inputs.shape[0], 1))
    additional_structure_function_inputs = structure_function_inputs + additive_pi
    extended_structure_function_inputs = np.zeros((2 * structure_function_inputs.shape[0], 2))
    extended_structure_function = np.zeros(2 * structure_function.shape[0])
    extended_structure_function_inputs[:structure_function_inputs.shape[0]] = structure_function_inputs
    extended_structure_function_inputs[structure_function_inputs.shape[0]:] = additional_structure_function_inputs
    extended_structure_function[:structure_function_inputs.shape[0]] = structure_function
    extended_structure_function[structure_function_inputs.shape[0]:] = structure_function

    # Convert from polar coordinates (r, theta) to rectangular coordinates (x, y):
    num_points = extended_structure_function_inputs.shape[0]
    x = np.zeros(num_points)
    y = np.zeros(num_points)
    for point_index in range(num_points):
        r = extended_structure_function_inputs[point_index, 0]
        theta = extended_structure_function_inputs[point_index, 1]
        x[point_index], y[point_index] = (r * np.cos(theta), r * np.sin(theta))

    # Sets the pixel lengths along the x- and y-axes:
    x_pixel_length, y_pixel_length = 1 / interpolation_scale, 1 / interpolation_scale

    # Sets the number of numbers of pixels along each axis:
    num_x_pixels = interpolation_scale * int(x.max()-x.min()+1)
    num_y_pixels = interpolation_scale * int(y.max()-y.min()+1)

    # Create pixel centers
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    x_values = np.linspace(x_min + 0.5*x_pixel_length, x_max - 0.5*x_pixel_length, num_x_pixels)
    y_values = np.linspace(y_min + 0.5*y_pixel_length, y_max - 0.5*y_pixel_length, num_y_pixels)

    # If "interpolate_using_zero" is True, add the origin and a value of zero to the list of structure function values:
    if interpolate_using_zero:
        x = np.insert(x, 0, 0)
        y = np.insert(y, 0, 0)
        extended_structure_function = np.insert(extended_structure_function, 0, 0)

    # Create the structure function image by interpolating (bi-linear) at the center of each pixel:
    X, Y = np.meshgrid(x_values, y_values)
    structure_function_image = scipy.interpolate.griddata((x, y), extended_structure_function, (X, Y),
                                                          method='linear')

    # Removes the pixels closest to zero from the image (by setting the values to "nan"):
    lowest_distance = np.min(np.sqrt(X ** 2 + Y ** 2))
    close_to_zero_indices = np.argwhere(np.abs(np.sqrt(X ** 2 + Y ** 2) - lowest_distance) < 1e-10)
    for i in range(close_to_zero_indices.shape[0]):
        zero_index = (close_to_zero_indices[i][0], close_to_zero_indices[i][1])
        structure_function_image[zero_index] = np.nan

    # Create image figure:
    fig = plt.figure()
    ax = plt.axes()
    im = ax.imshow(structure_function_image, interpolation='none', aspect='equal', origin='lower',
                   extent=[x_values[0] - 0.5*x_pixel_length, x_values[-1] + 0.5*x_pixel_length,
                           y_values[0] - 0.5*y_pixel_length, y_values[-1] + 0.5*y_pixel_length])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, shrink=colorbar_scale, label=colorbar_label)
    ax.set_xlabel(x_label, fontsize=13)
    ax.set_ylabel(y_label, fontsize=13)
    plt.title(image_title, loc='center', fontsize=15)
    plt.tight_layout()

    if savefile is not None:
        plt.savefig(savefile)

    if show:
        plt.show()

    return fig


def plot_psd(frequencies, psd_values, psd_values_2=None, x_label='', y_label='', title='', log_scale=False,
             label1='Input Data', label2='Synthetic Data', savefile=None, show=True):
    """Uses matplotlib.pyplot to plot the temporal power spectral density (PSD) of measured and/or synthetic data. This
    function plot a single array of values or two arrays in the same graph.

    Args:
        frequencies (ndarray): numpy 1-D array containing the frequency bins of the PSD values.
        psd_values (ndarray): numpy 1-D array containing the PSD values at each frequency bin.
        psd_values_2 (ndarray): numpy 1-D array containing a second set of PSD values at each frequency bin.
        x_label (str, optional): [Default=''] the label to display along the x-axis
        y_label (str, optional): [Default=''] the label to display along the y-axis
        title (str, optional): [Default=''] the title of the plot
        log_scale (bool, optional): [Default=False] whether to use a logarithmic scale on the x-axis
        label1 (str, optional): [Default=''] the label of the psd_values plot
        label2 (str, optional): [Default=''] the label of the psd_values_2 plot
        savefile (str, optional): [Default=None] the filename to save the figure to

            - If set to None, the figure is not saved.

        show (bool, optional): [Default=True] whether to display the image

    Returns:
        matplotlib.figure: The matplotlib.pyplot figure object containing the plot.
    """
    assert (len(frequencies) == len(psd_values))

    fig = plt.figure()

    if psd_values_2 is None:
        # Just include one plot
        plt.plot(frequencies, psd_values, '.')
        plt.plot(frequencies, psd_values, 'r', linewidth=0.5)
    else:
        # Include both plots
        assert (len(psd_values) == len(psd_values_2))
        plt.plot(frequencies, psd_values, '-o', label=label1, markersize=5, markerfacecolor='green',
                 linewidth=0.5)
        plt.plot(frequencies, psd_values_2, '-o', label=label2, markersize=5, markerfacecolor='red',
                 linewidth=0.5)
        plt.legend(fontsize=13)

    if log_scale:
        plt.xscale('log')

    plt.title(title, fontsize=15)
    plt.xlabel(x_label, fontsize=13)
    plt.ylabel(y_label, fontsize=13)
    plt.tight_layout()

    if savefile is not None:
        plt.savefig(savefile)

    if show:
        plt.show()

    return fig