import aomodel
import numpy as np
import matplotlib.pyplot as plt
from demo_utils import create_video, plot_psd, structure_function_image

"""
This file demonstrates a demo of the aomodel.ReVAR class, which applies the ReVAR (Re-Whitened Vector Auto-Regression)
algorithm to create a synthetic time-series of single-channel images with the same spatial and temporal statistics as
measured data.
"""

datasets = ['F06', 'F12']

# Sampling frequencies (Hz) for each data set:
sampling_frequencies = {'F06': 100000, 'F12': 130000}

# Number of time lags for each data set:
num_time_lags = 4

# Number of moving averages to use:
num_moving_averages = 2

# Block sizes for the PSD calculations:
psd_block_sizes = {'F06': (596, 298), 'F12': (994, 496)}

# Scale of the color-bar in the structure function images:
colorbar_scales = {'F06': 0.75, 'F12': 1}

for dataset in datasets:
    print("Data Set %s" % dataset + '\n'
          "============\n"
          "============\n")

    # Extracts sampling frequency, prediction window distance, number of lags, PSD time-block size, and post-processing
    # window size for the data set:
    sampling_frequency = sampling_frequencies[dataset]
    psd_block_size = psd_block_sizes[dataset]

    # Reads the data from the given file:
    print("Loading Data Set {} From File...".format(dataset))
    file = np.load('./demo/data/{}_pre_processed.npz'.format(dataset))
    OPD = file['OPD']
    x_coordinates = file['x_coordinates']
    print("Successfully Loaded Data Set {}.\n".format(dataset))

    # Sets the mask to be the intersection of valid (i.e., not "nan") data values for all images in the sequence:
    mask = (np.average(1 - np.uint8(np.isnan(OPD)), axis=0) == 1)

    # Uses 80% of image data for training the model:
    num_training_time_steps = int(0.8 * OPD.shape[0])

    # Uses the remaining 20% for validation:
    num_validation_time_steps = OPD.shape[0] - num_training_time_steps

    # Calculates the temporal PSD for both OPD and deflection angle for the input data:
    frequencies_opd, data_psd_opd = aomodel.temporal_psd(data_values=OPD[num_training_time_steps:],
                                                         time_block_size=psd_block_size[0],
                                                         sampling_frequency=sampling_frequency,
                                                         remove_mean=True)
    frequencies_theta_x, data_psd_theta_x = aomodel.slopes_psd(data_values=OPD[num_training_time_steps:],
                                                               time_block_size=psd_block_size[1],
                                                               axis=2,
                                                               sampling_frequency=sampling_frequency,
                                                               remove_mean=True)

    # Finds the spatial structure function values of the OPD data:
    data_structure_function = aomodel.anisotropic_structure_function(data=OPD[num_training_time_steps:], mask=mask)
    structure_function_inputs, data_structure_function_values = data_structure_function

    # Creates an instance of the class ReVAR and loads pre-trained model:
    model = aomodel.ReVAR(time_lags=num_time_lags,
                          mask=mask,
                          num_moving_averages=num_moving_averages,
                          load_file='./demo/pre_trained_models/{}_pre_trained_model.npz'.format(dataset))

    # Generates synthetic data using this model:
    synthetic_data = model.run(num_images=num_validation_time_steps)

    # Calculates the PSD of both OPD and deflection angle for the (synthetic) data:
    synthetic_psd_opd = aomodel.temporal_psd(data_values=synthetic_data,
                                             time_block_size=psd_block_size[0],
                                             sampling_frequency=sampling_frequency,
                                             remove_mean=True)[1]
    synthetic_psd_theta_x = aomodel.slopes_psd(data_values=synthetic_data,
                                               time_block_size=psd_block_size[1],
                                               axis=2,
                                               sampling_frequency=sampling_frequency,
                                               remove_mean=True)[1]

    # Finds the spatial structure function values of the synthetic data:
    synthetic_structure_function_values = aomodel.anisotropic_structure_function(data=synthetic_data, mask=mask)[1]

    # Creates a video of the (first 100 frames of the) data alongside synthetic data:
    vid = create_video(data=OPD[num_training_time_steps:num_training_time_steps + 100],
                       title='Data Set {}: Input Data'.format(dataset),
                       mask=mask,
                       data2=synthetic_data[:100],
                       title2='Synthetic Data')
    vid.save('./demo/output/data_set_{}_video.gif'.format(dataset))

    # Plots the temporal PSD of both the OPD values and the synthetic data:
    plot_psd(frequencies=frequencies_opd,
             psd_values=data_psd_opd,
             psd_values_2=synthetic_psd_opd,
             x_label='Frequency $f$ (Hz)',
             y_label='PSD Value $S_{OPD}(f)$',
             title=f'Data Set {dataset}: Temporal PSD of $OPD$',
             label1='Measured Data',
             label2='Synthetic Data',
             savefile=f'./demo/output/data_set_{dataset}_temporal_PSD_of_OPD.png')

    # Plots the temporal PSD of the deflection angle for both the OPD values and the synthetic data:
    plot_psd(frequencies=frequencies_theta_x,
             psd_values=data_psd_theta_x,
             psd_values_2=synthetic_psd_theta_x,
             x_label='Frequency $f$ (Hz)',
             y_label='PSD Value $S_{\\theta_x}(f)$',
             title=f'Data Set {dataset}: Temporal PSD of $\\theta_x$',
             label1='Measured Data',
             label2='Synthetic Data',
             savefile=f'./demo/output/data_set_{dataset}_temporal_PSD_of_deflection_angle.png')

    # Creates images of both structure functions:
    colorbar_scale = colorbar_scales[dataset]
    structure_function_image(structure_function_inputs=structure_function_inputs,
                             structure_function=data_structure_function_values,
                             interpolation_scale=2,
                             x_label='$x/d$',
                             y_label='$y/d$',
                             colorbar_label='$\\mathbf{D}_{\\phi/\\sigma}(x/d,y/d)$',
                             image_title=f'Data Set {dataset}: Structure Function of Measured Data',
                             colorbar_scale=colorbar_scale,
                             savefile=f'./demo/output/data_set_{dataset}_measured_data_structure_function.png')
    structure_function_image(structure_function_inputs=structure_function_inputs,
                             structure_function=synthetic_structure_function_values,
                             interpolation_scale=2,
                             x_label='$x/d$',
                             y_label='$y/d$',
                             colorbar_label='$\\mathbf{D}_{\\phi/\\sigma}(x/d,y/d)$',
                             image_title=f'Data Set {dataset}: Structure Function of Synthetic Data',
                             colorbar_scale=colorbar_scale,
                             savefile=f'./demo/output/data_set_{dataset}_synthetic_data_structure_function.png')

    print("\n\n")
