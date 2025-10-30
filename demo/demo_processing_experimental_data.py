import jax
import jax.numpy as jnp
import wind_density_tomo.simulation as sim
import wind_density_tomo.tomography as tomo
import wind_density_tomo.visualization_and_analysis as va
import wind_density_tomo.configuration_params as config
import matplotlib.pyplot as plt
from demo_utils import display_raw_data_and_processed_data

"""
This file is a demo script for experimental data processing. 
    1. First, we define the parameters of our optical set up
    2. Next, we simulate raw data collection by performing a forward projection of the simulated data, cropping views 
        to the aperture, and setting values outside the aperture to Nan.
    3. Lastly, we use tomography.generate_ct_model_sinogram_weights_from_experimental_data to get the raw data into the 
        correct form (undoing step 1) and produce appropriate FOV weights and mbirjax ct model.
"""

## Step 1: Define Optical set up
test_region_dims=(20,12.5,4) # in cms (depth axis, stream-wise axis ,vertical axis)
beam_diameter = 2  # in cms
pixel_pitch = 0.03125  # in cms

# Define sensor locations in cms (depth axis, stream-wise axis) with respect to the center of the test region
sensor_one = jnp.array([26, -4])
sensor_two = jnp.array([26, 0])
sensor_three = jnp.array([26, 4])
sensor_locations = [sensor_one, sensor_two, sensor_three]

# Define beam angles in radians for each sensor, defined with respect to the depth axis (i.e., 0 radians is along the depth axis)
sensor_one_angles = jnp.array([-13]) * jnp.pi / 180
sensor_two_angles = jnp.array([0]) * jnp.pi / 180
sensor_three_angles = jnp.array([13]) * jnp.pi / 180
beam_angles=[sensor_one_angles,sensor_two_angles,sensor_three_angles]

# collect optical setup information
optical_params=config.define_optical_setup(sensor_locations, beam_angles, test_region_dims, pixel_pitch,beam_fov=beam_diameter, windows=True)

# visualize the viewing configuration
va.display_viewing_configuration_schematic(optical_params,roi_thickness_and_num_regions=(optical_params.beam_diameter_cm,1))


## Step 2: Simulate Raw Data
print('Simulating raw data...')
# Simulate atmospheric phase volume
r0=0.05  # Fried parameter in meters
delta=pixel_pitch/100  # pixel pitch of the phase volume in meters
L0=0.02  # Outer scale in meters
seed=42  # random seed for phase volume generation

#check if CUDA-enabled GPU is available
devices = jax.devices()
# Filter for CUDA devices
cuda_devices = [d for d in devices if d.platform == 'cuda']
if cuda_devices:
    print("CUDA-enabled GPU(s) found. Will generate new atmospheric phase volume on GPU.")
    key = jax.random.PRNGKey(42)
    phase_volume = sim.generate_random_atmospheric_phase_volume(r0, optical_params.test_region_pixel_dims, delta, L0=L0, l0=0.0, key=key)
else:
    print("No CUDA-enabled GPU found for JAX. Using pre-generated atmospheric phase volume.")
    phase_volume=jnp.zeros((optical_params.test_region_pixel_dims))
    top_slice=int(optical_params.test_region_pixel_dims[2]//2+optical_params.beam_diameter_pixels/2)
    bottom_slice=int(optical_params.test_region_pixel_dims[2]//2-optical_params.beam_diameter_pixels/2)
    phase_volume= phase_volume.at[:,:,bottom_slice:top_slice].set(jnp.load("data/pre_generated_phase_volume.npy"))

# create mbirjax CT model and FOV mask
ct_model, FOV = sim.create_ct_model_and_weights_for_simulation(optical_params)

# simulate tip-tilt removed OPD views
OPD_views=sim.collect_projection_measurement(ct_model, FOV, phase_volume, projection_type='OPD_TT')
OPD_views=OPD_views.at[FOV==0].set(jnp.nan)

# Make views look like raw data
num_rows=max(jnp.where(FOV[0]==1)[0])-min(jnp.where(FOV[0]==1)[0])
num_cols=max(jnp.where(FOV[0]==1)[1])-min(jnp.where(FOV[0]==1)[1])
OPD_raw=jnp.zeros((FOV.shape[0],num_rows,num_cols))
for i in range(FOV.shape[0]):
    max_row = max(jnp.where(FOV[i] == 1)[0])
    min_row = min(jnp.where(FOV[i] == 1)[0])
    max_col = max(jnp.where(FOV[i] == 1)[1])
    min_col = min(jnp.where(FOV[i] == 1)[1])
    OPD_raw=OPD_raw.at[i].set(OPD_views[i,min_row:max_row,min_col:max_col])

### Step 3: process the raw data
print('Processing raw data')

#forget information about beam FOV
optical_params.beam_fov=None

#process raw data and produce ct model with weight matrix
ct_model, sinogram, weight_matrix=tomo.generate_ct_model_sinogram_weights_from_experimental_data(optical_params,OPD_raw)
display_raw_data_and_processed_data(OPD_raw,sinogram,weight_matrix)
plt.show()


