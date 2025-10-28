#%%
import jax
import jax.numpy as jnp
from wind_tomo.simulation import *
from wind_tomo.tomography import *
from demo_utils import display_N_planes_from_recon_and_ground_truth
from wind_tomo.utilities import generate_beam_path_ROI_mask, display_viewing_configuration_schematic
import matplotlib.pyplot as plt

"""
This file is a demo script for simulating OPD_TT measurements with simulated data and performing tomographic reconstruction.
"""

## Collect simulation parameters
test_region_dims=(20,12.5,2) # in cms (depth axis, stream-wise axis ,vertical axis)
beam_diameter = 2  # in cms
pixel_pitch = 0.03125  # in cms

# Define sensor locations in cms (depth axis, stream-wise axis) with respect to the center of the test region
sensor_one = jnp.array([26, -2.5])
sensor_two = jnp.array([26, 0])
sensor_three = jnp.array([26, 2.5])
sensor_locations = [sensor_one, sensor_two, sensor_three]

# Define beam angles in radians for each sensor, defined with respect to the depth axis (i.e., 0 radians is along the depth axis)
# sensor_one_angles = jnp.array([-6.5, -5.5, -4.5]) * jnp.pi / 180
sensor_one_angles = jnp.array([-1, 0, 1]) * jnp.pi / 180
sensor_two_angles = jnp.array([-1, 0, 1]) * jnp.pi / 180
sensor_three_angles = jnp.array([-1, 0, 1]) * jnp.pi / 180

# sensor_three_angles = jnp.array([4.5, 5.5, 6.5]) * jnp.pi / 180
beam_angles=[sensor_one_angles,sensor_two_angles,sensor_three_angles]

display_viewing_configuration_schematic(sensor_locations, beam_angles, diameter=None, title=f'{len(beam_angles)} sensor viewing configuration with {beam_diameter} cm beam diameter', dims=test_region_dims,roi_thickness_and_num_regions=(2,1))
plt.show()
#%%
# collect optical setup information
optical_params=define_optical_setup(sensor_locations, beam_angles, test_region_dims, pixel_pitch, windows=True)
beam_diameter_pixels=int(beam_diameter/pixel_pitch)

## Simulate atmospheric phase volume
r0=0.05  # Fried parameter in meters
dim=optical_params['test_region_pixel_dims']  # dimensions of the phase volume in pixels
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
    phase_volume = generate_random_atmospheric_phase_volume(r0, dim, delta, L0=L0, l0=0.0, key=key)
else:
    print("No CUDA-enabled GPU found for JAX. Using pre-generated atmospheric phase volume.")
    phase_volume = jnp.load("data/pre_generated_phase_volume.npy")

## Simulate OPD_TT measurements
# create mbirjax CT model and FOV mask
ct_model, FOV = create_ct_model_and_weights(optical_params, beam_diameter)
print('\nCT model and FOV mask created.')

# simulate tip-tilt removed OPD views
print('\nSimulating OPD_TT measurements...')
OPD_views=collect_projection_measurement(ct_model, FOV, phase_volume, projection_type='OPD_TT')
print('\nSimulated OPD_TT measurements collected.')

## Perform tomographic reconstruction using mbirjax
print('\nStarting tomographic reconstruction using mbirjax...')
reconstruction,_=ct_model.recon(OPD_views, weights=FOV)
print('Tomographic reconstruction completed.')
# visualize results
ROI=generate_beam_path_ROI_mask(dim, beam_diameter_pixels)
print('\nDisplaying reconstructed planes compared to ground truth planes...')
display_N_planes_from_recon_and_ground_truth(reconstruction,phase_volume,ROI,depth_axis_length=test_region_dims[0],N=4,plane_type='OPD_TT')
plt.show()
