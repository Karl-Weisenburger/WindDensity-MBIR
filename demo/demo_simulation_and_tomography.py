#Approved for public release; distribution is unlimited. Public Affairs release approval # 2025-5579

import jax
import jax.numpy as jnp
import winddensity_mbir.simulation as sim
import winddensity_mbir.visualization_and_analysis as va
import winddensity_mbir.utilities as utils
import winddensity_mbir.configuration_params as config
import matplotlib.pyplot as plt
from demo_utils import display_planes_from_recon_and_ground_truth
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, 'output')
os.makedirs(output_dir, exist_ok=True)

"""
This file is a demo script for simulating OPD_TT measurements with simulated data and performing tomographic
 reconstruction. There are three steps:
    1. First, we set the viewing configuration parameters and visual the geometry
    2. Next, we generate (or preload) a volume of atmospheric phase, and simulate OPD_TT measurements
    3. Lastly, we use the OPD_TT measurements and mbirjax to perform MBIR reconstruction.
"""

## Step 1: Set simulation parameters
test_region_dims=(20,12.5,2) # in cms (depth axis, stream-wise axis ,vertical axis)
beam_diameter = 2  # in cms
pixel_pitch = 0.03125  # in cms

#visualization parameters
num_planes=4
zernike_range=(2,11)

# Define sensor locations in cms (depth axis, stream-wise axis) with respect to the center of the test region
sensor_one = jnp.array([26, -2.5])
sensor_two = jnp.array([26, 0])
sensor_three = jnp.array([26, 2.5])
sensor_locations = [sensor_one, sensor_two, sensor_three]

# Define beam angles in radians for each sensor, defined with respect to the depth axis (i.e., 0 radians is along the depth axis)
sensor_one_angles = jnp.array([-6.5, -5.5, -4.5]) * jnp.pi / 180
sensor_two_angles = jnp.array([-1, 0, 1]) * jnp.pi / 180
sensor_three_angles = jnp.array([4.5, 5.5, 6.5]) * jnp.pi / 180

# sensor_three_angles = jnp.array([4.5, 5.5, 6.5]) * jnp.pi / 180
beam_angles=[sensor_one_angles,sensor_two_angles,sensor_three_angles]

# collect optical setup information
optical_params=config.define_optical_setup(sensor_locations, beam_angles, test_region_dims, pixel_pitch,beam_fov=beam_diameter, windows=True)

# visualize the viewing configuration
va.display_viewing_configuration_schematic(optical_params,roi_thickness_and_num_regions=(optical_params.beam_diameter_cm,1))
plt.savefig(os.path.join(output_dir, 'Viewing_Configuration_Schematic.png'))

## Step 2: Simulate atmospheric phase volume
cn2=1e-11  # Refractive-index structure parameter [m^{-2/3}]
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
    phase_volume = sim.generate_random_atmospheric_volume(cn2, optical_params.test_region_pixel_dims, delta, L0=L0, l0=0.0, key=key)
else:
    print("No CUDA-enabled GPU found for JAX. Using pre-generated atmospheric phase volume.")
    data_dir = os.path.join(script_dir, 'data')
    phase_volume_path = os.path.join(data_dir, 'pre_generated_phase_volume.npy')
    phase_volume = jnp.load(phase_volume_path)

## Simulate OPD_TT measurements
# create mbirjax CT model and FOV mask
ct_model, FOV = sim.create_ct_model_and_weights_for_simulation(optical_params)
print('\nCT model and FOV mask created.')

# simulate tip-tilt removed OPD views
print('\nSimulating OPD_TT measurements...')
OPD_views=sim.collect_projection_measurement(ct_model, FOV, phase_volume, projection_type='OPD_TT')
print('\nSimulated OPD_TT measurements collected.')

## Perform tomographic reconstruction using mbirjax
print('\nStarting tomographic reconstruction using mbirjax and FBP...')

#MBIR reconstruction
recon,_=ct_model.recon(OPD_views, weights=FOV)

# scale-corrected FBP reconstruction (uncomment to use)
# recon,_=ct_model.direct_recon(OPD_views)

print('Tomographic reconstruction completed.')


# visualize results
ROI=va.generate_beam_path_roi_mask(optical_params.test_region_pixel_dims, optical_params.beam_diameter_pixels)

print(f'\nReducing volumes into {num_planes} OPL planes...')
recon_planes = va.divide_into_sections_of_opl(recon, num_planes, test_region_dims[0])
gt_planes = va.divide_into_sections_of_opl(phase_volume, num_planes, test_region_dims[0])
ROI_planes = ROI[:num_planes]

print('\nConverting OPL planes into $\\text{OPD}_{\\text{TT}}$...')
recon_planes_OPD = utils.remove_tip_tilt_piston(recon_planes, ROI_planes)
gt_planes_OPD = utils.remove_tip_tilt_piston(gt_planes, ROI_planes)

display_planes_from_recon_and_ground_truth(recon_planes_OPD, gt_planes_OPD, ROI_planes, title=f'Reconstruction of {num_planes}' + ' $\\text{OPD}_{\\text{TT}}$ planes')
plt.savefig(os.path.join(output_dir, 'Recon_Planes.png'))


print(f'\nIsolating specific Zernike Modes of radial degree {zernike_range[0]} to {zernike_range[1]}')
recon_planes_zern=jnp.array(va.isolate_zernike_mode_range_for_volume(recon_planes, zernike_range[0], zernike_range[1], ROI_planes))
gt_planes_zern=jnp.array(va.isolate_zernike_mode_range_for_volume(gt_planes, zernike_range[0], zernike_range[1], ROI_planes))

display_planes_from_recon_and_ground_truth(recon_planes_zern,gt_planes_zern,ROI_planes,title=f'Reconstruction of {num_planes} planes, isolating zernike radial degrees {zernike_range[0]} to {zernike_range[1]}')
plt.savefig(os.path.join(output_dir, 'Zernike_Isolated_Recon_Planes.png'))
plt.show()
