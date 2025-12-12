#Approved for public release; distribution is unlimited. Public Affairs release approval # 2025-5579

import warnings
import math
import jax.numpy as jnp
import jax
import numpy as np
import matplotlib.pyplot as plt
import winddensity_mbir.configuration_params as config
from functools import lru_cache
from matplotlib.patches import Rectangle, Patch
from matplotlib.lines import Line2D
from collections import defaultdict
from typing import Optional, List, Tuple, Union, Dict


def refractive_index_2_density(refractive_index_volume, gladestone_dale_constant):
    """
    Convert refractive index volume to density volume using Gladstone-Dale relation.

    Args:
        refractive_index_volume (jnp.ndarray): 3D array of refractive index values.
        gladestone_dale_constant (float): Gladstone-Dale constant for the medium.

    Returns:
        jnp.ndarray: 3D array of density values.
    """
    if gladestone_dale_constant == 0:
        warnings.warn("Gladstone-Dale constant is zero. Returning zero density volume.")
        return jnp.zeros_like(refractive_index_volume)
    density_volume = (refractive_index_volume - 1) / gladestone_dale_constant
    return density_volume


def _jax_nrmse_roi_flat(GT_flat, recon_flat, indices, option=0):
    """
    Compute NRMSE using flattened arrays and integer indices.

    Args:
        GT_flat: Flattened ground truth array.
        recon_flat: Flattened reconstructed array.
        indices: Integer indices where roi is True.
        option: Denominator option (0: RMS of GT, 1: range, 2: interpercentile range).

    Returns:
        float: Normalized RMSE.
    """
    selected_GT = GT_flat[indices]
    selected_recon = recon_flat[indices]
    rmse = jnp.sqrt(jnp.mean((selected_recon - selected_GT) ** 2))

    if option == 0:
        denominator = jnp.sqrt(jnp.mean(selected_GT ** 2))
    elif option == 1:
        denominator = jnp.max(selected_GT) - jnp.min(selected_GT)
    else:
        q1, q2 = jnp.percentile(selected_GT, jnp.array([5, 95]))
        denominator = q2 - q1

    return rmse / denominator


def nrmse_over_roi(GT, recon, ROI, option=0):
    """
    Compute NRMSE over a region of interest (roi) using JAX.

    Args:
        GT (jax.numpy.ndarray): Ground truth array.
        recon (jax.numpy.ndarray): Reconstructed array.
        ROI (jax.numpy.ndarray): Boolean mask defining the region of interest.
        option (int): Denominator option (0: RMS of GT, 1: range, 2: interpercentile range).

    Returns:
        float: Normalized RMSE over the roi.
    """
    GT_flat = GT.flatten()
    recon_flat = recon.flatten()
    ROI_flat = ROI.flatten()
    indices = jnp.where(ROI_flat)[0]
    return _jax_nrmse_roi_flat(GT_flat, recon_flat, indices, option)

def divide_into_sections_of_opl(recon, sections, total_length):
    """
    Divide a refractive index volume into sections and compute the optical path length (OPL) for each section.

    Args:
        recon (jax.numpy.ndarray): 3D refractive index volume with shape (N, H, W).
        sections (int): Number of sections to divide the volume into.
        total_length(float): Total physical length of the volume along the first axis.

    Returns:
        jnp.ndarray: volume of OPL planes with shape (sections, H, W)
    """
    N = recon.shape[0]
    if sections == N:
        return recon
    S = sections
    L = N * 1.0 / S
    H, W = recon.shape[1:]
    zeros = jnp.zeros((1, H, W), dtype=recon.dtype)
    cumsum = jnp.concatenate([zeros, jnp.cumsum(recon, axis=0)], axis=0)

    def integral_to(x):
        floor_x = jnp.floor(x).astype(jnp.int32)
        frac_x = x - floor_x
        recon_idx = jnp.minimum(floor_x, N - 1)
        return cumsum[floor_x] + frac_x * recon[recon_idx]

    ks = jnp.arange(S)
    starts = ks * L
    ends = (ks + 1) * L
    integrals_starts = jax.vmap(integral_to)(starts)
    integrals_ends = jax.vmap(integral_to)(ends)
    weighted_sums = integrals_ends - integrals_starts
    OPLs = weighted_sums * total_length / (L * S)
    return OPLs

def generate_beam_path_roi_mask(recon_dim, beam_pixel_diam, location=(0, 0, 0), angle=0):
    """
    Generate a 3D mask for the beam path roi.

    Args:
        recon_dim (tuple): Dimensions of the reconstruction volume (rows, cols, slices).
        beam_pixel_diam (float): Beam diameter in pixels.
        location (tuple): (x, y, z) location of the beam center in pixels.
        angle (float): Beam angle from row axis in radians.

    Returns:
        jnp.ndarray: 3D boolean mask of the beam path roi.
    """

    rows, cols, slices = recon_dim
    if len(location) == 2:
        a, b = location
        c = 0
    else:
        a, b, c = location
    radius = beam_pixel_diam / 2
    center_r = rows / 2
    center_c = cols / 2
    center_s = slices / 2
    i = jnp.arange(rows)
    j = jnp.arange(cols)
    k = jnp.arange(slices)
    I, J, K = jnp.meshgrid(i, j, k, indexing="ij")
    x = I - center_r
    y = -(J - center_c)
    z = -(K - center_s)
    sin_theta = jnp.sin(angle)
    cos_theta = jnp.cos(angle)
    perp_dist_xy = sin_theta * (x - a) + cos_theta * (y - b)
    dist_z = z - c
    dist = jnp.sqrt(perp_dist_xy ** 2 + dist_z ** 2)
    mask = dist <= radius
    return mask

@lru_cache(maxsize=None)
def _fact(k):
    """Helper function for factorial, cached."""
    if k < 0:
        return 0
    return math.factorial(k)


def _radial_zernike(n, abs_m, rho):
    """Calculates the radial part of Zernike polynomial."""
    rad = np.zeros_like(rho)
    for s in range((n - abs_m) // 2 + 1):
        sign = (-1) ** s
        coeff = _fact(n - s) / (_fact(s) * _fact((n + abs_m) // 2 - s) * _fact((n - abs_m) // 2 - s))
        rad += sign * coeff * rho ** (n - 2 * s)
    return rad


def _zernike(n, m, rho, theta):
    """Calculates the orthonormal Zernike polynomial Z(n, m)."""
    abs_m = abs(m)
    kron_delta = 1 if m == 0 else 0
    norm = np.sqrt((2 * (n + 1)) / np.pi / (1 + kron_delta))
    radial = _radial_zernike(n, abs_m, rho)
    if m > 0:
        angular = np.cos(m * theta)
    elif m < 0:
        angular = np.sin(abs_m * theta)
    else:
        angular = np.ones_like(theta)
    return norm * radial * angular


def _get_zernike_modes(min_n, max_n):
    """
    Get (n, m) modes for a radial range, ensuring (n - |m|) is even.
    """
    modes = []
    if max_n < min_n:
        return modes
    for n in range(min_n, max_n + 1):
        for m in range(-n, n + 1):
            if (n - abs(m)) % 2 == 0:
                modes.append((n, m))
    return modes

def isolate_zernike_mode_range_for_img(image, roi, radial_degree_min, radial_degree_max):
    """
    Isolates a specific range of Zernike modes for a single 2D image within a
    region of interest (roi).

    Args:
        image (numpy.ndarray): The 2D input image.
        roi (numpy.ndarray): A 2D boolean mask (Region of Interest) with the same shape as `image`. The analysis is
            performed only on pixels where roi is True.
        radial_degree_min (int): The minimum radial degree (n) to include.
        radial_degree_max (int): The maximum radial degree (n) to include.

    Returns:
        np.ndarray: A 2D image of the same shape as the input, but with only the specified range of zernike modes present
    """
    if radial_degree_min > radial_degree_max:
        raise ValueError("radial_degree_min must be <= radial_degree_max")

    image = np.asarray(image, dtype=np.float64)
    roi = np.asarray(roi, dtype=bool)

    if image.shape != roi.shape:
        raise ValueError("Image and roi must have the same shape")

    if image.ndim != 2:
        raise ValueError(f"Input image must be 2D, but got {image.ndim} dimensions.")

    output = np.zeros_like(image, dtype=np.float64)

    # Find all pixels within the roi
    rows, cols = np.nonzero(roi)
    if len(rows) == 0:
        # No roi, return the all-zero image
        return output

    # Get the 1D vector of data from the roi
    y = image[rows, cols].flatten()

    # --- Coordinate Calculation ---
    # Calculate coordinates relative to the roi's centroid
    cy = np.mean(rows)
    cx = np.mean(cols)
    dy = rows - cy
    dx = cols - cx

    dist = np.sqrt(dx ** 2 + dy ** 2)
    r_max = np.max(dist)

    if r_max == 0:
        # roi is a single point.
        # Place the single value back if n=0 is in range
        if 0 >= radial_degree_min and 0 <= radial_degree_max:
            output[rows, cols] = y
        return output

    # Normalize coordinates for Zernike polynomials
    rho = dist / r_max
    theta = np.arctan2(dy, dx)

    # --- Zernike Fitting ---
    # Get the modes and build the system matrix A
    modes = _get_zernike_modes(radial_degree_min, radial_degree_max)
    if not modes:
        # No modes selected (e.g., min=5, max=4), return zero array
        return output

    num_modes = len(modes)
    A = np.zeros((len(y), num_modes), dtype=np.float64)
    for j, (n, m) in enumerate(modes):
        A[:, j] = _zernike(n, m, rho, theta)

    # Solve for coefficients using least squares
    try:
        c, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    except np.linalg.LinAlgError:
        # This can happen with a singular matrix
        print(f"Warning: Singular matrix encountered in least-squares fit. Returning zero array for this slice.")
        return output

    # Reconstruct the signal using only the fitted coefficients
    recon = A @ c

    # Place the reconstructed data back into the output image
    output[rows, cols] = recon

    return output

def isolate_zernike_mode_range_for_volume(volume, radial_degree_min, radial_degree_max, roi, axis=0):
    """
    Isolates a specific range of Zernike modes for each slice in a 3D volume
    along a specified axis.

    Args:
        volume (numpy.ndarray): The 3D input volume.
        radial_degree_min (int): The minimum radial degree (n) to include.
        radial_degree_max (int): The maximum radial degree (n) to include.
        roi (numpy.ndarray): A 3D boolean mask (Region of Interest) with the same shape as `Volume`.
        axis (int, optional): The axis to iterate over. Defaults to 0.

    Returns:
        np.ndarray: A 3D volume of the same shape as the input, but with only the specified zernike modes present in each slice
    """
    if radial_degree_min > radial_degree_max:
        raise ValueError("radial_degree_min must be <= radial_degree_max")

    volume = np.asarray(volume)
    roi = np.asarray(roi, dtype=bool)

    if volume.shape != roi.shape:
        raise ValueError("volume and roi must have the same shape")

    if volume.ndim != 3:
        raise ValueError(f"Input volume must be 3D, but got {volume.ndim} dimensions.")

    output = np.zeros_like(volume, dtype=np.float64)

    # Move the specified axis to the front (axis 0) for easy iteration
    volume_moved = np.moveaxis(volume, axis, 0)
    roi_moved = np.moveaxis(roi, axis, 0)
    output_moved = np.moveaxis(output, axis, 0)

    # Iterate over each 2D slice
    for i in range(volume_moved.shape[0]):
        # Process each 2D slice using the dedicated 2D function
        output_moved[i] = isolate_zernike_mode_range_for_img(
            image=volume_moved[i],
            roi=roi_moved[i],
            radial_degree_min=radial_degree_min,
            radial_degree_max=radial_degree_max
        )

    # Move the processed axis back to its original position
    output = np.moveaxis(output_moved, 0, axis)

    return output

def _is_point_in_rectangle(recty, point):
    """
    Check if a point is inside a rectangle.

    Args:
        recty (list): List of four tuples representing the corners of the rectangle in clockwise or counterclockwise order.
        point (tuple): Tuple representing the coordinates of the point (x, y).

    Returns:
        bool: True if the point is inside the rectangle, False otherwise.
    """
    def cross_product(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    def is_point_in_triangle(p, a, b, c):
        # Check if point p is inside the triangle formed by points a, b, and c
        return (cross_product(p, a, b) >= 0 and
                cross_product(p, b, c) >= 0 and
                cross_product(p, c, a) >= 0)

    # Split the rectangle into two triangles
    triangle1 = [recty[0], recty[1], recty[2]]
    triangle2 = [recty[2], recty[3], recty[0]]

    # Check if the point is inside either of the triangles
    return is_point_in_triangle(point, *triangle1) or is_point_in_triangle(point, *triangle2)

def display_viewing_configuration_schematic(
    optical_setup: Optional[config.OpticalSetup] = None,
    sensor_locations: Optional[List[Tuple[float, float]]] = None,
    angles: Optional[List[Union[List[float], np.ndarray, jnp.ndarray]]] = None,
    beam_diameter: Optional[float] = None,
    show_beam_diameter: bool = False,
    scale: float = 0.5,
    overlap_threshold: Optional[int] = None,
    title: Optional[str] = 'Beam Path Schematic',
    plane: str = 'transverse',
    dims: Optional[Tuple[float, float, float]] = None,
    outer_buffer: Tuple[float, float] = (2, 2),
    roi_thickness_and_num_regions: Optional[Tuple[float, int]] = None,
    legend_scale: float = 1
):
    """
    Display a schematic of the viewing configuration for wind tunnel tomography.

    Args:
        optical_setup (OpticalSetup): Optical setup parameters from `define_optical_setup`.
            If provided, extracts sensor_locations, angles (grouped), beam_diameter (from beam_fov if scalar),
            and dims (from test_region_dims if dims is None).
        sensor_locations (list of tuples, optional): List of (x, y) coordinates for sensor locations in cm.
            Required if optical_setup is None.
        angles (list, optional): List of beam angles for each sensor (grouped), where each group can be a list, np.ndarray, or jnp.ndarray.
            Required if optical_setup is None.
        beam_diameter (float, optional): Beam diameter in cm. Required if show_beam_diameter is True and
            optical_setup is None or beam_fov is not scalar.
        show_beam_diameter (bool, optional): Whether to show the beam diameter as a band (default: False).
        scale (float, optional): Scaling factor for the figure size and line widths (default: 0.75).
        overlap_threshold (int, optional): Threshold for highlighting overlap regions (default: None).
        title (str, optional): Title for the plot (default: 'Beam Path Schematic').
        plane (str, optional): Viewing plane, 'transverse' or 'sagittal' (default: 'transverse').
        dims (tuple, optional): Dimensions of the test region (rows, cols, slices) in cm.
            Defaults to (20, 25, 20) if not provided and no optical_setup.
        outer_buffer (tuple, optional): Buffer around the test region (default: (2, 2)).
        roi_thickness_and_num_regions (tuple, optional): roi thickness and number of regions (default: None).
        legend_scale (float, optional): Scaling factor for the legend font size (default: 1).

    Returns:
        tuple: (fig, ax)
    """
    if optical_setup is not None:
        # Reconstruct grouped angles from flattened beam_angles and repeated sensor_locations_pixels
        pixel_pitch = optical_setup.pixel_pitch
        grouped_angles_dict: Dict[Tuple[float, float], List[float]] = defaultdict(list)
        for this_center, angle in zip(optical_setup.sensor_locations_pixels, optical_setup.beam_angles):
            cm_center = (this_center[0] * pixel_pitch, this_center[1] * pixel_pitch)
            grouped_angles_dict[cm_center].append(float(angle))
        sensor_locations = list(grouped_angles_dict.keys())
        angles = list(grouped_angles_dict.values())

        # Set beam_diameter from beam_fov if scalar
        if beam_diameter is None and optical_setup.beam_fov is not None:
            if jnp.isscalar(optical_setup.beam_fov):
                beam_diameter = float(optical_setup.beam_fov)
            elif optical_setup.beam_diameter_cm is not None:
                beam_diameter = optical_setup.beam_diameter_cm
            else:
                warnings.warn("beam_fov is not scalar and beam_diameter_cm is not provided; cannot show beam diameter.")
                show_beam_diameter = False

        # Override dims if not explicitly provided
        if dims is None:
            dims = optical_setup.test_region_dims
    else:
        if sensor_locations is None or angles is None:
            raise ValueError("If optical_setup is not provided, sensor_locations and angles must be supplied.")
        if show_beam_diameter and beam_diameter is None:
            raise ValueError("beam_diameter must be provided if show_beam_diameter is True.")

    # Set default dims if still None
    if dims is None:
        dims = (20, 25, 20)

    if plane == 'transverse':
        height = dims[1]  # columns
        width = dims[0]  # rows
    elif plane == 'sagittal':
        height = dims[2]  # slices
        width = dims[0]  # rows
    else:
        raise ValueError("Plane options are only 'transverse' or 'sagittal'")

    # Create a figure and axis with the specified dimensions
    fig, ax = plt.subplots(figsize=((width + outer_buffer[0] * 2) * scale, (height + outer_buffer[1] * 2) * scale))
    h_bounds = height / 2 + outer_buffer[1]
    w_bounds = width / 2 + outer_buffer[0]

    # Set the limits of the plot
    ax.set_xlim(-w_bounds, w_bounds)
    ax.set_ylim(-h_bounds, h_bounds)

    # Set the overall background color to white
    ax.set_facecolor('white')

    if plane == 'transverse':
        background_rect = plt.Rectangle((-width / 2, -h_bounds * 1.2), width, h_bounds * 2 * 1.2, edgecolor='k', facecolor='0.8',
                                        zorder=0, linewidth=4 * scale)
        ax.add_patch(background_rect)

    rectangle = plt.Rectangle((-width / 2, -height / 2), width, height, edgecolor=None, facecolor='#87CEFA',
                              zorder=1)
    ax.add_patch(rectangle)

    custom_legend = [
        Patch(facecolor="#87CEFA", label="Wind Tunnel Test Region"),
    ]

    ax.plot([-width / 2, -width / 2], [-height / 2, height / 2], '#87CEFA', linewidth=4 * scale, zorder=1)
    ax.plot([width / 2, width / 2], [-height / 2, height / 2], '#87CEFA', linewidth=4 * scale, zorder=1)

    if roi_thickness_and_num_regions is not None:
        roi_thickness, num_regions = roi_thickness_and_num_regions
        rectangle = plt.Rectangle((-width / 2, -roi_thickness / 2), width, roi_thickness, edgecolor='#D0F0C0',
                                  facecolor='#D0F0C0',
                                  zorder=1)
        ax.add_patch(rectangle)
        custom_legend.append(Patch(facecolor="#D0F0C0", edgecolor='#D0F0C0', label="Region of Interest"))

        inc = width / num_regions
        for i in range(1, num_regions):
            x = -width / 2 + inc * i
            y_top = roi_thickness / 2
            y_bot = -roi_thickness / 2
            ax.plot([x, x], [y_bot, y_top], 'k', linewidth=2 * scale, zorder=1.5)

    # Store bands for overlap calculation
    bands = []

    # Plot lines from each location at specified angles
    for i, location in enumerate(sensor_locations):
        x, y = location
        for angle in angles[i]:
            # Convert angle to radians relative to the negative x-axis
            angle_rad = angle
            # Calculate the end point of the line to the outer edge of the plot
            # if line is vertical
            if np.cos(angle_rad) == 0:
                x1 = x
                x2 = x
                y1 = -h_bounds * 1.2 if np.sin(angle_rad) > 0 else h_bounds * 1.2
                y2 = -y1
            else:
                x1 = -w_bounds * 1.2 if np.cos(angle_rad) > 0 else w_bounds * 1.2
                x2 = -x1
                y1 = y - (w_bounds * 1.2 + x) * np.tan(angle_rad)
                y2 = y + (w_bounds * 1.2 - x) * np.tan(angle_rad)

            # Plot the line with increased thickness
            ax.plot([x1, x2], [y1, y2], 'r--', linewidth=3 * scale, zorder=2)

            # If show_beam_diameter is positive, add a translucent red band around the line
            if show_beam_diameter:
                length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                angle_deg = np.rad2deg(angle_rad)
                band = Rectangle((x1 + beam_diameter * np.sin(angle_rad) / 2, y1 - beam_diameter * np.cos(angle_rad) / 2), length, beam_diameter,
                                 angle=angle_deg, color='red', alpha=0.2, zorder=1.5, edgecolor=None)
                bands.append(band)

    if overlap_threshold is not None and overlap_threshold > 0 and show_beam_diameter:
        custom_legend.append(Patch(facecolor="red", alpha=0.3, label="Pixels seen by ≥ " + str(overlap_threshold) + " beams"))
        # Create a grid to check overlaps
        grid_sizex = int(18 * (width + 4))
        grid_sizey = int(18 * (height + 4))
        x_grid = np.linspace(-w_bounds, w_bounds, grid_sizex)
        y_grid = np.linspace(-h_bounds, h_bounds, grid_sizey)
        overlap_grid = np.zeros((grid_sizex, grid_sizey))

        for band in bands:
            band_coords = band.get_corners()
            for i in range(grid_sizex):
                for j in range(grid_sizey):
                    if _is_point_in_rectangle(band_coords, (x_grid[i], y_grid[j])):
                        overlap_grid[i, j] += 1

        for i in range(grid_sizex):
            for j in range(grid_sizey):
                if overlap_grid[i, j] >= overlap_threshold:
                    rect = Rectangle((x_grid[i], y_grid[j]), (x_grid[1] - x_grid[0]), (y_grid[1] - y_grid[0]), color='red', alpha=0.3)
                    ax.add_patch(rect)

    elif show_beam_diameter:
        for band in bands:
            ax.add_patch(band)
        custom_legend.append(Patch(facecolor="red", alpha=0.2, edgecolor=None, label="Beam Path"))

    if title is not None:
        plt.title(title, fontsize=50 * scale)

    space = ' '
    axes_ftsize = int(30 * scale)
    if plane == "transverse":
        plt.ylabel('<-- Flow Direction (x-axis) <--', fontsize=axes_ftsize)
        plt.xlabel(f'Laser Side{space * int(scale * w_bounds * 80 / axes_ftsize)}Depth Axis (y-axis){space * int(scale * w_bounds * 80 / axes_ftsize)}Camera Side', fontsize=axes_ftsize)
    elif plane == 'sagittal':
        plt.ylabel('DOWN <-- Vertical Axis (z-axis) --> UP', fontsize=axes_ftsize)
        plt.xlabel(f'Laser Side{space * int(scale * w_bounds * 80 / axes_ftsize)}Depth Axis{space * int(scale * w_bounds * 80 / axes_ftsize)}Camera Side', fontsize=axes_ftsize)
    else:
        raise ValueError("Plane options are only transverse or sagittal")

    custom_legend.append(Line2D([0], [0], color='red', linestyle='--', label='Beam Optical Axis'))
    plt.legend(handles=custom_legend, loc='upper right', bbox_to_anchor=(1 - 2 / (w_bounds * 2), 1 - 2 / (h_bounds * 2)), fontsize=int(25 * scale * legend_scale))
    # Show the plot
    plt.grid(False)

    return fig, ax