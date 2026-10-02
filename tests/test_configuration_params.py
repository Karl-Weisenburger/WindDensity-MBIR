import numpy as np
import pytest

import winddensity_mbir.configuration_params as config


def test_grouped_and_flat_angle_formats_agree():
    locs = [(26.0, -2.5), (26.0, 2.5)]
    grouped = [[-0.1, 0.0], [0.05]]
    flat_locs = [locs[0], locs[0], locs[1]]
    flat_angles = [-0.1, 0.0, 0.05]

    a = config.define_optical_setup(locs, grouped, (20, 12.5, 2), 0.03125, beam_fov=2.0)
    b = config.define_optical_setup(flat_locs, flat_angles, (20, 12.5, 2), 0.03125, beam_fov=2.0)

    np.testing.assert_allclose(a.beam_angles, b.beam_angles)
    assert a.sensor_locations_pixels == b.sensor_locations_pixels


def test_cm_to_pixel_conversion_matches_paper_geometry():
    cm_per_pixel = 25.0 / 800
    setup = config.define_optical_setup(
        [(0.0, 0.0)], [[0.0]], (640 * cm_per_pixel, 400 * cm_per_pixel, 64 * cm_per_pixel),
        cm_per_pixel, beam_fov=2.0,
    )
    assert setup.test_region_pixel_dims == (640, 400, 64)
    assert setup.beam_diameter_cm == 2.0
    assert setup.beam_diameter_pixels == pytest.approx(64.0)


def test_beam_diameter_inferred_from_2d_fov_mask():
    fov = np.zeros((20, 20), dtype=bool)
    fov[5:15, 4:16] = True          # 10 rows x 12 cols
    setup = config.define_optical_setup([(0.0, 0.0)], [[0.0]], (4, 4, 4), 0.5, beam_fov=fov)
    assert float(setup.beam_diameter_cm) == pytest.approx(12 * 0.5)


def test_mismatched_sensor_and_angle_groups_raise():
    with pytest.raises(ValueError):
        config.define_optical_setup([(0.0, 0.0)], [[0.0], [0.1]], (4, 4, 4), 0.5, beam_fov=1.0)
