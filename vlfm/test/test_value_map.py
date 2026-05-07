# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

import numpy as np

from vlfm.mapping.value_map import ValueMap


def test_get_value_peaks_returns_sorted_local_maxima() -> None:
    value_map = ValueMap(value_channels=1, size=20)
    value_map._value_map[4, 4, 0] = 0.2
    value_map._value_map[8, 8, 0] = 0.9
    value_map._value_map[14, 14, 0] = 0.7

    points, values = value_map.get_value_peaks(radius=0.05, max_points=2)
    peak_pixels = value_map._xy_to_px(points)

    assert np.allclose(values, [0.9, 0.7])
    assert np.array_equal(peak_pixels, np.array([[8, 8], [14, 14]]))
