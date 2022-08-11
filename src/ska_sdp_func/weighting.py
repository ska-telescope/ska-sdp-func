# See the LICENSE file at the top-level directory of this distribution.

"""Module for the weighting functions."""

import numpy as np


def get_uv_range(uvw, freq_hz):
    """Calculate uv-range in WL units given UVW-coordinates
    and frequency array."""
    max_abs_uv = np.amax(np.abs(uvw[:, 0:1]))
    max_abs_uv *= freq_hz[-1] / 299792458.0

    return max_abs_uv


def uniform_weights(uvw, freq_hz, max_abs_uv, grid_size, weights):
    """Calculate the number of hits per UV cell and use an invert
    as the weight"""
    grid_uv = np.zeros((grid_size + 1, grid_size + 1))

    uvw_range = range(len(uvw))
    freq_hz_range = range(len(freq_hz))
    for i in uvw_range:
        for j in freq_hz_range:
            grid_u = uvw[i, 0] * freq_hz[j] / 299792458.0
            grid_v = uvw[i, 1] * freq_hz[j] / 299792458.0
            idx_u = int(grid_u / max_abs_uv * grid_size / 2 + grid_size / 2)
            idx_v = int(grid_v / max_abs_uv * grid_size / 2 + grid_size / 2)
            grid_uv[idx_u, idx_v] += 1.0
            weights[i, j, 0] = idx_u
            weights[i, j, 1] = idx_v

    for i in uvw_range:
        for j in freq_hz_range:
            idx_u = int(weights[i, j, 0])
            idx_v = int(weights[i, j, 1])
            weight_g = 1.0 / grid_uv[idx_u, idx_v]
            weights[i, j, :] = weight_g

    return grid_uv, weights
