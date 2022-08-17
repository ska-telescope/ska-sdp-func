# See the LICENSE file at the top-level directory of this distribution.

"""Module for the weighting functions."""

import numpy as np


def get_uv_range(uvw, freq_hz):
    """
    Calculate uv-range in wavelength units given UVW-coordinates
    and frequency array.

    :param uvw: List of UVW coordinates in meters, real-valued.
                Dimensions are [num_times*num_baselines, 3]
    :type uvw: numpy.ndarray

    :param freq_hz: List of frequencies in Hz, real-valued.
                    Dimension is [num_channels]
    :type freq_hz: numpy.ndarray

    :returns max_abs_uv: Maximum absolute value of UV coordinates
                         in wavelength units, real-valued
    """
    max_abs_uv = np.amax(np.abs(uvw[:, 0:1]))
    max_abs_uv *= freq_hz[-1] / 299792458.0

    return max_abs_uv


def uniform_weights(uvw, freq_hz, max_abs_uv, grid_uv, weights):
    """
    Calculate the number of hits per UV cell and use the inverse of this
    as the weight

    :param uvw: List of UVW coordinates in meters, real-valued.
                Dimensions are [num_times*num_baselines, 3]
    :type uvw: numpy.ndarray

    :param freq_hz: List of frequencies in Hz, real-valued.
                    Dimension is [num_channels]
    :type freq_hz: numpy.ndarray

    :param max_abs_uv: Maximum absolute value of UV coordinates
                         in wavelength units, real-valued
    :type max_abs_uv: float

    :param grid_uv: A zero-valued 2D UV grid array, returns
                      the number of hits per UV cell
    :type grid_uv: numpy.ndarray

    :param weights: A zero-valued 3D array, returns the weights.
                    Dimensions are [num_times*num_baselines, num_channels, 4]
    :type weights: numpy.ndarray

    """

    grid_size = grid_uv.shape[0]

    uvw_range = range(len(uvw))
    freq_hz_range = range(len(freq_hz))
    for i in uvw_range:
        for j in freq_hz_range:
            grid_u = uvw[i, 0] * freq_hz[j] / 299792458.0
            grid_v = uvw[i, 1] * freq_hz[j] / 299792458.0
            idx_u = int(grid_u / max_abs_uv * grid_size / 2 + grid_size / 2)
            idx_v = int(grid_v / max_abs_uv * grid_size / 2 + grid_size / 2)
            if idx_u >= grid_size:
                idx_u = grid_size - 1
            if idx_v >= grid_size:
                idx_v = grid_size - 1
            grid_uv[idx_u, idx_v] += 1.0
            weights[i, j, 0] = idx_u
            weights[i, j, 1] = idx_v

    for i in uvw_range:
        for j in freq_hz_range:
            idx_u = int(weights[i, j, 0])
            idx_v = int(weights[i, j, 1])
            weight_g = 1.0 / grid_uv[idx_u, idx_v]
            weights[i, j, :] = weight_g
