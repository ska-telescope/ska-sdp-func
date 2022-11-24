# See the LICENSE file at the top-level directory of this distribution.

"""Module for the weighting functions."""

import ctypes

import numpy

from ..utility import Lib, Mem

Lib.wrap_func(
    "sdp_weighting_uniform",
    restype=None,
    argtypes=[
        Mem.handle_type(),
        Mem.handle_type(),
        ctypes.c_double,
        Mem.handle_type(),
        Mem.handle_type(),
    ],
    check_errcode=True,
)


def get_uv_range(uvw, freq_hz):
    """
    Calculate uv-range in wavelength units given UVW-coordinates
    and frequency array.

    :param uvw: List of UVW coordinates in metres, real-valued.
                Dimensions are [num_times, num_baselines, 3]
    :type uvw: numpy.ndarray

    :param freq_hz: List of frequencies in Hz, real-valued.
                    Dimension is [num_channels]
    :type freq_hz: numpy.ndarray

    :returns max_abs_uv: Maximum absolute value of UV coordinates
                         in wavelength units, real-valued
    """
    max_abs_uv = numpy.amax(numpy.abs(uvw[:, :, 0:1]))
    max_abs_uv *= freq_hz[-1] / 299792458.0

    return max_abs_uv


def uniform_weights(uvw, freq_hz, max_abs_uv, grid_uv, weights):
    """
    Calculate the number of hits per UV cell and use the inverse of this
    as the weight.

    :param uvw: List of UVW coordinates in metres, real-valued.
                Dimensions are [num_times, num_baselines, 3]
    :type uvw: numpy.ndarray

    :param freq_hz: List of frequencies in Hz, real-valued.
                    Dimension is [num_channels]
    :type freq_hz: numpy.ndarray

    :param max_abs_uv: Maximum absolute value of UV coordinates
                       in wavelength units, real-valued.
    :type max_abs_uv: float

    :param grid_uv: A initially zero-valued 2D UV grid array.
                    Returns the number of hits per UV cell.
    :type grid_uv: numpy.ndarray

    :param weights: A real-valued 4D array, returns the weights.
                    Dimensions are
                    [num_times, num_baselines, num_channels, num_pols]
    :type weights: numpy.ndarray
    """
    Lib.sdp_weighting_uniform(
        Mem(uvw), Mem(freq_hz), max_abs_uv, Mem(grid_uv), Mem(weights)
    )
