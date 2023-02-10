# See the LICENSE file at the top-level directory of this distribution.

"""Module for the weighting functions."""

import ctypes

import numpy

from ..utility import Lib, Mem

Lib.wrap_func(
    "sdp_weighting_briggs",
    restype=None,
    argtypes=[
        Mem.handle_type(),
        Mem.handle_type(),
        ctypes.c_double,
        ctypes.c_uint, 
        ctypes.c_double,
        Mem.handle_type(),
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


def briggs_weights(uvw, freq_hz, max_abs_uv, wt, robust_param, grid_uv, input_weights, output_weights):
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

    :param wt: Parameter for switching between Uniform and Robust Weighting.
               Value of 1(Robust) or 2(Uniform) 
    :type wt: enum/int

    :param robust_param: Parameter given by the user to gauge the robustness of the weighting function. 
                         A value of -2 would be closer to uniform weighting and 2 would be closer to natural weighting. 

    :param grid_uv: A initially zero-valued 2D UV grid array.
                    Returns the number of hits per UV cell.
    :type grid_uv: numpy.ndarray

    :param weights: A real-valued 4D array, returns the weights.
                    Dimensions are
                    [num_times, num_baselines, num_channels, num_pols]
    :type weights: numpy.ndarray
    """
    Lib.sdp_weighting_briggs(
        Mem(uvw), Mem(freq_hz), max_abs_uv, wt, robust_param, Mem(grid_uv), Mem(input_weights), Mem(output_weights)
    )
