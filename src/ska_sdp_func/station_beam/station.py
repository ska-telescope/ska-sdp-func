# See the LICENSE file at the top-level directory of this distribution.

"""Module for station beam functions."""

import ctypes

from ..utility import Lib, Mem

Lib.wrap_func(
    "sdp_station_beam_array_factor",
    restype=None,
    argtypes=[
        ctypes.c_double,
        Mem.handle_type(),
        Mem.handle_type(),
        Mem.handle_type(),
        Mem.handle_type(),
        ctypes.c_int32,
        ctypes.c_int32,
        Mem.handle_type(),
        Mem.handle_type(),
        Mem.handle_type(),
        Mem.handle_type(),
        Mem.handle_type(),
        ctypes.c_int32,
        Mem.handle_type(),
        ctypes.c_int32,
    ],
    check_errcode=True,
)


def array_factor(
    wavenumber: float,
    element_weights,
    element_x,
    element_y,
    element_z,
    point_x,
    point_y,
    point_z,
    data_index,
    data,
    beam,
    normalise: bool,
) -> None:
    """Evaluates a basic array factor.

    Args:
        wavenumber (float): Wavenumber for the current frequency channel.
        element_weights (numpy.ndarray or cupy.ndarray):
            Complex array of element beamforming weights.
        element_x (numpy.ndarray or cupy.ndarray):
            Element x coordinates, in metres.
        element_y (numpy.ndarray or cupy.ndarray):
            Element y coordinates, in metres.
        element_z (numpy.ndarray or cupy.ndarray):
            Element z coordinates, in metres.
        point_x (numpy.ndarray or cupy.ndarray):
            Source x direction cosines.
        point_y (numpy.ndarray or cupy.ndarray):
            Source y direction cosines.
        point_z (numpy.ndarray or cupy.ndarray):
            Source z direction cosines.
        data_index (numpy.ndarray or cupy.ndarray):
            Optional pointer to indirection indices. May be None.
        data (numpy.ndarray or cupy.ndarray):
            Optional pointer to element response matrix. May be None.
        beam (numpy.ndarray or cupy.ndarray):
            Output complex beam array.
        normalise (bool):
            If true, normalise output by dividing by the number of elements.

    """
    Lib.sdp_station_beam_array_factor(
        wavenumber,
        Mem(element_weights),
        Mem(element_x),
        Mem(element_y),
        Mem(element_z),
        0,
        point_x.size,
        Mem(point_x),
        Mem(point_y),
        Mem(point_z),
        Mem(data_index),
        Mem(data),
        0,
        Mem(beam),
        normalise,
    )
