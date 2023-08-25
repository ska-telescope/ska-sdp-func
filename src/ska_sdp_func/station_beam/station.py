# See the LICENSE file at the top-level directory of this distribution.

"""Module for station beam functions."""

import ctypes

from ..utility import Lib, Mem

Lib.wrap_func(
    "sdp_station_beam_aperture_array",
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
        ctypes.c_int32,
        ctypes.c_int32,
    ],
    check_errcode=True,
)


def aperture_array(
    wavenumber: float,
    element_weights,
    element_x,
    element_y,
    element_z,
    point_x,
    point_y,
    point_z,
    element_beam_index,
    element_beam,
    station_beam,
    normalise: bool=True,
    eval_x: bool=True,
    eval_y: bool=True,
) -> None:
    """Evaluates a station beam from an aperture array.

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
        element_beam_index (numpy.ndarray or cupy.ndarray):
            Optional pointer to element beam indices. May be None.
        element_beam (numpy.ndarray or cupy.ndarray):
            Optional pointer to element beam matrix. May be None.
        station_beam (numpy.ndarray or cupy.ndarray):
            Output complex station beam array.
        normalise (bool):
            If true, normalise output by dividing by the number of elements.
        eval_x (bool):
            If true, evaluate polarised beam using X antennas.
        eval_y (bool):
            If true, evaluate polarised beam using Y antennas.

    """
    Lib.sdp_station_beam_aperture_array(
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
        Mem(element_beam_index),
        Mem(element_beam),
        0,
        Mem(station_beam),
        normalise,
        eval_x,
        eval_y,
    )
