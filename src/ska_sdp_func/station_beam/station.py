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
    normalise: bool = True,
    eval_x: bool = True,
    eval_y: bool = True,
) -> None:
    """Evaluates a station beam from an aperture array.

    This function evaluates an aperture-array-based station beam for a
    given set of antenna/element coordinates, at a set of source positions.

    Element beam data can be supplied if required via the ``element_beam``
    parameter, and may be either complex scalar or fully polarised.
    If provided, this must be a complex matrix containing the element response
    for each element type and each direction on the sky.
    The matrix dimemsions of ``element_beam`` are therefore the number of
    element responses and the number of point directions, with the point
    directions the fastest-varying dimension.
    If there are fewer element types than there are elements, then
    the ``element_beam_index`` array should be used to specify which vector
    of the element response matrix is used for each element.
    If polarised element data is supplied, the output station beam is also
    polarised, and the fastest-varying dimension or dimensions of both must be
    either of size 4 or (2, 2).

    Example dimensions allowed for ``element_beam``, if not ``None``:

    - (``num_elements``, ``num_points``) for complex scalar responses.
    - (``num_elements``, ``num_points``, 1) for complex scalar responses.
    - (``num_elements``, ``num_points``, 4) for complex polarised responses.
    - (``num_elements``, ``num_points``, 2, 2) for complex polarised responses.

    The ``station_beam`` array must be consistent in shape, but without
    the element dimension.

    Example dimensions allowed for ``station_beam``:

    - (``num_points``) for complex scalar responses.
    - (``num_points``, 1) for complex scalar responses.
    - (``num_points``, 4) for complex polarised responses.
    - (``num_points``, 2, 2) for complex polarised responses.

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
