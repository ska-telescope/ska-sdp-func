# See the LICENSE file at the top-level directory of this distribution.

"""Module for example function."""

from .utility import Error, Lib, Mem, SkyCoord


def sky_function(sky_coordinates):
    """
    Simple example to add two vectors, element_wise.

    Parameters can be either numpy or cupy arrays, but must all be consistent,
    and with the same length and data type.
    Computation is performed either on the CPU or GPU as appropriate.

    :param input_a: First input vector.
    :type input_a: numpy.ndarray or cupy.ndarray

    :param input_b: Second input vector.
    :type input_b: numpy.ndarray or cupy.ndarray

    :param output: Output vector.
    :type output: numpy.ndarray or cupy.ndarray
    """
    error_status = Error()
    lib_vector_add = Lib.handle().sdp_sky_coordinate_test
    lib_vector_add.argtypes = [
        SkyCoord.handle_type(),
        Error.handle_type(),
    ]
    lib_vector_add(
        sky_coordinates.handle(),
        error_status.handle(),
    )
    error_status.check()
