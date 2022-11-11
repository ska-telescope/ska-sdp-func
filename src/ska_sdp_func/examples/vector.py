# See the LICENSE file at the top-level directory of this distribution.

"""Module for example function."""

from ..utility import Lib, Mem

Lib.wrap_func(
    "sdp_vector_add",
    restype=None,
    argtypes=[
        Mem.handle_type(),
        Mem.handle_type(),
        Mem.handle_type(),
    ],
    check_errcode=True,
)


def vector_add(input_a, input_b, output):
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
    Lib.sdp_vector_add(Mem(input_a), Mem(input_b), Mem(output))
