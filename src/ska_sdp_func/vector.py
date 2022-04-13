# See the LICENSE file at the top-level directory of this distribution.

from .utility import Error, Lib, Mem

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
    mem_input_a = Mem(input_a)
    mem_input_b = Mem(input_b)
    mem_output = Mem(output)
    error_status = Error()
    lib_vector_add = Lib.handle().sdp_vector_add
    lib_vector_add.argtypes = [
        Mem.handle_type(),
        Mem.handle_type(),
        Mem.handle_type(),
        Error.handle_type()
    ]
    lib_vector_add(
        mem_input_a.handle(),
        mem_input_b.handle(),
        mem_output.handle(),
        error_status.handle()
    )
    error_status.check()
