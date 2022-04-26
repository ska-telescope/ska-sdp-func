# See the LICENSE file at the top-level directory of this distribution.

from .utility import Error, Lib, Mem

def degridding(input, output):
    """
    This is just a placeholder.
    
    It doesn't do anything yet, it will be replaced later.
    
    :param input: An input
    :type input: numpy.ndarray or cupy.ndarray
    
    :param output: The output
    :type output: numpy.ndarray or cupy.ndarray
    """
    mem_input = Mem(input)
    mem_output = Mem(output)
    error_status = Error()
    lib_degridding = Lib.handle().sdp_degridding
    lib_degridding.argtypes = [
        Mem.handle_type(),
        Mem.handle_type(),
        Error.handle_type()
    ]
    lib_degridding(
        mem_input.handle(),
        mem_output.handle(),
        error_status.handle()
    )
    error_status.check()