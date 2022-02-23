# See the LICENSE file at the top-level directory of this distribution.

from .utility import Error, Lib, Mem



def rfi_flagger(vis,sequence,thresholds,flags):
    """
    Function to perform RFI flagging using sum threshold algorithm 

    Parameters can be either numpy or cupy arrays, but must all be consistent,
    and with the same length and data type.
    Computation is performed either on the CPU or GPU as appropriate.

    :param vis: Visibility data.
    :type vis: numpy.ndarray or cupy.ndarray

    :param sequence: The length of continious channels used for flagging.
    :type sequence: numpy.ndarray or cupy.ndarray

    :param thresholds: Threshold for each sequence.
    :type thresholds: numpy.ndarray or cupy.ndarray

    :param flags: Output flags.
    :type flags: numpy.ndarray or cupy.ndarray

    """

    mem_vis = Mem(vis)
    mem_sequence = Mem(sequence)
    mem_thresholds = Mem(thresholds)
    mem_flags = Mem(flags)
    error_status = Error()
    lib_rfi_flagger = Lib.handle().sdp_rfi_flagger
    lib_rfi_flagger.argtypes = [
        Mem.handle_type(),
        Mem.handle_type(),
        Mem.handle_type(),
        Mem.handle_type(),
        Error.handle_type()
    ]
    lib_rfi_flagger(
        mem_vis.handle(),
        mem_sequence.handle(),
        mem_thresholds.handle(),
        mem_flags.handle(),
        error_status.handle()
    )
    error_status.check()
