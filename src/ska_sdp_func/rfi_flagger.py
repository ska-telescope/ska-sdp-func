# See the LICENSE file at the top-level directory of this distribution.

import ctypes
from .utility import Error, Lib, Mem



def sum_threshold_rfi_flagger(vis, thresholds, flags, max_sequence_length):
    """
    Function to perform RFI flagging using sum threshold algorithm 

    Parameters can be either numpy or cupy arrays, but must all be consistent,
    and with the same length and data type.
    Computation is performed either on the CPU or GPU as appropriate.

    
    Array dimensions must be as follows:

    * ``vis`` is 4D and complex-valued, with shape:
        * [ num_timesamples, num_baselines, num_channels, num_polarisations ]
      
    
    * ``thresholds`` is 1D and real valued, with shape:
        *[num_seq_len]
    
    * ``flags`` is 4D and integer-valued, with shape:
        * [ num_timesamples, num_baselines, num_channels, num_polarisations ]
    
    * ``max_sequence_length`` is integer defining size of the window
    
    :param vis: Visibility data.
    :type vis: numpy.ndarray or cupy.ndarray
     
    :param thresholds: Threshold for each sequence.
    :type thresholds: numpy.ndarray or cupy.ndarray

    :param flags: Output flags.
    :type flags: numpy.ndarray or cupy.ndarray
    
    :param max_sequence_length: Window size.
    :type max_sequence_length: integer

    """

    mem_vis = Mem(vis)
    mem_thresholds = Mem(thresholds)
    mem_flags = Mem(flags)
    error_status = Error()
    lib_rfi_flagger = Lib.handle().sdp_sum_threshold_rfi_flagger
    lib_rfi_flagger.argtypes = [
        Mem.handle_type(),
        Mem.handle_type(),
        Mem.handle_type(),
        ctypes.c_int,
        Error.handle_type()
    ]
    lib_rfi_flagger(
        mem_vis.handle(),
        mem_thresholds.handle(),
        mem_flags.handle(),
        ctypes.c_int(max_sequence_length),
        error_status.handle()
    )
    error_status.check()
