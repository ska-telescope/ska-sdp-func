# See the LICENSE file at the top-level directory of this distribution.

"""Module for RFI flagging functions."""

import ctypes

from ..utility import Error, Lib, Mem


def sum_threshold_rfi_flagger(vis, thresholds, flags, max_sequence_length):
    """
    Basic RFI flagger based on sum-threshold algorithm.

    Array dimensions are as follows, from slowest to fastest varying:

    * ``vis`` is 4D and complex-valued, with shape:

      * [ num_timesamples, num_baselines, num_channels, num_polarisations ]

    * ``thresholds`` is 1D and real-valued.

      * The size of the array is n, where 2^(n-1) = ``max_sequence_length``.

    * ``flags`` is 4D and integer-valued, with the same shape as ``vis``.

    * ``max_sequence_length`` is the maximum length of the sum performed
      by the algorithm.

    :param vis: Complex valued visibilities. Dimensions as above.
    :type vis: numpy.ndarray

    :param thresholds: List of thresholds, one for each sequence length.
    :type thresholds: numpy.ndarray

    :param flags: Output flags. Dimensions as above.
    :type flags: numpy.ndarray

    :param max_sequence_length: Maximum length of the partial sum.
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
        ctypes.c_int64,
        Error.handle_type(),
    ]
    lib_rfi_flagger(
        mem_vis.handle(),
        mem_thresholds.handle(),
        mem_flags.handle(),
        ctypes.c_int64(max_sequence_length),
        error_status.handle(),
    )
    error_status.check()
