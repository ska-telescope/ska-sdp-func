# See the LICENSE file at the top-level directory of this distribution.

"""Module for RFI flagging functions."""

import ctypes

from ..utility import Lib, Mem

Lib.wrap_func(
    "sdp_sum_threshold_rfi_flagger",
    restype=None,
    argtypes=[
        Mem.handle_type(),
        Mem.handle_type(),
        Mem.handle_type(),
        ctypes.c_int64,
    ],
    check_errcode=True,
)


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
    Lib.sdp_sum_threshold_rfi_flagger(
        Mem(vis), Mem(thresholds), Mem(flags), max_sequence_length
    )
