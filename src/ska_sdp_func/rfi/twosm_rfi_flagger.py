# See the LICENSE file at the top-level directory of this distribution.

"""Module for RFI flagging functions."""

from ..utility import Lib, Mem


Lib.wrap_func(
    "sdp_twosm_algo_flagger",
    restype=None,
    argtypes=[
        Mem.handle_type(),
        Mem.handle_type(),
        Mem.handle_type(),
        Mem.handle_type(),
    ],
    check_errcode=True
)


def twosm_rfi_flagger(vis, thresholds, antennas, flags):
    """
    Basic RFI flagger based on sum-threshold algorithm.

    Array dimensions are as follows, from slowest to fastest varying:

    * ``vis`` is 4D and complex-valued, with shape:

      * [ num_timesamples, num_baselines, num_channels, num_polarisations ]

    * ``thresholds`` is 1D and real-valued.

      * The size of the array is 2``.

    * ``antennas`` is 1D and integer.

      * The size of the array is 2``.

    * ``flags`` is 4D and integer-valued, with the same shape as ``vis``.

    :param vis: Complex valued visibilities. Dimensions as above.
    :type vis: numpy.ndarray

    :param thresholds: Thresholds for first-order two-state machine model and
    extrapolation-based method

    :param flags: Output flags. Dimensions as above.
    :type flags: numpy.ndarray
    """
    Lib.sdp_twosm_algo_flagger(Mem(vis), Mem(thresholds), Mem(antennas), Mem(flags))
