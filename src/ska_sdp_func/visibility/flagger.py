# See the LICENSE file at the top-level directory of this distribution.

"""Module for RFI flagging functions."""

from ..utility import Lib, Mem

Lib.wrap_func(
    "sdp_flagger_fixed_threshold",
    restype=None,
    argtypes=[
        Mem.handle_type(),
        Mem.handle_type(),
        Mem.handle_type(),
    ],
    check_errcode=True,
)

Lib.wrap_func(
    "sdp_flagger_dynamic_threshold",
    restype=None,
    argtypes=[
        Mem.handle_type(),
        Mem.handle_type(),
        Mem.handle_type(),
    ],
    check_errcode=True,
)


def flagger_fixed_threshold(vis, parameters, flags):
    """
    Basic RFI flagger based on sum-threshold algorithm.

    Array dimensions are as follows, from slowest to fastest varying:

    * ``vis`` is 4D and complex-valued, with shape:

      * [ num_timesamples, num_baselines, num_channels, num_polarisations ]

    * ``parameters`` is 1D and real-valued.

      * The size of the array is 5``.

    * ``flags`` is 4D and integer-valued, with the same shape as ``vis``.

    :param parameters:
    :param vis: Complex valued visibilities. Dimensions as above.
    :type vis: numpy.ndarray

    :param thresholds: Thresholds for first-order two-state machine model and
    extrapolation-based method

    :param flags: Output flags. Dimensions as above.
    :type flags: numpy.ndarray
    """

    Lib.sdp_flagger_fixed_threshold(
        Mem(vis),
        Mem(parameters),
        Mem(flags),
    )


def flagger_dynamic_threshold(vis, parameters, flags):
    """
    Basic RFI flagger based on sum-threshold algorithm.

    Array dimensions are as follows, from slowest to fastest varying:

    * ``vis`` is 4D and complex-valued, with shape:

      * [ num_timesamples, num_baselines, num_channels, num_polarisations ]

    * ``parameters`` is 1D and real-valued.

      * The size of the array is 5``.

    * ``flags`` is 4D and integer-valued, with the same shape as ``vis``.

    :param parameters:
    :param vis: Complex valued visibilities. Dimensions as above.
    :type vis: numpy.ndarray

    :param thresholds: Thresholds for first-order two-state machine model and
    extrapolation-based method

    :param flags: Output flags. Dimensions as above.
    :type flags: numpy.ndarray
    """

    Lib.sdp_flagger_dynamic_threshold(
        Mem(vis),
        Mem(parameters),
        Mem(flags),
    )
