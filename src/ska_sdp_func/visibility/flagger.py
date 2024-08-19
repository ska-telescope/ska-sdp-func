# See the LICENSE file at the top-level directory of this distribution.

"""Module for RFI flagging functions."""
import ctypes

from ..utility import Lib, Mem

Lib.wrap_func(
    "sdp_flagger_dynamic_threshold",
    restype=None,
    argtypes=[
        Mem.handle_type(),
        Mem.handle_type(),
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ],
    check_errcode=True,
)


def flagger_dynamic_threshold(
    vis,
    flags,
    alpha,
    threshold_magnitudes,
    threshold_variations,
    threshold_broadband,
    sampling_step,
    window,
    window_median_history,
):
    """
    A leightweight RFI flagger to statistically flag the unusually
    larger absolute values of visibilities, the unusually fluctuating
    absolute values, and unusually large collection of channels with a
    sudden jump in their absolute values to detect broadband RFI

    Array dimensions are as follows, from slowest to fastest varying:

    * ``vis`` is 4D and complex-valued, with shape:

      * [ num_timesamples, num_baselines, num_channels, num_polarisations ]


    * ``flags`` is 4D and integer-valued, with the same shape as ``vis``.

    :param parameters:
    :param vis: Complex valued visibilities. Dimensions as above.
    :type vis: numpy.ndarray

    :param parameters: A numpy array containing the parameters for the
    flagger in the following order.

    [alpha, threshold_magnitudes, threshold_variations,
    threshold_broadband, sampling_step, window, window_median_history]

    :param alpha: the coefficient for the recursive equation for measuring
    the rate of fluctuations in the 'recent time samples', between
    0 and 1.
    :param threshold_magnitudes: the threshold on modified z-score to flag
    based on magnitudes of visibilities. A recommended rule of theumb
    in the statistics textbooks for this value is 3.5.

    :param threshold_variations: the threshold on modified z-score to flag
    based on the rate of fluctuations in the magnitudes in the recent
    time samples (similar to last one, 3.5 is an appropriate value)

    :param threshold_broadband: A threshold on the modified z-score of the
    medians of each time slot across all channels to detect broadband
    RFI.

    :param sampling_step: integer value, shows at how many steps we take
    sample to compute the medians and z-scores across all channels
    in a time slot.

    :param window: is the number of side channels of a flagged visibility
    on each side to be flagged

    :param window_median_history: the window size that the history of
    the medians of time slots is maintained for the broadband RFI
    detection.

    :param flags: Output flags. Dimensions as above.
    :type flags: numpy.ndarray
    """

    Lib.sdp_flagger_dynamic_threshold(
        Mem(vis),
        Mem(flags),
        alpha,
        threshold_magnitudes,
        threshold_variations,
        threshold_broadband,
        sampling_step,
        window,
        window_median_history,
    )
