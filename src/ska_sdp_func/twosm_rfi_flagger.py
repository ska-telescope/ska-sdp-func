# See the LICENSE file at the top-level directory of this distribution.

from .utility import Error, Lib, Mem
import ctypes


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

    mem_vis = Mem(vis)
    mem_thresholds = Mem(thresholds)
    mem_antennas = Mem(antennas)
    mem_flags = Mem(flags)
    error_status = Error()
    lib_rfi_flagger = Lib.handle().sdp_twosm_algo_flagger
    lib_rfi_flagger.argtypes = [
        Mem.handle_type(),
        Mem.handle_type(),
        Mem.handle_type(),
        Mem.handle_type(),
        Error.handle_type(),
    ]
    lib_rfi_flagger(
        mem_vis.handle(),
        mem_thresholds.handle(),
        mem_antennas.handle(),
        mem_flags.handle(),
        error_status.handle(),
    )
    error_status.check()
