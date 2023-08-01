# See the LICENSE file at the top-level directory of this distribution.

"""Module for RFI flagging functions."""

from .utility import Error, Lib, Mem


def twosm_rfi_flagger(vis, parameters, flags, antennas, baselines1, baselines2):
    """
    Basic RFI flagger based on sum-threshold algorithm.

    Array dimensions are as follows, from slowest to fastest varying:

    * ``vis`` is 4D and complex-valued, with shape:

      * [ num_timesamples, num_baselines, num_channels, num_polarisations ]

    * ``parameters`` is 1D and real-valued.

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
    mem_parameters = Mem(parameters)
    mem_flags = Mem(flags)
    mem_baselines1 = Mem(baselines1)
    mem_baselines2 = Mem(baselines2)
    mem_antennas = Mem(antennas)
    error_status = Error()
    lib_rfi_flagger = Lib.handle().sdp_twosm_algo_flagger
    lib_rfi_flagger.argtypes = [
        Mem.handle_type(),
        Mem.handle_type(),
        Mem.handle_type(),
        Mem.handle_type(),
        Mem.handle_type(),
        Mem.handle_type(),
        Error.handle_type(),
    ]
    lib_rfi_flagger(
        mem_vis.handle(),
        mem_parameters.handle(),
        mem_flags.handle(),
        mem_antennas.handle(),
        mem_baselines1.handle(),
        mem_baselines2.handle(),
        error_status.handle(),
    )
    error_status.check()
