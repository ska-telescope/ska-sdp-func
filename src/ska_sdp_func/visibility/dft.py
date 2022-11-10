# See the LICENSE file at the top-level directory of this distribution.

"""Module for DFT functions."""

import ctypes

from ..utility import Lib, Mem


Lib.wrap_func(
    "sdp_dft_point_v00",
    restype=None,
    argtypes=[
        Mem.handle_type(),
        Mem.handle_type(),
        Mem.handle_type(),
        Mem.handle_type(),
    ],
    check_errcode=True,
)


Lib.wrap_func(
    "sdp_dft_point_v01",
    restype=None,
    argtypes=[
        Mem.handle_type(),
        Mem.handle_type(),
        Mem.handle_type(),
        ctypes.c_double,
        ctypes.c_double,
        Mem.handle_type(),
    ],
    check_errcode=True,
)


def dft_point_v00(source_directions, source_fluxes, uvw_lambda, vis):
    """Basic prediction of visibilities from point sources using a DFT.

    .. deprecated:: 0.0.3
       Use :func:`dft_point_v01` instead.

    This version of the function is compatible with the memory layout of
    arrays used by RASCIL.

    Parameters can be either numpy or cupy arrays, but must all be consistent.
    Computation is performed either on the CPU or GPU as appropriate.

    Input parameters ``source_directions`` and ``uvw_lambda`` are arrays
    of packed 3D coordinates.

    Array dimensions must be as follows:

    * ``source_directions`` is 2D and real-valued, with shape:

      * [ num_components, 3 ]

    * ``source_fluxes`` is 3D and complex-valued, with shape:

      * [ num_components, num_channels, num_pols ]

    * ``uvw_lambda`` is 4D and real-valued, with shape:

      * [ num_times, num_baselines, num_channels, 3 ]

    * ``vis`` is 4D and complex-valued, with shape:

      * [ num_times, num_baselines, num_channels, num_pols ]

    :param source_directions: Source direction cosines.
    :type source_directions: numpy.ndarray or cupy.ndarray

    :param source_fluxes: Complex source flux values.
    :type source_fluxes: numpy.ndarray or cupy.ndarray

    :param uvw_lambda: Baseline (u,v,w) coordinates, in wavelengths.
    :type uvw_lambda: numpy.ndarray or cupy.ndarray

    :param vis: Output complex visibilities.
    :type vis: numpy.ndarray or cupy.ndarray
    """
    Lib.sdp_dft_point_v00(
        Mem(source_directions), Mem(source_fluxes), Mem(uvw_lambda), Mem(vis)
    )


def dft_point_v01(
    source_directions,
    source_fluxes,
    uvw,
    channel_start_hz,
    channel_step_hz,
    vis,
):
    """Basic prediction of visibilities from point sources using a DFT.

    Parameters can be either numpy or cupy arrays, but must all be consistent.
    Computation is performed either on the CPU or GPU as appropriate.

    Input parameters ``source_directions`` and ``uvw`` are arrays
    of packed 3D coordinates.

    Array dimensions must be as follows:

    * ``source_directions`` is 2D and real-valued, with shape:

      * [ num_components, 3 ]

    * ``source_fluxes`` is 3D and complex-valued, with shape:

      * [ num_components, num_channels, num_pols ]

    * ``uvw`` is 3D and real-valued, with shape:

      * [ num_times, num_baselines, 3 ]

    * ``vis`` is 4D and complex-valued, with shape:

      * [ num_times, num_baselines, num_channels, num_pols ]

    :param source_directions: Source direction cosines.
    :type source_directions: numpy.ndarray or cupy.ndarray

    :param source_fluxes: Complex source flux values.
    :type source_fluxes: numpy.ndarray or cupy.ndarray

    :param uvw: Baseline (u,v,w) coordinates, in metres.
    :type uvw: numpy.ndarray or cupy.ndarray

    :param channel_start_hz: Frequency of first channel, in Hz.
    :type channel_start_hz: float

    :param channel_step_hz: Frequency increment between channels, in Hz.
    :type channel_step_hz: float

    :param vis: Output complex visibilities.
    :type vis: numpy.ndarray or cupy.ndarray
    """
    Lib.sdp_dft_point_v01(
        Mem(source_directions),
        Mem(source_fluxes),
        Mem(uvw),
        channel_start_hz,
        channel_step_hz,
        Mem(vis),
    )
