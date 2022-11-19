# See the LICENSE file at the top-level directory of this distribution.

"""Module for phase rotation functions."""

import ctypes

from .utility import Error, Lib, Mem, SkyCoord


def phase_rotate_uvw(
    phase_centre_orig,
    phase_centre_new,
    uvw_in,
    uvw_out,
):
    """Rotate (u,v,w) coordinates to a new phase centre.

    Parameters can be either numpy or cupy arrays, but must all be consistent.
    Computation is performed either on the CPU or GPU as appropriate.

    Input parameters ``uvw_in`` and ``uvw_out`` are arrays
    of packed 3D coordinates.

    Array dimensions must be as follows:

    * ``uvw_in`` is 3D and real-valued, with shape:

      * [ num_times, num_baselines, 3 ]

    * ``uvw_out`` is 3D and real-valued, with shape:

      * [ num_times, num_baselines, 3 ]

    :param phase_centre_orig: Original phase centre.
    :type phase_centre_orig: SkyCoord, or astropy.coordinates.SkyCoord

    :param phase_centre_new: New phase centre.
    :type phase_centre_new: SkyCoord, or astropy.coordinates.SkyCoord

    :param uvw_in: Input baseline (u,v,w) coordinates.
    :type uvw_in: numpy.ndarray or cupy.ndarray

    :param uvw_out: Output baseline (u,v,w) coordinates.
    :type uvw_in: numpy.ndarray or cupy.ndarray
    """
    phase_centre_orig = SkyCoord(phase_centre_orig)
    phase_centre_new = SkyCoord(phase_centre_new)
    mem_uvw_in = Mem(uvw_in)
    mem_uvw_out = Mem(uvw_out)
    error_status = Error()
    lib_rotate_uvw = Lib.handle().sdp_phase_rotate_uvw
    lib_rotate_uvw.argtypes = [
        SkyCoord.handle_type(),
        SkyCoord.handle_type(),
        Mem.handle_type(),
        Mem.handle_type(),
        Error.handle_type(),
    ]
    lib_rotate_uvw(
        phase_centre_orig.handle(),
        phase_centre_new.handle(),
        mem_uvw_in.handle(),
        mem_uvw_out.handle(),
        error_status.handle(),
    )
    error_status.check()


def phase_rotate_vis(
    phase_centre_orig,
    phase_centre_new,
    channel_start_hz,
    channel_step_hz,
    uvw,
    vis_in,
    vis_out,
):
    """Rotate visibilities to a new phase centre.

    Parameters can be either numpy or cupy arrays, but must all be consistent.
    Computation is performed either on the CPU or GPU as appropriate.

    Input parameter ``uvw`` is an array of packed 3D coordinates.

    Array dimensions must be as follows:

    * ``uvw`` is 3D and real-valued, with shape:

      * [ num_times, num_baselines, 3 ]

    * ``vis_in`` is 4D and complex-valued, with shape:

      * [ num_times, num_baselines, num_channels, num_pols ]

    * ``vis_out`` is 4D and complex-valued, with shape:

      * [ num_times, num_baselines, num_channels, num_pols ]

    :param phase_centre_orig: Original phase centre.
    :type phase_centre_orig: SkyCoord, or astropy.coordinates.SkyCoord

    :param phase_centre_new: New phase centre.
    :type phase_centre_new: SkyCoord, or astropy.coordinates.SkyCoord

    :param channel_start_hz: Frequency of first channel, in Hz.
    :type channel_start_hz: float

    :param channel_step_hz: Frequency incremenet between channels, in Hz.
    :type channel_step_hz: float

    :param uvw: Original baseline (u,v,w) coordinates, in metres.
    :type uvw: numpy.ndarray or cupy.ndarray

    :param vis_in: Input visibility data.
    :type vis_in: numpy.ndarray or cupy.ndarray

    :param vis_out: Output visibility data.
    :type vis_out: numpy.ndarray or cupy.ndarray
    """
    phase_centre_orig = SkyCoord(phase_centre_orig)
    phase_centre_new = SkyCoord(phase_centre_new)
    mem_uvw = Mem(uvw)
    mem_vis_in = Mem(vis_in)
    mem_vis_out = Mem(vis_out)
    error_status = Error()
    lib_rotate_vis = Lib.handle().sdp_phase_rotate_vis
    lib_rotate_vis.argtypes = [
        SkyCoord.handle_type(),
        SkyCoord.handle_type(),
        ctypes.c_double,
        ctypes.c_double,
        Mem.handle_type(),
        Mem.handle_type(),
        Mem.handle_type(),
        Error.handle_type(),
    ]
    lib_rotate_vis(
        phase_centre_orig.handle(),
        phase_centre_new.handle(),
        ctypes.c_double(channel_start_hz),
        ctypes.c_double(channel_step_hz),
        mem_uvw.handle(),
        mem_vis_in.handle(),
        mem_vis_out.handle(),
        error_status.handle(),
    )
    error_status.check()