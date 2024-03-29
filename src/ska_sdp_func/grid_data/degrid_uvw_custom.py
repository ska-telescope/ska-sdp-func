# See the LICENSE file at the top-level directory of this distribution.

"""Module for degridding functions which use custom kernels."""

import ctypes

from ..utility import Lib, Mem


def degrid_uvw_custom(
    grid,
    uvw,
    uv_kernel,
    w_kernel,
    theta,
    wstep,
    channel_start_hz,
    channel_step_hz,
    conjugate,
    vis,
):
    """
    Degrid visibilities.

    Degrids previously gridded visibilities using supplied convolution kernels.

    :param grid: Input grid data with shape [chan][w][v][u][pol]
    :type grid: numpy.ndarray or cupy.ndarray

    :param uvw: Visibility (u,v,w) coordinates with shape [time][baseline][3]
    :type uvw: numpy.ndarray or cupy.ndarray

    :param uv_kernel: (u,v)-plane kernel with shape [oversampling][stride]
    :type uv_kernel: numpy.ndarray or cupy.ndarray

    :param w_kernel: w-plane kernel with shape [oversampling][stride]
    :type w_kernel: numpy.ndarray or cupy.ndarray

    :param theta: Conversion parameter from (u,v)-coordinates
     to (x,y)-coordinates (i.e. x=u*theta)
    :type theta: float

    :param wstep: Conversion parameter from w-coordinates
     to z-coordinates (i.e. z=w*wstep)
    :type wstep: float

    :param channel_start_hz: Frequency of first channel, in Hz.
    :type channel_start_hz: float

    :param channel_step_hz: Frequency increment between channels, in Hz.
    :type channel_step_hz: float

    :param conjugate: Whether to generate conjugated visibilities
    :type conjugate: bool

    :param vis: Output visibilities with shape [time][baseline][chan][pol]
    :type vis: numpy.ndarray or cupy.ndarray
    """
    Lib.sdp_degrid_uvw_custom(
        Mem(grid),
        Mem(uvw),
        Mem(uv_kernel),
        Mem(w_kernel),
        theta,
        wstep,
        channel_start_hz,
        channel_step_hz,
        conjugate,
        Mem(vis),
    )


Lib.wrap_func(
    "sdp_degrid_uvw_custom",
    restype=None,
    argtypes=[
        Mem.handle_type(),
        Mem.handle_type(),
        Mem.handle_type(),
        Mem.handle_type(),
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_int32,
        Mem.handle_type(),
    ],
    check_errcode=True,
)
