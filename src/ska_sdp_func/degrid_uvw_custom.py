# See the LICENSE file at the top-level directory of this distribution.

import ctypes
from .utility import Error, Lib, Mem

def degrid_uvw_custom(
    grid,
    uvw,
    uv_kernel,
    w_kernel,
    uv_kernel_oversampling,
    w_kernel_oversampling,
    theta,
    wstep,
    channel_start_hz,
    channel_step_hz,
    conjugate,
    vis,
):
    """
    Degridding visibilities.

    Degrids a previously gridded visabilty, based on the UV and W kernel input and returns the result.

    :param grid: Input grid data with shape [chan][w][v][u][pol]
    :type grid: numpy.ndarray or cupy.ndarray

    :param uvw: u,v,w coordinates of the visibilities with shape [time][baseline][chan][uvw]
    :type vis_coordinates: numpy.ndarray or cupy.ndarray
    
    :param uv_kernel: u,v plane kernel
    :type uv_kernel: numpy.ndarray or cupy.ndarray

    :param w_kernel: w plane Kernel
    :type w_kernel: numpy.ndarray or cupy.ndarray

    :param uv_kernel_oversampling: u,v plane kernel oversampling
    :type uv_kernel_oversampling: int

    :param w_kernel_oversampling: w plane Kernel oversampling
    :type w_kernel_oversampling: int

    :param theta: Conversion parameter from uv coordinates to xy coordinates (i.e. x=u*theta)
    :type theta: float

    :param wstep: Conversion parameter from w coordinates to z coordinates (i.e. z=w*wstep)
    :type wstep: float

    :param channel_start_hz: Frequency of first channel, in Hz.
    :type channel_start_hz: float

    :param channel_step_hz: Frequency increment between channels, in Hz.
    :type channel_step_hz: float

    :param conjugate: Whether to generate conjugated visibilities
    :type conjugate: bool

    :param vis: Output Visabilities with shape [time][baseline][chan][pol]
    :type vis: numpy.ndarray or cupy.ndarray
    """
    mem_grid = Mem(grid)
    mem_vis_coordinates = Mem(uvw)
    mem_uv_kernel = Mem(uv_kernel)
    mem_w_kernel = Mem(w_kernel)
    mem_vis = Mem(vis)
    error_status = Error()
    lib_degridding = Lib.handle().sdp_degrid_uvw_custom
    lib_degridding.argtypes = [
        Mem.handle_type(),
        Mem.handle_type(),
        Mem.handle_type(),
        Mem.handle_type(),
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_bool,
        Mem.handle_type(),
        Error.handle_type(),
    ]
    lib_degridding(
        mem_grid.handle(),
        mem_vis_coordinates.handle(),
        mem_uv_kernel.handle(),
        mem_w_kernel.handle(),
        ctypes.c_int64(uv_kernel_oversampling),
        ctypes.c_int64(w_kernel_oversampling),
        ctypes.c_double(theta),
        ctypes.c_double(wstep),
        ctypes.c_double(channel_start_hz),
        ctypes.c_double(channel_step_hz),
        ctypes.c_bool(conjugate),
        mem_vis.handle(),
        error_status.handle(),
    )
    error_status.check()
