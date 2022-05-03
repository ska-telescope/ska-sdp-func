# See the LICENSE file at the top-level directory of this distribution.

import ctypes
from .utility import Error, Lib, Mem

def degridding(
    grid,
    u0,
    v0,
    w0,
    theta,
    wstep,
    uv_kernel,
    uv_kernel_stride,
    uv_kernel_oversampling,
    w_kernel,
    w_kernel_stride,
    w_kernel_oversampling,
    conjugate,
    vis_out,
):
    """
    Degridding visabilities.

    Degrids a previously gridded visabilty, based on the UV and W kernel input and returns the result.

    :param grid: Input grid data
    :type grid: numpy.ndarray or cupy.ndarray

    :param u0: Index of first coordinate on U axis
    :type u0: int

    :param v0: Index of first coordinate on V axis
    :type v0: int

    :param w0: Index of first coordinate on W axis
    :type w0: int

    :param theta: Conversion parameter from uv coordinates to xy coordinates (i.e. x=u*theta)
    :type theta: int

    :param wstep: Conversion parameter from w coordinates to z coordinates (i.e. z=w*wstep)
    :type wstep: int

    :param uv_kernel: U,V plane kernel
    :type uv_kernel: numpy.ndarray or cupy.ndarray

    :param uv_kernel_stride: U,V plane kernel padding
    :type uv_kernel_stride: int

    :param uv_kernel_oversampling: U,V plane kernel oversampling
    :type uv_kernel_oversampling: int

    :param w_kernel: W plane Kernel
    :type w_kernel: numpy.ndarray or cupy.ndarray

    :param w_kernel_stride: W plane Kernel padding
    :type w_kernel_stride: int

    :param w_kernel_oversampling: W plane Kernel oversampling
    :type w_kernel_oversampling: int

    :param conjugate: Whether to generate conjugated visibilities
    :type conjugate: bool

    :param vis_out: Output Visabilities
    :type vis_out: numpy.ndarray or cupy.ndarray
    """
    mem_grid = Mem(grid)
    mem_uv_kernel = Mem(uv_kernel)
    mem_w_kernel = Mem(w_kernel)
    mem_vis_out = Mem(vis_out)
    error_status = Error()
    lib_degridding = Lib.handle().sdp_degridding
    lib_degridding.argtypes = [
        Mem.handle_type(),
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        Mem.handle_type(),
        ctypes.c_int64,
        ctypes.c_int64,
        Mem.handle_type(),
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_bool,
        Mem.handle_type(),
        Error.handle_type(),
    ]
    lib_degridding(
        mem_grid.handle(),
        ctypes.c_int64(u0),
        ctypes.c_int64(v0),
        ctypes.c_int64(w0),
        ctypes.c_int64(theta),
        ctypes.c_int64(wstep),
        mem_uv_kernel.handle(),
        ctypes.c_int64(uv_kernel),
        ctypes.c_int64(uv_kernel_oversampling),
        mem_w_kernel.handle(),
        ctypes.c_int64(w_kernel_stride),
        ctypes.c_int64(w_kernel_oversampling),
        ctypes.c_int64(conjugate),
        mem_vis_out.handle(),
        error_status.handle(),
    )
    error_status.check()
