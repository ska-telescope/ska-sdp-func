# See the LICENSE file at the top-level directory of this distribution.
"""Module for (de)gridding utility functions used by w-towers gridders."""

import ctypes

import numpy

from ..utility import Lib, Mem


def clamp_channels_single(
    uvws: numpy.ndarray,
    dim: int,
    freq0_hz: float,
    dfreq_hz: float,
    start_ch: numpy.ndarray,
    end_ch: numpy.ndarray,
    min_u: float,
    max_u: float,
):
    """
    Clamp channels for a single dimension of an array of uvw coordinates.

    Restricts a channel range such that all visibilities lie in
    the given range in u or v or w.

    :param uvws:
        ``float[uvw_count, 3]`` UVW coordinates of visibilities (in m).
    :param dim: Dimension index (0, 1 or 2) of uvws to check.
    :param freq0_hz: Frequency of first channel, in Hz.
    :param dfreq_hz: Channel width, in Hz.
    :param start_ch: Channel range to clamp (excluding end).
    :param end_ch: Channel range to clamp (excluding end).
    :param min_u: Minimum value for u or v or w (inclusive).
    :param max_u: Maximum value for u or v or w (exclusive).
    """
    Lib.sdp_gridder_clamp_channels_single(
        Mem(uvws),
        dim,
        freq0_hz,
        dfreq_hz,
        Mem(start_ch),
        Mem(end_ch),
        min_u,
        max_u,
    )


def make_kernel(window: numpy.ndarray, kernel: numpy.ndarray):
    """
    Convert image-space window function to oversampled kernel.

    This uses a DFT to do the transformation to Fourier space.
    The supplied input window function is 1-dimensional, with a shape
    of (support), and the output kernel is 2-dimensional, with a shape
    of (oversample + 1, support), and it must be sized appropriately on entry.

    :param window: Image-space window function in 1D.
    :param kernel: Oversampled output kernel in Fourier space.
    """
    Lib.sdp_gridder_make_kernel(Mem(window), Mem(kernel))


def make_pswf_kernel(support: int, kernel: numpy.ndarray):
    """
    Generate an oversampled kernel using PSWF window function.

    This uses a DFT to do the transformation to Fourier space.
    The output kernel is 2-dimensional, with a shape
    of (oversample + 1, support), and it must be sized appropriately on entry.

    :param support: Kernel support size.
    :param kernel: Oversampled output kernel in Fourier space.
    """
    Lib.sdp_gridder_make_pswf_kernel(support, Mem(kernel))


def make_w_pattern(
    subgrid_size: int,
    theta: float,
    shear_u: float,
    shear_v: float,
    w_step: float,
    w_pattern: numpy.ndarray,
):
    """
    Generate w-pattern.

    This is the iDFT of a single visibility at (0, 0, w).

    :param subgrid_size: Subgrid size.
    :param theta: Total image size in direction cosines.
    :param shear_u: Shear parameter in u.
    :param shear_v: Shear parameter in v.
    :param w_step: Distance between w-planes.
    :param w_pattern:
        Complex w-pattern, dimensions (subgrid_size, subgrid_size).
    """
    Lib.sdp_gridder_make_w_pattern(
        subgrid_size, theta, shear_u, shear_v, w_step, Mem(w_pattern)
    )


def subgrid_add(
    grid: numpy.ndarray,
    offset_u: int,
    offset_v: int,
    subgrid: numpy.ndarray,
    factor: float = 1.0,
):
    """
    Add the supplied sub-grid to the grid.

    :param grid: Output grid.
    :param offset_u: Offset in u.
    :param offset_v: Offset in v.
    :param subgrid: Input sub-grid.
    :param factor:
        Factor by which to multiply elements of sub-grid before adding.
    """
    Lib.sdp_gridder_subgrid_add(
        Mem(grid), offset_u, offset_v, Mem(subgrid), factor
    )


def subgrid_cut_out(
    grid: numpy.ndarray,
    offset_u: int,
    offset_v: int,
    subgrid: numpy.ndarray,
):
    """
    Cut out a sub-grid from the supplied grid.

    :param grid: Input grid.
    :param offset_u: Offset in u.
    :param offset_v: Offset in v.
    :param subgrid: Output sub-grid.
    """
    Lib.sdp_gridder_subgrid_cut_out(
        Mem(grid), offset_u, offset_v, Mem(subgrid)
    )


def uvw_bounds_all(
    uvws: numpy.ndarray,
    freq0_hz: float,
    dfreq_hz: float,
    start_ch: numpy.ndarray,
    end_ch: numpy.ndarray,
):
    """
    Determine (scaled) min and max values in uvw coordinates.

    :param uvws:
        ``float[uvw_count, 3]`` UVW coordinates of visibilities (in m).
    :param freq0_hz: Frequency of first channel, in Hz.
    :param dfreq_hz: Channel width, in Hz.
    :param start_ch: First channel to degrid for every uvw.
    :param end_ch: Channel at which to stop degridding for every uvw.

    :return (uvw_min, uvw_max): A tuple of two 3-element lists
        containing minimum and maximum (u,v,w)-values.
    """
    min_uvw = (ctypes.c_double * 3)(0.0, 0.0, 0.0)
    max_uvw = (ctypes.c_double * 3)(0.0, 0.0, 0.0)
    Lib.sdp_gridder_uvw_bounds_all(
        Mem(uvws),
        freq0_hz,
        dfreq_hz,
        Mem(start_ch),
        Mem(end_ch),
        min_uvw,
        max_uvw,
    )
    return (min_uvw, max_uvw)


Lib.wrap_func(
    "sdp_gridder_clamp_channels_single",
    restype=None,
    argtypes=[
        Mem.handle_type(),
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_double,
        Mem.handle_type(),
        Mem.handle_type(),
        ctypes.c_double,
        ctypes.c_double,
    ],
    check_errcode=True,
)

Lib.wrap_func(
    "sdp_gridder_make_kernel",
    restype=None,
    argtypes=[Mem.handle_type(), Mem.handle_type()],
    check_errcode=True,
)

Lib.wrap_func(
    "sdp_gridder_make_pswf_kernel",
    restype=None,
    argtypes=[ctypes.c_int, Mem.handle_type()],
    check_errcode=True,
)

Lib.wrap_func(
    "sdp_gridder_make_w_pattern",
    restype=None,
    argtypes=[
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        Mem.handle_type(),
    ],
    check_errcode=True,
)

Lib.wrap_func(
    "sdp_gridder_subgrid_add",
    restype=None,
    argtypes=[
        Mem.handle_type(),
        ctypes.c_int,
        ctypes.c_int,
        Mem.handle_type(),
        ctypes.c_double,
    ],
    check_errcode=True,
)

Lib.wrap_func(
    "sdp_gridder_subgrid_cut_out",
    restype=None,
    argtypes=[
        Mem.handle_type(),
        ctypes.c_int,
        ctypes.c_int,
        Mem.handle_type(),
    ],
    check_errcode=True,
)

Lib.wrap_func(
    "sdp_gridder_uvw_bounds_all",
    restype=None,
    argtypes=[
        Mem.handle_type(),
        ctypes.c_double,
        ctypes.c_double,
        Mem.handle_type(),
        Mem.handle_type(),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
    ],
    check_errcode=True,
)
