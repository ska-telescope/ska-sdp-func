# See the LICENSE file at the top-level directory of this distribution.
"""Module for (de)gridding utility functions used by w-towers gridders."""

import ctypes
from typing import Optional

import numpy

from ..utility import Lib, Mem
from .gridder_wtower_uvw import GridderWtowerUVW


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


def clamp_channels_uv(
    uvws: numpy.ndarray,
    freq0_hz: float,
    dfreq_hz: float,
    start_ch: numpy.ndarray,
    end_ch: numpy.ndarray,
    min_u: float,
    max_u: float,
    min_v: float,
    max_v: float,
):
    """
    Clamp channels for (u,v) in an array of uvw coordinates.

    Restricts a channel range such that all visibilities lie in
    the given range in u and v.

    :param uvws:
        ``float[uvw_count, 3]`` UVW coordinates of visibilities (in m).
    :param freq0_hz: Frequency of first channel, in Hz.
    :param dfreq_hz: Channel width, in Hz.
    :param start_ch: Channel range to clamp (excluding end).
    :param end_ch: Channel range to clamp (excluding end).
    :param min_u: Minimum value for u (inclusive).
    :param max_u: Maximum value for u (exclusive).
    :param min_v: Minimum value for v (inclusive).
    :param max_v: Maximum value for v (exclusive).
    """
    Lib.sdp_gridder_clamp_channels_uv(
        Mem(uvws),
        freq0_hz,
        dfreq_hz,
        Mem(start_ch),
        Mem(end_ch),
        min_u,
        max_u,
        min_v,
        max_v,
    )


def determine_max_w_tower_height(
    subgrid_size: int,
    theta: float,
    fov: float,
    w_step: float,
    support: int,
    oversampling: int,
    w_support: int,
    w_oversampling: int,
    image_size: Optional[int] = None,
    shear_u: float = 0.0,
    shear_v: float = 0.0,
    subgrid_frac: float = 2.0 / 3.0,
    num_samples: int = 3,
    target_err: Optional[float] = None,
):
    """
    Find maximum w-tower height of a given configuration by trial-and-error.

    This is the same as :func:`find_max_w_tower_height()`,
    but without needing to create and supply a gridder kernel.

    :param subgrid_size: Sub-grid size in pixels.
    :param theta: Total image size in direction cosines.
    :param fov: Un-padded image field of view, in direction cosines.
    :param w_step: Spacing between w-planes.
    :param support: Kernel support size in (u, v).
    :param oversampling: Oversampling factor for uv-kernel.
    :param w_support: Support size in w.
    :param w_oversampling: Oversampling factor for w-kernel.
    :param image_size: Size of test image, in pixels.
        If not specified, defaults to twice the subgrid size.
    :param shear_u: Shear parameter in u (use zero for no shear).
    :param shear_v: Shear parameter in v (use zero for no shear).
    :param subgrid_frac: Fraction of sub-grid to use.
    :param num_samples: Number of sample points to test in u and v directions.
    :param target_err: Target error to use.
        If None, it is determined automatically.
    """
    if not image_size:
        image_size = 2 * subgrid_size
    if not target_err:
        target_err = 0.0
    return Lib.sdp_gridder_determine_max_w_tower_height(
        image_size,
        subgrid_size,
        theta,
        w_step,
        shear_u,
        shear_v,
        support,
        oversampling,
        w_support,
        w_oversampling,
        fov,
        subgrid_frac,
        num_samples,
        target_err,
    )


def determine_w_step(
    theta: float,
    fov: float,
    shear_u: float = 0.0,
    shear_v: float = 0.0,
    x_0: Optional[float] = None,
):
    """
    Determine a value for the w_step parameter.

    :param theta: Size of padded field of view, in direction cosines.
    :param fov: Size of imaged field of view, in direction cosines.
    :param shear_u: Shear parameter in u (use zero for no shear).
    :param shear_v: Shear parameter in v (use zero for no shear).
    :param x_0: Scaling factor for fov_n; defaults to fov / theta.
    """
    if not x_0:
        x_0 = 0.0
    return float(
        Lib.sdp_gridder_determine_w_step(theta, fov, shear_u, shear_v, x_0)
    )


def find_max_w_tower_height(
    grid_kernel: GridderWtowerUVW,
    fov: float,
    subgrid_frac: float = 2.0 / 3.0,
    num_samples: int = 3,
    target_err: Optional[float] = None,
):
    """
    Find maximum w-tower height of a given configuration by trial-and-error.

    This matches the interface with the function in the notebook.

    :param grid_kernel: Gridder kernel to use for the evaluation.
    :param fov: Un-padded image field of view, in direction cosines.
    :param subgrid_frac: Fraction of sub-grid to use.
    :param num_samples: Number of sample points to test in u and v directions.
    :param target_err: Target error to use.
        If None, it is determined automatically.
    """
    if not target_err:
        target_err = 0.0
    return Lib.sdp_gridder_determine_max_w_tower_height(
        grid_kernel.image_size,
        grid_kernel.subgrid_size,
        grid_kernel.theta,
        grid_kernel.w_step,
        grid_kernel.shear_u,
        grid_kernel.shear_v,
        grid_kernel.support,
        grid_kernel.oversampling,
        grid_kernel.w_support,
        grid_kernel.w_oversampling,
        fov,
        subgrid_frac,
        num_samples,
        target_err,
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


def rms_diff(array_a: numpy.ndarray, array_b: numpy.ndarray):
    """
    Returns the RMS of the difference between two 2D arrays: rms(a - b).

    The two arrays must be 2D and have the same shape.

    :param array_a: The first input array.
    :param array_b: The second input array.
    """
    return Lib.sdp_gridder_rms_diff(Mem(array_a), Mem(array_b))


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

    :return: A tuple of two 3-element lists
        containing minimum and maximum (u,v,w)-values: (uvw_min, uvw_max)
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
    "sdp_gridder_clamp_channels_uv",
    restype=None,
    argtypes=[
        Mem.handle_type(),
        ctypes.c_double,
        ctypes.c_double,
        Mem.handle_type(),
        Mem.handle_type(),
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
    ],
    check_errcode=True,
)

Lib.wrap_func(
    "sdp_gridder_determine_w_step",
    restype=ctypes.c_double,
    argtypes=[
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
    ],
    check_errcode=False,
)

Lib.wrap_func(
    "sdp_gridder_determine_max_w_tower_height",
    restype=ctypes.c_double,
    argtypes=[
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_int,
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
    "sdp_gridder_rms_diff",
    restype=ctypes.c_double,
    argtypes=[
        Mem.handle_type(),
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
