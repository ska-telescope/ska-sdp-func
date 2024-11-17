# See the LICENSE file at the top-level directory of this distribution.
"""Module for (de)gridding functions using w-stacking with w-towers."""

import ctypes
from typing import Optional

import numpy

from ..utility import Lib, Mem


def wstack_wtower_degrid_all(
    image: numpy.ndarray,
    freq0_hz: float,
    dfreq_hz: float,
    uvw: numpy.ndarray,
    subgrid_size: int,
    theta: float,
    w_step: float,
    shear_u: float,
    shear_v: float,
    support: int,
    oversampling: int,
    w_support: int,
    w_oversampling: int,
    subgrid_frac: float,
    w_tower_height: float,
    verbosity: int,
    vis: numpy.ndarray,
    num_threads: Optional[int] = None,
):
    """
    Degrid visibilities using w-stacking with w-towers.

    :param image: Image to degrid from.
    :param freq0_hz: Frequency of first channel (Hz).
    :param dfreq_hz: Channel separation (Hz).
    :param uvw: ``float[uvw_count, 3]`` UVW coordinates of visibilities (in m).
    :param subgrid_size: Sub-grid size in pixels.
    :param theta: Total image size in direction cosines.
    :param w_step: Spacing between w-planes.
    :param shear_u: Shear parameter in u (use zero for no shear).
    :param shear_v: Shear parameter in v (use zero for no shear).
    :param support: Kernel support size in (u, v).
    :param oversampling: Oversampling factor for uv-kernel.
    :param w_support: Support size in w.
    :param w_oversampling: Oversampling factor for w-kernel.
    :param subgrid_frac: Fraction of subgrid size that should be used.
    :param w_tower_height: Height of w-tower to use.
    :param verbosity: Verbosity level.
    :param vis: ``complex[uvw_count, ch_count]`` Output degridded visibilities.
    :param num_threads: Number of threads to use.
        If 0 or None, all available threads will be used.
    """
    if not num_threads:
        num_threads = 0
    Lib.sdp_grid_wstack_wtower_degrid_all(
        Mem(image),
        freq0_hz,
        dfreq_hz,
        Mem(uvw),
        subgrid_size,
        theta,
        w_step,
        shear_u,
        shear_v,
        support,
        oversampling,
        w_support,
        w_oversampling,
        subgrid_frac,
        w_tower_height,
        verbosity,
        Mem(vis),
        num_threads,
    )


def wstack_wtower_grid_all(
    vis: numpy.ndarray,
    freq0_hz: float,
    dfreq_hz: float,
    uvw: numpy.ndarray,
    subgrid_size: int,
    theta: float,
    w_step: float,
    shear_u: float,
    shear_v: float,
    support: int,
    oversampling: int,
    w_support: int,
    w_oversampling: int,
    subgrid_frac: float,
    w_tower_height: float,
    verbosity: int,
    image: numpy.ndarray,
    num_threads: Optional[int] = None,
):
    """
    Grid visibilities using w-stacking with w-towers.

    :param vis: ``complex[uvw_count, ch_count]`` Input visibilities.
    :param freq0_hz: Frequency of first channel (Hz).
    :param dfreq_hz: Channel separation (Hz).
    :param uvw: ``float[uvw_count, 3]`` UVW coordinates of visibilities (in m).
    :param subgrid_size: Sub-grid size in pixels.
    :param theta: Total image size in direction cosines.
    :param w_step: Spacing between w-planes.
    :param shear_u: Shear parameter in u (use zero for no shear).
    :param shear_v: Shear parameter in v (use zero for no shear).
    :param support: Kernel support size in (u, v).
    :param oversampling: Oversampling factor for uv-kernel.
    :param w_support: Support size in w.
    :param w_oversampling: Oversampling factor for w-kernel.
    :param subgrid_frac: Fraction of subgrid size that should be used.
    :param w_tower_height: Height of w-tower to use.
    :param verbosity: Verbosity level.
    :param image: Output image.
    :param num_threads: Number of threads to use.
        If 0 or None, all available threads will be used.
    """
    if not num_threads:
        num_threads = 0
    Lib.sdp_grid_wstack_wtower_grid_all(
        Mem(vis),
        freq0_hz,
        dfreq_hz,
        Mem(uvw),
        subgrid_size,
        theta,
        w_step,
        shear_u,
        shear_v,
        support,
        oversampling,
        w_support,
        w_oversampling,
        subgrid_frac,
        w_tower_height,
        verbosity,
        Mem(image),
        num_threads,
    )


Lib.wrap_func(
    "sdp_grid_wstack_wtower_degrid_all",
    restype=None,
    argtypes=[
        Mem.handle_type(),
        ctypes.c_double,
        ctypes.c_double,
        Mem.handle_type(),
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
        Mem.handle_type(),
        ctypes.c_int,
    ],
    check_errcode=True,
)

Lib.wrap_func(
    "sdp_grid_wstack_wtower_grid_all",
    restype=None,
    argtypes=[
        Mem.handle_type(),
        ctypes.c_double,
        ctypes.c_double,
        Mem.handle_type(),
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
        Mem.handle_type(),
        ctypes.c_int,
    ],
    check_errcode=True,
)
