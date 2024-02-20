# See the LICENSE file at the top-level directory of this distribution.
""" Module for stand-alone gridding functions. """

import ctypes

from ..utility import Lib, Mem


def grid_uvw_es(
    uvw,
    vis,
    weight,
    freq_hz,
    image_size: int,
    epsilon: float,
    cell_size_rad: float,
    w_scale: float,
    min_plane_w: float,
    sub_grid_start_u: int,
    sub_grid_start_v: int,
    sub_grid_w: int,
    sub_grid,
):
    """Grids visibilities onto the supplied sub-grid.

    Parameters
    ==========
    uvw: cupy.ndarray((num_rows, 3), dtype=numpy.float32 or numpy.float64)
        (u,v,w) coordinates.
    vis: cupy.ndarray((num_rows, num_chan), dtype=numpy.complex64 or
        numpy.complex128)
        The input visibility data.
        Its data type determines the precision used for the gridding.
    weight: cupy.ndarray((num_rows, num_chan), same precision as **vis**)
        Its values are used to multiply the input.
    freq_hz: cupy.ndarray((num_chan,), dtype=numpy.float32 or numpy.float64)
        Channel frequencies.
    image_size: int
        The required number of pixels on one side of the whole output image.
        (This is needed to calculate other kernel parameters.)
    epsilon: float
        Accuracy at which the computation should be done.
        Must be larger than 2e-13.
        If **vis** has type numpy.complex64, it must be larger than 1e-5.
    cell_size_rad: float
        Angular pixel size (in radians) of the image.
    w_scale: float
        Factor to convert w-coordinates to w-layer index.
    min_plane_w: float
        The w-coordinate of the first w-layer.
    sub_grid_start_u: int
        Start index of sub-grid in u dimension.
    sub_grid_start_v: int
        Start index of sub-grid in v dimension.
    sub_grid_w: int
        Index of sub-grid in w-layer stack.
    sub_grid: cupy.ndarray((num_pix, num_pix), dtype=numpy.complex64 or
        numpy.complex128)
        The output sub grid, **must be square**.
    """
    Lib.sdp_grid_uvw_es(
        Mem(uvw),
        Mem(vis),
        Mem(weight),
        Mem(freq_hz),
        image_size,
        epsilon,
        cell_size_rad,
        w_scale,
        min_plane_w,
        sub_grid_start_u,
        sub_grid_start_v,
        sub_grid_w,
        Mem(sub_grid),
    )


Lib.wrap_func(
    "sdp_grid_uvw_es",
    restype=None,
    argtypes=[
        Mem.handle_type(),
        Mem.handle_type(),
        Mem.handle_type(),
        Mem.handle_type(),
        ctypes.c_int32,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
        Mem.handle_type(),
    ],
    check_errcode=True,
)
