# See the LICENSE file at the top-level directory of this distribution.

"""Module for the bucket sort and tiling functions."""

import ctypes

from ..utility import Lib, Mem

Lib.wrap_func(
    "sdp_tile_simple",
    restype=None,
    argtypes=[
        Mem.handle_type(),
        Mem.handle_type(),
        Mem.handle_type(),
        Mem.handle_type(),
        ctypes.c_int,
        ctypes.c_int64,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_double,
        ctypes.c_int,
        Mem.handle_type(),
        Mem.handle_type(),
        Mem.handle_type(),
        ctypes.POINTER(ctypes.c_int),
    ],
    check_errcode=True,
)

Lib.wrap_func(
    "sdp_bucket_simple",
    restype=None,
    argtypes=[
        ctypes.c_int64,
        ctypes.c_int,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_double,
        Mem.handle_type(),
        Mem.handle_type(),
        Mem.handle_type(),
        Mem.handle_type(),
        Mem.handle_type(),
        Mem.handle_type(),
        Mem.handle_type(),
        Mem.handle_type(),
        Mem.handle_type(),
        Mem.handle_type(),
    ],
    check_errcode=True,
)


def tile_simple(
    uvw,
    freqs,
    vis,
    weights,
    grid_size,
    support,
    inv_tile_size_u,
    inv_tile_size_v,
    num_tiles_u,
    top_left_u,
    top_left_v,
    cell_size_rad,
    num_tiles,
    tile_offsets,
    num_points_in_tiles,
    num_skipped,
    num_visibilites,
):
    """
    Calculate the number of visibilies in each tile.

    :param uvw: List of UVW coordinates in metres, real-valued.
            Dimensions are [num_times, num_baselines, 3]
    :type uvw: numpy.ndarray

    :param freqs: List of frequencies in Hz, real-valued.
                    Dimension is [num_channels]
    :type freqs: numpy.ndarray

    :param vis: A complex valued 4D array, returns the visibilites.
                Dimensions are
                [time samples, baselines, channels, polarizations]
    :type vis: numpy.ndarray

    :param weights: A real-valued 4D array, returns the weights.
                    Dimensions are
                    [num_times, num_baselines, num_channels, num_pols]
    :type weights: numpy.ndarray

    :param num_points_in_tiles: A real valued 1D array, returns the number
                of visibilties in each tile. Dimensions are
                [num_tiles]
    :type num_points_in_tiles: numpy.ndarray

    :param num_skipped: A real valued 1D array, returns the number of
                visibilities that are skipped for each tile. Dimensions are
                [num_tiles]
    :type num_skipped: numpy.ndarray
    """

    Lib.sdp_tile_simple(
        Mem(uvw),
        Mem(freqs),
        Mem(vis),
        Mem(weights),
        grid_size,
        support,
        inv_tile_size_u,
        inv_tile_size_v,
        num_tiles_u,
        top_left_u,
        top_left_v,
        cell_size_rad,
        num_tiles,
        Mem(tile_offsets),
        Mem(num_points_in_tiles),
        Mem(num_skipped),
        ctypes.byref(num_visibilites),
    )


def bucket_simple(
    support,
    grid_size,
    inv_tile_size_u,
    inv_tile_size_v,
    top_left_u,
    top_left_v,
    num_tiles_u,
    cell_size_rad,
    uvw,
    vis,
    weights,
    freqs,
    tile_offsets,
    sorted_uu,
    sorted_vv,
    sorted_vis,
    sorted_weight,
    sorted_tile,
):
    """
    Sort the visibilities according to the tile they fall into

    :param uvw: List of UVW coordinates in metres, real-valued.
            Dimensions are [num_times, num_baselines, 3]
    :type uvw: numpy.ndarray

    :param freqs: List of frequencies in Hz, real-valued.
                    Dimension is [num_channels]
    :type freqs: numpy.ndarray

    :param vis: A complex valued 4D array, returns the visibilites.
                Dimensions are
                [time samples, baselines, channels, polarizations]
    :type vis: numpy.ndarray

    :param weights: A real-valued 4D array, returns the weights.
                    Dimensions are
                    [num_times, num_baselines, num_channels, num_pols]
    :type weights: numpy.ndarray

    :param num_points_in_tiles: A real valued 1D array, returns the number
                of visibilties in each tile. Dimensions are
                [num_tiles]
    :type num_points_in_tiles: numpy.ndarray

    :param num_skipped: A real valued 1D array, returns the number of
                visibilities that are skipped for each tile. Dimensions are
                [num_tiles]
    :type num_skipped: numpy.ndarray
    """

    Lib.sdp_bucket_simple(
        support,
        grid_size,
        inv_tile_size_u,
        inv_tile_size_v,
        top_left_u,
        top_left_v,
        num_tiles_u,
        cell_size_rad,
        Mem(uvw),
        Mem(vis),
        Mem(weights),
        Mem(freqs),
        Mem(tile_offsets),
        Mem(sorted_uu),
        Mem(sorted_vv),
        Mem(sorted_vis),
        Mem(sorted_weight),
        Mem(sorted_tile),
    )
