# See the LICENSE file at the top-level directory of this distribution.

"""Module for the tiled functions."""

import ctypes

from ..utility import Lib, Mem

Lib.wrap_func(
    "sdp_count_and_prefix_sum",
    restype=None,
    argtypes=[
        Mem.handle_type(),
        Mem.handle_type(),
        Mem.handle_type(),
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_int64,
        ctypes.POINTER(ctypes.c_int),
        Mem.handle_type(),
        Mem.handle_type(),
        Mem.handle_type(),
    ],
    check_errcode=True,
)

Lib.wrap_func(
    "sdp_bucket_sort",
    restype=None,
    argtypes=[
        Mem.handle_type(),
        Mem.handle_type(),
        Mem.handle_type(),
        Mem.handle_type(),
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_int64,
        ctypes.POINTER(ctypes.c_int),
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

Lib.wrap_func(
    "sdp_tiled_indexing",
    restype=None,
    argtypes=[
        Mem.handle_type(),
        Mem.handle_type(),
        Mem.handle_type(),
        Mem.handle_type(),
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_int64,
        ctypes.POINTER(ctypes.c_int),
        Mem.handle_type(),
        Mem.handle_type(),
        Mem.handle_type(),
        Mem.handle_type(),
        Mem.handle_type(),
    ],
    check_errcode=True,
)


def count_and_prefix_sum(
    uvw,
    freqs,
    vis,
    grid_size,
    cell_size_rad,
    support,
    num_visibilities,
    tile_offsets,
    num_points_in_tiles,
    num_skipped,
):
    """Function that separates the grid into tiles
    and performs a prefix sum

    :param uvw: UVW coordinates of visibilties (in m).
    Dimensions are [num_times, num_baselines, 3]
    :param freqs: Number of channels, in hertz.
    Dimensions are [ num_channels ]
    :param vis: Array of complex valued visibilities.
    Dimensions are [num_times, num_baselines, num_channels, num_pols]
    :param grid_size: Size of the grid,
    the grid is assumed to be square,
    so only one dimensional size is expected.
    :param cell_size_rad: Cell size, in radians.
    :param support: Number of grid points a visibility contributes to.
    :param num_visibilities: Number of visibilities that needs to be processed,
    this will be calculated by this function and expects a ctypes integer.
    :param tile_offsets: Array that results in the prefix summed
    number of visibilities, dimensions are [num_tiles + 1]
    :param num_points_in_tiles: Array that stores how many visibilities
    contribute to each tile, dimensions are [ num_tiles ]
    :param num_skipped: Array that stores how many visibilities are skipped
    by the function, dimensions are [1]

    """
    Lib.sdp_count_and_prefix_sum(
        Mem(uvw),
        Mem(freqs),
        Mem(vis),
        grid_size,
        cell_size_rad,
        support,
        ctypes.byref(num_visibilities),
        Mem(tile_offsets),
        Mem(num_points_in_tiles),
        Mem(num_skipped),
    )


def bucket_sort(
    uvw,
    freqs,
    vis,
    weights,
    grid_size,
    cell_size_rad,
    support,
    num_visibilities,
    sorted_uu,
    sorted_vv,
    sorted_weight,
    sorted_tile,
    sorted_vis,
    tile_offsets,
    num_points_in_tiles,
):
    """Performs a bucket sort per tile in the grid,
    duplicates visibilities in overlapping regions

    :param uvw: UVW coordinates of visibilties (in m).
    Dimensions are [num_times, num_baselines, 3]
    :param freqs: Number of channels, in hertz.
    Dimensions are [ num_channels ]
    :param vis: Array of complex valued visibilities.
    Dimensions are [num_times, num_baselines, num_channels, num_pols]
    :param weights: Array of weights for each visibility.
    Dimensions are [num_times, num_baselines, num_channels, num_pols]
    :param grid_size: Size of the grid, the grid is assumed to be square,
    so only one dimensional size is expected.
    :param cell_size_rad: Cell size, in radians.
    :param support: Number of grid points a visibility contributes to.
    :param num_visibilities: Number of visibilities that needs to be processed,
    this will be calculated by count_and_prefix_sum(),expects a ctypes integer.
    :param sorted_uu: Array that stores the sorted u-coordiantes
    for each visibility. Dimensions are [num_visibilities]
    :param sorted_vv: Array that stores the sorted v-coordiantes
    for each visibility. Dimensions are [num_visibilities]
    :param sorted_weights: Array that stores the sorted weights
    for each visibility. Dimensions are [num_visibilities]
    :param sorted_tiles: Array that stores the sorted tile coordinates
    for each visibility. Dimensions are [num_visibilities]
    :param sorted_vis: Array that stores the sorted visibilities.
    Dimensions are [num_visibilities]
    :param tile_offsets: Array that results in the prefix summed
    number of visibilities, dimensions are [num_tiles + 1]
    :param num_points_in_tiles: Array that stores how many visibilities
    contribute to each tile, dimensions are [ num_tiles ]
    """
    Lib.sdp_bucket_sort(
        Mem(uvw),
        Mem(freqs),
        Mem(vis),
        Mem(weights),
        grid_size,
        cell_size_rad,
        support,
        ctypes.byref(num_visibilities),
        Mem(sorted_uu),
        Mem(sorted_vv),
        Mem(sorted_weight),
        Mem(sorted_tile),
        Mem(sorted_vis),
        Mem(tile_offsets),
        Mem(num_points_in_tiles),
    )


def tiled_indexing(
    uvw,
    freqs,
    vis,
    weights,
    grid_size,
    cell_size_rad,
    support,
    num_visibilties,
    sorted_tile,
    sorted_uu,
    sorted_vv,
    sorted_vis_index,
    tile_offsets,
):
    """Performs a bucket sort pert tile on the grid,
    doesn't duplicate visibilities themselves only it's indices

    :param uvw: UVW coordinates of visibilties (in m).
    Dimensions are [num_times, num_baselines, 3]
    :param freqs: Number of channels, in hertz.
    Dimensions are [ num_channels ]
    :param vis: Array of complex valued visibilities.
    Dimensions are [num_times, num_baselines, num_channels, num_pols]
    :param weights: Array of weights for each visibility.
    Dimensions are [num_times, num_baselines, num_channels, num_pols]
    :param grid_size: Size of the grid, the grid is assumed to be square,
    so only one dimensional size is expected.
    :param cell_size_rad: Cell size, in radians.
    :param support: Number of grid points a visibility contributes to.
    :param num_visibilities: Number of visibilities that needs to be processed,
    this will be calculated by count_and_prefix_sum(),expects a ctypes integer.
    :param sorted_tiles: Array that stores the sorted tile coordinates
    for each visibility. Dimensions are [num_visibilities]
    :param sorted_uu: Array that stores the sorted u-coordinates
    for each visibility. Dimensions are [num_visibilities]
    :param sorted_vv: Array that stores the sorted v-coordiantes
    for each visibility. Dimensions are [num_visibilities]
    :param sorted_vis: Array that stores the sorted indices
    for each visibility. Dimensions are [num_visibilities]
    :param tile_offsets: Array that results in the prefix summed
    number of visibilities, dimensions are [num_tiles + 1]
    """
    Lib.sdp_tiled_indexing(
        Mem(uvw),
        Mem(freqs),
        Mem(vis),
        Mem(weights),
        grid_size,
        cell_size_rad,
        support,
        ctypes.byref(num_visibilties),
        Mem(sorted_tile),
        Mem(sorted_uu),
        Mem(sorted_vv),
        Mem(sorted_vis_index),
        Mem(tile_offsets),
    )
