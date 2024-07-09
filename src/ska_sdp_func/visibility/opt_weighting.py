# See the LICENSE file at the top-level directory of this distribution.

"""Module for the optimised weighting functions."""

import ctypes

from ..utility import Lib, Mem

Lib.wrap_func(
    "sdp_optimized_weighting",
    restype=None,
    argtypes=[
        Mem.handle_type(),
        Mem.handle_type(),
        Mem.handle_type(),
        Mem.handle_type(),
        ctypes.c_double,
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
        Mem.handle_type(),
    ],
    check_errcode=True,
)

Lib.wrap_func(
    "sdp_tile_and_prefix_sum",
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
        ctypes.c_double,
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
        ctypes.c_double,
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

Lib.wrap_func(
    "sdp_optimised_indexed_weighting",
    restype=None,
    argtypes=[
        Mem.handle_type(),
        Mem.handle_type(),
        Mem.handle_type(),
        ctypes.c_double,
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


def tile_and_prefix_sum(
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
    and performs a prefix sum"""
    Lib.sdp_tile_and_prefix_sum(
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


def optimized_weighting(
    uvw,
    freqs,
    vis,
    weights,
    robust_param,
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
    output_weights,
):
    """Optimised briggs weighting algorithm
    that performs weighting after a bucket sort"""
    Lib.sdp_optimized_weighting(
        Mem(uvw),
        Mem(freqs),
        Mem(vis),
        Mem(weights),
        robust_param,
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
        Mem(output_weights),
    )


def bucket_sort(
    uvw,
    freqs,
    vis,
    weights,
    robust_param,
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
    output_weights,
):
    """Performs a bucket sort per tile in the grid,
    duplicates visibilities in overlapping regions"""
    Lib.sdp_bucket_sort(
        Mem(uvw),
        Mem(freqs),
        Mem(vis),
        Mem(weights),
        robust_param,
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
        Mem(output_weights),
    )


def tiled_indexing(
    uvw,
    freqs,
    vis,
    weights,
    robust_param,
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
    doesn't duplicate visibilities themselves only it's indices"""
    Lib.sdp_tiled_indexing(
        Mem(uvw),
        Mem(freqs),
        Mem(vis),
        Mem(weights),
        robust_param,
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


def optimised_indexed_weighting(
    uvw,
    vis,
    weights,
    robust_param,
    grid_size,
    cell_size_rad,
    support,
    num_visibilities,
    sorted_tile,
    sorted_uu,
    sorted_vv,
    sorted_vis_index,
    tile_offsets,
    num_points_in_tiles,
    output_weights,
):
    """Utilises the indexed visibiliies/weights
    to better optimise performance for briggs weighting"""
    Lib.sdp_optimised_indexed_weighting(
        Mem(uvw),
        Mem(vis),
        Mem(weights),
        robust_param,
        grid_size,
        cell_size_rad,
        support,
        ctypes.byref(num_visibilities),
        Mem(sorted_tile),
        Mem(sorted_uu),
        Mem(sorted_vv),
        Mem(sorted_vis_index),
        Mem(tile_offsets),
        Mem(num_points_in_tiles),
        Mem(output_weights),
    )
