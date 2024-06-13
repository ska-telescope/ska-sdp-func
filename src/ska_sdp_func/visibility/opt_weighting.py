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
    check_errcode=True
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
        num_skipped
):
    """
    """
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
        Mem(num_skipped)
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
        output_weights      
):
    """
    """
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
        Mem(output_weights)
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
        output_weights      
):
    """
    """
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
        Mem(output_weights)
    )

