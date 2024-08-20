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
    that performs weighting after a bucket sort

    :param: uvw : List of UVW coordinates in metres, real-valued.
    Dimensions are [num_times, num_baselines, 3]
    :param freqs: List of frequencies in Hz, real-valued.
    Dimension is [num_channels]
    :param vis: Array of complex valued visibilities.
    Dimensions are [num_times, num_baselines, num_channels, num_pols]
    :param weights: A real-valued 4D array, returns the weights.
    Dimensions are [num_times, num_baselines, num_channels, num_pols]
    :param robust_param: Parameter given by the user to gauge the robustness
    of the weighting function.
    A value of -2 would be closer to uniform weighting and
    2 would be closer to natural weighting.
    :param grid_size: Size of the grid,
    the grid is assumed to be square,
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
    :param sorted_tile: Array that stores the sorted tile coordinates
    for each visibility. Dimensions are [num_visibilities]
    :param sorted_vis: Array that stores the sorted visibilities.
    Dimensions are [num_visibilities]
    :param tile_offsets: Array that results in the prefix summed
    number of visibilities, dimensions are [num_tiles + 1]
    :param num_points_in_tiles: Array that stores how many visibilities
    contribute to each tile, dimensions are [ num_tiles ]
    :param output_weights: A real-valued 4D array, returns the
    calculated weights.
    Dimensions are [num_times, num_baselines, num_channels, num_pols]
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
        Mem(output_weights),
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
    to better optimise performance for briggs weighting

    :param: uvw : List of UVW coordinates in metres, real-valued.
    Dimensions are [num_times, num_baselines, 3]
    :param vis: Array of complex valued visibilities.
    Dimensions are [num_times, num_baselines, num_channels, num_pols]
    :param weights: A real-valued 4D array, returns the weights.
    Dimensions are [num_times, num_baselines, num_channels, num_pols]
    :param robust_param: Parameter given by the user to gauge the robustness
    of the weighting function.
    :param grid_size: Size of the grid,
    the grid is assumed to be square,
    so only one dimensional size is expected.
    :param cell_size_rad: Cell size, in radians.
    :param support: Number of grid points a visibility contributes to.
    :param num_visibilities: Number of visibilities that needs to be processed,
    this will be calculated by count_and_prefix_sum(),expects a ctypes integer.
    :param sorted_tile: Array that stores the sorted tile coordinates
    for each visibility. Dimensions are [num_visibilities]
    :param sorted_uu: Array that stores the sorted u-coordiantes
    for each visibility. Dimensions are [num_visibilities]
    :param sorted_vv: Array that stores the sorted v-coordiantes
    for each visibility. Dimensions are [num_visibilities]
    :param sorted_vis_index: Array that stores the sorted indices
    for each visibility. Dimensions are [num_visibilities]
    :param tile_offsets: Array that results in the prefix summed
    number of visibilities, dimensions are [num_tiles + 1]
    :param num_points_in_tiles: Array that stores how many visibilities
    contribute to each tile, dimensions are [ num_tiles ]
    :param output_weights: A real-valued 4D array, returns the
    calculated weights.
    Dimensions are [num_times, num_baselines, num_channels, num_pols]

    """
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
