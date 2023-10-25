# See the LICENSE file at the top-level directory of this distribution.

"""Module for the bucket sort and tiling functions."""

import ctypes

from ..utility import Lib, Mem

Lib.wrap_func(
    "sdp_tile_and_bucket_sort_simple",
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


def tile_and_bucket_sort_simple(
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
    tile_offsets,
    sorted_vis,
    sorted_uu,
    sorted_vv,
    sorted_weight,
    num_points_in_tiles,
    num_skipped,
    sorted_tile,
):
    """
    Calculate the number of tiles and get the histogram for
    how many visibilites fall into those tiles.

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

    :param sorted_vis: A complex valued 4D array, gets the sorted visibilites.
                 Dimensions are
                 [time samples, baselines, channels, polarizations]
    :type sorted_vis: numpy.ndarray

    :param num_points_in_tiles: A real valued 1D array, returns the number
                of visibilties in each tile. Dimensions are
                [num_vis]
    :type num_points_in_tiles: numpy.ndarray

    :param num_skipped: A real valued 1D array, returns the number of
                visibilities that are skipped for each tile. Dimensions are
                [num_vis]
    :type num_skipped: numpy.ndarray
    """

    Lib.sdp_tile_and_bucket_sort_simple(
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
        Mem(tile_offsets),
        Mem(sorted_vis),
        Mem(sorted_uu),
        Mem(sorted_vv),
        Mem(sorted_weight),
        Mem(num_points_in_tiles),
        Mem(num_skipped),
        Mem(sorted_tile),
    )
