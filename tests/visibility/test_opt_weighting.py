"""Test the optimised briggs weighting function"""

import ctypes
import time

import cupy
import numpy as np

from ska_sdp_func.visibility.opt_weighting import (
    bucket_sort,
    optimised_indexed_weighting,
    optimized_weighting,
    tile_and_prefix_sum,
    tiled_indexing,
)


def bucket_sort_params(grid_size):
    # Prepare parameters for bucket sort
    tile_size_u = cupy.int64(32)
    tile_size_v = cupy.int64(16)
    num_tiles_u = cupy.int64((grid_size + tile_size_u - 1) / tile_size_u)
    num_tiles_v = cupy.int64((grid_size + tile_size_v - 1) / tile_size_v)
    num_tiles = cupy.int64(num_tiles_u * num_tiles_v)
    tile_offsets = cupy.zeros((num_tiles + 1), dtype=cupy.int32)
    num_points_in_tiles = cupy.zeros(num_tiles, dtype=cupy.int32)
    num_skipped = cupy.zeros(1, dtype=cupy.int32)

    return num_points_in_tiles, num_skipped, tile_offsets


def gen_gpu_arrays(num_visibilities, indexed):
    sorted_uu = np.zeros(num_visibilities.value, dtype=np.float64)
    sorted_vv = np.zeros(num_visibilities.value, dtype=np.float64)
    sorted_vis = np.zeros(num_visibilities.value, dtype=np.float64)
    sorted_weight = np.zeros(num_visibilities.value, dtype=np.float64)
    sorted_tile = np.zeros(num_visibilities.value, dtype=np.int32)
    sorted_vis_index = np.zeros(num_visibilities.value, dtype=np.int32)

    sorted_uu_gpu = cupy.asarray(sorted_uu)
    sorted_vv_gpu = cupy.asarray(sorted_vv)
    sorted_vis_gpu = cupy.asarray(sorted_vis)
    sorted_weight_gpu = cupy.asarray(sorted_weight)
    output_weights = cupy.asarray(sorted_weight)
    sorted_tile_gpu = cupy.asarray(sorted_tile)
    sorted_vis_index_gpu = cupy.asarray(sorted_vis_index)

    if indexed == 0:
        return (
            sorted_uu_gpu,
            sorted_vv_gpu,
            sorted_vis_gpu,
            sorted_weight_gpu,
            output_weights,
            sorted_tile_gpu,
        )

    if indexed == 1:
        return (
            sorted_tile_gpu,
            sorted_vis_index_gpu,
            output_weights,
            sorted_uu_gpu,
            sorted_vv_gpu,
        )


def input_gen():
    """
    Generate data for testing optimised and indexed weighting
    """
    freqs_np = np.array([1e9, 1.1e9, 1.2e9, 1.3e9, 1.4e9, 1.5e9])

    uvw_general = [
        [
            [10, 31, 21],
            [10, 31, 21],
            [10, 31, 21],
            [10, 31, 21],
            [10, 31, 21],
            [10, 31, 21],
            [10, 31, 21],
            [10, 31, 21],
        ],
        [
            [10, 31, 21],
            [10, 31, 21],
            [10, 31, 21],
            [10, 31, 21],
            [10, 31, 21],
            [10, 31, 21],
            [10, 31, 21],
            [10, 31, 21],
        ],
        [
            [10, 31, 21],
            [10, 31, 21],
            [10, 31, 21],
            [10, 31, 21],
            [10, 31, 21],
            [10, 31, 21],
            [10, 31, 21],
            [10, 31, 21],
        ],
        [
            [10, 31, 21],
            [10, 31, 21],
            [10, 31, 21],
            [10, 31, 21],
            [10, 31, 21],
            [10, 31, 21],
            [10, 31, 21],
            [10, 31, 21],
        ],
        [
            [10, 31, 21],
            [10, 31, 21],
            [10, 31, 21],
            [10, 31, 21],
            [10, 31, 21],
            [10, 31, 21],
            [10, 31, 21],
            [10, 31, 21],
        ],
        [
            [10, 31, 21],
            [10, 31, 21],
            [10, 31, 21],
            [10, 31, 21],
            [10, 31, 21],
            [10, 31, 21],
            [10, 31, 21],
            [10, 31, 21],
        ],
        [
            [10, 31, 21],
            [10, 31, 21],
            [10, 31, 21],
            [10, 31, 21],
            [10, 31, 21],
            [10, 31, 21],
            [10, 31, 21],
            [10, 31, 21],
        ],
        [
            [10, 31, 21],
            [10, 31, 21],
            [10, 31, 21],
            [10, 31, 21],
            [10, 31, 21],
            [10, 31, 21],
            [10, 31, 21],
            [10, 31, 21],
        ],
    ]

    baselines = 8
    times = 8
    pol = 1
    uvw_np = np.asarray(uvw_general, dtype=np.float64)
    vis_np = np.full((times, baselines, len(freqs_np), pol), 1j, dtype=complex)
    weights_np = np.ones_like(vis_np, dtype=np.float64)
    robust_param = -2
    grid_size = 500
    cell_size_rad = 4.06e-5
    support = 4

    uvw = cupy.asarray(uvw_np)
    freqs = cupy.asarray(freqs_np)
    vis = cupy.asarray(vis_np)
    weights = cupy.asarray(weights_np)

    return (
        freqs,
        uvw,
        vis,
        baselines,
        pol,
        times,
        weights,
        robust_param,
        grid_size,
        cell_size_rad,
        support,
    )


def test_optmised_weighting():
    """
    Test optimised weighting
    """
    indexed = 0
    (
        freqs,
        uvw,
        vis,
        baselines,
        pol,
        times,
        weights,
        robust_param,
        grid_size,
        cell_size_rad,
        support,
    ) = input_gen()
    num_visibilities = ctypes.c_int(0)
    num_points_in_tiles, num_skipped, tile_offsets = bucket_sort_params(
        grid_size
    )

    tile_and_prefix_sum(
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
    )

    (
        sorted_uu,
        sorted_vv,
        sorted_vis,
        sorted_weights,
        output_weights,
        sorted_tile,
    ) = gen_gpu_arrays(num_visibilities, indexed)

    bucket_sort(
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
        sorted_weights,
        sorted_tile,
        sorted_vis,
        tile_offsets,
        num_points_in_tiles,
        output_weights,
    )

    optimized_weighting(
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
        sorted_weights,
        sorted_tile,
        sorted_vis,
        tile_offsets,
        num_points_in_tiles,
        output_weights,
    )

    return output_weights


def test_indexed_weighting():
    """
    Test indexed weighting
    """
    indexed = 1
    (
        freqs,
        uvw,
        vis,
        baselines,
        pol,
        times,
        weights,
        robust_param,
        grid_size,
        cell_size_rad,
        support,
    ) = input_gen()
    num_visibilities = ctypes.c_int(0)
    num_points_in_tiles, num_skipped, tile_offsets = bucket_sort_params(
        grid_size
    )

    tile_and_prefix_sum(
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
    )

    (
        sorted_tile,
        sorted_vis_index,
        output_weights,
        sorted_uu,
        sorted_vv,
    ) = gen_gpu_arrays(num_visibilities, indexed)

    tiled_indexing(
        uvw,
        freqs,
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
    )

    optimised_indexed_weighting(
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
    )

    return output_weights


if __name__ == "__main__":
    n_indexed_start = time.time()
    n_indexed_w = test_optmised_weighting()
    n_indexed_end = time.time()

    indexed_start = time.time()
    indexed_w = test_indexed_weighting()
    indexed_end = time.time()

    print("Bucket sorted weighting time")
    print(n_indexed_end - n_indexed_start)
    print("Indexed weighting time")
    print(indexed_end - indexed_start)
