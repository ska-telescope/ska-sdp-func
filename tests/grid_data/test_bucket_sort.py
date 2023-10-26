# See the LICENSE file at the top-level directory of this distribution.

"""Test bucket sort functions."""

import numpy as np

try:
    import cupy as cp
except ImportError:
    cp = None

from ska_sdp_func.grid_data.tiling_and_bucket_sort import (
    tile_and_bucket_sort_simple,
)


def create_test_data_for_bucket_sort():

    # Defining frequency and uvw for tile and bucket sort
    freqs = np.array([1e9, 1.1e9, 1.2e9])
    uvw = np.array([[[2, 3, 5], [7, 11, 13], [17, 19, 23]]], dtype=np.float64)

    # Parameters for generating visibilties
    num_baselines = uvw.shape[1]
    num_times = uvw.shape[0]
    num_pols = 4  # Bucket sort only works with 4 polarisations
    num_channels = freqs.shape[0]

    # Generate visibilities and weights for sorting
    vis_rand_r = np.random.rand(
        num_times, num_baselines, num_channels, num_pols
    )
    vis_rand_i = np.random.rand(
        num_times, num_baselines, num_channels, num_pols
    )
    vis = vis_rand_r + 1j * vis_rand_i
    weights = np.ones_like(vis, dtype=np.float64)

    # Defining parameters for tile and bucket sort
    grid_size = np.int64(1000)
    grid_centre = np.int64(grid_size / 2)
    tile_size_u = np.int64(32)
    tile_size_v = np.int64(16)
    c_tile_u = np.int64(grid_centre / tile_size_u)
    c_tile_v = np.int64(grid_centre / tile_size_v)
    support = np.int64(16)
    inv_tile_size_u = np.int64(1 / tile_size_u)
    inv_tile_size_v = np.int64(1 / tile_size_v)
    num_tiles_u = np.int64((grid_size + tile_size_u - 1) / tile_size_u)
    num_tiles_v = np.int64((grid_size + tile_size_v - 1) / tile_size_v)
    num_tiles = np.int64(num_tiles_u * num_tiles_v)
    top_left_u = np.int64(
        grid_centre - c_tile_u * tile_size_u - tile_size_u / 2
    )
    top_left_v = np.int64(
        grid_centre - c_tile_v * tile_size_v - tile_size_v / 2
    )

    # Generating output arrays
    tile_offsets = np.empty((num_tiles + 1), dtype=np.float64)
    sorted_vis = np.empty(
        (uvw.shape[0], uvw.shape[1], freqs.shape[0], 4), dtype=np.complex128
    )
    sorted_uu = np.empty(vis.size, dtype=np.float32)
    sorted_vv = np.empty(vis.size, dtype=np.float32)
    sorted_weight = np.empty(weights.shape, dtype=np.float64)
    num_points_in_tiles = np.empty(num_tiles, dtype=np.float64)
    num_skipped = np.empty(num_tiles, dtype=np.float64)
    sorted_tile = np.empty(vis.size, dtype=np.float64)

    return (
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
    )


def test_tile_and_bucket_sort():

    (
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
    ) = create_test_data_for_bucket_sort()

    tile_and_bucket_sort_simple(
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
    )

    # Same tests for the GPU
    if cp:
        uvw_gpu = cp.asarray(uvw)
        freqs_gpu = cp.asarray(freqs)
        vis_gpu = cp.asarray(vis)
        weights_gpu = cp.asarray(weights)
        tile_offsets_gpu = cp.asarray(tile_offsets)
        sorted_vis_gpu = cp.asarray(sorted_vis)
        sorted_uu_gpu = cp.asarray(sorted_uu)
        sorted_vv_gpu = cp.asarray(sorted_vv)
        sorted_weight_gpu = cp.asarray(sorted_weight)
        num_points_in_tiles_gpu = cp.asarray(num_points_in_tiles)
        num_skipped_gpu = cp.asarray(num_skipped)
        sorted_tile_gpu = cp.asarray(sorted_tile)

        tile_and_bucket_sort_simple(
            uvw_gpu,
            freqs_gpu,
            vis_gpu,
            weights_gpu,
            grid_size,
            support,
            inv_tile_size_u,
            inv_tile_size_v,
            num_tiles_u,
            top_left_u,
            top_left_v,
            tile_offsets_gpu,
            sorted_vis_gpu,
            sorted_uu_gpu,
            sorted_vv_gpu,
            sorted_weight_gpu,
            num_points_in_tiles_gpu,
            num_skipped_gpu,
            sorted_tile_gpu,
        )

        assert np.allclose(
            sorted_vis, sorted_vis_gpu
        ), "The GPU bucket sort kernel and CPU code are not the same"


if __name__ == "__main__":
    test_tile_and_bucket_sort()
