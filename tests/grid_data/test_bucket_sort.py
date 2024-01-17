# See the LICENSE file at the top-level directory of this distribution.

"""Test bucket sort functions."""
import ctypes
import numpy as np

try:
    import cupy as cp
except ImportError:
    cp = None

from ska_sdp_func.grid_data.tiling_and_bucket_sort import (
    tile_simple,
    bucket_simple
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
    grid_size = np.int64(160)
    grid_centre = np.int64(grid_size / 2)
    tile_size_u = np.int64(32)
    tile_size_v = np.int64(16)
    c_tile_u = np.int64(grid_centre / tile_size_u)
    c_tile_v = np.int64(grid_centre / tile_size_v)
    support = np.int64(20)
    inv_tile_size_u = 1.0 / tile_size_u
    inv_tile_size_v = 1.0 / tile_size_v
    num_tiles_u = np.int64((grid_size + tile_size_u - 1) / tile_size_u)
    num_tiles_v = np.int64((grid_size + tile_size_v - 1) / tile_size_v)
    num_tiles = np.int64(num_tiles_u * num_tiles_v)
    top_left_u = np.int64(
        grid_centre - c_tile_u * tile_size_u - tile_size_u / 2
    )
    top_left_v = np.int64(
        grid_centre - c_tile_v * tile_size_v - tile_size_v / 2
    )

    assert(top_left_u <= 0)
    assert(top_left_v <= 0)

    # Generating output arrays
    tile_offsets = np.zeros((num_tiles + 1), dtype=np.int32)
    num_points_in_tiles = np.zeros(num_tiles, dtype=np.int32)
    num_skipped = np.zeros(1, dtype=np.int32)


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
        num_tiles,
        tile_offsets,
        num_points_in_tiles,
        num_skipped,
        num_tiles_v
    )


def test_tile_and_bucket_sort():

    (   uvw,
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
        num_tiles,
        tile_offsets,
        num_points_in_tiles,
        num_skipped, 
        num_tiles_v
    ) = create_test_data_for_bucket_sort()

    cell_size_rad = 4.84814e-6
    num_visibilites = ctypes.c_int(0)
    num_visibilites_gpu = ctypes.c_int(0)

       # Same tests for the GPU
    if cp:
        uvw_gpu = cp.asarray(uvw)
        freqs_gpu = cp.asarray(freqs)
        vis_gpu = cp.asarray(vis)
        weights_gpu = cp.asarray(weights)
        tile_offsets_gpu = cp.zeros_like(tile_offsets)
        num_points_in_tiles_gpu = cp.zeros_like(num_points_in_tiles)
        num_skipped_gpu = cp.zeros_like(num_skipped)

        tile_simple(
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
            cell_size_rad,
            num_tiles,
            tile_offsets_gpu,
            num_points_in_tiles_gpu,
            num_skipped_gpu,
            num_visibilites_gpu
        )

        sorted_uu = np.zeros(num_visibilites_gpu.value, dtype = np.float64)
        sorted_vv = np.zeros(num_visibilites_gpu.value, dtype = np.float64)
        sorted_vis = np.zeros(num_visibilites_gpu.value, dtype = np.float64)
        sorted_tile = np.zeros(num_visibilites_gpu.value, dtype = np.int32)
        sorted_weight = np.zeros(num_visibilites_gpu.value, dtype = np.float64)

        sorted_uu_gpu = cp.asarray(sorted_uu)
        sorted_vv_gpu = cp.asarray(sorted_vv)
        sorted_vis_gpu = cp.asarray(sorted_vis)
        sorted_weight_gpu = cp.asarray(sorted_weight)
        sorted_tile_gpu = cp.asarray(sorted_tile)

        bucket_simple(
            support, 
            grid_size, 
            inv_tile_size_u, 
            inv_tile_size_v,
            top_left_u,
            top_left_v,
            num_tiles_u,
            cell_size_rad,
            uvw_gpu,
            vis_gpu,
            weights_gpu,
            freqs_gpu,
            tile_offsets_gpu, 
            sorted_uu_gpu, 
            sorted_vv_gpu, 
            sorted_vis_gpu, 
            sorted_weight_gpu, 
            sorted_tile_gpu
        )

    tile_simple(
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
        num_visibilites
    )

    sorted_uu = np.zeros(num_visibilites.value, dtype = np.float64)
    sorted_vv = np.zeros(num_visibilites.value, dtype = np.float64)
    sorted_vis = np.zeros(num_visibilites.value, dtype = np.float64)
    sorted_tile = np.zeros(num_visibilites.value, dtype = np.int32)
    sorted_weight = np.zeros(num_visibilites.value, dtype = np.float64)

    bucket_simple(
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
        sorted_tile   
    )

    assert np.allclose(
        num_points_in_tiles, num_points_in_tiles_gpu
    ), "The GPU bucket sort kernel and CPU code are not the same"


    
if __name__ == "__main__":
    test_tile_and_bucket_sort()
