"""Test the tiling functions"""

import ctypes

import cupy
import numpy as np

from ska_sdp_func.visibility.tiled_functions import (
    bucket_sort,
    count_and_prefix_sum,
    tiled_indexing,
)


def calc_grid_u(
    uvw,
    freqs,
    num_channels,
    num_times,
    num_baselines,
    grid_size,
    cell_size_rad,
):
    """
    Calculates the expected grid points in u for the visiblities
    """
    c_0 = 299792458.0
    grid_scale = grid_size * cell_size_rad
    expected_uu = np.empty((num_times, num_baselines, 1))

    for i_time in range(num_times):
        for i_baseline in range(num_baselines):
            for i_channel in range(num_channels):
                inv_wavelength = freqs[i_channel] / c_0
                grid_u = (
                    uvw[i_time, i_baseline, 0] * inv_wavelength * grid_scale
                )
                expected_uu[i_time, i_baseline, 0] = grid_u
    return expected_uu


def bucket_sort_params(grid_size):
    """
    Prepares the parameters for the bucket sort such as tile size and
    other arrays like the number of points in each tile
    """
    # Prepare parameters for bucket sort
    tile_size_u = cupy.int64(10)
    tile_size_v = cupy.int64(10)
    num_tiles_u = cupy.int64((grid_size + tile_size_u - 1) / tile_size_u)
    num_tiles_v = cupy.int64((grid_size + tile_size_v - 1) / tile_size_v)
    num_tiles = cupy.int64(num_tiles_u * num_tiles_v)
    tile_offsets = cupy.zeros((num_tiles + 1), dtype=cupy.int32)
    num_points_in_tiles = cupy.zeros(num_tiles, dtype=cupy.int32)
    num_skipped = cupy.zeros(1, dtype=cupy.int32)

    return num_points_in_tiles, num_skipped, tile_offsets


def gen_gpu_arrays(num_visibilities, indexed):
    """
    Generates arrays to be processed on the gpu with cupy,
    takes in an indexed parameter to decide what arrays to return
    """
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
    sorted_tile_gpu = cupy.asarray(sorted_tile)
    sorted_vis_index_gpu = cupy.asarray(sorted_vis_index)

    if indexed == 0:
        return (
            sorted_uu_gpu,
            sorted_vv_gpu,
            sorted_vis_gpu,
            sorted_weight_gpu,
            sorted_tile_gpu,
        )

    if indexed == 1:
        return (
            sorted_tile_gpu,
            sorted_vis_index_gpu,
            sorted_uu_gpu,
            sorted_vv_gpu,
        )


def gen_data():
    """
    Generate data for testing bucket sort,
    tiled indexing, counting and prefix sum.
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
    grid_size = 400
    cell_size_rad = 4.06e-5
    support = 4
    tile_size_u = 32
    tile_size_v = 16

    uvw = cupy.asarray(uvw_np)
    freqs = cupy.asarray(freqs_np)
    vis = cupy.asarray(vis_np)
    weights = cupy.asarray(weights_np)

    return (
        freqs,
        uvw,
        vis,
        weights,
        grid_size,
        cell_size_rad,
        support,
        tile_size_u,
        tile_size_v,
    )


def test_bucket_sort():
    """
    Tests bucket sorting of visibilities on
    fixed tile sizes.
    """

    (
        freqs,
        uvw,
        vis,
        weights,
        grid_size,
        cell_size_rad,
        support,
        tile_size_u,
        tile_size_v,
    ) = gen_data()
    num_points_in_tiles, num_skipped, tile_offsets = bucket_sort_params(
        grid_size
    )
    num_visibilities = ctypes.c_int(0)

    count_and_prefix_sum(
        uvw,
        freqs,
        vis,
        grid_size,
        tile_size_u,
        tile_size_v,
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
        sorted_weight,
        sorted_tile,
    ) = gen_gpu_arrays(num_visibilities, 0)

    num_channels = freqs.shape[0]
    num_times = uvw.shape[0]
    num_baselines = uvw.shape[1]

    bucket_sort(
        uvw,
        freqs,
        vis,
        weights,
        grid_size,
        tile_size_u,
        tile_size_v,
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
    )

    expected_uu = calc_grid_u(
        uvw,
        freqs,
        num_channels,
        num_times,
        num_baselines,
        grid_size,
        cell_size_rad,
    )
    assert num_visibilities.value == vis.size
    assert expected_uu.all() == sorted_uu.all()


def test_tiled_indexing():
    """
    Tests tiled indexing on fixed tile sizes.
    """
    (
        freqs,
        uvw,
        vis,
        weights,
        grid_size,
        cell_size_rad,
        support,
        tile_size_u,
        tile_size_v,
    ) = gen_data()
    num_points_in_tiles, num_skipped, tile_offsets = bucket_sort_params(
        grid_size
    )
    num_visibilities = ctypes.c_int(0)

    count_and_prefix_sum(
        uvw,
        freqs,
        vis,
        grid_size,
        tile_size_u,
        tile_size_v,
        cell_size_rad,
        support,
        num_visibilities,
        tile_offsets,
        num_points_in_tiles,
        num_skipped,
    )

    sorted_tile, sorted_vis_index, sorted_uu, sorted_vv = gen_gpu_arrays(
        num_visibilities, 1
    )

    num_channels = freqs.shape[0]
    num_times = uvw.shape[0]
    num_baselines = uvw.shape[1]
    num_pol = 1

    tiled_indexing(
        uvw,
        freqs,
        grid_size,
        tile_size_u,
        tile_size_v,
        cell_size_rad,
        support,
        num_channels,
        num_baselines,
        num_times,
        num_pol,
        num_visibilities,
        sorted_tile,
        sorted_uu,
        sorted_vv,
        sorted_vis_index,
        tile_offsets,
    )

    expected_uu = calc_grid_u(
        uvw,
        freqs,
        num_channels,
        num_times,
        num_baselines,
        grid_size,
        cell_size_rad,
    )
    assert num_visibilities.value == vis.size
    assert expected_uu.all() == sorted_uu.all()
