"""Test the optimised briggs weighting function"""

import ctypes
from math import floor

import cupy
import numpy as np

from ska_sdp_func.visibility.opt_weighting import (
    optimised_indexed_weighting,
    optimized_weighting,
)
from ska_sdp_func.visibility.tiled_functions import (
    bucket_sort,
    count_and_prefix_sum,
    tiled_indexing,
)


def reference_briggs_weights(
    uvw,
    freq_hz,
    max_abs_uv,
    weights_grid_uv,
    grid_size,
    robust_param,
    input_weight,
    output_weight,
    num_channels,
    num_baselines,
    num_pol,
    num_times,
):
    """
    Reference briggs weighting function implemented in python and
    checked against DDFacet/WSClean
    """
    sum_weight = 0
    sum_weight2 = 0
    c_0 = 299792458.0

    # Generate grid of weights
    for i_time in range(num_times):
        for i_baseline in range(num_baselines):
            for i_channel in range(num_channels):
                grid_u = uvw[i_time, i_baseline, 0] * freq_hz[i_channel] / c_0
                grid_v = uvw[i_time, i_baseline, 1] * freq_hz[i_channel] / c_0
                idx_u = int(
                    floor(grid_u / max_abs_uv * grid_size / 2) + grid_size / 2
                )
                idx_v = int(
                    floor(grid_v / max_abs_uv * grid_size / 2) + grid_size / 2
                )
                if idx_u >= grid_size or idx_v >= grid_size:
                    continue
                for i_pol in range(num_pol):
                    w = input_weight[i_time, i_baseline, i_channel, i_pol]
                    weights_grid_uv[idx_u, idx_v, i_pol] += w

    # Calculate the sum of weights and sum of the gridded weights squared
    for i_time in range(num_times):
        for i_baseline in range(num_baselines):
            for i_channel in range(num_channels):
                grid_u = uvw[i_time, i_baseline, 0] * freq_hz[i_channel] / c_0
                grid_v = uvw[i_time, i_baseline, 1] * freq_hz[i_channel] / c_0
                idx_u = int(
                    floor(grid_u / max_abs_uv * grid_size / 2) + grid_size / 2
                )
                idx_v = int(
                    floor(grid_v / max_abs_uv * grid_size / 2) + grid_size / 2
                )
                if idx_u >= grid_size or idx_v >= grid_size:
                    continue
                for i_pol in range(num_pol):
                    sum_weight += weights_grid_uv[idx_u, idx_v, i_pol]
                    sum_weight2 += (
                        weights_grid_uv[idx_u, idx_v, i_pol]
                        * weights_grid_uv[idx_u, idx_v, i_pol]
                    )

    # Calculate the robustness function
    numerator = (5.0 * (1 / (10.0**robust_param))) ** 2
    division_param = sum_weight2 / sum_weight
    robustness = numerator / division_param

    # Read from the grid of weights according to the enum type
    for i_time in range(num_times):
        for i_baseline in range(num_baselines):
            for i_channel in range(num_channels):
                grid_u = uvw[i_time, i_baseline, 0] * freq_hz[i_channel] / c_0
                grid_v = uvw[i_time, i_baseline, 1] * freq_hz[i_channel] / c_0
                idx_u = int(
                    floor(grid_u / max_abs_uv * grid_size / 2) + grid_size / 2
                )
                idx_v = int(
                    floor(grid_v / max_abs_uv * grid_size / 2) + grid_size / 2
                )
                for i_pol in range(num_pol):
                    if idx_u >= grid_size or idx_v >= grid_size:
                        weight_g = 1.0
                    else:
                        weight_g = weights_grid_uv[idx_u, idx_v, i_pol]
                        w = input_weight[i_time, i_baseline, i_channel, i_pol]
                        weight_x = w / (1 + (robustness * weight_g))
                        output_weight[
                            i_time, i_baseline, i_channel, i_pol
                        ] = weight_x


def bucket_sort_params(grid_size):
    """
    Prepares the parameters for the bucket sort such as tile size and
    other arrays like the number of points in each tile
    """
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
            sorted_uu_gpu,
            sorted_vv_gpu,
        )


def input_gen():
    """
    Generate data and input parameters
    for testing optimised and indexed briggs weighting
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
    robust_param = 2
    grid_size = 40
    cell_size_rad = 4.06e-5
    support = 4

    uvw = cupy.asarray(uvw_np)
    freqs = cupy.asarray(freqs_np)
    vis = cupy.asarray(vis_np)
    weights = cupy.asarray(weights_np)

    return (
        uvw_np,
        freqs_np,
        weights_np,
        freqs,
        uvw,
        vis,
        weights,
        robust_param,
        grid_size,
        cell_size_rad,
        support,
    )


# This function currently gives inaccurate results
# due to some bugs and is thus not tested only timed
def test_optmised_weighting():
    """
    Test for the briggs weighting function which
    bucket sorts the weights,the output weight array is
    of a different shape than the input weights array
    """
    indexed = 0
    (
        uvw_np,
        freqs_np,
        weights_np,
        freqs,
        uvw,
        vis,
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

    tile_size_u = 32
    tile_size_v = 16

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
        sorted_weights,
        output_weights,
        sorted_tile,
    ) = gen_gpu_arrays(num_visibilities, indexed)

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
        sorted_weights,
        sorted_tile,
        sorted_vis,
        tile_offsets,
        num_points_in_tiles,
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


def test_indexed_weighting():
    """
    Tests the briggs weighting function that only
    bucket sorts the indices for the weights and visibilites,
    the output weight array has the same shape as the input weight
    """

    # Prepare the data and run the indexed weighting function
    indexed = 1
    (
        uvw_np,
        freqs_np,
        weights_np,
        freqs,
        uvw,
        vis,
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
    tile_size_u = 32
    tile_size_v = 16

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
        sorted_tile,
        sorted_vis_index,
        sorted_uu,
        sorted_vv,
    ) = gen_gpu_arrays(num_visibilities, indexed)

    output_weights_cpu = np.zeros_like(weights, dtype=np.float64)
    output_weights = cupy.asarray(output_weights_cpu, dtype=np.float64)

    tiled_indexing(
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

    # Prepare the data and run the reference weighting function
    max_abs_uv = 16011.076569511299
    num_pols = 1
    num_baselines = uvw.shape[1]
    num_times = uvw.shape[0]
    num_channels = freqs.shape[0]
    weights_grid_uv = np.zeros(
        (grid_size, grid_size, num_pols), dtype=np.float64
    )

    reference_briggs_weights(
        uvw_np,
        freqs_np,
        max_abs_uv,
        weights_grid_uv,
        grid_size,
        robust_param,
        weights_np,
        output_weights_cpu,
        num_channels,
        num_baselines,
        num_pols,
        num_times,
    )
    # Test results
    assert np.allclose(output_weights_cpu, output_weights)


if __name__ == "__main__":
    test_optmised_weighting()
    test_indexed_weighting()
