# See the LICENSE file at the top-level directory of this distribution.

"""Test weighting functions."""

from math import floor

import numpy as np

try:
    import cupy as cp
except ImportError:
    cp = None


from ska_sdp_func.visibility.weighting import briggs_weights, uniform_weights


def prime_3x3_input():
    freqs = np.array([1e9, 1.1e9, 1.2e9])

    uvwgeneral = [[[2, 3, 5], [7, 11, 13], [17, 19, 23]]]
    inputweight = [
        [[[10], [31], [21]], [[10], [31], [21]], [[10], [31], [21]]]
    ]

    uvw = np.asarray(uvwgeneral, dtype=np.float64)
    input_weights = np.asarray(inputweight, dtype=np.float64)

    max_abs_uv_control = 16011.076569511299

    return freqs, uvw, max_abs_uv_control, input_weights


def gpu_input():
    freqs = np.array([1e9, 1.1e9, 1.2e9, 1.3e9, 1.4e9, 1.5e9])

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

    input_weight = [
        [
            [[24], [38], [47], [81], [21], [41]],
            [[24], [38], [47], [81], [21], [41]],
            [[24], [38], [47], [81], [21], [41]],
            [[24], [38], [47], [81], [21], [41]],
            [[24], [38], [47], [81], [21], [41]],
            [[24], [38], [47], [81], [21], [41]],
            [[24], [38], [47], [81], [21], [41]],
            [[24], [38], [47], [81], [21], [41]],
        ],
        [
            [[24], [38], [47], [81], [21], [41]],
            [[24], [38], [47], [81], [21], [41]],
            [[24], [38], [47], [81], [21], [41]],
            [[24], [38], [47], [81], [21], [41]],
            [[24], [38], [47], [81], [21], [41]],
            [[24], [38], [47], [81], [21], [41]],
            [[24], [38], [47], [81], [21], [41]],
            [[24], [38], [47], [81], [21], [41]],
        ],
        [
            [[24], [38], [47], [81], [21], [41]],
            [[24], [38], [47], [81], [21], [41]],
            [[24], [38], [47], [81], [21], [41]],
            [[24], [38], [47], [81], [21], [41]],
            [[24], [38], [47], [81], [21], [41]],
            [[24], [38], [47], [81], [21], [41]],
            [[24], [38], [47], [81], [21], [41]],
            [[24], [38], [47], [81], [21], [41]],
        ],
        [
            [[24], [38], [47], [81], [21], [41]],
            [[24], [38], [47], [81], [21], [41]],
            [[24], [38], [47], [81], [21], [41]],
            [[24], [38], [47], [81], [21], [41]],
            [[24], [38], [47], [81], [21], [41]],
            [[24], [38], [47], [81], [21], [41]],
            [[24], [38], [47], [81], [21], [41]],
            [[24], [38], [47], [81], [21], [41]],
        ],
        [
            [[24], [38], [47], [81], [21], [41]],
            [[24], [38], [47], [81], [21], [41]],
            [[24], [38], [47], [81], [21], [41]],
            [[24], [38], [47], [81], [21], [41]],
            [[24], [38], [47], [81], [21], [41]],
            [[24], [38], [47], [81], [21], [41]],
            [[24], [38], [47], [81], [21], [41]],
            [[24], [38], [47], [81], [21], [41]],
        ],
        [
            [[24], [38], [47], [81], [21], [41]],
            [[24], [38], [47], [81], [21], [41]],
            [[24], [38], [47], [81], [21], [41]],
            [[24], [38], [47], [81], [21], [41]],
            [[24], [38], [47], [81], [21], [41]],
            [[24], [38], [47], [81], [21], [41]],
            [[24], [38], [47], [81], [21], [41]],
            [[24], [38], [47], [81], [21], [41]],
        ],
        [
            [[24], [38], [47], [81], [21], [41]],
            [[24], [38], [47], [81], [21], [41]],
            [[24], [38], [47], [81], [21], [41]],
            [[24], [38], [47], [81], [21], [41]],
            [[24], [38], [47], [81], [21], [41]],
            [[24], [38], [47], [81], [21], [41]],
            [[24], [38], [47], [81], [21], [41]],
            [[24], [38], [47], [81], [21], [41]],
        ],
        [
            [[24], [38], [47], [81], [21], [41]],
            [[24], [38], [47], [81], [21], [41]],
            [[24], [38], [47], [81], [21], [41]],
            [[24], [38], [47], [81], [21], [41]],
            [[24], [38], [47], [81], [21], [41]],
            [[24], [38], [47], [81], [21], [41]],
            [[24], [38], [47], [81], [21], [41]],
            [[24], [38], [47], [81], [21], [41]],
        ],
    ]

    max_abs_uv_control = 16011.076569511299

    uvw = np.asarray(uvw_general, dtype=np.float64)
    input_weights = np.asarray(input_weight, dtype=np.float64)

    return freqs, uvw, max_abs_uv_control, input_weights


def reference_briggs_weights(
    uvw,
    freq_hz,
    max_abs_uv,
    weights_grid_uv,
    robust_param,
    input_weight,
    output_weight,
    num_channels,
    num_baselines,
    num_pol,
    num_times,
):
    sum_weight = 0
    sum_weight2 = 0
    grid_size = weights_grid_uv.shape[0]
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
                    sum_weight += input_weight[
                        i_time, i_baseline, i_channel, i_pol
                    ]
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


def reference_uniform_weights(
    uvw,
    freq_hz,
    max_abs_uv,
    weights_grid_uv,
    input_weight,
    output_weight,
    num_channels,
    num_baselines,
    num_pol,
    num_times,
):

    grid_size = weights_grid_uv.shape[0]
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
                        output_weight[i_time, i_baseline, i_channel, i_pol] = (
                            1.0 / weight_g
                        )


def test_briggs_weights():
    # Generate inputs
    freqs, uvw, max_abs_uv_control, input_weights = prime_3x3_input()
    grid_size = 3
    num_pols = 1
    num_baselines = uvw.shape[1]
    num_times = uvw.shape[0]
    num_channels = freqs.shape[0]
    robust_param = -2.0

    # Call reference python function
    weights_grid_uv = np.zeros(
        (grid_size, grid_size, num_pols), dtype=np.float64
    )
    output_weight_ref_briggs = np.zeros(
        (num_times, num_baselines, num_channels, num_pols), dtype=np.float64
    )

    reference_briggs_weights(
        uvw,
        freqs,
        max_abs_uv_control,
        weights_grid_uv,
        robust_param,
        input_weights,
        output_weight_ref_briggs,
        num_channels,
        num_baselines,
        num_pols,
        num_times,
    )

    # Call PFL function
    weights_grid_uv = np.zeros(
        (grid_size, grid_size, num_pols), dtype=np.float64
    )
    output_weight_pfl_briggs = np.zeros(
        (num_times, num_baselines, num_channels, num_pols), dtype=np.float64
    )
    briggs_weights(
        uvw,
        freqs,
        max_abs_uv_control,
        robust_param,
        weights_grid_uv,
        input_weights,
        output_weight_pfl_briggs,
    )

    # Print the output of both functions

    assert np.allclose(
        output_weight_pfl_briggs, output_weight_ref_briggs
    ), "The briggs weights are not identical"

    if cp:
        uvw_gpu = cp.asarray(uvw)
        freqs_gpu = cp.asarray(freqs)
        weights_grid_uv_gpu = cp.zeros(
            (grid_size, grid_size, num_pols), dtype=cp.float64
        )
        input_weights_gpu = cp.asarray(input_weights)
        output_weight_pfl_gpu_briggs = cp.zeros(
            (num_times, num_baselines, num_channels, num_pols),
            dtype=cp.float64,
        )

        briggs_weights(
            uvw_gpu,
            freqs_gpu,
            max_abs_uv_control,
            robust_param,
            weights_grid_uv_gpu,
            input_weights_gpu,
            output_weight_pfl_gpu_briggs,
        )

        assert np.allclose(
            output_weight_pfl_gpu_briggs, output_weight_ref_briggs
        ), "The briggs gpu weights are not identical"


def test_uniform_weights():
    # Generate inputs
    freqs, uvw, max_abs_uv_control, input_weights = prime_3x3_input()
    grid_size = 3
    num_pols = 1
    num_baselines = uvw.shape[1]
    num_times = uvw.shape[0]
    num_channels = freqs.shape[0]

    # Call reference python function
    weights_grid_uv = np.zeros(
        (grid_size, grid_size, num_pols), dtype=np.float64
    )

    output_weight_ref_uniform = np.zeros(
        (num_times, num_baselines, num_channels, num_pols), dtype=np.float64
    )

    reference_uniform_weights(
        uvw,
        freqs,
        max_abs_uv_control,
        weights_grid_uv,
        input_weights,
        output_weight_ref_uniform,
        num_channels,
        num_baselines,
        num_pols,
        num_times,
    )

    # Call PFL function
    weights_grid_uv = np.zeros(
        (grid_size, grid_size, num_pols), dtype=np.float64
    )
    output_weight_pfl_uniform = np.zeros(
        (num_times, num_baselines, num_channels, num_pols), dtype=np.float64
    )

    uniform_weights(
        uvw,
        freqs,
        max_abs_uv_control,
        weights_grid_uv,
        input_weights,
        output_weight_pfl_uniform,
    )

    # Print the output of both functions

    assert np.allclose(
        output_weight_pfl_uniform, output_weight_ref_uniform
    ), "The uniform weights are not identical"

    if cp:
        uvw_gpu = cp.asarray(uvw)
        freqs_gpu = cp.asarray(freqs)
        weights_grid_uv_gpu = cp.zeros(
            (grid_size, grid_size, num_pols), dtype=cp.float64
        )
        input_weights_gpu = cp.asarray(input_weights)
        output_weight_pfl_gpu_uniform = cp.zeros(
            (num_times, num_baselines, num_channels, num_pols),
            dtype=cp.float64,
        )

        uniform_weights(
            uvw_gpu,
            freqs_gpu,
            max_abs_uv_control,
            weights_grid_uv_gpu,
            input_weights_gpu,
            output_weight_pfl_gpu_uniform,
        )

        assert np.allclose(
            output_weight_pfl_gpu_uniform, output_weight_ref_uniform
        ), "The uniform gpu weights are not identical"


if __name__ == "__main__":
    test_briggs_weights()
    test_uniform_weights()
