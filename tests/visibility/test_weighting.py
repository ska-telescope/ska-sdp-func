# See the LICENSE file at the top-level directory of this distribution.

"""Test weighting functions."""

import numpy as np

from ska_sdp_func.visibility.weighting import get_uv_range, uniform_weights


def _make_inputs():
    """Make mock frequency array and uv coverage."""

    # Create frequency array - do not change, hardcoded asserts
    freqs = np.array([1e9, 1.1e9, 1.2e9])

    # Create uv track - do not change, hardcoded asserts
    angle = np.arange(-np.pi, np.pi, np.pi / 10)
    uv_size = len(angle)
    uvw = np.zeros((1, uv_size, 3))
    uv_max_meters = 4000
    for i in range(uv_size):
        uvw[0, i, 0] = uv_max_meters * np.sin(angle[i])
        uvw[0, i, 1] = uv_max_meters * np.cos(angle[i])

    max_abs_uv_control = 16011.076569511299

    return freqs, uvw, max_abs_uv_control


def reference_uniform_weights(uvw, freq_hz, max_abs_uv, grid_uv, weights):
    """
    Calculate the number of hits per UV cell and use the inverse of this
    as the weight.

    :param uvw: List of UVW coordinates in metres, real-valued.
                Dimensions are [num_times, num_baselines, 3]
    :type uvw: numpy.ndarray

    :param freq_hz: List of frequencies in Hz, real-valued.
                    Dimension is [num_channels]
    :type freq_hz: numpy.ndarray

    :param max_abs_uv: Maximum absolute value of UV coordinates
                       in wavelength units, real-valued
    :type max_abs_uv: float

    :param grid_uv: A zero-valued 2D UV grid array, returns
                      the number of hits per UV cell
    :type grid_uv: numpy.ndarray

    :param weights: A zero-valued 3D array, returns the weights.
                    Dimensions are
                    [num_times, num_baselines, num_channels, num_pols]
    :type weights: numpy.ndarray
    """

    grid_size = grid_uv.shape[0]
    num_times = uvw.shape[0]
    num_baselines = uvw.shape[1]
    num_channels = freq_hz.shape[0]
    c_0 = 299792458.0

    # Generate the grid of weights.
    for i_time in range(num_times):
        for i_baseline in range(num_baselines):
            for i_channel in range(num_channels):
                grid_u = uvw[i_time, i_baseline, 0] * freq_hz[i_channel] / c_0
                grid_v = uvw[i_time, i_baseline, 1] * freq_hz[i_channel] / c_0
                idx_u = int(
                    grid_u / max_abs_uv * grid_size / 2 + grid_size / 2
                )
                idx_v = int(
                    grid_v / max_abs_uv * grid_size / 2 + grid_size / 2
                )
                if idx_u >= grid_size or idx_v >= grid_size:
                    continue
                grid_uv[idx_u, idx_v] += 1.0

    # Read from the grid of weights.
    for i_time in range(num_times):
        for i_baseline in range(num_baselines):
            for i_channel in range(num_channels):
                grid_u = uvw[i_time, i_baseline, 0] * freq_hz[i_channel] / c_0
                grid_v = uvw[i_time, i_baseline, 1] * freq_hz[i_channel] / c_0
                idx_u = int(
                    grid_u / max_abs_uv * grid_size / 2 + grid_size / 2
                )
                idx_v = int(
                    grid_v / max_abs_uv * grid_size / 2 + grid_size / 2
                )
                if idx_u >= grid_size or idx_v >= grid_size:
                    weight_g = 1.0
                else:
                    weight_g = 1.0 / grid_uv[idx_u, idx_v]
                weights[i_time, i_baseline, i_channel, :] = weight_g


def test_get_uv_range():
    """Test get_uv_range function"""

    freqs, uvw, max_abs_uv_control = _make_inputs()

    tol = 1.0e-8

    # Run get_uv_range
    max_abs_uv = get_uv_range(uvw, freqs)
    diff_max_abs_uv = max_abs_uv - max_abs_uv_control
    assert (
        np.abs(diff_max_abs_uv)
    ) < tol, f"diff < {tol} expected, got: {np.abs(diff_max_abs_uv)}"


def test_uniform_weights():
    """Test uniform weights calculation."""

    # Generate inputs.
    freqs, uvw, max_abs_uv_c = _make_inputs()
    grid_size = 4
    grid_uv = np.zeros((grid_size, grid_size))

    # Run the reference Python version.
    weights_ref = np.zeros((uvw.shape[0], uvw.shape[1], freqs.shape[0], 4))
    reference_uniform_weights(uvw, freqs, max_abs_uv_c, grid_uv, weights_ref)

    # Run the library version.
    grid_uv[:, :] = 0
    weights = np.zeros_like(weights_ref)
    uniform_weights(uvw, freqs, max_abs_uv_c, grid_uv, weights)

    # Check results are the same.
    assert np.allclose(weights, weights_ref), "The weights are not identical"
