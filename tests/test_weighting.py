# See the LICENSE file at the top-level directory of this distribution.

"""Test weighting functions."""

import numpy as np

from ska_sdp_func.weighting import get_uv_range, uniform_weights


def _make_inputs():
    """Make mock frequency array and uv coverage."""

    # Create frequency array - do not change, hardcoded asserts
    freqs = np.array([1e9, 1.1e9, 1.2e9])

    # Create uv track - do not change, hardcoded asserts
    angle = np.arange(-np.pi, np.pi, np.pi / 10)
    uv_size = len(angle)
    uvw = np.zeros((uv_size, 3))
    uvw[:, 2] = 0.0
    uv_max_meters = 4000
    for i in range(uv_size):
        uvw[i, 0] = uv_max_meters * np.sin(angle[i])
        uvw[i, 1] = uv_max_meters * np.cos(angle[i])

    return freqs, uvw


def _define_control_outputs():
    """Define control outputs to compare."""

    max_abs_uv_control = 16011.076569511299

    weights_control = np.array(
        [
            [
                [0.14285714, 0.14285714, 0.14285714, 0.14285714],
                [0.14285714, 0.14285714, 0.14285714, 0.14285714],
                [0.14285714, 0.14285714, 0.14285714, 0.14285714],
            ],
            [
                [0.14285714, 0.14285714, 0.14285714, 0.14285714],
                [0.14285714, 0.14285714, 0.14285714, 0.14285714],
                [0.14285714, 0.14285714, 0.14285714, 0.14285714],
            ],
            [
                [0.14285714, 0.14285714, 0.14285714, 0.14285714],
                [0.25, 0.25, 0.25, 0.25],
                [0.25, 0.25, 0.25, 0.25],
            ],
            [
                [0.25, 0.25, 0.25, 0.25],
                [0.25, 0.25, 0.25, 0.25],
                [0.25, 0.25, 0.25, 0.25],
            ],
            [
                [0.25, 0.25, 0.25, 0.25],
                [0.25, 0.25, 0.25, 0.25],
                [0.25, 0.25, 0.25, 0.25],
            ],
            [
                [0.14285714, 0.14285714, 0.14285714, 0.14285714],
                [0.14285714, 0.14285714, 0.14285714, 0.14285714],
                [0.14285714, 0.14285714, 0.14285714, 0.14285714],
            ],
            [
                [0.14285714, 0.14285714, 0.14285714, 0.14285714],
                [0.14285714, 0.14285714, 0.14285714, 0.14285714],
                [0.14285714, 0.14285714, 0.14285714, 0.14285714],
            ],
            [
                [0.14285714, 0.14285714, 0.14285714, 0.14285714],
                [0.25, 0.25, 0.25, 0.25],
                [0.25, 0.25, 0.25, 0.25],
            ],
            [
                [0.25, 0.25, 0.25, 0.25],
                [0.25, 0.25, 0.25, 0.25],
                [0.25, 0.25, 0.25, 0.25],
            ],
            [
                [0.25, 0.25, 0.25, 0.25],
                [0.25, 0.25, 0.25, 0.25],
                [0.25, 0.25, 0.25, 0.25],
            ],
            [
                [0.14285714, 0.14285714, 0.14285714, 0.14285714],
                [0.14285714, 0.14285714, 0.14285714, 0.14285714],
                [0.14285714, 0.14285714, 0.14285714, 0.14285714],
            ],
            [
                [0.14285714, 0.14285714, 0.14285714, 0.14285714],
                [0.14285714, 0.14285714, 0.14285714, 0.14285714],
                [0.14285714, 0.14285714, 0.14285714, 0.14285714],
            ],
            [
                [0.14285714, 0.14285714, 0.14285714, 0.14285714],
                [0.25, 0.25, 0.25, 0.25],
                [0.25, 0.25, 0.25, 0.25],
            ],
            [
                [0.14285714, 0.14285714, 0.14285714, 0.14285714],
                [0.25, 0.25, 0.25, 0.25],
                [0.25, 0.25, 0.25, 0.25],
            ],
            [
                [0.14285714, 0.14285714, 0.14285714, 0.14285714],
                [0.14285714, 0.14285714, 0.14285714, 0.14285714],
                [0.14285714, 0.14285714, 0.14285714, 0.14285714],
            ],
            [
                [0.14285714, 0.14285714, 0.14285714, 0.14285714],
                [0.14285714, 0.14285714, 0.14285714, 0.14285714],
                [0.14285714, 0.14285714, 0.14285714, 0.14285714],
            ],
            [
                [0.25, 0.25, 0.25, 0.25],
                [0.25, 0.25, 0.25, 0.25],
                [0.25, 0.25, 0.25, 0.25],
            ],
            [
                [0.25, 0.25, 0.25, 0.25],
                [0.25, 0.25, 0.25, 0.25],
                [0.25, 0.25, 0.25, 0.25],
            ],
            [
                [0.25, 0.25, 0.25, 0.25],
                [0.25, 0.25, 0.25, 0.25],
                [0.25, 0.25, 0.25, 0.25],
            ],
            [
                [0.25, 0.25, 0.25, 0.25],
                [0.25, 0.25, 0.25, 0.25],
                [0.25, 0.25, 0.25, 0.25],
            ],
        ]
    )

    return max_abs_uv_control, weights_control


def test_get_uv_range():
    """Test get_uv_range function"""

    freqs, uvw = _make_inputs()
    max_abs_uv_control, _ = _define_control_outputs()

    tol = 1.0e-8

    # Run get_uv_range
    max_abs_uv = get_uv_range(uvw, freqs)
    diff_max_abs_uv = max_abs_uv - max_abs_uv_control
    assert (
        np.abs(diff_max_abs_uv)
    ) < tol, f"diff < {tol} expected, got: {np.abs(diff_max_abs_uv)}"


def test_uniform_weights():
    """Test uniform weights calculation."""

    freqs, uvw = _make_inputs()
    max_abs_uv_c, weights_c = _define_control_outputs()
    weights = np.zeros((uvw.shape[0], freqs.shape[0], 4))
    grid_size = 4
    _, weights = uniform_weights(uvw, freqs, max_abs_uv_c, grid_size, weights)
    assert np.allclose(weights, weights_c), "The weights are not identical"
