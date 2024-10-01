# See the LICENSE file at the top-level directory of this distribution.

"""Test RFI flagger (fixed threshold and dynamic) functions."""

import numpy
import pytest

from ska_sdp_func.visibility import flagger_dynamic_threshold


@pytest.fixture(scope="module", name="vis_data")
def visibility_data():
    """visibility data fixture"""
    num_times = 50
    num_baselines = 3
    num_freqs = 100
    num_pols = 4

    vis = numpy.zeros(
        shape=(num_times, num_baselines, num_freqs, num_pols),
        dtype=numpy.complex128,
    )
    vis[:, :, :, :] = complex(1, 1)

    vis[10, 0, 28, :] = 20 + 4j
    vis[36, 0, 14, 0] = vis[36, 0, 14, 0] + 0.08 + 0.08j
    vis[27, 1, :, 2] = 20 + 30j

    return vis


def test_dynamic_flagger(vis_data):
    """Test dynamic RFI flagger"""

    # below line is added to consider broadband RFI

    alpha = 0.5
    threshold_magnitudes = 3.5
    threshold_variations = 3.5
    threshold_broadband = 3.5
    sampling_step = 1
    window = 0
    window_median_history = 20

    flags = numpy.zeros(vis_data.shape, dtype=numpy.int32)
    expected_flags = numpy.zeros(vis_data.shape, dtype=numpy.int32)

    expected_flags[9, 0, 28, :] = 1
    expected_flags[10, 0, 28, :] = 1
    expected_flags[11, 0, 28, :] = 1
    expected_flags[36, 0, 14, 0] = 1
    expected_flags[27, 1, :, 2] = 1

    flagger_dynamic_threshold(
        vis_data,
        flags,
        alpha,
        threshold_magnitudes,
        threshold_variations,
        threshold_broadband,
        sampling_step,
        window,
        window_median_history,
    )
    assert (expected_flags == flags).all()
