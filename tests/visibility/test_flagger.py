# See the LICENSE file at the top-level directory of this distribution.

"""Test RFI flagger (fixed threshold and dynamic) functions."""

import numpy
import pytest

from ska_sdp_func.visibility import flagger_fixed_threshold, flagger_dynamic_threshold

@pytest.fixture(scope="module",name="visibility_data")
def visibility_data():
    """visibility data fixture"""
    num_times = 50
    num_baselines = 3
    num_freqs = 100
    num_pols = 4

    vis = numpy.zeros(shape=(num_times, num_baselines, num_freqs, num_pols), dtype=numpy.complex128)
    
    for t in range(num_times):
        for b in range(num_baselines):
            for c in range(num_freqs):
                vis[t, b, c, 0] = numpy.complex(1 + 0.01 * t + 0.01 * numpy.mod(c, 2) + b * 0.1 , 0.01 * t)
                vis[t, b, c, 1] = numpy.complex(0.1 * (1 + 0.01 * t +  b * 0.1) + 0.01 * t)
                vis[t, b, c, 2] = numpy.complex(0.1 * (1 + 0.01 * t +  b * 0.1) + 0.01 * t)
                vis[t, b, c, 3] = numpy.complex(1 + 0.01 * t + 0.01 * numpy.mod(c, 2) + b * 0.1 , 0.01 * t)
    
    vis[10, 0, 28, :] = 20 + 4j
    vis[36, 0, 14, 0] = vis[36, 0, 14, 0] + 0.08 + 0.08j
    
    return vis


def test_fixed_flagger(visibility_data):
    """Test fixed threshold RFI flagger."""
    
    what_quantile_for_vis = 0.98
    what_quantile_for_changes = 0.989
    sampling_step = 1
    alpha = 0.5
    window = 0
    parameters = numpy.array([what_quantile_for_vis, what_quantile_for_changes, sampling_step, alpha, window], dtype=numpy.float64)

    flags =  numpy.zeros(visibility_data.shape, dtype=numpy.int32)
    expected_flags =  numpy.zeros(visibility_data.shape, dtype=numpy.int32)
    
    expected_flags[10, 0, 28, :] = 1
    expected_flags[36, 0, 14, 0] = 1
    
    flagger_fixed_threshold(visibility_data, parameters, flags)
    for t in range(visibility_data.shape[0]):
        for b in range(visibility_data.shape[1]):
            for c in range(visibility_data.shape[2]):
                for p in range(visibility_data.shape[3]):
                    if expected_flags[t, b, c, p] != flags[t, b, c, p]:
                        print("expected_flags: ", expected_flags[t, b, c, p], " at  ", t, ",",b, ",", c, ",",p)
                        print("flags: ", flags[t, b, c, p], " at  ", t, ",",b, ",", c, ",",p)
    
    assert (expected_flags == flags).all()