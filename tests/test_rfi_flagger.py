# See the LICENSE file at the top-level directory of this distribution.

import numpy
try:
    import cupy
except ImportError:
    cupy = None

from ska_sdp_func import sum_threshold_rfi_flagger

def test_rfi_flagger():
    # Run DFT test on CPU, using numpy arrays.
    num_timesamples = 100
    num_baselines = 15
    num_channels = 128
    num_polarisations = 4
    max_sequence_length = 32
    num_sequence_el = 5
    
    vis = numpy.zeros([num_timesamples, num_baselines, num_channels, num_polarisations], dtype=numpy.complex128)
    thresholds = numpy.zeros([num_sequence_el], dtype=numpy.float64)
    flags = numpy.zeros([num_timesamples, num_baselines, num_channels, num_polarisations], dtype=numpy.int32)
    
    print("Testing DFT on CPU from ska-sdp-func...")
    sum_threshold_rfi_flagger(vis, thresholds, flags, max_sequence_length)
    print("DFT on CPU: Test passed")

