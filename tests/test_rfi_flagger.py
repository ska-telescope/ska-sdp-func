# See the LICENSE file at the top-level directory of this distribution.

import numpy

from ska_sdp_func import sum_threshold_rfi_flagger


def threshold_calc(initial_value, rho, seq_lengths):
    thresholds = numpy.zeros(len(seq_lengths), dtype=numpy.float64)
    for i in range(len(seq_lengths)):
        m = pow(rho, numpy.log2(seq_lengths[i]))
        thresholds[i] = initial_value / m
    return thresholds


def data_preparation(spectro, flags, threshold, num_timesamples, num_channels, num_samples, num_baselines):
    for i in range(num_samples):
        time = numpy.random.randint(0, num_timesamples - 1)
        freq = numpy.random.randint(0, num_channels - 1)
        for b in range(num_baselines):
            spectro[time][b][freq][0] = threshold + 0.01
            flags[time][b][freq][0] = 1


def test_rfi_flagger():
    num_channels = 200
    num_baselines = 21
    num_timesamples = 1000
    num_polarisations = 4
    num_samples = 20
    max_sequence_length = 1
    num_sequence_el = 1
    sequence_lengths = numpy.array([1], dtype=numpy.int32)
    rho1 = 1.5

    # Initialise thresholds
    initial_threshold = 20
    thresholds = threshold_calc(initial_threshold, rho1, sequence_lengths)

    # Initialise numpy arrays
    spectrogram = numpy.zeros(
            [num_timesamples, num_baselines, num_channels, num_polarisations]) + 0j
    flags_by_algo = numpy.zeros(spectrogram.shape, dtype=numpy.int32)
    flags_as_expected = numpy.zeros(spectrogram.shape, dtype=numpy.int32)

    data_preparation(
            spectrogram, flags_as_expected, thresholds[0],
            num_timesamples, num_channels, num_samples, num_baselines)
    sum_threshold_rfi_flagger(spectrogram, thresholds, flags_by_algo, max_sequence_length)

    print(numpy.sum(flags_by_algo), "   ", numpy.sum(flags_as_expected))
    numpy.testing.assert_array_equal(flags_by_algo, flags_as_expected)
    print("test passed!")
