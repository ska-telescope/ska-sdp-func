# See the LICENSE file at the top-level directory of this distribution.

"""Test two-state RFI functions."""

import numpy as np

from ska_sdp_func import twosm_rfi_flagger


def data_preparation(
    spectro,
    flags,
    num_timesamples,
    num_channels,
):
    """Prepares data for RFI test."""
    arr = np.array([0.1, 0.2, 0.3, 2.8, 2.81, 2.805, 0.1])
    insertion_time = np.random.randint(0, num_timesamples - 9)
    freq = np.random.randint(0, num_channels - 1)
    for i in range(7):
        spectro[insertion_time + i][freq] = arr[i]

    for i in range(7):
        flags[insertion_time + i][freq] = 1

def test_rfi_flagger():
    """Prepares data for RFI test."""
    num_channels = 200
    num_timesamples = 1000
    thresholds = np.array([0.05, 0.05])
    antennas = np.array([0, 1, 2, 3, 4, 5], dtype=np.int32)

    # Initialise numpy arrays
    spectrogram = (
        np.zeros([num_timesamples, num_channels]) + 0j
    )
    flags_by_algo = np.zeros(spectrogram.shape, dtype=np.int32)
    flags_as_expected = np.zeros(spectrogram.shape, dtype=np.int32)

    data_preparation(
        spectrogram,
        flags_as_expected,
        num_timesamples,
        num_channels,
    )
    twosm_rfi_flagger(spectrogram, thresholds, flags_by_algo)
    np.testing.assert_array_equal(flags_by_algo, flags_as_expected)
    print("test passed!")
