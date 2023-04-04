# See the LICENSE file at the top-level directory of this distribution.
"""Test for FFT convolution."""

import numpy as np
import scipy.signal as sig
from ska_sdp_func.numeric_functions import fft_convolution


def test_fft_convolution():
    """Test the FFT convolution function"""

    in1_dim = 1024
    in2_dim = 2048

    in1 = np.random.random_sample([in1_dim, in1_dim])
    in1 = in1 + np.random.random_sample([in1_dim, in1_dim]) * 1j
    in1.astype(dtype=np.complex128)

    in2 = np.random.random_sample([in2_dim, in2_dim])
    in2 = in2 + np.random.random_sample([in2_dim, in2_dim]) * 1j
    in2.astype(dtype=np.complex128)

    out = np.zeros_like(in1)

    out_reference = sig.convolve(in1, in2, mode="same")

    print("Performing convolution for complex double on CPU using ska-sdp-func...")
    fft_convolution(in1, in2, out)

    np.testing.assert_allclose(out, out_reference)

    in1_float = np.random.random_sample([in1_dim, in1_dim])
    in1_float = in1_float + np.random.random_sample([in1_dim, in1_dim]) * 1j
    in1_float.astype(dtype=np.complex64)

    in2_float = np.random.random_sample([in2_dim, in2_dim])
    in2_float = in2_float + np.random.random_sample([in2_dim, in2_dim]) * 1j
    in2_float.astype(dtype=np.complex64)

    out_float = np.zeros_like(in1_float)

    out_reference_float = sig.convolve(in1_float, in2_float, mode="same")

    print("Performing convolution for complex float on CPU using ska-sdp-func...")
    fft_convolution(in1_float, in2_float, out_float)

    np.testing.assert_allclose(out_float, out_reference_float)

    print("FFT convlution on CPU: Test passed")
