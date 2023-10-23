# See the LICENSE file at the top-level directory of this distribution.
"""Test for FFT convolution."""

import numpy as np
import scipy.signal as sig
from ska_sdp_func.numeric_functions import fft_convolution

try:
    import cupy
except ImportError:
    cupy = None


def test_fft_convolution():
    """Test the FFT convolution function"""
    # Run FFT convolution test on CPU using numpy arrays.

    in1_dim = 156
    in2_dim = 512

    # Test for complex double
    in1 = np.random.random_sample([in1_dim, in1_dim])
    in1 = in1 + np.random.random_sample([in1_dim, in1_dim]) * 1j
    in1 = in1.astype(dtype=np.complex128)

    in2 = np.random.random_sample([in2_dim, in2_dim])
    in2 = in2 + np.random.random_sample([in2_dim, in2_dim]) * 1j
    in2 = in2.astype(dtype=np.complex128)

    out = np.zeros_like(in1)

    out_reference = sig.fftconvolve(in1, in2, mode="same")

    print("Performing convolution at double precision on CPU using ska-sdp-func...")
    fft_convolution(in1, in2, out)

    np.testing.assert_allclose(out, out_reference)

    print("FFT convlution at double precision on CPU: Test passed")

    # # Test for complex float
    in1_float = in1.astype(dtype=np.complex64)
    in2_float = in2.astype(dtype=np.complex64)
    out_float = np.zeros_like(in1_float)

    out_reference_float = sig.fftconvolve(in1_float, in2_float, mode="same")

    print("Performing convolution at float precision on CPU using ska-sdp-func...")
    fft_convolution(in1_float, in2_float, out_float)

    np.testing.assert_array_almost_equal(out_float, out_reference_float, decimal=0)

    print("FFT convlution at float precision on CPU: Test passed")

    # Run FFT convolution test on GPU using cumpy arrays.
    if cupy:
        # Test for complex double
        in1_gpu = cupy.asarray(in1)
        in2_gpu = cupy.asarray(in2)
        out_gpu = cupy.zeros_like(in1_gpu)

        print("Performing convolution on GPU using ska-sdp-func...")
        fft_convolution(in1_gpu, in2_gpu, out_gpu)
        
        output_gpu_check = cupy.asnumpy(out_gpu)

        np.testing.assert_array_almost_equal(output_gpu_check, out_reference)

        print("FFT convlution at double precision on GPU: Test passed")

        # Test for complex float
        in1_gpu_float = cupy.asarray(in1_float)
        in2_gpu_float = cupy.asarray(in2_float)
        out_gpu_float = cupy.zeros_like(in1_gpu_float)

        print("Performing convolution at float precision on GPU using ska-sdp-func...")
        fft_convolution(in1_gpu_float, in2_gpu_float, out_gpu_float)
        
        output_gpu_check = cupy.asnumpy(out_gpu_float)

        np.testing.assert_array_almost_equal(output_gpu_check, out_reference_float, decimal=0)

        print("FFT convlution at float precision on GPU: Test passed")