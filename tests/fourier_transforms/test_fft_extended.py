# See the LICENSE file at the top-level directory of this distribution.

"""Test FFT_extended functions."""

import numpy

try:
    import cupy
except ImportError:
    cupy = None

from ska_sdp_func.fourier_transforms.fft import Fft_extended


def test_fft_2d_ones():
    """Test 2D FFT with input all 1.0."""
    input_data = numpy.ones((512, 512)) + 0j
    output_ref = numpy.fft.fft2(input_data)

    # Test CPU version.
    if cupy:
        idata_1d_gpu = cupy.zeros(512 * 4 * 128, dtype=numpy.complex128)
        odata_1d_gpu = cupy.zeros_like(idata_1d_gpu)
        output_cpu = numpy.zeros_like(input_data)
        fft_cpu = Fft_extended(
            idata_1d_gpu, odata_1d_gpu, input_data, output_cpu, 1, True, 4, 128
        )
        fft_cpu.exec(input_data, output_cpu)
        numpy.testing.assert_allclose(output_cpu, output_ref)


def test_fft_2d():
    """Test 2D FFT for consistency with numpy."""
    input_real = numpy.random.random((512, 512))
    input_imag = numpy.random.random(input_real.shape)
    input_data = input_real + 1j * input_imag
    output_ref = numpy.fft.fft2(input_data)

    # Test CPU version.
    if cupy:
        idata_1d_gpu = cupy.zeros(512 * 4 * 128, dtype=numpy.complex128)
        odata_1d_gpu = cupy.zeros_like(idata_1d_gpu)
        output_cpu = numpy.zeros_like(input_data)
        fft_cpu = Fft_extended(
            idata_1d_gpu, odata_1d_gpu, input_data, output_cpu, 1, True, 4, 128
        )
        fft_cpu.exec(input_data, output_cpu)
        numpy.testing.assert_allclose(output_cpu, output_ref)


def test_fft_2d_inverse():
    """Test 2D iFFT for consistency with numpy."""
    input_real = numpy.random.random((512, 512))
    input_imag = numpy.random.random(input_real.shape)
    input_data = input_real + 1j * input_imag
    output_ref = numpy.fft.ifft2(input_data)

    # Test CPU version.
    if cupy:
        idata_1d_gpu = cupy.zeros(512 * 4 * 128, dtype=numpy.complex128)
        odata_1d_gpu = cupy.zeros_like(idata_1d_gpu)
        output_cpu = numpy.zeros_like(input_data)
        fft_cpu = Fft_extended(
            idata_1d_gpu,
            odata_1d_gpu,
            input_data,
            output_cpu,
            1,
            False,
            4,
            128,
        )
        fft_cpu.exec(input_data, output_cpu)
        output_cpu /= input_data.size
        numpy.testing.assert_allclose(output_cpu, output_ref)
