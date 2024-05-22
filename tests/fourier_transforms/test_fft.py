# See the LICENSE file at the top-level directory of this distribution.

"""Test FFT functions."""

import numpy

try:
    import cupy
except ImportError:
    cupy = None

from ska_sdp_func.fourier_transforms import Fft


def test_fft_1d():
    """Test 1D FFT for consistency with numpy."""
    input_data = numpy.random.random(256) + 0j
    output_ref = numpy.fft.fft(input_data)

    # Test CPU version.
    output_cpu = numpy.zeros_like(input_data)
    fft_cpu = Fft(input_data, output_cpu, 1, True)
    fft_cpu.exec(input_data, output_cpu)
    numpy.testing.assert_allclose(output_cpu, output_ref)

    # Test GPU version.
    if cupy:
        input_gpu = cupy.asarray(input_data)
        output_gpu = cupy.zeros_like(input_gpu)
        fft_gpu = Fft(input_gpu, output_gpu, 1, True)
        fft_gpu.exec(input_gpu, output_gpu)
        output = cupy.asnumpy(output_gpu)
        numpy.testing.assert_allclose(output, output_ref)


def test_fft_2d_ones():
    """Test 2D FFT with input all 1.0."""
    input_data = numpy.ones((256, 256)) + 0j
    output_ref = numpy.fft.fft2(input_data)

    # Test CPU version.
    output_cpu = numpy.zeros_like(input_data)
    fft_cpu = Fft(input_data, output_cpu, 2, True)
    fft_cpu.exec(input_data, output_cpu)
    numpy.testing.assert_allclose(output_cpu, output_ref)

    # Test GPU version.
    if cupy:
        input_gpu = cupy.asarray(input_data)
        output_gpu = cupy.zeros_like(input_gpu)
        fft_gpu = Fft(input_gpu, output_gpu, 2, True)
        fft_gpu.exec(input_gpu, output_gpu)
        output = cupy.asnumpy(output_gpu)
        numpy.testing.assert_allclose(output, output_ref)
        numpy.testing.assert_allclose(output[0, 0], input_data.size)


def test_fft_2d():
    """Test 2D FFT for consistency with numpy."""
    input_real = numpy.random.random((256, 512))
    input_imag = numpy.random.random(input_real.shape)
    input_data = input_real + 1j * input_imag
    output_ref = numpy.fft.fft2(input_data)

    # Test CPU version.
    output_cpu = numpy.zeros_like(input_data)
    fft_cpu = Fft(input_data, output_cpu, 2, True)
    fft_cpu.exec(input_data, output_cpu)
    numpy.testing.assert_allclose(output_cpu, output_ref)

    # Test GPU version.
    if cupy:
        input_gpu = cupy.asarray(input_data)
        output_gpu = cupy.zeros_like(input_gpu)
        fft_gpu = Fft(input_gpu, output_gpu, 2, True)
        fft_gpu.exec(input_gpu, output_gpu)
        output = cupy.asnumpy(output_gpu)
        numpy.testing.assert_allclose(output, output_ref)


def test_fft_2d_inverse():
    """Test 2D iFFT for consistency with numpy."""
    input_real = numpy.random.random((256, 512))
    input_imag = numpy.random.random(input_real.shape)
    input_data = input_real + 1j * input_imag
    output_ref = numpy.fft.ifft2(input_data)

    # Test CPU version.
    output_cpu = numpy.zeros_like(input_data)
    fft_cpu = Fft(input_data, output_cpu, 2, False)
    fft_cpu.exec(input_data, output_cpu)
    output_cpu /= input_data.size
    numpy.testing.assert_allclose(output_cpu, output_ref)

    # Test GPU version.
    if cupy:
        input_gpu = cupy.asarray(input_data)
        output_gpu = cupy.zeros_like(input_gpu)
        fft_gpu = Fft(input_gpu, output_gpu, 2, False)
        fft_gpu.exec(input_gpu, output_gpu)
        output = cupy.asnumpy(output_gpu)
        output /= input_data.size
        numpy.testing.assert_allclose(output, output_ref)


def test_fft_2d_stack():
    """Test multiple stacked 2D FFTs for consistency with numpy."""
    num_slices = 4
    input_real = numpy.random.random((num_slices, 256, 512))
    input_imag = numpy.random.random(input_real.shape)
    input_data = input_real + 1j * input_imag
    output_ref = numpy.zeros_like(input_data)
    for i in range(num_slices):
        output_ref[i, :, :] = numpy.fft.fft2(input_data[i, :, :])

    # Test CPU version.
    output_cpu = numpy.zeros_like(input_data)
    fft_cpu = Fft(input_data, output_cpu, 2, True)
    fft_cpu.exec(input_data, output_cpu)
    numpy.testing.assert_allclose(output_cpu, output_ref)

    # Test GPU version.
    if cupy:
        input_gpu = cupy.asarray(input_data)
        output_gpu = cupy.zeros_like(input_gpu)
        fft_gpu = Fft(input_gpu, output_gpu, 2, True)
        fft_gpu.exec(input_gpu, output_gpu)
        output = cupy.asnumpy(output_gpu)
        numpy.testing.assert_allclose(output, output_ref)
