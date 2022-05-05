# See the LICENSE file at the top-level directory of this distribution.

import numpy
try:
    import cupy
except ImportError:
    cupy = None

from ska_sdp_func import Fft

def test_fft_1d():
    input = numpy.random.random(256) + 0j
    output_ref = numpy.fft.fft(input)
    if cupy:
        input_gpu = cupy.asarray(input)
        output_gpu = cupy.zeros_like(input_gpu)
        fft = Fft(input_gpu, output_gpu, 1, True)
        fft.exec(input_gpu, output_gpu)
        output = cupy.asnumpy(output_gpu)
        numpy.testing.assert_allclose(output, output_ref)


def test_fft_2d_ones():
    input = numpy.ones((256, 256)) + 0j
    output_ref = numpy.fft.fft2(input)
    if cupy:
        input_gpu = cupy.asarray(input)
        output_gpu = cupy.zeros_like(input_gpu)
        fft = Fft(input_gpu, output_gpu, 2, True)
        fft.exec(input_gpu, output_gpu)
        output = cupy.asnumpy(output_gpu)
        numpy.testing.assert_allclose(output, output_ref)
        numpy.testing.assert_allclose(output[0, 0], input.size)


def test_fft_2d():
    input = numpy.random.random((256, 512)) + 0j
    output_ref = numpy.fft.fft2(input)
    if cupy:
        input_gpu = cupy.asarray(input)
        output_gpu = cupy.zeros_like(input_gpu)
        fft = Fft(input_gpu, output_gpu, 2, True)
        fft.exec(input_gpu, output_gpu)
        output = cupy.asnumpy(output_gpu)
        numpy.testing.assert_allclose(output, output_ref)


def test_fft_2d_stack():
    num_slices = 4
    input = numpy.random.random((num_slices, 256, 512)) + 0j
    output_ref = numpy.zeros_like(input)
    for i in range(num_slices):
        output_ref[i, :, :] = numpy.fft.fft2(input[i, :, :])
    if cupy:
        input_gpu = cupy.asarray(input)
        output_gpu = cupy.zeros_like(input_gpu)
        fft = Fft(input_gpu, output_gpu, 2, True)
        fft.exec(input_gpu, output_gpu)
        output = cupy.asnumpy(output_gpu)
        numpy.testing.assert_allclose(output, output_ref)
