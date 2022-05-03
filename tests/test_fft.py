# See the LICENSE file at the top-level directory of this distribution.

import numpy
try:
    import cupy
except ImportError:
    cupy = None

from ska_sdp_func import Fft

def test_fft():
    num_points = 8192
    input = numpy.random.random(num_points) + 0j
    if cupy:
        input_gpu = cupy.asarray(input)
        output_gpu = cupy.zeros_like(input_gpu)
        fft = Fft(input.dtype, "GPU", "C2C", input.ndim, input.shape, 1, True)
        fft.exec(input_gpu, output_gpu)
        output = cupy.asnumpy(output_gpu)
        numpy.testing.assert_allclose(output, numpy.fft.fft(input))
