# See the LICENSE file at the top-level directory of this distribution.

import numpy

try:
    import cupy
except ImportError:
    cupy = None

from ska_sdp_func import vector_add


def test_vector_add():
    # Run vector add test on CPU, using numpy arrays.
    input_a = numpy.random.random_sample([1000])
    input_b = numpy.random.random_sample(input_a.shape)
    output_vector = numpy.zeros(input_a.shape, dtype=input_a.dtype)
    print("Adding vectors on CPU using ska-sdp-func...")
    vector_add(input_a, input_b, output_vector)
    numpy.testing.assert_array_almost_equal(output_vector, input_a + input_b)
    print("Vector addition on CPU: Test passed")

    # Run vector add test on GPU, using cupy arrays.
    if cupy:
        input_a_gpu = cupy.asarray(input_a)
        input_b_gpu = cupy.asarray(input_b)
        output_vector_gpu = cupy.zeros(input_a.shape, dtype=input_a.dtype)
        print("Adding vectors on GPU using ska-sdp-func...")
        vector_add(input_a_gpu, input_b_gpu, output_vector_gpu)
        output_gpu_check = cupy.asnumpy(output_vector_gpu)
        numpy.testing.assert_array_almost_equal(
            output_gpu_check, input_a + input_b
        )
        print("Vector addition on GPU: Test passed")
