# See the LICENSE file at the top-level directory of this distribution.

import numpy
try:
    import cupy
except ImportError:
    cupy = None

from ska.sdp.func import FunctionA


def test_function_a():

    # Define parameters and create plan
    a = 5
    b = 10
    c = 0.1
    function_a = FunctionA(a, b, c)

    # Run function_a
    output_vector = numpy.zeros(a*b, dtype=numpy.float32)
    function_a.exec(output_vector)

    print("Got to the end so it must work...")
