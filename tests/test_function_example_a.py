# See the LICENSE file at the top-level directory of this distribution.

import numpy

from ska.sdp.func import FunctionExampleA


def test_function_example_a():

    # Define parameters and create plan
    a = 5
    b = 10
    c = 0.1
    function_example_a = FunctionExampleA(a, b, c)

    # Run function_example_a
    output_vector = numpy.zeros(a*b, dtype=numpy.float32)
    function_example_a.exec(output_vector)
