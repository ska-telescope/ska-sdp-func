# See the LICENSE file at the top-level directory of this distribution.

"""Test example function."""

import numpy

from ska_sdp_func import FunctionExampleA


def test_function_example_a():
    """Test FunctionExampleA."""
    # Define parameters and create plan
    par_a = 5
    par_b = 10
    par_c = 0.1
    function_example_a = FunctionExampleA(par_a, par_b, par_c)

    # Run function_example_a
    output_vector = numpy.zeros(par_a * par_b, dtype=numpy.float32)
    function_example_a.exec(output_vector)
