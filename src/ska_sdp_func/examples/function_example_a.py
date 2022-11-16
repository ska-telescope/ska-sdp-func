# See the LICENSE file at the top-level directory of this distribution.

"""Module for example function."""

import ctypes

from ..utility import Lib, Mem, StructWrapper


class FunctionExampleA(StructWrapper):
    """Processing function example A."""

    def __init__(self, par_a: int, par_b: int, par_c: float) -> None:
        """Creates processing function A.

        Args:
            par_a: Value of a
            par_b: Value of b
            par_c: Value of c
        """
        create_args = (par_a, par_b, par_c)
        super().__init__(
            Lib.sdp_function_example_a_create_plan,
            create_args,
            Lib.sdp_function_example_a_free_plan,
        )

    def exec(self, output) -> None:
        """Demonstrate a function utilising a plan.

        :param output: Output buffer.
        :type output: numpy.ndarray
        """
        Lib.sdp_function_example_a_exec(self, Mem(output))


Lib.wrap_func(
    "sdp_function_example_a_create_plan",
    restype=FunctionExampleA.handle_type(),
    argtypes=[
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_float,
    ],
    check_errcode=True,
)

Lib.wrap_func(
    "sdp_function_example_a_free_plan",
    restype=None,
    argtypes=[FunctionExampleA.handle_type()],
    check_errcode=True,
)


Lib.wrap_func(
    "sdp_function_example_a_exec",
    restype=None,
    argtypes=[
        FunctionExampleA.handle_type(),
        Mem.handle_type(),
    ],
    check_errcode=True,
)
