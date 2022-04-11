# See the LICENSE file at the top-level directory of this distribution.

import ctypes
from .utility import Error, Lib, Mem

class FunctionExampleA:
    """Processing function example A.
    """
    class Handle(ctypes.Structure):
        pass

    def __init__(self, par_a, par_b, par_c):
        """Creates processing function A.

        :param par_a: Value of a.
        :type par_a: int

        :param par_b: Value of b.
        :type par_b: int

        :param par_c: Value of c.
        :type par_c: float
        """
        self._handle = None
        error_status = Error()
        function_create = Lib.handle().sdp_function_example_a_create_plan
        function_create.restype = FunctionExampleA.handle_type()
        function_create.argtypes = [
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_float,
            Error.handle_type()
        ]
        self._handle = function_create(
            ctypes.c_int(par_a),
            ctypes.c_int(par_b),
            ctypes.c_float(par_c),
            error_status.handle()
        )
        error_status.check()

    def __del__(self):
        """Releases handle to the processing function.
        """
        if self._handle:
            function_free = Lib.handle().sdp_function_example_a_free_plan
            function_free.argtypes = [FunctionExampleA.handle_type()]
            function_free(self._handle)

    def handle(self):
        """Returns a handle to the wrapped processing function.

        Use this handle when calling the function in the compiled library.

        :return: Handle to wrapped function.
        :rtype: ctypes.POINTER(FunctionExampleA.Handle)
        """
        return self._handle

    @staticmethod
    def handle_type():
        """Static convenience method to return the ctypes handle type.

        Use this when defining the list of argument types.

        :return: Type of the function handle.
        :rtype: ctypes.POINTER(FunctionExampleA.Handle)
        """
        return ctypes.POINTER(FunctionExampleA.Handle)

    def exec(self, output):
        """Demonstrate a function utilising a plan.

        :param output: Output buffer.
        :type output: numpy.ndarray
        """
        if self._handle is None:
            raise RuntimeError("Function plan not ready")

        mem_output = Mem(output)
        error_status = Error()
        function_exec = Lib.handle().sdp_function_example_a_exec
        function_exec.argtypes = [
            FunctionExampleA.handle_type(),
            Mem.handle_type(),
            Error.handle_type()
        ]
        function_exec(
            self._handle,
            mem_output.handle(),
            error_status.handle()
        )
        error_status.check()
