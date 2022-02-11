# See the LICENSE file at the top-level directory of this distribution.

from .utility import Error, Lib, Mem
import ctypes

class FunctionA:
    class Handle(ctypes.Structure):
        pass

    def __init__(self, par_a, par_b, par_c):
        self._handle = None
        error_status = Error()
        function_a_create = Lib.handle().sdp_function_a_create_plan
        function_a_create.restype = FunctionA.handle_type()
        function_a_create.argtypes = [
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_float,
            Error.handle_type()
        ]
        self._handle = function_a_create(
            ctypes.c_int(par_a),
            ctypes.c_int(par_b),
            ctypes.c_float(par_c),
            error_status.handle()
        )
        error_status.check()

    def __del__(self):
        if self._handle:
            function_a_free = Lib.handle().sdp_function_a_free_plan
            function_a_free.argtypes = [FunctionA.handle_type()]
            function_a_free(self._handle)

    def handle(self):
        return self._handle

    @staticmethod
    def handle_type():
        return ctypes.POINTER(FunctionA.Handle)

    def exec(self, output):
        if self._handle is None:
            raise RuntimeError("Function plan not ready")

        mem_output = Mem(output)
        error_status = Error()
        sdp_function_a = Lib.handle().sdp_function_a_exec
        sdp_function_a.argtypes = [
            FunctionA.handle_type(),
            Mem.handle_type(),
            Error.handle_type()
        ]
        sdp_function_a(
            self._handle,
            mem_output.handle(),
            error_status.handle()
        )
        error_status.check()
