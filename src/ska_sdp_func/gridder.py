# See the LICENSE file at the top-level directory of this distribution.

import ctypes
from .utility import Error, Lib, Mem

class Gridder:
    """Processing function example A.
    """
    class Handle(ctypes.Structure):
        pass

    def __init__(self, uvw, freq, vis, weight, pixsize_x_rad, pixsize_y_rad, epsilon: float,
             do_wstacking: bool):
        """Creates processing function A.

        :param par_a: Value of a.
        :type par_a: int

        :param par_b: Value of b.
        :type par_b: int

        :param par_c: Value of c.
        :type par_c: float
        """
        print("BITB")

        self._handle = None
        mem_uvw = Mem(uvw)
        mem_freq = Mem(freq)
        mem_vis = Mem(vis)
        mem_weight = Mem(weight)
        error_status = Error()

        function_create = Lib.handle().sdp_gridder_create_plan
        function_create.restype = Gridder.handle_type()
        function_create.argtypes = [
            Mem.handle_type(),
            Mem.handle_type(),
            Mem.handle_type(),
            Mem.handle_type(),
            ctypes.c_float,  # 5
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_bool,
            Error.handle_type()  # 9
        ]
        self._handle = function_create(
            mem_uvw.handle(),
            mem_freq.handle(),
            mem_vis.handle(),
            mem_weight.handle(),
            ctypes.c_float(pixsize_x_rad),  # 5
            ctypes.c_float(pixsize_y_rad),
            ctypes.c_float(epsilon),
            ctypes.c_bool(do_wstacking),
            error_status.handle() # 9
        )
        error_status.check()

    def __del__(self):
        """Releases handle to the processing function.
        """
        if self._handle:
            function_free = Lib.handle().sdp_gridder_free_plan
            function_free.argtypes = [Gridder.handle_type()]
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
        return ctypes.POINTER(Gridder.Handle)

    def exec(self, uvw, freq, vis, weight, output):
        """Demonstrate a function utilising a plan.

        :param output: Output buffer.
        :type output: numpy.ndarray
        """
        if self._handle is None:
            raise RuntimeError("Function plan not ready")

        mem_uvw = Mem(uvw)
        mem_freq = Mem(freq)
        mem_vis = Mem(vis)
        mem_weight = Mem(weight)
        mem_output = Mem(output)
        error_status = Error()
        function_exec = Lib.handle().sdp_gridder_exec
        function_exec.argtypes = [
            Mem.handle_type(),
            Mem.handle_type(),
            Mem.handle_type(),
            Mem.handle_type(),
            Gridder.handle_type(),
            Mem.handle_type(),
            Error.handle_type()
        ]
        function_exec(
            mem_uvw.handle(),
            mem_freq.handle(),
            mem_vis.handle(),
            mem_weight.handle(),
            self._handle,
            mem_output.handle(),
            error_status.handle()
        )
        error_status.check()
