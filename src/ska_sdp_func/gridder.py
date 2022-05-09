# See the LICENSE file at the top-level directory of this distribution.

import ctypes
from .utility import Error, Lib, Mem
import numpy as np

try:
    import cupy
    print("All good!")
except ImportError:
    cupy = None

class Gridder:
    """Processing function example A.
    """
    class Handle(ctypes.Structure):
        pass

    def __init__(self, uvw, freq, vis, weight, pixsize_x_rad, pixsize_y_rad, epsilon: float,
             do_wstacking: bool, dirty_image):
        """Creates processing function A.

        :param par_a: Value of a.
        :type par_a: int

        :param par_b: Value of b.
        :type par_b: int

        :param par_c: Value of c.
        :type par_c: float
        """

        self._handle = None
        mem_uvw = Mem(uvw)
        mem_freq = Mem(freq)
        mem_vis = Mem(vis)
        mem_weight = Mem(weight)
        mem_dirty_image = Mem(dirty_image)
        error_status = Error()

        # check types consistent here???

        if do_wstacking:
            min_abs_w, max_abs_w = Gridder.get_w_range(uvw, freq_hz)
        else:
            min_abs_w = 0
            max_abs_w = 0

        function_create = Lib.handle().sdp_gridder_create_plan
        function_create.restype = Gridder.handle_type()
        function_create.argtypes = [
            Mem.handle_type(),
            Mem.handle_type(),
            Mem.handle_type(),
            Mem.handle_type(),
            ctypes.c_double,  # 5
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_bool,    # 10
            Mem.handle_type(),
            Error.handle_type()
        ]
        self._handle = function_create(
            mem_uvw.handle(),
            mem_freq.handle(),
            mem_vis.handle(),
            mem_weight.handle(),
            ctypes.c_double(pixsize_x_rad),  # 5
            ctypes.c_double(pixsize_y_rad),
            ctypes.c_double(epsilon),
            ctypes.c_double(min_abs_w),
            ctypes.c_double(max_abs_w),
            ctypes.c_bool(do_wstacking),  # 10
            mem_dirty_image.handle(),
            error_status.handle()
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
    def get_w_range(uvw, freq_hz):

        if type(uvw) == np.ndarray:
            min_abs_w = np.amin(np.abs(uvw[:, 2]))
            max_abs_w = np.amax(np.abs(uvw[:, 2]))
        elif cupy and type(uvw) == cupy.ndarray:
            min_abs_w = cupy.amin(cupy.abs(uvw[:, 2]))
            max_abs_w = cupy.amax(cupy.abs(uvw[:, 2]))
        else:
            print("bad type!!")
            return -1, -1

        min_abs_w *= freq_hz[ 0] / 299792458.0
        max_abs_w *= freq_hz[-1] / 299792458.0

        return min_abs_w, max_abs_w

    @staticmethod
    def handle_type():
        """Static convenience method to return the ctypes handle type.

        Use this when defining the list of argument types.

        :return: Type of the function handle.
        :rtype: ctypes.POINTER(FunctionExampleA.Handle)
        """
        return ctypes.POINTER(Gridder.Handle)

    def exec(self, uvw, freq, vis, weight, dirty_image):
        """Demonstrate a function utilising a plan.

        :param dirty_image: Output buffer.
        :type dirty_image: numpy.ndarray
        """
        if self._handle is None:
            raise RuntimeError("Function plan not ready")

        mem_uvw = Mem(uvw)
        mem_freq = Mem(freq)
        mem_vis = Mem(vis)
        mem_weight = Mem(weight)
        mem_dirty_image = Mem(dirty_image)
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
            mem_dirty_image.handle(),
            error_status.handle()
        )
        error_status.check()
