# See the LICENSE file at the top-level directory of this distribution.

import ctypes
import numpy

from .utility import Error, Lib, Mem

class Fft:
    """Interface to SDP FFT.
    """
    class Handle(ctypes.Structure):
        pass

    class FftType:
        SDP_FFT_C2C = 0

    def __init__(self, precision, location, fft_type,
                 num_dims, dim_size, batch_size, is_forward):
        """Creates a plan for FFTs.

        This wraps cuFFT, so only GPU FFTs are currently supported.

        :param precision: The precision for the FFT (single or double).
        :type precision: numpy.dtype("float32") or numpy.dtype("float64")

        :param location: The location for the FFT, either "GPU" or "CPU"
                         (currently only "GPU" is supported).
        :type location: string

        :param fft_type: The FFT type (currently only "C2C" is supported).
        :type fft_type: string

        :param num_dims: The number of dimensions for the FFT.
        :type num_dims: int

        :param dim_size: The size of each dimension.
        :type dim_size: list[int]

        :param batch_size: The batch size.
        :type batch_size: int

        :param is_forward: Set true if FFT should be "forward",
                           false for "inverse".
        :type is_forward: bool
        """
        self._handle = None
        error_status = Error()
        function_create = Lib.handle().sdp_fft_create
        function_create.restype = Fft.handle_type()
        function_create.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int64),
            ctypes.c_int,
            ctypes.c_int,
            Error.handle_type()
        ]
        if (precision == numpy.dtype(numpy.float32) or
            precision == numpy.dtype(numpy.complex64)):
            enum_precision = Mem.MemType.SDP_MEM_FLOAT
        elif (precision == numpy.dtype(numpy.float64) or
              precision == numpy.dtype(numpy.complex128)):
            enum_precision = Mem.MemType.SDP_MEM_DOUBLE
        else:
            raise TypeError("Unsupported FFT precision")
        if location.lower() == "gpu":
            enum_location = Mem.MemLocation.SDP_MEM_GPU
        else:
            raise RuntimeError("Unsupported FFT location")
        if (fft_type.lower() == "c2c"):
            enum_fft_type = self.FftType.SDP_FFT_C2C
        else:
            raise RuntimeError("Unsupported FFT type")
        dims = (ctypes.c_int64 * num_dims)(*dim_size)
        self._handle = function_create(
            ctypes.c_int(enum_precision),
            ctypes.c_int(enum_location),
            ctypes.c_int(enum_fft_type),
            ctypes.c_int(num_dims),
            dims,
            ctypes.c_int(batch_size),
            ctypes.c_int(is_forward),
            error_status.handle()
        )
        error_status.check()

    def __del__(self):
        """Releases handle to the processing function.
        """
        if self._handle:
            function_free = Lib.handle().sdp_fft_free
            function_free.argtypes = [Fft.handle_type()]
            function_free(self._handle)

    def handle(self):
        """Returns a handle to the wrapped processing function.

        Use this handle when calling the function in the compiled library.

        :return: Handle to wrapped function.
        :rtype: ctypes.POINTER(Fft.Handle)
        """
        return self._handle

    @staticmethod
    def handle_type():
        """Static convenience method to return the ctypes handle type.

        Use this when defining the list of argument types.

        :return: Type of the function handle.
        :rtype: ctypes.POINTER(Fft.Handle)
        """
        return ctypes.POINTER(Fft.Handle)

    def exec(self, input, output):
        """Executes FFT using plan and supplied data.

        :param input: Input data.
        :type input: cupy.ndarray

        :param output: Output data.
        :type output: cupy.ndarray
        """
        if not self._handle:
            raise RuntimeError("FFT plan not ready")

        mem_input = Mem(input)
        mem_output = Mem(output)
        error_status = Error()
        function_exec = Lib.handle().sdp_fft_exec
        function_exec.argtypes = [
            Fft.handle_type(),
            Mem.handle_type(),
            Mem.handle_type(),
            Error.handle_type()
        ]
        function_exec(
            self._handle,
            mem_input.handle(),
            mem_output.handle(),
            error_status.handle()
        )
        error_status.check()
