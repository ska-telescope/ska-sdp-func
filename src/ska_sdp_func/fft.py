# See the LICENSE file at the top-level directory of this distribution.

import ctypes

from .utility import Error, Lib, Mem


class Fft:
    """Interface to SDP FFT."""

    class Handle(ctypes.Structure):
        pass

    def __init__(self, input, output, num_dims_fft, is_forward):
        """Creates a plan for FFTs using the supplied input and output buffers.

        The number of dimensions used for the FFT is specified using the
        num_dims_fft parameter. If this is less than the number of dimensions
        in the arrays, then the FFT batch size is assumed to be the size of the
        first (slowest varying) dimension.

        This wraps cuFFT, so only GPU FFTs are currently supported.

        :param input: Input data.
        :type input: cupy.ndarray

        :param output: Output data.
        :type output: cupy.ndarray

        :param num_dims_fft: The number of dimensions for the FFT.
        :type num_dims_fft: int

        :param is_forward: Set true if FFT should be "forward",
                           false for "inverse".
        :type is_forward: bool
        """
        self._handle = None
        mem_input = Mem(input)
        mem_output = Mem(output)
        error_status = Error()
        function_create = Lib.handle().sdp_fft_create
        function_create.restype = Fft.handle_type()
        function_create.argtypes = [
            Mem.handle_type(),
            Mem.handle_type(),
            ctypes.c_int32,
            ctypes.c_int32,
            Error.handle_type(),
        ]
        self._handle = function_create(
            mem_input.handle(),
            mem_output.handle(),
            ctypes.c_int32(num_dims_fft),
            ctypes.c_int32(is_forward),
            error_status.handle(),
        )
        error_status.check()

    def __del__(self):
        """Releases handle to the processing function."""
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
            Error.handle_type(),
        ]
        function_exec(
            self._handle,
            mem_input.handle(),
            mem_output.handle(),
            error_status.handle(),
        )
        error_status.check()
