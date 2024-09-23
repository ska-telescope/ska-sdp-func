# See the LICENSE file at the top-level directory of this distribution.

"""Module for FFT functions."""

import ctypes

from ..utility import Lib, Mem, StructWrapper


class Fft(StructWrapper):
    """Interface to SDP FFT."""

    def __init__(self, input_data, output_data, num_dims_fft, is_forward):
        """Creates a plan for FFTs using the supplied input and output buffers.

        The number of dimensions used for the FFT is specified using the
        num_dims_fft parameter. If this is less than the number of dimensions
        in the arrays, then the FFT batch size is assumed to be the size of the
        first (slowest varying) dimension.

        :param input_data: Input data.
        :type input_data: numpy.ndarray or cupy.ndarray

        :param output_data: Output data.
        :type output_data: numpy.ndarray or cupy.ndarray

        :param num_dims_fft: The number of dimensions for the FFT.
        :type num_dims_fft: int

        :param is_forward: Set true if FFT should be "forward",
                           false for "inverse".
        :type is_forward: bool
        """
        create_args = (
            Mem(input_data),
            Mem(output_data),
            num_dims_fft,
            is_forward,
        )
        super().__init__(Lib.sdp_fft_create, create_args, Lib.sdp_fft_free)

    def exec(self, input_data, output_data):
        """Executes FFT using plan and supplied data.

        :param input_data: Input data.
        :type input_data: cupy.ndarray

        :param output_data: Output data.
        :type output_data: cupy.ndarray
        """
        Lib.sdp_fft_exec(self, Mem(input_data), Mem(output_data))


def padded_fft_size(n: int, padding_factor: float):
    """Returns the next largest even that is a power of 2, 3, 5, 7 or 11.

    :param n: Minimum input grid size.
    :param padding_factor: Padding factor to multiply input grid size.
    """
    return Lib.sdp_fft_padded_size(n, padding_factor)


Lib.wrap_func(
    "sdp_fft_create",
    restype=Fft.handle_type(),
    argtypes=[
        Mem.handle_type(),
        Mem.handle_type(),
        ctypes.c_int32,
        ctypes.c_int32,
    ],
    check_errcode=True,
)

Lib.wrap_func(
    "sdp_fft_free",
    restype=None,
    argtypes=[Fft.handle_type()],
)

Lib.wrap_func(
    "sdp_fft_exec",
    restype=None,
    argtypes=[
        Fft.handle_type(),
        Mem.handle_type(),
        Mem.handle_type(),
    ],
    check_errcode=True,
)

Lib.wrap_func(
    "sdp_fft_padded_size",
    restype=ctypes.c_int,
    argtypes=[
        ctypes.c_int,
        ctypes.c_double,
    ],
)
