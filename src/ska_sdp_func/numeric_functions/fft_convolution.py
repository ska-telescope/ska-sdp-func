"""Python bindings for fft convolution function."""

from ..utility import Lib, Mem

Lib.wrap_func(
    "sdp_fft_convolution",
    restype=None,
    argtypes=[Mem.handle_type(), Mem.handle_type(), Mem.handle_type()],
    check_errcode=True,
)


def fft_convolution(in1, in2, out):
    """Implimentation of the convolution theorem. Assumes that inputs
    are square. returned convolution is the same size as in1,
    similar to scipy.signal.convolve "same" mode.

    Parameters can be either numpy or cupy arrays, but must all be consistent.
    Computation is performed either on the CPU or GPU as appropriate.

    :param in1: First input image.
    :type in1: numpy.ndarray or cupy.ndarray
    :param in2: Second input image.
    :type in2: numpy.ndarray or cupy.ndarray
    :param out: Result of convolution
    :type out: numpy.ndarray or cupy.ndarray
    """

    Lib.sdp_fft_convolution(
        Mem(in1),
        Mem(in2),
        Mem(out),
    )
