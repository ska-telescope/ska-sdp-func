# See the LICENSE file at the top-level directory of this distribution.
"""Module for (de)gridding utility functions used by w-towers gridders."""

import ctypes

import numpy

from ..utility import Lib, Mem


def make_kernel(window: numpy.ndarray, kernel: numpy.ndarray):
    """
    Convert image-space window function to oversampled kernel.

    This uses a DFT to do the transformation to Fourier space.
    The supplied input window function is 1-dimensional, with a shape
    of (support), and the output kernel is 2-dimensional, with a shape
    of (oversample + 1, support), and it must be sized appropriately on entry.

    :param window: Image-space window function in 1D.
    :param kernel: Oversampled output kernel in Fourier space.
    """
    Lib.sdp_gridder_make_kernel(Mem(window), Mem(kernel))


def make_pswf_kernel(support: int, kernel: numpy.ndarray):
    """
    Generate an oversampled kernel using PSWF window function.

    This uses a DFT to do the transformation to Fourier space.
    The output kernel is 2-dimensional, with a shape
    of (oversample + 1, support), and it must be sized appropriately on entry.

    :param support: Kernel support size.
    :param kernel: Oversampled output kernel in Fourier space.
    """
    Lib.sdp_gridder_make_pswf_kernel(support, Mem(kernel))


def make_w_pattern(
    subgrid_size: int,
    theta: float,
    shear_u: float,
    shear_v: float,
    w_step: float,
    w_pattern: numpy.ndarray,
):
    """
    Generate w-pattern.

    This is the iDFT of a single visibility at (0, 0, w).

    :param subgrid_size: Subgrid size.
    :param theta: Total image size in direction cosines.
    :param shear_u: Shear parameter in u.
    :param shear_v: Shear parameter in v.
    :param w_step: Distance between w-planes.
    :param w_pattern:
        Complex w-pattern, dimensions (subgrid_size, subgrid_size).
    """
    Lib.sdp_gridder_make_w_pattern(
        subgrid_size, theta, shear_u, shear_v, w_step, Mem(w_pattern)
    )


Lib.wrap_func(
    "sdp_gridder_make_kernel",
    restype=None,
    argtypes=[Mem.handle_type(), Mem.handle_type()],
    check_errcode=True,
)

Lib.wrap_func(
    "sdp_gridder_make_pswf_kernel",
    restype=None,
    argtypes=[ctypes.c_int, Mem.handle_type()],
    check_errcode=True,
)

Lib.wrap_func(
    "sdp_gridder_make_w_pattern",
    restype=None,
    argtypes=[
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        Mem.handle_type(),
    ],
    check_errcode=True,
)

Lib.wrap_func(
    "sdp_generate_pswf_at_x",
    restype=None,
    argtypes=[
        ctypes.c_int,
        ctypes.c_double,
        Mem.handle_type(),
        Mem.handle_type(),
    ],
    check_errcode=True,
)

Lib.wrap_func(
    "sdp_gridder_subgrid_add",
    restype=None,
    argtypes=[
        Mem.handle_type(),
        ctypes.c_int,
        ctypes.c_int,
        Mem.handle_type(),
        ctypes.c_double,
    ],
    check_errcode=True,
)

Lib.wrap_func(
    "sdp_gridder_subgrid_cut_out",
    restype=None,
    argtypes=[
        Mem.handle_type(),
        ctypes.c_int,
        ctypes.c_int,
        Mem.handle_type(),
    ],
    check_errcode=True,
)
