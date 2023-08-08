# See the LICENSE file at the top-level directory of this distribution.

"""Module for element beam functions."""

import ctypes

from ..utility import Lib, Mem

Lib.wrap_func(
    "sdp_element_beam_dipole",
    restype=None,
    argtypes=[
        ctypes.c_int32,
        Mem.handle_type(),
        Mem.handle_type(),
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_int32,
        ctypes.c_int32,
        Mem.handle_type(),
    ],
    check_errcode=True,
)

Lib.wrap_func(
    "sdp_element_beam_spherical_wave_harp",
    restype=None,
    argtypes=[
        ctypes.c_int32,
        Mem.handle_type(),
        Mem.handle_type(),
        Mem.handle_type(),
        ctypes.c_int32,
        Mem.handle_type(),
        ctypes.c_int32,
        Mem.handle_type(),
    ],
    check_errcode=True,
)


def dipole(
    theta_rad,
    phi_rad,
    freq_hz: float,
    dipole_length_m: float,
    stride_element_beam: int,
    index_offset_element_beam: int,
    element_beam,
) -> None:
    """Evaluates an element beam from a dipole.

    Args:
        theta_rad (numpy.ndarray or cupy.ndarray):
            Source theta values.
        phi_rad (numpy.ndarray or cupy.ndarray):
            Source phi values.
        freq_hz (float):
            Frequency of observation, in Hz.
        dipole_length_m (float):
            Length of dipole, in metres.
        stride_element_beam (int):
            Stride into output array (normally 1 or 4).
        index_offset_element_beam (int):
            Start offset into output array.
        element_beam (numpy.ndarray or cupy.ndarray):
            Output complex element beam array.

    """
    Lib.sdp_element_beam_dipole(
        theta_rad.size,
        Mem(theta_rad),
        Mem(phi_rad),
        freq_hz,
        dipole_length_m,
        stride_element_beam,
        index_offset_element_beam,
        Mem(element_beam),
    )


def spherical_wave_harp(
    theta_rad,
    phi_x_rad,
    phi_y_rad,
    l_max: int,
    coeffs,
    index_offset_element_beam: int,
    element_beam,
) -> None:
    """
    Evaluates an element beam using spherical wave coefficients (HARP version).

    Args:
        theta_rad (numpy.ndarray or cupy.ndarray):
            Source theta values, in rad.
        phi_x_rad (numpy.ndarray or cupy.ndarray):
            Source phi values for X, in rad.
        phi_y_rad (numpy.ndarray or cupy.ndarray):
            Source phi values for Y, in rad.
        l_max (int):
            Maximum order of spherical wave.
        coeffs (numpy.ndarray or cupy.ndarray):
            TE and TM mode coefficients for X and Y antennas.
        index_offset_element_beam (int):
            Start offset into output array.
        element_beam (numpy.ndarray or cupy.ndarray):
            Output complex element beam array.

    """
    Lib.sdp_element_beam_spherical_wave_harp(
        theta_rad.size,
        Mem(theta_rad),
        Mem(phi_x_rad),
        Mem(phi_y_rad),
        l_max,
        Mem(coeffs),
        index_offset_element_beam,
        Mem(element_beam),
    )
