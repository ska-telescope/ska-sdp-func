# See the LICENSE file at the top-level directory of this distribution.

"""Test element beam functions."""

import numpy

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

from ska_sdp_func.station_beam import dipole


def test_element_beam_dipole():
    """Test dipole pattern."""
    # Run dipole beam test on CPU, using numpy arrays.
    print("Testing dipole beam on CPU from ska-sdp-func...")
    freq_hz = 100e6
    dipole_length_m = 1.5  # Half-wavelength at 100 MHz.

    # Generate source positions.
    x = numpy.linspace(-1.0, 1.0, 50)
    point_x, point_y = numpy.meshgrid(x, x)
    point_z = numpy.sqrt(1.0 - point_x**2 - point_y**2)
    theta_rad = numpy.arctan2(numpy.sqrt(point_x**2 + point_y**2), point_z)
    phi_rad = numpy.arctan2(point_y, point_x)
    num_points = point_x.size

    # Polarised dipole beams.
    beam = numpy.zeros((num_points, 4), dtype=numpy.complex128)
    dipole(theta_rad, phi_rad, freq_hz, dipole_length_m, 4, 0, beam)
    phi_rad += numpy.pi / 2.0
    dipole(theta_rad, phi_rad, freq_hz, dipole_length_m, 4, 2, beam)

    # Scalar dipole beam.
    beam_scalar = numpy.zeros((num_points), dtype=numpy.complex128)
    dipole(theta_rad, phi_rad, freq_hz, dipole_length_m, 1, 0, beam_scalar)

    # Plot results.
    if plt:
        fig, ax = plt.subplots(2, 2)
        fig.suptitle("Dipole patterns")
        ax[0, 0].set_title("x_theta")
        ax[0, 0].scatter(point_x, point_y, c=numpy.abs(beam[:, 0]))  # x_theta
        ax[0, 1].set_title("x_phi")
        ax[0, 1].scatter(point_x, point_y, c=numpy.abs(beam[:, 1]))  # x_phi
        ax[1, 1].set_title("y_theta")
        ax[1, 1].scatter(point_x, point_y, c=numpy.abs(beam[:, 2]))  # y_theta
        ax[1, 0].set_title("y_phi")
        ax[1, 0].scatter(point_x, point_y, c=numpy.abs(beam[:, 3]))  # y_phi
        for axis in ax.flat:
            axis.axis("equal")
            axis.axis("off")
        plt.savefig("test_dipole_cpu.png")

        fig, ax = plt.subplots(1, 1)
        fig.suptitle("Scalar dipole pattern")
        ax.scatter(point_x, point_y, c=numpy.abs(beam_scalar))
        ax.axis("equal")
        ax.axis("off")
        plt.savefig("test_dipole_scalar_cpu.png")
