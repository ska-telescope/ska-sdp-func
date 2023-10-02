# See the LICENSE file at the top-level directory of this distribution.

"""Test station beam functions."""

import numpy

try:
    import cupy
except ImportError:
    cupy = None

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

from ska_sdp_func.station_beam import aperture_array


def test_station_beam_aperture_array():
    """Test station beam aperture array."""
    # Run aperture array beam test on CPU, using numpy arrays.
    freq_hz = 100e6
    wavenumber = 2.0 * numpy.pi * freq_hz / 299792458.0

    # Generate station layout and beamforming weights.
    x = numpy.linspace(-5.0, 5.0, 10)
    element_x, element_y = numpy.meshgrid(x, x)
    element_z = numpy.zeros_like(element_x)
    element_weights = numpy.ones_like(element_x, dtype=numpy.complex128)

    # Generate source positions.
    x = numpy.linspace(-1.0, 1.0, 50)
    point_x, point_y = numpy.meshgrid(x, x)
    point_z = numpy.sqrt(1.0 - point_x**2 - point_y**2)

    # Call library function to evaluate array factor.
    print("Testing scalar aperture array beam on CPU from ska-sdp-func...")
    beam_scalar = numpy.zeros((point_x.size), dtype=numpy.complex128)
    aperture_array(
        wavenumber,
        element_weights,
        element_x,
        element_y,
        element_z,
        point_x,
        point_y,
        point_z,
        None,
        None,
        beam_scalar,
    )
    beam_pol = numpy.zeros((point_x.size, 4), dtype=numpy.complex128)
    aperture_array(
        wavenumber,
        element_weights,
        element_x,
        element_y,
        element_z,
        point_x,
        point_y,
        point_z,
        None,
        None,
        beam_pol,
        normalise=True,
        eval_x=False,
        eval_y=True,
    )
    if plt:
        plt.figure()
        plt.scatter(point_x, point_y, c=numpy.abs(beam_scalar))
        plt.colorbar()
        plt.savefig("test_aperture_array_cpu.png")

        fig, ax = plt.subplots(2, 2)
        fig.suptitle("Polarised station beam patterns")
        ax[0, 0].set_title("x_theta")
        sc00 = ax[0, 0].scatter(point_x, point_y, c=numpy.abs(beam_pol[:, 0]))
        plt.colorbar(sc00, ax=ax[0, 0])
        ax[0, 1].set_title("x_phi")
        sc01 = ax[0, 1].scatter(point_x, point_y, c=numpy.abs(beam_pol[:, 1]))
        plt.colorbar(sc01, ax=ax[0, 1])
        ax[1, 1].set_title("y_theta")
        sc11 = ax[1, 1].scatter(point_x, point_y, c=numpy.abs(beam_pol[:, 2]))
        plt.colorbar(sc11, ax=ax[1, 1])
        ax[1, 0].set_title("y_phi")
        sc10 = ax[1, 0].scatter(point_x, point_y, c=numpy.abs(beam_pol[:, 3]))
        plt.colorbar(sc10, ax=ax[1, 0])
        plt.savefig("test_aperture_array_cpu_pol.png")

    # Run aperture array beam test on GPU, using cupy arrays.
    if cupy:
        element_x_gpu = cupy.asarray(element_x)
        element_y_gpu = cupy.asarray(element_y)
        element_z_gpu = cupy.asarray(element_z)
        element_weights_gpu = cupy.asarray(element_weights)
        point_x_gpu = cupy.asarray(point_x)
        point_y_gpu = cupy.asarray(point_y)
        point_z_gpu = cupy.asarray(point_z)
        print("Testing aperture array beam on GPU from ska-sdp-func...")
        beam_gpu_scalar = cupy.zeros((point_x_gpu.size), dtype=cupy.complex128)
        aperture_array(
            wavenumber,
            element_weights_gpu,
            element_x_gpu,
            element_y_gpu,
            element_z_gpu,
            point_x_gpu,
            point_y_gpu,
            point_z_gpu,
            None,
            None,
            beam_gpu_scalar,
        )
        beam_gpu_copy = cupy.asnumpy(beam_gpu_scalar)
        numpy.testing.assert_allclose(
            beam_gpu_copy, beam_scalar, equal_nan=True
        )
        print("aperture array beam on GPU: Test passed")
        if plt:
            plt.figure()
            plt.scatter(point_x, point_y, c=numpy.abs(beam_gpu_copy))
            plt.colorbar()
            plt.savefig("test_aperture_array_gpu.png")
