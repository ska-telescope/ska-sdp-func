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

from ska_sdp_func.station_beam import array_factor

def test_station_beam_array_factor():
    """Test station beam array factor."""
    # Run array factor test on CPU, using numpy arrays.
    freq_hz = 100e6
    wavenumber = 2.0 * numpy.pi * freq_hz / 299792458.0

    # Generate station layout and beamforming weights.
    x = numpy.linspace(-5.0, 5.0, 10)
    element_x, element_y = numpy.meshgrid(x, x)
    element_z = numpy.zeros_like(element_x)
    element_weights = numpy.ones_like(element_x, dtype=numpy.complex128)

    # Generate source positions and empty output array.
    x = numpy.linspace(-1.0, 1.0, 50)
    point_x, point_y = numpy.meshgrid(x, x)
    point_z = numpy.sqrt(1.0 - point_x**2 - point_y**2)
    beam = numpy.zeros_like(point_x, dtype=numpy.complex128)

    # Call library function to evaluate array factor.
    print("Testing array factor on CPU from ska-sdp-func...")
    array_factor(
        wavenumber, element_weights, element_x, element_y, element_z,
        point_x, point_y, point_z, None, None, beam, normalise=True
    )
    if plt:
        plt.scatter(point_x, point_y, c=numpy.abs(beam))
        plt.colorbar()
        plt.savefig("test_array_factor_cpu.png")

    # Run DFT test on GPU, using cupy arrays.
    if cupy:
        element_x_gpu = cupy.asarray(element_x)
        element_y_gpu = cupy.asarray(element_y)
        element_z_gpu = cupy.asarray(element_z)
        element_weights_gpu = cupy.asarray(element_weights)
        point_x_gpu = cupy.asarray(point_x)
        point_y_gpu = cupy.asarray(point_y)
        point_z_gpu = cupy.asarray(point_z)
        beam_gpu = cupy.zeros_like(point_x_gpu, dtype=cupy.complex128)
        print("Testing array factor on GPU from ska-sdp-func...")
        array_factor(
            wavenumber, element_weights_gpu,
            element_x_gpu, element_y_gpu, element_z_gpu,
            point_x_gpu, point_y_gpu, point_z_gpu,
            None, None, beam_gpu, normalise=True
        )
        beam_gpu_copy = cupy.asnumpy(beam_gpu)
        numpy.testing.assert_array_almost_equal(beam_gpu_copy, beam)
        print("station beam array factor on GPU: Test passed")
        if plt:
            plt.scatter(point_x, point_y, c=numpy.abs(beam_gpu_copy))
            plt.colorbar()
            plt.savefig("test_array_factor_gpu.png")

