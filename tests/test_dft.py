# See the LICENSE file at the top-level directory of this distribution.

"""Test DFT functions."""

import numpy

try:
    import cupy
except ImportError:
    cupy = None

from ska_sdp_func import dft_point_v00, dft_point_v01

C_0 = 299792458.0


def reference_dft_v00(directions, fluxes, uvw_lambda):
    """Generate reference data for DFT comparison."""
    num_times, num_baselines, num_channels, _ = uvw_lambda.shape
    num_components, _, num_pols = fluxes.shape
    vis = numpy.zeros(
        [num_times, num_baselines, num_channels, num_pols],
        dtype=numpy.complex128,
    )
    for i_comp in range(num_components):
        phasor = numpy.exp(
            -2j
            * numpy.pi
            * numpy.sum(uvw_lambda.data * directions[i_comp, :], axis=-1)
        )
        for i_pol in range(num_pols):
            vis[..., i_pol] += fluxes[i_comp, :, i_pol] * phasor
    return vis


def test_dft_v00():
    """Test DFT function."""
    # Run DFT test on CPU, using numpy arrays.
    num_components = 20
    num_pols = 4
    num_channels = 10
    num_baselines = 351
    num_times = 10
    fluxes = (
        numpy.random.random_sample([num_components, num_channels, num_pols])
        + 0j
    )
    directions = numpy.random.random_sample([num_components, 3])
    uvw_lambda = numpy.random.random_sample(
        [num_times, num_baselines, num_channels, 3]
    )
    vis = numpy.zeros(
        [num_times, num_baselines, num_channels, num_pols],
        dtype=numpy.complex128,
    )
    print("Testing DFT on CPU from ska-sdp-func...")
    dft_point_v00(directions, fluxes, uvw_lambda, vis)
    vis_reference = reference_dft_v00(directions, fluxes, uvw_lambda)
    numpy.testing.assert_array_almost_equal(vis, vis_reference)
    print("DFT on CPU: Test passed")

    # Run DFT test on GPU, using cupy arrays.
    if cupy:
        fluxes_gpu = cupy.asarray(fluxes)
        directions_gpu = cupy.asarray(directions)
        uvw_lambda_gpu = cupy.asarray(uvw_lambda)
        vis_gpu = cupy.zeros(
            [num_times, num_baselines, num_channels, num_pols],
            dtype=numpy.complex128,
        )
        print("Testing DFT on GPU from ska-sdp-func...")
        dft_point_v00(directions_gpu, fluxes_gpu, uvw_lambda_gpu, vis_gpu)
        output_gpu_check = cupy.asnumpy(vis_gpu)
        numpy.testing.assert_array_almost_equal(
            output_gpu_check, vis_reference
        )
        print("DFT on GPU: Test passed")


def reference_dft_v01(
    directions, fluxes, uvw, channel_start_hz, channel_step_hz
):
    """Generate reference data for DFT comparison."""
    num_times, num_baselines, _ = uvw.shape
    num_components, num_channels, num_pols = fluxes.shape
    vis = numpy.zeros(
        [num_times, num_baselines, num_channels, num_pols],
        dtype=numpy.complex128,
    )
    inv_wavelength = (
        channel_start_hz + numpy.arange(0, num_channels) * channel_step_hz
    ) / C_0
    uvw_lambda = (
        uvw[:, :, numpy.newaxis, :]
        * inv_wavelength[numpy.newaxis, numpy.newaxis, :, numpy.newaxis]
    )
    for i_comp in range(num_components):
        phasor = numpy.exp(
            -2j
            * numpy.pi
            * numpy.sum(uvw_lambda.data * directions[i_comp, :], axis=-1)
        )
        for i_pol in range(num_pols):
            vis[..., i_pol] += fluxes[i_comp, :, i_pol] * phasor
    return vis


def test_dft_v01():
    """Test DFT function."""
    # Run DFT test on CPU, using numpy arrays.
    num_components = 20
    num_pols = 4
    num_channels = 10
    num_baselines = 351
    num_times = 10
    channel_start_hz = 100e6
    channel_step_hz = 100e3
    fluxes = (
        numpy.random.random_sample([num_components, num_channels, num_pols])
        + 0j
    )
    directions = numpy.random.random_sample([num_components, 3])
    uvw = numpy.random.random_sample([num_times, num_baselines, 3])
    vis = numpy.zeros(
        [num_times, num_baselines, num_channels, num_pols],
        dtype=numpy.complex128,
    )
    print("Testing DFT on CPU from ska-sdp-func...")
    dft_point_v01(
        directions, fluxes, uvw, channel_start_hz, channel_step_hz, vis
    )
    vis_reference = reference_dft_v01(
        directions, fluxes, uvw, channel_start_hz, channel_step_hz
    )
    numpy.testing.assert_array_almost_equal(vis, vis_reference)
    print("DFT on CPU: Test passed")

    # Run DFT test on GPU, using cupy arrays.
    if cupy:
        fluxes_gpu = cupy.asarray(fluxes)
        directions_gpu = cupy.asarray(directions)
        uvw_lambda_gpu = cupy.asarray(uvw)
        vis_gpu = cupy.zeros(
            [num_times, num_baselines, num_channels, num_pols],
            dtype=numpy.complex128,
        )
        print("Testing DFT on GPU from ska-sdp-func...")
        dft_point_v01(
            directions_gpu,
            fluxes_gpu,
            uvw_lambda_gpu,
            channel_start_hz,
            channel_step_hz,
            vis_gpu,
        )
        output_gpu_check = cupy.asnumpy(vis_gpu)
        numpy.testing.assert_array_almost_equal(
            output_gpu_check, vis_reference
        )
        print("DFT on GPU: Test passed")
