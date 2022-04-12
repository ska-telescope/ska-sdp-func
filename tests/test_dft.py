# See the LICENSE file at the top-level directory of this distribution.

import numpy
try:
    import cupy
except ImportError:
    cupy = None

from ska_sdp_func import dft_point_v00

def reference_dft(directions, fluxes, uvw_lambda):
    num_times, num_baselines, num_channels, _ = uvw_lambda.shape
    num_components, _, num_pols = fluxes.shape
    vis = numpy.zeros(
        [num_times, num_baselines, num_channels, num_pols],
        dtype=numpy.complex128)
    for i_comp in range(num_components):
        phasor = numpy.exp(
            -2j * numpy.pi *
            numpy.sum(uvw_lambda.data * directions[i_comp, :], axis=-1)
        )
        for i_pol in range(num_pols):
            vis[..., i_pol] += fluxes[i_comp, :, i_pol] * phasor
    return vis


def test_dft():
    # Run DFT test on CPU, using numpy arrays.
    num_components = 20
    num_pols = 4
    num_channels = 10
    num_baselines = 351
    num_times = 10
    fluxes = numpy.random.random_sample(
        [num_components, num_channels, num_pols]) + 0j
    directions = numpy.random.random_sample([num_components, 3])
    uvw_lambda = numpy.random.random_sample(
        [num_times, num_baselines, num_channels, 3])
    vis = numpy.zeros(
        [num_times, num_baselines, num_channels, num_pols],
        dtype=numpy.complex128)
    print("Testing DFT on CPU from ska-sdp-func...")
    dft_point_v00(directions, fluxes, uvw_lambda, vis)
    vis_reference = reference_dft(directions, fluxes, uvw_lambda)
    numpy.testing.assert_array_almost_equal(vis, vis_reference)
    print("DFT on CPU: Test passed")

    # Run DFT test on GPU, using cupy arrays.
    if cupy:
        fluxes_gpu = cupy.asarray(fluxes)
        directions_gpu = cupy.asarray(directions)
        uvw_lambda_gpu = cupy.asarray(uvw_lambda)
        vis_gpu = cupy.zeros(
            [num_times, num_baselines, num_channels, num_pols],
            dtype=numpy.complex128)
        print("Testing DFT on GPU from ska-sdp-func...")
        dft_point_v00(directions_gpu, fluxes_gpu, uvw_lambda_gpu, vis_gpu)
        output_gpu_check = cupy.asnumpy(vis_gpu)
        numpy.testing.assert_array_almost_equal(output_gpu_check, vis_reference)
        print("DFT on GPU: Test passed")
