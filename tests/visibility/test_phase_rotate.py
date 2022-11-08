# See the LICENSE file at the top-level directory of this distribution.

"""Test phase centre rotation functions."""

import numpy

try:
    import cupy
except ImportError:
    cupy = None

from ska_sdp_func.utility import SkyCoord
from ska_sdp_func.visibility import phase_rotate_uvw, phase_rotate_vis


def test_phase_rotate():
    """Test phase centre rotation."""
    # Run test on CPU, using numpy arrays.
    original_phase_centre = SkyCoord(
        "icrs", 123.5 * numpy.pi / 180.0, 17.8 * numpy.pi / 180.0
    )
    new_phase_centre = SkyCoord(
        "icrs", 148.3 * numpy.pi / 180.0, 38.9 * numpy.pi / 180.0
    )
    channel_start_hz = 100e6
    channel_step_hz = 10e6
    num_pols = 4
    num_channels = 10
    num_baselines = 351
    num_times = 10
    uvw_in = numpy.random.random_sample([num_times, num_baselines, 3])
    uvw_out = numpy.zeros_like(uvw_in)
    vis_in = (
        numpy.random.random_sample(
            [num_times, num_baselines, num_channels, num_pols]
        )
        + 0j
    )
    vis_out = numpy.zeros_like(vis_in)
    print("Testing phase rotation on CPU from ska-sdp-func...")
    phase_rotate_uvw(
        original_phase_centre,
        new_phase_centre,
        uvw_in,
        uvw_out,
    )
    phase_rotate_vis(
        original_phase_centre,
        new_phase_centre,
        channel_start_hz,
        channel_step_hz,
        uvw_in,
        vis_in,
        vis_out,
    )

    # Run test on GPU, using cupy arrays.
    if cupy:
        uvw_in_gpu = cupy.asarray(uvw_in)
        uvw_out_gpu = cupy.zeros_like(uvw_in_gpu)
        vis_in_gpu = cupy.asarray(vis_in)
        vis_out_gpu = cupy.zeros_like(vis_in_gpu)
        print("Testing phase rotation on GPU from ska-sdp-func...")
        phase_rotate_uvw(
            original_phase_centre,
            new_phase_centre,
            uvw_in_gpu,
            uvw_out_gpu,
        )
        phase_rotate_vis(
            original_phase_centre,
            new_phase_centre,
            channel_start_hz,
            channel_step_hz,
            uvw_in_gpu,
            vis_in_gpu,
            vis_out_gpu,
        )
        output_gpu_uvw_check = cupy.asnumpy(uvw_out_gpu)
        output_gpu_vis_check = cupy.asnumpy(vis_out_gpu)
        numpy.testing.assert_array_almost_equal(output_gpu_uvw_check, uvw_out)
        numpy.testing.assert_array_almost_equal(output_gpu_vis_check, vis_out)
        print("Phase rotation on GPU: Test passed")
