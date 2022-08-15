# See the LICENSE file at the top-level directory of this distribution.

"""Test degridding functions."""

import numpy

try:
    import cupy
except ImportError:
    cupy = None

from ska_sdp_func import degrid_uvw_custom


# pylint: disable=too-many-locals
def calculate_coordinates(
    grid_size,
    x_stride,
    y_stride,
    kernel_size,
    kernel_stride,
    oversample,
    wkernel_stride,
    oversample_w,
    theta,
    wstep,
    u,
    v,
    w,
):
    """Calculate coordinates in grid for visibility point."""
    # u or x coordinate
    o_x = theta * u * oversample
    iox = round(o_x) + (grid_size // 2 + 1) * oversample - 1
    home_x = iox // oversample
    frac_x = oversample - 1 - (iox % oversample)

    # v or y coordinate
    o_y = theta * v * oversample
    ioy = round(o_y) + (grid_size // 2 + 1) * oversample - 1
    home_y = ioy // oversample
    frac_y = oversample - 1 - (ioy % oversample)

    # w or z coordinate
    o_z = (1.0 + w / wstep) * oversample_w
    ioz = round(o_z) + oversample_w - 1
    frac_z = oversample_w - 1 - (ioz % oversample_w)

    grid_offset = (home_y - kernel_size // 2) * y_stride + (
        home_x - kernel_size // 2
    ) * x_stride
    sub_offset_x = kernel_stride * frac_x
    sub_offset_y = kernel_stride * frac_y
    sub_offset_z = wkernel_stride * frac_z

    return (
        int(grid_offset),
        int(sub_offset_x),
        int(sub_offset_y),
        int(sub_offset_z),
    )


# pylint: disable=too-many-locals,too-many-nested-blocks
def reference_degrid_uvw_custom(
    uv_kernel_stride,
    w_kernel_stride,
    x_size,
    y_size,
    num_times,
    num_baselines,
    num_channels,
    num_pols,
    grid,
    uvw,
    uv_kernel,
    w_kernel,
    uv_kernel_oversampling,
    w_kernel_oversampling,
    theta,
    wstep,
    channel_start_hz,
    channel_step_hz,
    conjugate,
):
    """Generate reference data for degridding comparison."""
    vis = numpy.zeros(
        [num_times, num_baselines, num_channels, num_pols],
        dtype=numpy.complex128,
    )

    grid = grid.flatten()
    uv_kernel = uv_kernel.flatten()
    w_kernel = w_kernel.flatten()

    for i_baseline in range(num_baselines):
        for i_channel in range(num_channels):
            for i_time in range(num_times):
                for i_pol in range(num_pols):
                    inv_wavelength = (
                        channel_start_hz + i_channel * channel_step_hz
                    ) / 299792458.0

                    (
                        grid_offset,
                        sub_offset_x,
                        sub_offset_y,
                        sub_offset_z,
                    ) = calculate_coordinates(
                        x_size,
                        1,
                        y_size,
                        uv_kernel_stride,
                        uv_kernel_stride,
                        uv_kernel_oversampling,
                        w_kernel_stride,
                        w_kernel_oversampling,
                        theta,
                        wstep,
                        inv_wavelength * uvw[i_time][i_baseline][0],
                        inv_wavelength * uvw[i_time][i_baseline][1],
                        inv_wavelength * uvw[i_time][i_baseline][2],
                    )

                    vis_local = complex(0, 0)
                    for z in range(w_kernel_stride):
                        visz = complex(0, 0)
                        for y in range(uv_kernel_stride):
                            visy = complex(0, 0)
                            for x in range(uv_kernel_stride):
                                grid_value = grid[
                                    z * x_size * y_size
                                    + grid_offset
                                    + y * y_size
                                    + x
                                ]
                                visy += (
                                    uv_kernel[sub_offset_x + x] * grid_value
                                )
                            visz += uv_kernel[sub_offset_y + y] * visy
                        vis_local += w_kernel[sub_offset_z + z] * visz

                    if conjugate:
                        vis_local = vis_local.conjugate()

                    vis[i_time][i_baseline][i_channel][i_pol] = vis_local

    return vis


# pylint: disable=too-many-locals
def test_degrid_uvw_custom():
    """Test degridding function."""
    # Run degridding test on CPU using numpy arrays.
    uv_kernel_oversampling = 16000
    w_kernel_oversampling = 16000
    theta = 0.1
    wstep = 250
    channel_start_hz = 100
    channel_step_hz = 0.1
    conjugate = False
    x_size = 512
    y_size = 512
    z_size = 4
    num_channels = 1
    num_pols = 1
    num_baselines = 14
    num_times = 4
    uv_kernel_stride = 8
    w_kernel_stride = 4
    rng = numpy.random.default_rng()
    grid_real = rng.random(
        (num_channels, z_size, y_size, x_size, num_pols), dtype=numpy.float64
    )
    grid_imag = 1j * rng.random(
        (num_channels, z_size, y_size, x_size, num_pols), dtype=numpy.float64
    )
    grid = grid_real + grid_imag
    uvw = rng.random((num_times, num_baselines, 3), dtype=numpy.float64)
    uv_kernel = rng.random(
        (uv_kernel_oversampling, uv_kernel_stride), dtype=numpy.float64
    )
    w_kernel = rng.random(
        (w_kernel_oversampling, w_kernel_stride), dtype=numpy.float64
    )
    vis = numpy.zeros(
        [num_times, num_baselines, num_channels, num_pols],
        dtype=numpy.complex128,
    )
    print("Testing degridding on CPU from ska-sdp-func...")
    degrid_uvw_custom(
        grid,
        uvw,
        uv_kernel,
        w_kernel,
        theta,
        wstep,
        channel_start_hz,
        channel_step_hz,
        conjugate,
        vis,
    )
    vis_reference = reference_degrid_uvw_custom(
        uv_kernel_stride,
        w_kernel_stride,
        x_size,
        y_size,
        num_times,
        num_baselines,
        num_channels,
        num_pols,
        grid,
        uvw,
        uv_kernel,
        w_kernel,
        uv_kernel_oversampling,
        w_kernel_oversampling,
        theta,
        wstep,
        channel_start_hz,
        channel_step_hz,
        conjugate,
    )

    numpy.testing.assert_array_almost_equal(vis, vis_reference)

    # Run degridding on GPU using cupy arrays.
    if cupy:
        grid_gpu = cupy.asarray(grid)
        uvw_gpu = cupy.asarray(uvw)
        uv_kernel_gpu = cupy.asarray(uv_kernel)
        w_kernel_gpu = cupy.asarray(w_kernel)
        vis_gpu = cupy.zeros(
            [num_times, num_baselines, num_channels, num_pols],
            dtype=numpy.complex128,
        )
        print("Testing degridding on GPU from ska-sdp-func...")
        degrid_uvw_custom(
            grid_gpu,
            uvw_gpu,
            uv_kernel_gpu,
            w_kernel_gpu,
            theta,
            wstep,
            channel_start_hz,
            channel_step_hz,
            conjugate,
            vis_gpu,
        )
        output_gpu_check = cupy.asnumpy(vis_gpu)
        numpy.testing.assert_array_almost_equal(
            output_gpu_check, vis_reference
        )
        print("Degridding on GPU: Test passed")
