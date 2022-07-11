# See the LICENSE file at the top-level directory of this distribution.

from matplotlib import testing
import numpy

try:
    import cupy
except ImportError:
    cupy = None

from ska_sdp_func import degrid_uvw_custom


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
    grid_offset,
    sub_offset_x,
    sub_offset_y,
    sub_offset_z,
):

    # x coordinate
    x = theta * u
    ox = x * oversample
    iox = round(ox)
    iox += (grid_size / 2 + 1) * oversample - 1
    home_x = iox / oversample
    home_x = round(home_x)
    frac_x = oversample - 1 - (iox % oversample)

    # y coordinate
    y = theta * v
    oy = y * oversample
    ioy = round(oy)
    ioy += (grid_size / 2 + 1) * oversample - 1
    home_y = ioy / oversample
    home_y = round(home_y)
    frac_y = oversample - 1 - (ioy % oversample)

    # w coordinate
    z = 1.0 + w / wstep
    oz = z * oversample_w
    ioz = round(oz)
    ioz += oversample_w - 1
    frac_z = oversample_w - 1 - (ioz % oversample_w)

    grid_offset = (home_y - kernel_size / 2) * y_stride + (
        home_x - kernel_size / 2
    ) * x_stride
    sub_offset_x = kernel_stride * frac_x
    sub_offset_y = kernel_stride * frac_y
    sub_offset_z = wkernel_stride * frac_z

    return int(grid_offset), int(sub_offset_x), int(sub_offset_y), int(sub_offset_z)


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
    conjugate,
):

    grid_offset = 0
    sub_offset_x = 0
    sub_offset_y = 0
    sub_offset_z = 0

    vis = numpy.zeros(
        [num_times, num_baselines, num_channels, num_pols], dtype=numpy.complex128
    )

    grid = grid.flatten()

    for i_time in range(num_times):
        for i_channel in range(num_channels):
            for i_baseline in range(num_baselines):

                u_vis_coordinate = uvw[i_time][i_baseline][i_channel][0]

                v_vis_coordinate = uvw[i_time][i_baseline][i_channel][1]

                w_vis_coordinate = uvw[i_time][i_baseline][i_channel][2]

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
                    u_vis_coordinate,
                    v_vis_coordinate,
                    w_vis_coordinate,
                    grid_offset,
                    sub_offset_x,
                    sub_offset_y,
                    sub_offset_z,
                )

                vis_r = 0
                vis_i = 0
                for z in range(w_kernel_stride):
                    visz_r = 0
                    visz_i = 0
                    for y in range(uv_kernel_stride):
                        visy_r = 0
                        visy_i = 0
                        for x in range(uv_kernel_stride):
                            temp = grid[
                                z * x_size * y_size + grid_offset + y * y_size + x
                            ]
                            visy_r += uv_kernel[sub_offset_x + x] * temp.real
                            visy_i += uv_kernel[sub_offset_x + x] * temp.imag
                        visz_r += uv_kernel[sub_offset_y + y] * visy_r
                        visz_i += uv_kernel[sub_offset_y + y] * visy_i
                    vis_r += w_kernel[sub_offset_z + z] * visz_r
                    vis_i += w_kernel[sub_offset_z + z] * visz_i

                if conjugate:
                    temp_vis = vis_r - vis_i * 1j
                else:
                    temp_vis = vis_r + vis_i * 1j

                for i_pol in range(num_pols):
                    vis[i_time][i_baseline][i_channel][i_pol] = temp_vis

    return vis


def test_degrid_uvw_custom():
    # Run degridding test on CPU using numpy arrays.
    uv_kernel_oversampling = 16000
    w_kernel_oversampling = 16000
    theta = 0.1
    wstep = 250
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
    grid_imag = (
        rng.random(
            (num_channels, z_size, y_size, x_size, num_pols), dtype=numpy.float64
        )
        * 1j
    )
    grid = grid_real + grid_imag
    uvw = rng.random((num_times, num_baselines, num_channels, 3), dtype=numpy.float64)
    uv_kernel = rng.random(
        (uv_kernel_oversampling * uv_kernel_stride), dtype=numpy.float64
    )
    w_kernel = rng.random(
        (w_kernel_oversampling * w_kernel_stride), dtype=numpy.float64
    )
    vis = numpy.zeros(
        [num_times, num_baselines, num_channels, num_pols], dtype=numpy.complex128
    )
    print("Testing Degridding on CPU from ska-sdp-func...")
    degrid_uvw_custom(
        grid,
        uvw,
        uv_kernel,
        w_kernel,
        uv_kernel_oversampling,
        w_kernel_oversampling,
        theta,
        wstep,
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
        conjugate,
    )

    numpy.testing.assert_array_almost_equal(vis, vis_reference)
    
    # Run degridding on GPU using cuppy arrays.
    if cupy:
        grid_gpu = cupy.asarray(grid)
        uvw_gpu = cupy.asarray(uvw)
        uv_kernel_gpu = cupy.asarray(uv_kernel)
        w_kernel_gpu = cupy.asarray(w_kernel)
        vis_gpu = cupy.zeros(
            [num_times, num_baselines, num_channels, num_pols], dtype=numpy.complex128
        )
        print("Testing Degridding on GPU from ska-sdp-func...")
        degrid_uvw_custom(
            grid_gpu,
            uvw_gpu,
            uv_kernel_gpu,
            w_kernel_gpu,
            uv_kernel_oversampling,
            w_kernel_oversampling,
            theta,
            wstep,
            conjugate,
            vis_gpu,
        )
        output_gpu_check = cupy.asnumpy(vis_gpu)
        numpy.testing.assert_array_almost_equal(output_gpu_check, vis_reference)
        print("Degridding on GPU: Test passed")
