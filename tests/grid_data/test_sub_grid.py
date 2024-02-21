# See the LICENSE file at the top-level directory of this distribution.

"""Test gridding with a sub-grid."""

from math import cos, pi, sin

import numpy

try:
    import cupy
except ImportError:
    cupy = None

# try:
#     import matplotlib.pyplot as plt
# except ImportError:
#     plt = None

from ska_sdp_func.grid_data import grid_uvw_es


def generate_uvw(ant_config, num_times, length_hours, longitude_rad):
    num_antennas = ant_config.shape[0]
    num_baselines = (num_antennas * (num_antennas - 1)) // 2
    ant_x, ant_y, ant_z = ant_config[:, 0], ant_config[:, 1], ant_config[:, 2]

    # Observation parameters.
    ra0 = 0.0
    dec0 = 70.0 * pi / 180.0
    dt = length_hours / (num_times - 1) if num_times != 1 else 0
    t0 = (dt - length_hours) / 2.0

    # (u,v,w) coordinates to return.
    uvw = numpy.zeros((num_times * num_baselines, 3))

    # Loop over time samples.
    for t in range(num_times):
        # Current time and HA.
        # Note: approximate required GST!
        gst = (t0 + t * dt) * 2 * pi / 24.0 - longitude_rad
        ha0 = gst - ra0

        # Rotation matrix to convert from (x,y,z) to (u,v,w).
        mat = numpy.array(
            [
                [sin(ha0), cos(ha0), 0],
                [-sin(dec0) * cos(ha0), sin(dec0) * sin(ha0), cos(dec0)],
                [cos(dec0) * cos(ha0), -cos(dec0) * sin(ha0), sin(dec0)],
            ]
        )
        (u, v, w) = numpy.dot(mat, numpy.array((ant_x, ant_y, ant_z)))

        # Generate baseline pairs.
        b = 0
        for a1 in range(num_antennas):
            for a2 in range(a1):
                uvw[t * num_baselines + b, 0] = u[a1] - u[a2]
                uvw[t * num_baselines + b, 1] = v[a1] - v[a2]
                uvw[t * num_baselines + b, 2] = w[a1] - w[a2]
                b += 1
    return uvw


def test_sub_grid():
    """Test gridding on a sub-grid."""

    # VLA D-configuration.
    ant_config = numpy.array(
        [
            [-1601188.989351, -5042000.518599, 3554843.384480],
            [-1601225.230987, -5041980.390730, 3554855.657987],
            [-1601265.110332, -5041982.563379, 3554834.816409],
            [-1601315.874282, -5041985.324465, 3554808.263784],
            [-1601376.950042, -5041988.682890, 3554776.344871],
            [-1601447.176774, -5041992.529191, 3554739.647266],
            [-1601526.335275, -5041996.876364, 3554698.284889],
            [-1601614.061201, -5042001.676547, 3554652.455603],
            [-1601709.987416, -5042006.942534, 3554602.306306],
            [-1601192.424192, -5042022.883542, 3554810.383317],
            [-1601150.027460, -5042000.630731, 3554860.703495],
            [-1601114.318178, -5042023.187696, 3554844.922416],
            [-1601068.771188, -5042051.929370, 3554824.767363],
            [-1601014.405657, -5042086.261585, 3554800.768970],
            [-1600951.545716, -5042125.927280, 3554772.987195],
            [-1600880.545264, -5042170.376845, 3554741.425036],
            [-1600801.880602, -5042219.386677, 3554706.382285],
            [-1600715.918854, -5042273.142150, 3554668.128757],
            [-1601185.553970, -5041978.191573, 3554876.382645],
            [-1601180.820941, -5041947.459898, 3554921.573373],
            [-1601177.368455, -5041925.069104, 3554954.532566],
            [-1601173.903632, -5041902.679083, 3554987.485762],
            [-1601168.735762, -5041869.062707, 3555036.885577],
            [-1601162.553007, -5041829.021602, 3555095.854771],
            [-1601155.593706, -5041783.860938, 3555162.327771],
            [-1601147.885235, -5041733.855114, 3555235.914849],
            [-1601139.483292, -5041679.021042, 3555316.478099],
        ]
    )

    # Generate (u,v,w) coordinates.
    longitude_rad = -107.566 * pi / 180.0
    num_antennas = ant_config.shape[0]
    num_baselines = (num_antennas * (num_antennas - 1)) // 2
    num_times = 10
    num_channels = 1
    length_hours = 4
    uvw = generate_uvw(ant_config, num_times, length_hours, longitude_rad)

    # Plot (u,v,w) coordinates to check.
    # plt.scatter(
    #     numpy.concatenate((uvw[:, 0], -uvw[:, 0])),
    #     numpy.concatenate((uvw[:, 1], -uvw[:, 1]))
    # )
    # plt.show()

    # Define other test parameters.
    vis = numpy.random.randn(
        num_times * num_baselines, num_channels
    ) + 1j * numpy.random.randn(num_times * num_baselines, num_channels)
    vis[:, :] = 1.0
    weight = numpy.ones(vis.shape)
    image_size = 2000
    epsilon = 1e-4
    cell_size_rad = numpy.radians(10.0 / 3600.0)
    w_scale = 0.01
    min_plane_w = numpy.min(uvw[:, 2])  # -260.31 in compiled version
    sub_grid_w = 10  # w-plane index
    freq = numpy.array([1e9])

    if cupy:
        # Copy input data.
        uvw_gpu = cupy.asarray(uvw)
        vis_gpu = cupy.asarray(vis)
        weight_gpu = cupy.asarray(weight)
        freq_gpu = cupy.asarray(freq)

        # Do gridding onto the full grid.
        full_grid_size = 2816
        full_grid = cupy.zeros(
            [full_grid_size, full_grid_size], dtype=cupy.complex128
        )
        grid_uvw_es(
            uvw_gpu,
            vis_gpu,
            weight_gpu,
            freq_gpu,
            image_size,
            epsilon,
            cell_size_rad,
            w_scale,
            min_plane_w,
            0,
            0,
            sub_grid_w,
            full_grid,
        )

        # Do gridding onto a sub-grid.
        sub_grid_size = 300
        sub_grid_start_u = 1100
        sub_grid_start_v = 1400
        sub_grid = cupy.zeros(
            [sub_grid_size, sub_grid_size], dtype=cupy.complex128
        )
        grid_uvw_es(
            uvw_gpu,
            vis_gpu,
            weight_gpu,
            freq_gpu,
            image_size,
            epsilon,
            cell_size_rad,
            w_scale,
            min_plane_w,
            sub_grid_start_u,
            sub_grid_start_v,
            sub_grid_w,
            sub_grid,
        )

        # Check results match.
        full_grid_cpu = cupy.asnumpy(full_grid)
        sub_grid_cpu = cupy.asnumpy(sub_grid)
        numpy.testing.assert_array_almost_equal(
            sub_grid_cpu,
            full_grid_cpu[
                sub_grid_start_u : sub_grid_start_u + sub_grid_size,
                sub_grid_start_v : sub_grid_start_v + sub_grid_size,
            ],
        )

        # Plot results to check.
        # if plt:
        #     plt.imshow(numpy.real(sub_grid_cpu))
        #     plt.colorbar()
        #     plt.show()

        #     image_gpu = cupy.zeros([image_size, image_size])
        #     gridder = GridderUvwEsFft(
        #         uvw_gpu, freq_gpu, vis_gpu, weight_gpu, image_gpu,
        #         cell_size_rad, cell_size_rad, epsilon, True
        #     )
        #     gridder.grid_uvw_es_fft(
        #         uvw_gpu, freq_gpu, vis_gpu, weight_gpu, image_gpu
        #     )
        #     image_cpu = cupy.asnumpy(image_gpu)
        #     plt.imshow(image_cpu)
        #     plt.show()
