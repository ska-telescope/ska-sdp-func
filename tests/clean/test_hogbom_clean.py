# See the LICENSE file at the top-level directory of this distribution.

"""Integration test for CLEAN using DTF and gridder functions."""

try:
    import cupy
except ImportError:
    cupy = None
    print('no cupy')

import numpy as np
# import matplotlib.pyplot as plt
from ska_sdp_func.visibility import dft_point_v01
from ska_sdp_func.grid_data import GridderUvwEsFft


def create_test_data():
    """Test DFT function."""
    # Initialise settings
    num_components = 20
    num_pols = 1
    num_channels = 1
    num_baselines = 350
    num_times = 1
    channel_start_hz = 100e6
    channel_step_hz = 100e3
    np.random.seed(12)  # seed for generating data

    # initialise empty output array
    vis = np.zeros(
        [num_times, num_baselines, num_channels, num_pols],
        dtype=np.complex128,
    )

    vis_psf = np.zeros(
        [num_times, num_baselines, num_channels, num_pols],
        dtype=np.complex128,
    )

    # initialise empty arrays for intermediate data
    fluxes = (
        np.zeros([num_components, num_channels, num_pols])
        + 0j
    )

    fluxes_psf = (
        np.ones([1, num_channels, num_pols])
        + 0j
    )
    directions = np.zeros([num_components, 3])
    directions_psf = np.zeros([1, 3])
    uvw = np.zeros([num_times, num_baselines, 3])

    # UVW coverage as a random filled circle of given length, centered on (0,0)
    y0 = 0
    x0 = 0
    r = 3000

    # random filled circle
    for i in range(num_baselines):

        # random angle
        theta = 2 * np.pi * np.random.random()

        # random radius
        radius = r * np.random.random()

        x = x0 + radius * np.cos(theta)
        y = y0 + radius * np.sin(theta)

        uvw[0, i, :] = x, y, 0

    # directions[0,:] = 0.015,0.015,0.9998
    # directions[1,:] = -0.001,-0.005,0.9999
    # directions[2,:] = -0.01,-0.01,0.9998

    # Create random fluxes between 0 + 0j and 10 + 0j
    for i in range(num_components):
        fluxes[i, 0, 0] = np.random.uniform(0, 10) + 0j  # ((np.random.uniform(0, 1)) * 1j)

    # create random lmn co-ordinates between (-0.015,-0.015) and (0.015,0.015)
    for i in range(num_components):
        directions[i, :] = np.random.uniform(-0.015, 0.015), np.random.uniform(-0.015, 0.015), 0

        directions[i, 2] = np.sqrt(1 - (directions[i, 0])**2 - (directions[i, 1] ** 2))

    print("Creating visibilities using dft_point_v01 from ska-sdp-func...")
    # dirty image
    dft_point_v01(
        directions, fluxes, uvw, channel_start_hz, channel_step_hz, vis
    )

    # psf
    dft_point_v01(
        directions_psf, fluxes_psf, uvw, channel_start_hz, channel_step_hz, vis_psf
    )

    # initialise settings for gridder
    nxydirty = 1024
    fov = 2  # degrees
    pixel_size_rad = fov * np.pi / 180 / nxydirty
    freqs = channel_start_hz + np.arange(num_channels) * channel_step_hz
    do_w_stacking = False
    epsilon = 1e-5
    num_vis = vis.shape[1]

    # reshape vis output from DFT to fit input of gridder
    vis = vis[0, :, :]
    vis_psf = vis_psf[0, :, :]

    # reshape UVW to fit input of gridder
    uvw = uvw[0, :, :]

    # Run gridder test on GPU, using cupy arrays.
    if cupy:
        freqs_gpu = cupy.asarray(freqs)
        uvw_gpu = cupy.asarray(uvw)
        dirty_image_gpu = cupy.zeros([nxydirty, nxydirty], np.float64)
        psf_gpu = cupy.zeros([nxydirty, nxydirty], np.float64)
        vis_gpu = cupy.array(vis, np.complex128, order='C')
        vis_psf_gpu = cupy.array(vis_psf, np.complex128, order='C')
        weight_gpu = cupy.ones([num_vis, num_channels])

        # Create gridder for dirty image
        gridder = GridderUvwEsFft(
            uvw_gpu,
            freqs_gpu,
            vis_gpu,
            weight_gpu,
            dirty_image_gpu,
            pixel_size_rad,
            pixel_size_rad,
            epsilon,
            do_w_stacking,
        )

        # Run gridder
        print("Creating Dirty image using GridderUvwEsFft from ska-sdp-func...")
        gridder.grid_uvw_es_fft(
            uvw_gpu, freqs_gpu, vis_gpu, weight_gpu, dirty_image_gpu
                )

        # Create gridder for PSF
        gridder = GridderUvwEsFft(
            uvw_gpu,
            freqs_gpu,
            vis_psf_gpu,
            weight_gpu,
            psf_gpu,
            pixel_size_rad,
            pixel_size_rad,
            epsilon,
            do_w_stacking,
        )

        # Run gridder
        print("Creating PSF using GridderUvwEsFft from ska-sdp-func...")
        gridder.grid_uvw_es_fft(
            uvw_gpu, freqs_gpu, vis_psf_gpu, weight_gpu, psf_gpu
                )

    dirty_image = cupy.asnumpy(dirty_image_gpu)
    psf = cupy.asnumpy(psf_gpu)

    # plt.figure()
    # plt.title('Dirty Image')
    # plt.imshow(dirty_image)

    # plt.figure()
    # plt.title('PSF')
    # plt.imshow(psf)

    # plt.figure()
    # plt.title('UV Plot')
    # plt.scatter(uvw[:, 0], uvw[:, 1])

    # plt.show()

    return dirty_image, psf

# create_test_data()
