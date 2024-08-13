# See the LICENSE file at the top-level directory of this distribution.

"""Integration test for Hogbom CLEAN using DFT and gridder functions."""

try:
    import cupy
except ImportError:
    cupy = None

import numpy as np
import scipy.signal as sig

from ska_sdp_func.clean import hogbom_clean
from ska_sdp_func.grid_data import GridderUvwEsFft
from ska_sdp_func.visibility import dft_point_v01


def create_test_data(dirty_size, psf_size):
    """create a test image and corresponding PSF using PFL functions.
    This image and PSF can then be used to test the Hogbom CLEAN
    against known data.
    Creates a simulated UVW coverage of randomly sized baselines,
    random lmn coordinates for sources are generated and processed
    by DFT and gridder functions in to a dirty image."""
    # Initialise settings
    num_components = 10
    num_pols = 1
    num_channels = 1
    num_baselines = 2000
    num_times = 1
    channel_start_hz = 100e6
    channel_step_hz = 100e3
    np.random.seed(12)  # seed for generating data

    # initialise empty output array for visibilities
    vis = np.zeros(
        [num_times, num_baselines, num_channels, num_pols],
        dtype=np.complex128,
    )

    vis_psf = np.zeros(
        [num_times, num_baselines, num_channels, num_pols],
        dtype=np.complex128,
    )

    # initialise empty arrays for intermediate data
    fluxes = np.zeros([num_components, num_channels, num_pols]) + 0j

    fluxes_psf = np.ones([1, num_channels, num_pols]) + 0j
    directions = np.zeros([num_components, 3])
    directions_psf = np.zeros([1, 3])
    uvw = np.zeros([num_times, num_baselines, 3])

    # UVW coverage as a random filled circle of given length, centered on (0,0)
    y0 = 0
    x0 = 0
    r = 3000

    # random filled circle representing baselines
    for i in range(num_baselines):

        # random angle
        theta = 2 * np.pi * np.random.uniform()

        # random radius
        radius = r * np.random.uniform(0, 1)

        x = x0 + radius * np.cos(theta)
        y = y0 + radius * np.sin(theta)

        uvw[0, i, :] = x, y, 0

    # Create random fluxes between 0 + 0j and 10 + 0j for sources
    for i in range(num_components):
        fluxes[i, 0, 0] = np.random.uniform(1, 10) + 0j

    # create random lmn co-ordinates between (-0.015,-0.015) and (0.015,0.015)
    for i in range(num_components):
        directions[i, :] = (
            np.random.uniform(-0.015, 0.015),
            np.random.uniform(-0.015, 0.015),
            0,
        )

        directions[i, 2] = np.sqrt(
            1 - (directions[i, 0]) ** 2 - (directions[i, 1] ** 2)
        )

    # process in to visibilities
    print("Creating visibilities using dft_point_v01 from ska-sdp-func...")
    # dirty image
    dft_point_v01(
        directions, fluxes, uvw, channel_start_hz, channel_step_hz, vis
    )

    # psf
    dft_point_v01(
        directions_psf,
        fluxes_psf,
        uvw,
        channel_start_hz,
        channel_step_hz,
        vis_psf,
    )

    # grid the generated visibilities
    # initialise settings for gridder
    nxydirty = dirty_size
    nxydirty_psf = psf_size
    fov = 2  # degrees
    pixel_size_rad = fov * np.pi / 180 / nxydirty
    pixel_size_rad_psf = fov * np.pi / 180 / nxydirty_psf
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
        psf_gpu = cupy.zeros([nxydirty_psf, nxydirty_psf], np.float64)
        vis_gpu = cupy.array(vis, np.complex128, order="C")
        vis_psf_gpu = cupy.array(vis_psf, np.complex128, order="C")
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
        print(
            "Creating Dirty image using GridderUvwEsFft from ska-sdp-func..."
        )
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
            pixel_size_rad_psf,
            pixel_size_rad_psf,
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

    # normalise fluxes by number of baselines
    psf = psf / num_baselines
    dirty_image = dirty_image / num_baselines

    return dirty_image, psf


def create_cbeam(coeffs, size):
    """create clean beam"""

    center = size // 2

    cbeam = np.zeros([size, size])

    A = 1
    x0 = center
    y0 = center
    sigma_X = coeffs[0]
    sigma_Y = coeffs[1]
    theta = (np.pi / 180) * coeffs[2]

    X = np.arange(0, size, 1)
    Y = np.arange(0, size, 1)

    a = np.cos(theta) ** 2 / (2 * sigma_X ** 2) + np.sin(theta) ** 2 / (
        2 * sigma_Y ** 2
    )
    b = np.sin(2 * theta) / (4 * sigma_X ** 2) - np.sin(2 * theta) / (
        4 * sigma_Y ** 2
    )
    c = np.sin(theta) ** 2 / (2 * sigma_X ** 2) + np.cos(theta) ** 2 / (
        2 * sigma_Y ** 2
    )

    for x in X:
        for y in Y:
            cbeam[x, y] = A * np.exp(
                -(
                    a * (x - x0) ** 2
                    + 2 * b * (x - x0) * (y - y0)
                    + c * (y - y0) ** 2
                )
            )

    return cbeam


def reference_hogbom_clean(
    dirty_img, psf, cbeam_details, loop_gain, threshold, cycle_limit
):
    """Reference python implementation of Hogbom CLEAN to
    test against c version"""
    # calculate useful shapes and sizes
    dirty_size = dirty_img.shape[0]

    # set up so loop variables
    cur_cycle = 0
    stop = False

    # Create intermediate data arrays
    maximum = np.zeros((cycle_limit, 3))
    clean_comp = np.zeros(dirty_img.shape)
    residual = np.copy(dirty_img)

    # create CLEAN beam
    cbeam = create_cbeam(cbeam_details, dirty_size)

    # begin CLEAN while thereshold and number of iterations not exceded
    while cur_cycle < cycle_limit and stop is False:

        # Find index of the maximum value in residual
        max_idx_flat = residual.argmax()
        max_idx = np.unravel_index(max_idx_flat, residual.shape)

        # check maximum value against threshold
        if residual[max_idx] < threshold:
            # Set stop flag if threshold reached
            stop = True
            break

        # Save position and peak value
        maximum[cur_cycle] = residual[max_idx], max_idx[0], max_idx[1]

        # Add fraction of maximum to clean components list
        clean_comp[max_idx[0], max_idx[1]] += loop_gain * maximum[cur_cycle, 0]

        # Shift the center of the PSF to the position of the maxmium
        shifted_psf = psf[
            (dirty_size - max_idx[0]) : (dirty_size - max_idx[0] + dirty_size),
            (dirty_size - max_idx[1]) : (dirty_size - max_idx[1] + dirty_size),
        ]

        # Subtract scaled version of PSF from residual (loop gain x maximum)
        residual = np.subtract(
            residual, (loop_gain * residual[max_idx] * shifted_psf)
        )

        cur_cycle += 1

    # Convolve clean beam with clean components
    skymodel = sig.convolve(clean_comp, cbeam, mode="same")

    # Add remaining residual
    skymodel = np.add(skymodel, residual)

    return skymodel, residual, clean_comp


def test_hogbom_clean():
    """Test the Hogbom CLEAN function python version against c"""

    # initalise settings
    dirty_size = 256
    psf_size = 512
    cbeam_details = np.array([2.0, 2.0, 1.0, 128.0], dtype=np.float64)
    loop_gain = 0.1
    threshold = 0.001
    cycle_limit = 10000

    # create empty array for result
    skymodel = np.zeros((dirty_size, dirty_size))
    clean_model = np.zeros((dirty_size, dirty_size))
    residual = np.zeros((dirty_size, dirty_size))

    # create test data
    print("Creating test data on CPU from ska-sdp-func...")
    dirty_img, psf = create_test_data(dirty_size, psf_size)

    print("Creating reference data on CPU from ska-sdp-func...")
    (
        skymodel_reference,
        residual_reference,
        clean_model_reference,
    ) = reference_hogbom_clean(
        dirty_img, psf, cbeam_details, loop_gain, threshold, cycle_limit
    )

    print("Reference created")

    print("Testing Hogbom CLEAN on CPU from ska-sdp-func...")

    hogbom_clean(
        dirty_img,
        psf,
        cbeam_details,
        loop_gain,
        threshold,
        cycle_limit,
        clean_model,
        residual,
        skymodel,
    )

    np.testing.assert_array_almost_equal(
        clean_model, clean_model_reference, decimal=6
    )
    np.testing.assert_array_almost_equal(
        residual, residual_reference, decimal=6
    )
    np.testing.assert_array_almost_equal(
        skymodel, skymodel_reference, decimal=6
    )

    print("Hogbom CLEAN double precision on CPU: Test passed")

    dirty_img_float = dirty_img.astype(np.float32)
    psf_float = psf.astype(np.float32)
    skymodel_float = skymodel.astype(np.float32)
    clean_model_float = clean_model.astype(np.float32)
    residual_float = residual.astype(np.float32)
    cbeam_details_float = cbeam_details.astype(np.float32)

    hogbom_clean(
        dirty_img_float,
        psf_float,
        cbeam_details_float,
        loop_gain,
        threshold,
        cycle_limit,
        clean_model_float,
        residual_float,
        skymodel_float,
    )

    np.testing.assert_array_almost_equal(
        clean_model_float, clean_model_reference, decimal=4
    )
    np.testing.assert_array_almost_equal(
        residual_float, residual_reference, decimal=4
    )
    np.testing.assert_array_almost_equal(
        skymodel_float, skymodel_reference, decimal=4
    )

    print("Hogbom CLEAN float precision on CPU: Test passed")

    if cupy:
        dirty_img_gpu = cupy.asarray(dirty_img)
        psf_gpu = cupy.asarray(psf)
        # cbeam_details_gpu = cupy.asarray(cbeam_details)
        clean_model_gpu = cupy.zeros_like(dirty_img_gpu)
        residual_gpu = cupy.zeros_like(dirty_img_gpu)
        skymodel_gpu = cupy.zeros_like(dirty_img_gpu)

        print("Testing Hogbom CLEAN on GPU from ska-sdp-func...")

        hogbom_clean(
            dirty_img_gpu,
            psf_gpu,
            cbeam_details,
            loop_gain,
            threshold,
            cycle_limit,
            clean_model_gpu,
            residual_gpu,
            skymodel_gpu,
        )

        clean_model_check = cupy.asnumpy(clean_model_gpu)
        residual_check = cupy.asnumpy(residual_gpu)
        skymodel_check = cupy.asnumpy(skymodel_gpu)

        np.testing.assert_array_almost_equal(
            clean_model_check, clean_model_reference, decimal=6
        )
        np.testing.assert_array_almost_equal(
            residual_check, residual_reference, decimal=6
        )
        np.testing.assert_array_almost_equal(
            skymodel_check, skymodel_reference, decimal=6
        )

        print("Hogbom CLEAN double precision on GPU: Test passed")

        dirty_img_gpu_float = cupy.asarray(dirty_img_float)
        psf_gpu_float = cupy.asarray(psf_float)
        # cbeam_details_gpu_float = cupy.asarray(cbeam_details_float)
        clean_comp_gpu_float = cupy.zeros_like(dirty_img_gpu_float)
        residual_gpu_float = cupy.zeros_like(dirty_img_gpu_float)
        skymodel_gpu_float = cupy.zeros_like(dirty_img_gpu_float)

        hogbom_clean(
            dirty_img_gpu_float,
            psf_gpu_float,
            cbeam_details_float,
            loop_gain,
            threshold,
            cycle_limit,
            clean_comp_gpu_float,
            residual_gpu_float,
            skymodel_gpu_float,
        )

        clean_model_check_float = cupy.asnumpy(clean_comp_gpu_float)
        residual_check_float = cupy.asnumpy(residual_gpu_float)
        skymodel_check_float = cupy.asnumpy(skymodel_gpu_float)

        np.testing.assert_array_almost_equal(
            clean_model_check_float, clean_model_reference, decimal=4
        )
        np.testing.assert_array_almost_equal(
            residual_check_float, residual_reference, decimal=4
        )
        np.testing.assert_array_almost_equal(
            skymodel_check_float, skymodel_reference, decimal=4
        )

        print("Hogbom CLEAN float precision on GPU: Test passed")
