# See the LICENSE file at the top-level directory of this distribution.

"""Integration test for CLEAN using DTF and gridder functions."""

try:
    import cupy
except ImportError:
    cupy = None
    print("no cupy")

import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from ska_sdp_func.visibility import dft_point_v01
from ska_sdp_func.grid_data import GridderUvwEsFft
from ska_sdp_func.clean import hogbom_clean


def create_test_data():
    """Test DFT function."""
    # Initialise settings
    num_components = 10
    num_pols = 1
    num_channels = 1
    num_baselines = 1000
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
    fluxes = np.zeros([num_components, num_channels, num_pols]) + 0j

    fluxes_psf = np.ones([1, num_channels, num_pols]) + 0j
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
        theta = 2 * np.pi * np.random.uniform()

        # random radius
        radius = r * np.random.uniform(0, 1)

        x = x0 + radius * np.cos(theta)
        y = y0 + radius * np.sin(theta)

        uvw[0, i, :] = x, y, 0

    # Create random fluxes between 0 + 0j and 10 + 0j
    for i in range(num_components):
        fluxes[i, 0, 0] = (
            np.random.uniform(1, 10) + 0j
        )  # ((np.random.uniform(0, 1)) * 1j)
        # fluxes[i, 0, 0] = 10

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

    print("Creating visibilities using dft_point_v01 from ska-sdp-func...")
    # dirty image
    dft_point_v01(directions, fluxes, uvw, channel_start_hz, channel_step_hz, vis)

    # psf
    dft_point_v01(
        directions_psf, fluxes_psf, uvw, channel_start_hz, channel_step_hz, vis_psf
    )

    # initialise settings for gridder
    nxydirty = 1024
    nxydirty_psf = 2048
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
            pixel_size_rad_psf,
            pixel_size_rad_psf,
            epsilon,
            do_w_stacking,
        )

        # Run gridder
        print("Creating PSF using GridderUvwEsFft from ska-sdp-func...")
        gridder.grid_uvw_es_fft(uvw_gpu, freqs_gpu, vis_psf_gpu, weight_gpu, psf_gpu)

    dirty_image = cupy.asnumpy(dirty_image_gpu)
    psf = cupy.asnumpy(psf_gpu)

    # normalise fluxes by number of baselines
    psf = psf / num_baselines
    dirty_image = dirty_image / num_baselines

    return dirty_image, psf


def create_cbeam(coeffs):
    # create clean beam

    size = 200
    center = 100

    cbeam = np.zeros([size, size])

    A = 1
    x0 = center
    y0 = center
    sigma_X = coeffs[0]
    sigma_Y = coeffs[1]
    theta = (np.pi / 180) * coeffs[2]

    X = np.arange(0, size, 1)
    Y = np.arange(0, size, 1)

    a = np.cos(theta) ** 2 / (2 * sigma_X**2) + np.sin(theta) ** 2 / (
        2 * sigma_Y**2
    )
    b = np.sin(2 * theta) / (4 * sigma_X**2) - np.sin(2 * theta) / (4 * sigma_Y**2)
    c = np.sin(theta) ** 2 / (2 * sigma_X**2) + np.cos(theta) ** 2 / (
        2 * sigma_Y**2
    )

    for x in X:
        for y in Y:
            cbeam[x, y] = A * np.exp(
                -(a * (x - x0) ** 2 + 2 * b * (x - x0) * (y - y0) + c * (y - y0) ** 2)
            )

    return cbeam


def error_residuals(coeffs, psf):

    half_width = 100

    c = create_cbeam(coeffs)
    c = c.flatten()

    p = psf[
        1024 - half_width : 1024 + half_width, 1024 - half_width : 1024 + half_width
    ]
    p = p.flatten()

    return np.subtract(p, c)


def reference_hogbom_clean(
    dirty_img, psf, cbeam_details, loop_gain, threshold, cycle_limit
):

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
    cbeam = create_cbeam(cbeam_details)

    # begin CLEAN while thereshold and number of iterations not exceded
    while cur_cycle < cycle_limit and stop is False:

        print(
            f"Current Cycle: {cur_cycle} of {cycle_limit} Cycles Limit",
            end="\r",
        )

        # Find index of the maximum value in residual
        max_idx_flat = residual.argmax()
        max_idx = np.unravel_index(max_idx_flat, residual.shape)

        print("\n")
        print(residual[max_idx])

        # check maximum value against threshold
        if residual[max_idx] < threshold:
            # Set stop flag if threshold reached
            stop = True
            break

        # Save position and peak value
        maximum[cur_cycle] = residual[max_idx], max_idx[0], max_idx[1]

        # Add fraction of maximum to clean components list
        clean_comp[max_idx[0], max_idx[1]] += loop_gain * maximum[cur_cycle, 0]

        # Shift the center of the PSF to the position of the maxmium, to allow easy subtraction
        shifted_psf = psf[
            (dirty_size - max_idx[0]) : (dirty_size - max_idx[0] + dirty_size),
            (dirty_size - max_idx[1]) : (dirty_size - max_idx[1] + dirty_size),
        ]

        # Subtract scaled version of PSF from residual (loop gain x maximum)
        residual = np.subtract(residual, (loop_gain * residual[max_idx] * shifted_psf))

        cur_cycle += 1

    # Convolve clean beam with clean components
    skymodel = sig.convolve(clean_comp, cbeam, mode="same")

    # Add remaining residual
    # skymodel = np.add(inbetween, residual)

    return skymodel


def test_hogbom_clean():
    """Test the Hogbom CLEAN function"""

    # initalise settings
    cbeam_details = np.ones(3)
    loop_gain = 0.1
    threshold = 0.001
    cycle_limit = 10000

    # create empty array for result
    skymodel = np.zeros([1024, 1024])

    # create test data
    print("Creating test data on CPU from ska-sdp-func...")
    dirty_img, psf = create_test_data()

    print("Fitting CLEAN beam to PSF ...")
    fit = least_squares(
        error_residuals, cbeam_details, args=([psf]), ftol=1e-03, xtol=1e-3, verbose=0
    )

    # [23.89807183 24.63014124 23.59091923]
    cbeam_details = fit.x

    print("Creating reference data on CPU from ska-sdp-func...")
    skymodel_reference = reference_hogbom_clean(
        dirty_img, psf, cbeam_details, loop_gain, threshold, cycle_limit
    )

    print("Testing Hogbom CLEAN on CPU from ska-sdp-func...")

    hogbom_clean(
        dirty_img, psf, cbeam_details, loop_gain, threshold, cycle_limit, skymodel
    )

    np.testing.assert_array_almost_equal(skymodel, skymodel_reference)
    print("DFT on CPU: Test passed")
