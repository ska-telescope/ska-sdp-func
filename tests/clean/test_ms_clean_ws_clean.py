# See the LICENSE file at the top-level directory of this distribution.

"""Integration test for Multi-scale CLEAN, wsclean version using DFT and gridder functions."""

try:
    import cupy
except ImportError:
    cupy = None

import numpy as np
import scipy.signal as sig
from scipy.ndimage import gaussian_filter
from ska_sdp_func.visibility import dft_point_v01
from ska_sdp_func.grid_data import GridderUvwEsFft
from ska_sdp_func.clean import ms_clean_ws_clean


def create_test_data(nxydirty, nxydirty_psf):
    """create a test image and corresponding PSF"""
    # Initialise settings
    num_components = 10
    num_pols = 1
    num_channels = 1
    num_baselines = 2000
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
    # nxydirty = 512
    # nxydirty_psf = 1024
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


def create_cbeam(coeffs, size):
    """create clean beam with Bmaj, Bmin and Bpa"""

    # size = 512
    center = size / 2

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


def sub_minor_loop(residual, psf, loop_gain, threshold, cycle_limit):
    """perform the sub-minor loop of WSCLEAN msCLEAN - essentially a standard Hogbom"""

    stop = False
    maximum = np.zeros((cycle_limit, 3))
    clean_comp = np.zeros(residual.shape)
    dirty_size = len(residual)
    cur_cycle_sub_minor = 0

    while cur_cycle_sub_minor < cycle_limit and stop is False:

        # Find index of the maximum value in residual
        max_idx_flat = residual.argmax()
        max_idx = np.unravel_index(max_idx_flat, residual.shape)

        # print(
        #     f"Current Sub-Minor Cycle: {cur_cycle_sub_minor} of {cycle_limit} Cycles Limit\n"
        #     f"Current Max = {residual[max_idx]}\n"
        #     f"Current position = {max_idx}\n",
        #     end="\n")

        # check maximum value against threshold
        if residual[max_idx] < threshold:
            # Set stop flag if threshold reached
            stop = True
            break

        # Save position and peak value
        maximum[cur_cycle_sub_minor] = residual[max_idx], max_idx[0], max_idx[1]

        # Add fraction of maximum to clean components list
        clean_comp[max_idx[0], max_idx[1]] += (
            loop_gain * maximum[cur_cycle_sub_minor, 0]
        )

        # Shift the center of the PSF to the position of the maxmium, to allow easy subtraction
        shifted_psf = psf[
            (dirty_size - max_idx[0]) : (dirty_size - max_idx[0] + dirty_size),
            (dirty_size - max_idx[1]) : (dirty_size - max_idx[1] + dirty_size),
        ]

        # Subtract scaled version of PSF from residual (loop gain x maximum)
        residual = np.subtract(residual, (loop_gain * residual[max_idx] * shifted_psf))

        cur_cycle_sub_minor += 1

    return clean_comp


def create_scale_bias(scales):
    """calculate a set of biases for each scale"""

    scale_bias = np.zeros(len(scales))
    first_scale = 0

    for i, scale in enumerate(scales):
        if i == 0 and scale != 0:
            first_scale = scale
        if i == 1 and scale != 0 and first_scale == 0:
            first_scale = scale

        if scale == 0:
            scale_bias[i] = 1
        else:
            scale_bias[i] = 0.6 ** (-1 - np.log2(scale / first_scale))

    # scale_bias = np.array([1, 0.6**-1, 0.6**-2, 0.6**-3, 0.6**-4])

    return scale_bias


def create_scale_kern(scales, length, scale_kern_list):
    """created a set of 2D gaussians at a specifed scales"""

    # calculate sigma
    sigma_list = []
    for scale in scales:
        if scale == 0:
            sigma_list.append(0)
        else:
            sigma_list.append((3 / 16) * scale)

    # calculate gaussians
    for i, sigma in enumerate(sigma_list):

        if sigma == 0:
            kernel = np.zeros((length, length))
            kernel[length // 2, length // 2] = 1
            scale_kern_list[i, :, :] = kernel
        else:
            kernel = np.zeros((length, length))
            kernel[length // 2, length // 2] = 1
            kernel = gaussian_filter(kernel, sigma)
            scale_kern_list[i, :, :] = kernel

    return scale_kern_list


def reference_ms_clean_ws_clean(
    dirty_img,
    psf,
    cbeam_details,
    clean_gain,
    ms_gain,
    threshold,
    cycle_limit,
    cycle_limit_minor,
    scales,
):
    """python implementation of msCLEAN from WSCLEAN"""

    # set up some loop variables
    cur_cycle = 0
    stop = False

    # calculate useful shapes and sizes
    dirty_size = dirty_img.shape[0]
    psf_size = psf.shape[0]

    # create intermediate data arrays
    model = np.zeros(dirty_img.shape)
    residual = np.copy(dirty_img)
    scaled_residuals = np.zeros([len(scales), dirty_size, dirty_size])
    scale_kern_list = np.zeros([len(scales), dirty_size, dirty_size])

    # create CLEAN beam
    cbeam = create_cbeam(cbeam_details, dirty_size)

    # calculate scale kernels
    scale_kern_list = create_scale_kern(scales, dirty_size, scale_kern_list)

    # scale psf with scale kernel twice
    scaled_psf = np.zeros([len(scales), psf_size, psf_size])

    for i, scale_kern in enumerate(scale_kern_list):
        scaled_psf[i] = sig.convolve(psf, scale_kern, mode="same")
        scaled_psf[i] = sig.convolve(scaled_psf[i], scale_kern, mode="same")        

    # calculate scale bias
    scale_bias = create_scale_bias(scales)

    # begin cycle
    while cur_cycle < cycle_limit and stop is False:

        # calculate scaled dirty images
        for i, scale_kern in enumerate(scale_kern_list):
            scaled_residuals[i] = sig.convolve(residual, scale_kern, mode="same")

        # Find index of the maximum value at each scale and then use scale bias to find overall maximum
        max_val = np.zeros(len(scaled_residuals))
        max_val_scaled = np.zeros(len(scaled_residuals))

        for i, scaled in enumerate(scaled_residuals):
            max_val[i] = np.max(scaled)
            max_val_scaled[i] = max_val[i] * scale_bias[i]

        # find max scale after scaling
        max_scale = max_val_scaled.argmax()

        # find value for sub-minor to clean to
        stop_val = max_val[max_scale] * ms_gain

        # print(
        #     f"Current Minor Cycle: {cur_cycle} of {cycle_limit} Cycles Limit.\n"
        #     f"Current Scale = {max_scale} \n"
        #     f"Current max value = {max_val[max_scale]} \n"
        #     f"Target value = {stop_val} \n",
        #     end="\n",
        # )

        if max_val[max_scale] < threshold:
            stop = True
            break

        # call subminor loop
        clean_comp_ret = sub_minor_loop(
            scaled_residuals[max_scale, :, :],
            scaled_psf[max_scale, :, :],
            clean_gain,
            stop_val,
            cycle_limit_minor,
        )

        # convolve the returned model with the scale kernel for the used scale
        clean_comp_ret = sig.convolve(
            clean_comp_ret, scale_kern_list[max_scale, :, :], mode="same"
        )

        # add the convolved returned model to the overall model
        model = np.add(model, clean_comp_ret)

        # convolve the convolved returned model with the psf
        clean_ret_psf = sig.convolve(clean_comp_ret, psf, mode="same")

        # subtract the above from the residual
        residual = residual - clean_ret_psf

        cur_cycle += 1

    print("")
    print("convolving........")
    # Convolve clean beam with clean components
    skymodel = sig.convolve(model, cbeam, mode="same")

    # print("adding........")
    # # Add remaining residual
    # skymodel = np.add(skymodel, residual)

    return skymodel


def test_ms_clean_ws_clean():
    """Test the ms CLEAN from wsclean function"""

    # initalise settings
    cbeam_details = np.array([10.0, 10.0, 1.0], dtype=np.float64)
    scales = np.array([0, 8, 16, 32, 64], dtype=np.float64)
    clean_gain = 0.1
    ms_gain = 0.8
    threshold = 0.01
    cycle_limit_minor = 100
    cycle_limit_major = 100
    # size of dirty image
    nxydirty = 256
    # size of psf
    nxydirty_psf = 512

    # create empty array for result
    skymodel = np.zeros([nxydirty, nxydirty])

    # create test data
    print("Creating test data on CPU from ska-sdp-func...")
    dirty_img, psf = create_test_data(nxydirty, nxydirty_psf)

    print("Creating reference data on CPU from ska-sdp-func...")
    skymodel_reference = reference_ms_clean_ws_clean(
        dirty_img,
        psf,
        cbeam_details,
        clean_gain,
        ms_gain,
        threshold,
        cycle_limit_major,
        cycle_limit_minor,
        scales,
    )

    print("Testing msCLEAN WSCLEAN version on CPU from ska-sdp-func...")
    ms_clean_ws_clean(
        dirty_img,
        psf,
        cbeam_details,
        scales,
        clean_gain,
        threshold,
        cycle_limit_major,
        cycle_limit_minor,
        ms_gain,
        skymodel
    )

    np.testing.assert_almost_equal(skymodel_reference, skymodel, decimal=2)
    print("msCLEAN WSCLEAN version on CPU: Test passed")
