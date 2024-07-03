"""Integration test for Multi-scale CLEAN, Tim Cornwell version using DFT
and gridder functions."""

try:
    import cupy

    print("cupy imported")
except ImportError:
    cupy = None
    print("cupy not imported")

import numpy as np
import scipy.signal as sig
from scipy.ndimage import gaussian_filter

from ska_sdp_func.clean import ms_clean_cornwell
from ska_sdp_func.grid_data import GridderUvwEsFft
from ska_sdp_func.visibility import dft_point_v01


def create_test_data(dirty_size, psf_size):
    """Create a test data set using pfl DFT and gridder"""
    # Initialise settings
    num_components = 500
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
        if i < num_components - 10:
            fluxes[i, 0, 0] = (
                np.random.uniform(0.01, 0.2) + 0j
            )  # ((np.random.uniform(0, 1)) * 1j)
            # fluxes[i, 0, 0] = 10
        else:
            fluxes[i, 0, 0] = np.random.uniform(0.5, 1) + 0j

    # create random lmn co-ordinates between (-0.015,-0.015) and (0.015,0.015)
    max_radius = 0.010
    turns = 2

    for i in range(num_components):

        if i < num_components - 10:
            fraction = i / num_components
            angle = fraction * turns * 2 * np.pi  # Angle for the spiral
            radius = (
                fraction * max_radius
            )  # Radius increases linearly with fraction
            directions[i, 0] = radius * np.cos(angle) + np.random.normal(
                0, 0.001
            )  # Add slight randomness
            directions[i, 1] = radius * np.sin(angle) + np.random.normal(
                0, 0.001
            )  # Add slight randomness
            directions[i, 2] = np.sqrt(
                1 - (directions[i, 0]) ** 2 - (directions[i, 1] ** 2)
            )

        else:
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


def create_cbeam(coeffs):
    """Create clean beam using vectorized operations."""
    A = 1
    sigma_X, sigma_Y, theta, size = coeffs

    if size % 2 == 1:
        center = size / 2
    else:
        center = size / 2 - 1

    theta = np.radians(theta)  # Convert degrees to radians

    # Generate grid coordinates
    x, y = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    x0 = y0 = center

    # Pre-compute constants for the Gaussian formula
    a = np.cos(theta) ** 2 / (2 * sigma_X**2) + np.sin(theta) ** 2 / (
        2 * sigma_Y**2
    )
    b = -np.sin(2 * theta) / (4 * sigma_X**2) + np.sin(2 * theta) / (
        4 * sigma_Y**2
    )
    c = np.sin(theta) ** 2 / (2 * sigma_X**2) + np.cos(theta) ** 2 / (
        2 * sigma_Y**2
    )

    # Compute the Gaussian
    cbeam = A * np.exp(
        -(a * (x - x0) ** 2 + 2 * b * (x - x0) * (y - y0) + c * (y - y0) ** 2)
    )

    return cbeam


def scale_kern_calc(scales, length):
    """Create the msCLEAN scale bias kernels"""

    sigma_list = []
    for scale in scales:
        if scale == 0:
            sigma_list.append(0)
        else:
            sigma_list.append((3 / 16) * scale)

    scale_kern_list = []
    for sigma in sigma_list:

        if sigma == 0:
            kernel = np.zeros((length, length))
            kernel[length // 2, length // 2] = 1
            scale_kern_list.append(kernel)
        else:
            kernel = np.zeros((length, length))
            kernel[length // 2, length // 2] = 1
            kernel = gaussian_filter(kernel, sigma)
            scale_kern_list.append(kernel)

    return scale_kern_list


def reference_ms_clean_cornwell(
    dirty_img, psf, cbeam_details, loop_gain, threshold, cycle_limit, scales
):
    """Perform a reference msCLEAN minor cycle on the test data set,
    to compare results with PFL version"""

    # set up some loop variables
    cur_cycle = 0

    # calculate useful shapes and sizes
    dirty_size = dirty_img.shape[0]
    psf_size = psf.shape[0]

    # create intermediate data arrays
    clean_comp = np.zeros(dirty_img.shape)
    residual = np.copy(dirty_img)
    scaled_residuals = np.zeros([len(scales), dirty_size, dirty_size])
    coupling_matrix = np.zeros([len(scales), len(scales)])
    scaled_psf = np.zeros([len(scales), len(scales), psf_size, psf_size])

    # create CLEAN beam
    cbeam = create_cbeam(cbeam_details)

    # calculate scale kernels
    scale_kern_list = scale_kern_calc(scales, psf_size)

    # scale psf with each scale kernel twice, in all combinations
    for s in range(len(scales)):
        for p in range(len(scales)):

            scaled_psf[s, p, :, :] = sig.convolve(
                psf, scale_kern_list[p], mode="same"
            )
            scaled_psf[s, p, :, :] = sig.convolve(
                scaled_psf[s, p, :, :], scale_kern_list[s], mode="same"
            )

    # Evaluate the coupling matrix between the various scale sizes.
    for iscale in range(len(scales)):
        for iscale1 in range(len(scales)):
            coupling_matrix[iscale, iscale1] = np.max(
                scaled_psf[iscale, iscale1, :, :]
            )

    print(f"coupling matrix = \n {coupling_matrix}")

    # calculate scaled dirty images
    for i, scale_kern in enumerate(scale_kern_list):
        scaled_residuals[i] = sig.convolve(residual, scale_kern, mode="same")

    # begin cycle
    while cur_cycle < cycle_limit:

        max_idx_per_scale = []
        max_val_per_scale = []
        max_val_biased_per_scale = []
        max_scale_this_iter = np.nan

        for i, cur_residual in enumerate(scaled_residuals):
            max_idx_flat = cur_residual.argmax()
            max_idx_per_scale.append(
                np.unravel_index(max_idx_flat, cur_residual.shape)
            )
            max_val_per_scale.append(cur_residual[max_idx_per_scale[i]])

        for i, max in enumerate(max_val_per_scale):
            max_val_biased_per_scale.append(max / coupling_matrix[i, i])

        max_scale_this_iter = np.argmax(max_val_biased_per_scale)

        if max_val_biased_per_scale[max_scale_this_iter] < threshold:
            print("loop stopped at minimum threshold")
            break

        # calculate PSF window to be subtracted
        x_window_start = dirty_size - max_idx_per_scale[max_scale_this_iter][0]
        x_window_end = (
            dirty_size - max_idx_per_scale[max_scale_this_iter][0] + dirty_size
        )
        y_window_start = dirty_size - max_idx_per_scale[max_scale_this_iter][1]
        y_window_end = (
            dirty_size - max_idx_per_scale[max_scale_this_iter][1] + dirty_size
        )

        # add clean component
        # (loop gain x maximum x scale kernel at scale of current max)
        cur_scale_kern = np.asarray(scale_kern_list[max_scale_this_iter])
        clean_comp += (
            loop_gain
            * max_val_biased_per_scale[max_scale_this_iter]
            * cur_scale_kern[
                x_window_start:x_window_end, y_window_start:y_window_end
            ]
        )

        # Cross subtract psf from other scales
        for i in range(len(scales)):

            # Shift the center of the PSF to the position of the maxmium
            shifted_psf = scaled_psf[
                max_scale_this_iter,
                i,
                x_window_start:x_window_end,
                y_window_start:y_window_end,
            ]

            # Subtract scaled version of PSF from residual
            # (loop gain x maximum)
            scaled_residuals[i, :, :] -= (
                loop_gain
                * max_val_biased_per_scale[max_scale_this_iter]
                * shifted_psf
            )

        cur_cycle += 1

    print("")
    print("convolving CLEAN componets and beam........")
    # Convolve clean beam with clean components
    clean_comp_convolved = sig.convolve(clean_comp, cbeam, mode="same")

    print("adding residuals........")
    # # Add remaining residual
    skymodel = clean_comp_convolved + scaled_residuals[0]

    return skymodel, clean_comp, scaled_residuals[0]


def test_ms_clean_cornwell():
    """Test the msCLEAN from Cornwell function"""

    # initalise settings
    dirty_size = 256
    psf_size = 512
    cbeam_details = np.array([5.0, 5.0, 1.0, 128.0], dtype=np.float64)
    scales = np.array([0, 2, 4, 8, 16], dtype=np.intc)
    loop_gain = 0.1
    threshold = 0.001
    cycle_limit = 10000

    # create empty array for result
    skymodel = np.zeros([dirty_size, dirty_size])
    clean_model = np.zeros((dirty_size, dirty_size))
    residual = np.zeros((dirty_size, dirty_size))

    # create test data
    print("Creating test data on CPU from ska-sdp-func...")
    dirty_img, psf = create_test_data(dirty_size, psf_size)

    dirty_img = dirty_img.astype(np.float64)
    psf = psf.astype(np.float64)
    skymodel = skymodel.astype(np.float64)

    print("Creating reference data on CPU from ska-sdp-func...")
    (
        skymodel_reference,
        clean_comp_reference,
        residual_reference,
    ) = reference_ms_clean_cornwell(
        dirty_img,
        psf,
        cbeam_details,
        loop_gain,
        threshold,
        cycle_limit,
        scales,
    )

    print("Testing msCLEAN from Cornwell on CPU from ska-sdp-func...")

    ms_clean_cornwell(
        dirty_img,
        psf,
        cbeam_details,
        scales,
        loop_gain,
        threshold,
        cycle_limit,
        clean_model,
        residual,
        skymodel,
    )

    np.testing.assert_array_almost_equal(
        skymodel, skymodel_reference, decimal=3
    )
    np.testing.assert_array_almost_equal(
        clean_model, clean_comp_reference, decimal=2
    )
    np.testing.assert_array_almost_equal(
        residual, residual_reference, decimal=2
    )
    print("msCLEAN from Cornwell double precision on CPU: Test passed")

    dirty_img_float = dirty_img.astype(np.float32)
    psf_float = psf.astype(np.float32)
    skymodel_float = skymodel.astype(np.float32)
    clean_model_float = clean_model.astype(np.float32)
    residual_float = residual.astype(np.float32)
    cbeam_details_float = cbeam_details.astype(np.float32)

    ms_clean_cornwell(
        dirty_img_float,
        psf_float,
        cbeam_details_float,
        scales,
        loop_gain,
        threshold,
        cycle_limit,
        clean_model_float,
        residual_float,
        skymodel_float,
    )

    np.testing.assert_array_almost_equal(
        skymodel_float, skymodel_reference, decimal=3
    )
    np.testing.assert_array_almost_equal(
        clean_model_float, clean_comp_reference, decimal=2
    )
    np.testing.assert_array_almost_equal(
        residual_float, residual_reference, decimal=2
    )

    print("msCLEAN from Cornwell float precision on CPU: Test passed")
