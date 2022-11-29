# See the LICENSE file at the top-level directory of this distribution.
""" Module to test MSMFS deconvolution. """

try:
    import cupy
except ImportError:
    cupy = None

import math

import numpy as np

from ska_sdp_func import msmfs_perform


def add_source_to_image(image, amplitude, variance, centre, half_source_size):
    """

    :param image: assumed square
    :param amplitude:
    :param variance:
    :param centre:
    :param half_source_size:
    """
    image_size = image.shape[1]
    x_min = max(centre[0] - half_source_size, 0)
    x_max = min(centre[0] + half_source_size, image_size - 1)
    y_min = max(centre[1] - half_source_size, 0)
    y_max = min(centre[1] + half_source_size, image_size - 1)

    for this_x in range(x_min, x_max + 1):
        for this_y in range(y_min, y_max + 1):
            r_squared = (this_x - centre[0]) ** 2 + (this_y - centre[1]) ** 2
            if variance > 0:
                image[0, this_x, this_y] += (
                    amplitude
                    / (2 * math.pi * variance)
                    * math.exp(-0.5 * r_squared / variance)
                )
            else:
                image[0, this_x, this_y] += 0 if r_squared > 0 else amplitude

            # print(f"image[{x}, {y}] = {image[0, x, y]}")


def calculate_simple_psf_image(image):
    """
    :param image:  2D array to put a simple PSF in
    """
    psf_max_radius = 1.5  # radius of psf
    psf_moment_dropoff = (
        0.004  # amplitude dropoff with even taylor terms of the psf
    )
    psf_size = image.shape[1]
    num_psf = image.shape[0]
    psf_centre = psf_size // 2
    for taylor in range(0, num_psf + 1, 2):
        central_peak = psf_moment_dropoff ** math.sqrt(
            taylor
        )  # note MSMFS won't work with strict power law on PSF peaks
        for this_x in range(0, psf_size):
            for this_y in range(0, psf_size):
                radius = math.sqrt(
                    (this_x - psf_centre) ** 2 + (this_y - psf_centre) ** 2
                )
                image[taylor, this_x, this_y] = max(
                    central_peak * (1 - (radius / psf_max_radius) ** 2), 0
                )

    psf_x = psf_centre
    psf_y = psf_centre
    print(f"image[0, {psf_x}, {psf_y}] = {image[0, psf_x, psf_y]}")
    print(f"image[1, {psf_x}, {psf_y}] = {image[1, psf_x, psf_y]}")
    print(f"image[2, {psf_x}, {psf_y}] = {image[2, psf_x, psf_y]}")
    print(f"image[3, {psf_x}, {psf_y}] = {image[3, psf_x, psf_y]}")
    print(f"image[4, {psf_x}, {psf_y}] = {image[4, psf_x, psf_y]}")


def test_msmfs_perform():
    """Test function."""
    print(" ")  # just for separation of debug output
    print("Let's go!!")
    print(" ")

    dirty_moment_size = 1024
    psf_moment_size = dirty_moment_size // 4
    num_taylor = 3
    num_scales = 6
    image_border = 0
    convolution_accuracy = 1.2e-3
    clean_loop_gain = 0.35
    max_gaussian_sources = 100
    scale_bias_factor = 0.6
    clean_threshold = 0.001

    dirty_moment_images = np.zeros(
        (num_taylor, dirty_moment_size, dirty_moment_size)
    )
    psf_moment_images = np.zeros(
        (2 * num_taylor - 1, psf_moment_size, psf_moment_size)
    )

    print("dirty_moment_images is :", dirty_moment_images.shape)
    print("psf_moment_images is: ", psf_moment_images.shape)

    add_source_to_image(
        dirty_moment_images,
        10,
        1,
        [dirty_moment_size // 2, dirty_moment_size // 2],
        2,
    )
    add_source_to_image(
        dirty_moment_images,
        2,
        0,
        [dirty_moment_size // 2 - 4, dirty_moment_size // 2 - 4],
        2,
    )
    add_source_to_image(
        dirty_moment_images,
        5,
        4,
        [dirty_moment_size // 2 + 20, dirty_moment_size // 2],
        2,
    )

    calculate_simple_psf_image(psf_moment_images)

    # create single versions
    # uvw_s = uvw.astype(np.float32)
    # freqs_s = freqs.astype(np.float32)
    # vis_s = vis.astype(np.complex64)
    # weight_s = weight.astype(np.float32)
    # dirty_image_s = dirty_image.astype(np.float32)

    if cupy:
        dirty_moment_images_gpu = cupy.asarray(dirty_moment_images)
        psf_moment_images_gpu = cupy.asarray(psf_moment_images)

        num_sources, sources = msmfs_perform(
            dirty_moment_images_gpu,
            psf_moment_images_gpu,
            num_scales,
            image_border,
            convolution_accuracy,  # 5
            clean_loop_gain,
            max_gaussian_sources,
            scale_bias_factor,
            clean_threshold,
        )

        print(f"Finished, {num_sources} distinct sources were found.")

        for i in range(num_sources):
            print(
                f"Source {i:3d} has scale variance "
                f"{sources['variances'][i]:8.1f} at "
                f"{sources['positions'][i,:]} with Taylor term intensities "
                f"{sources['taylor_intensities'][i,:]}"
            )

        # test the output, only true for max_gaussian_sources = 100
        assert num_sources == 28
        assert abs(np.amax(sources["taylor_intensities"]) - 104.0194) < 0.001
        assert abs(np.amin(sources["taylor_intensities"]) - -195.7728) < 0.001
        assert np.amax(sources["variances"]) == 64
        assert np.amin(sources["variances"]) == 0
        assert all(sources["positions"][0, :] == [512, 512])
        assert all(sources["positions"][5, :] == [512, 532])
        assert all(sources["positions"][27, :] == [509, 509])
