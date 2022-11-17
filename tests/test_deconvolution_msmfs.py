# See the LICENSE file at the top-level directory of this distribution.
""" Module to test MSMFS deconvolution. """
from numpy import int32

try:
    import cupy
except ImportError:
    cupy = None

import numpy as np
import pytest

from ska_sdp_func import msmfs_perform

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
    convolution_accuracy = 1.2E-3
    clean_loop_gain = 0.35
    max_gaussian_sources = 200
    scale_bias_factor = 0.6
    clean_threshold = 0.001

    dirty_moment_images = np.random.rand(num_taylor, dirty_moment_size, dirty_moment_size)
    # dirty_moment_images = np.zeros((num_taylor, dirty_moment_size, dirty_moment_size))
    psf_moment_images = np.random.rand(2*num_taylor-1, psf_moment_size, psf_moment_size)
    # psf_moment_images = np.zeros((2*num_taylor-1, psf_moment_size, psf_moment_size))

    print("dirty_moment_images is :", dirty_moment_images.shape)
    print("psf_moment_images is: ", psf_moment_images.shape)

    num_gaussian_sources = 0

    gaussian_source_positions =  np.zeros((max_gaussian_sources, 2), dtype=int32)
    gaussian_source_variances =  np.zeros(max_gaussian_sources)
    gaussian_source_taylor_intensities =  np.zeros((max_gaussian_sources, num_taylor))

    # create single versions
    # uvw_s = uvw.astype(np.float32)
    # freqs_s = freqs.astype(np.float32)
    # vis_s = vis.astype(np.complex64)
    # weight_s = weight.astype(np.float32)
    # dirty_image_s = dirty_image.astype(np.float32)

    if cupy:
        dirty_moment_images_gpu = cupy.asarray(dirty_moment_images)
        psf_moment_images_gpu = cupy.asarray(psf_moment_images)
        gaussian_source_positions_gpu = cupy.asarray(gaussian_source_positions)
        gaussian_source_variances_gpu = cupy.asarray(gaussian_source_variances)
        gaussian_source_taylor_intensities_gpu = cupy.asarray(gaussian_source_taylor_intensities)

        msmfs_perform(
                dirty_moment_images_gpu,
                psf_moment_images_gpu,
                dirty_moment_size,
                num_scales,
                num_taylor,  # 5
                psf_moment_size,
                image_border,
                convolution_accuracy,
                clean_loop_gain,
                max_gaussian_sources,  # 10
                scale_bias_factor,
                clean_threshold,
                num_gaussian_sources,
                gaussian_source_positions,
                gaussian_source_variances,  #15
                gaussian_source_taylor_intensities,
        )
