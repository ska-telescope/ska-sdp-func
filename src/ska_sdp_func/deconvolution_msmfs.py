# See the LICENSE file at the top-level directory of this distribution.
""" Module for MSMFS deconvolution function. """


import ctypes

import numpy as np

from .utility import Error, Lib, Mem


def msmfs_perform(
    dirty_moment_images,
    psf_moment_images,
    num_scales: int,
    image_border: int,
    convolution_accuracy,  # 5
    clean_loop_gain,
    max_gaussian_sources: int,
    scale_bias_factor,
    clean_threshold,
) -> (int, dict):
    """Performs the entire MSMFS deconvolution.

        dirty_moment_images and psf_moment_images assumed to be square and
        centred around origin with dirty_moment_images and have sufficient
        border for convolutions.

    :param dirty_moment_images: cupy.ndarray((num_taylor, dirty_moment_size,
        dirty_moment_size), dtype=numpy.float32 or numpy.float64)
        Taylor coefficient dirty images to be convolved.
    :param psf_moment_images: cupy.ndarray((num_taylor, psf_moment_size,
        psf_moment_size), dtype=numpy.float32 or numpy.float64)
        Taylor coefficient PSF images to be convolved.
    :param num_scales:  Number of scales to use in MSMFS cleaning.
    :param image_border:  Border around dirty moment images and PSFs to clip
        when using convolved images or convolved PSFs.
    :param convolution_accuracy:
    :param clean_loop_gain:  Loop gain fraction of peak point to clean from
        the peak each minor cycle.
    :param max_gaussian_sources:  Upper bound on the number of gaussian
        sources to find.
    :param scale_bias_factor:  Bias multiplicative factor to favour cleaning
        with smaller scales.
    :param clean_threshold:  Set clean_threshold to 0 to disable checking
        whether source to clean below cutoff threshold.
    :returns:
        num_gaussian_sources: The number of Gaussian sources found.
        sources: Dictionary with the keys 'positions', 'variances',
        'taylor_intensities'
    """
    mem_dirty_moment_images = Mem(dirty_moment_images)
    mem_psf_moment_images = Mem(psf_moment_images)

    num_taylor = dirty_moment_images.shape[0]
    dirty_moment_size = dirty_moment_images.shape[1]
    psf_moment_size = psf_moment_images.shape[1]

    gaussian_source_positions = np.zeros(
        (max_gaussian_sources, 2), dtype=np.int32
    )
    gaussian_source_variances = np.zeros(max_gaussian_sources)
    gaussian_source_taylor_intensities = np.zeros(
        (max_gaussian_sources, num_taylor)
    )

    mem_gaussian_source_positions = Mem(gaussian_source_positions)
    mem_gaussian_source_variances = Mem(gaussian_source_variances)
    mem_gaussian_source_taylor_intensities = Mem(
        gaussian_source_taylor_intensities
    )

    num_gaussian_sources_uint = ctypes.c_uint()

    error_status = Error()
    lib_msmfs_perform = Lib.handle().sdp_msmfs_perform
    lib_msmfs_perform.argtypes = [
        Mem.handle_type(),
        Mem.handle_type(),
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_uint,  # 5
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_uint,  # 10
        ctypes.c_double,
        ctypes.c_double,
        ctypes.POINTER(ctypes.c_uint),
        Mem.handle_type(),
        Mem.handle_type(),  # 15
        Mem.handle_type(),
        Error.handle_type(),
    ]
    lib_msmfs_perform(
        mem_dirty_moment_images.handle(),
        mem_psf_moment_images.handle(),
        ctypes.c_uint(dirty_moment_size),
        ctypes.c_uint(num_scales),
        ctypes.c_uint(num_taylor),  # 5
        ctypes.c_uint(psf_moment_size),
        ctypes.c_uint(image_border),
        ctypes.c_double(convolution_accuracy),
        ctypes.c_double(clean_loop_gain),
        ctypes.c_uint(max_gaussian_sources),  # 10
        ctypes.c_double(scale_bias_factor),
        ctypes.c_double(clean_threshold),
        ctypes.byref(num_gaussian_sources_uint),
        mem_gaussian_source_positions.handle(),
        mem_gaussian_source_variances.handle(),  # 15
        mem_gaussian_source_taylor_intensities.handle(),
        error_status.handle(),
    )
    error_status.check()

    num_sources = num_gaussian_sources_uint.value

    sources = {
        "positions": gaussian_source_positions[0:num_sources, :],
        "variances": gaussian_source_variances[0:num_sources],
        "taylor_intensities": gaussian_source_taylor_intensities[
            0:num_sources, :
        ],
    }

    return num_sources, sources
