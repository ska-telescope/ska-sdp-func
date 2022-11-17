# See the LICENSE file at the top-level directory of this distribution.
""" Module for MSMFS deconvolution function. """

import ctypes

from .utility import Error, Lib, Mem

# try:
#     import cupy
# except ImportError:
#     cupy = None
#
# import numpy as np


def msmfs_perform(
    dirty_moment_images,
    psf_moment_images,
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
    gaussian_source_variances,  # 15
    gaussian_source_taylor_intensities,
):
    """Performs the entire MSMFS deconvolution.

        dirty_moment_images and psf_moment_images assumed to be centred
        around origin with dirty_moment_images and have sufficient border
        for convolutions.

    :param dirty_moment_images: cupy.ndarray((num_taylor, dirty_moment_size,
        dirty_moment_size), dtype=numpy.float32 or numpy.float64)
        Taylor coefficient dirty images to be convolved.
    :param psf_moment_images: cupy.ndarray((num_taylor, psf_moment_size,
        psf_moment_size), dtype=numpy.float32 or numpy.float64)
        Taylor coefficient PSF images to be convolved.
    :param dirty_moment_size:  One dimensional size of image, assumed square.
    :param num_scales:  Number of scales to use in MSMFS cleaning.
    :param num_taylor:  Number of Taylor moments.
    :param psf_moment_size:  One dimensional size of PSF, assumed square.
    :param image_border:  Border around dirty moment images and PSFs to clip
        when using convolved images or convolved PSFs.
    :param convolution_accuracy:
    :param clean_loop_gain:  Loop gain fraction of peak point to clean from
        the peak each minor cycle.
    :param max_gaussian_sources:  Upper bound on the number of gaussian
        sources the list data structure will hold.
    :param scale_bias_factor:  Bias multiplicative factor to favour cleaning
        with smaller scales.
    :param clean_threshold:  Set clean_threshold to 0 to disable checking
        whether source to clean below cutoff threshold.
    """
    mem_dirty_moment_images = Mem(dirty_moment_images)
    mem_psf_moment_images = Mem(psf_moment_images)
    mem_gaussian_source_positions = Mem(gaussian_source_positions)
    mem_gaussian_source_variances = Mem(gaussian_source_variances)
    mem_gaussian_source_taylor_intensities = Mem(gaussian_source_taylor_intensities)

    num_gaussian_sources_uint = ctypes.c_uint(num_gaussian_sources)

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
        ctypes.pointer(num_gaussian_sources_uint),
        mem_gaussian_source_positions.handle(),
        mem_gaussian_source_variances.handle(),  # 15
        mem_gaussian_source_taylor_intensities.handle(),
        error_status.handle(),
    )
    error_status.check()
