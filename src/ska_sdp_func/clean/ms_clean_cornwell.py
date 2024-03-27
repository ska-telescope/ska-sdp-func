"""Module for msCLEAN Cornwell version functions."""

import ctypes

from ..utility import Lib, Mem

Lib.wrap_func(
    "sdp_ms_clean_cornwell",
    restype=None,
    argtypes=[
        Mem.handle_type(),
        Mem.handle_type(),
        Mem.handle_type(),
        Mem.handle_type(),
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        Mem.handle_type()
    ],
    check_errcode=True,
)


def ms_clean_cornwell(
    dirty_img,
    psf,
    cbeam_details,
    scale_list,
    loop_gain,
    threshold,
    cycle_limit,
    skymodel,
 ):
    """Implimentation of the Tim Cornwell version of msCLEAN, also found in RASCIL.
    Requires dirty image, psf and detals of the CLEAN beam

    Parameters can be either numpy or cupy arrays, but must all be consistent.
    Computation is performed either on the CPU or GPU as appropriate.

    :param dirty_img: Input dirty image.
    :type dirty_img: numpy.ndarray or cupy.ndarray
    :param psf: Input Point Spread Function.
    :type psf: numpy.ndarray or cupy.ndarray
    :param cbeam_details: Input shape of cbeam [BMAJ, BMINN, THETA]
    :type cbeam_deatils: numpy.ndarray or cupy.ndarray
    :param scale_list: List of scales to use [scale1, scale2, scale3 ........]
    :type scale_list: numpy.ndarray or cupy.ndarray
    :param loop_gain: Gain to be used in the CLEAN loop (typically 0.1)
    :type loop_gain: float
    :param threshold: Minimum intensity of peak to search for,
    loop terminates if peak is found under this threshold.
    :type threshold: float
    :param cycle_limit: Maximum nuber of loops to perform, if the stop
    threshold is not reached first.
    :type cycle_limit: float
    :param skymodel: Output Skymodel (CLEANed image).:type
    :type skymodel: numpy.ndarray or cupy.ndarray
    """

    Lib.sdp_ms_clean_cornwell(
        Mem(dirty_img),
        Mem(psf),
        Mem(cbeam_details),
        Mem(scale_list),
        loop_gain,
        threshold,
        cycle_limit,
        Mem(skymodel)
    )