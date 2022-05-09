# See the LICENSE file at the top-level directory of this distribution.

import numpy as np
import pytest

try:
    import cupy
    print("All good!")
except ImportError:
    cupy = None

from ska_sdp_func.utility import Error
from ska_sdp_func import Gridder


def rrmse(x, y):
    return np.linalg.norm(x - y)/np.linalg.norm(y)


def atest_gridder_plan():
    print(" ")  # just for separation of debug output
    print(" ")

    # load dataset
    test_data = np.load("tests/test_data/vla_d_3_chan.npz")
    vis = test_data["vis"]
    freqs = test_data["freqs"]
    uvw = test_data["uvw"]
    weight = np.ones(vis.shape)
    # parameters
    imSize = 1024
    pixsize_deg = 1.94322419749866394E-02
    pixsize_rad = pixsize_deg * np.pi / 180.0

    print("pixsize_rad is %.12e" % pixsize_rad)

    epsilon = 1e-5

    dirty_image = np.zeros([imSize, imSize], dtype=np.float64)

    # Run gridder test on GPU, using cupy arrays.
    if cupy:
        print("hi!!")
        vis_gpu = cupy.asarray(vis)
        freqs_gpu = cupy.asarray(freqs)
        uvw_gpu = cupy.asarray(uvw)
        weight_gpu = cupy.asarray(weight)
        dirty_image_gpu = cupy.zeros([imSize, imSize], dtype=np.float64)

        ## tests for plan creation

        # test for memory mismatch on inputs
        error_string = "Memory location mismatch"
        with pytest.raises(RuntimeError, match = error_string):
            gridder = Gridder(uvw, freqs_gpu, vis_gpu, weight_gpu, pixsize_rad, pixsize_rad, epsilon, False)
        with pytest.raises(RuntimeError, match = error_string):
            gridder = Gridder(uvw_gpu, freqs, vis_gpu, weight_gpu, pixsize_rad, pixsize_rad, epsilon, False)
        with pytest.raises(RuntimeError, match = error_string):
            gridder = Gridder(uvw_gpu, freqs_gpu, vis, weight_gpu, pixsize_rad, pixsize_rad, epsilon, False)
        with pytest.raises(RuntimeError, match = error_string):
            gridder = Gridder(uvw_gpu, freqs_gpu, vis_gpu, weight, pixsize_rad, pixsize_rad, epsilon, False)

        # test for wrong type on inputs
        error_string = "Unsupported data type\\(s\\)"
        with pytest.raises(RuntimeError, match = error_string):
            gridder = Gridder(vis_gpu, freqs_gpu, vis_gpu, weight_gpu, pixsize_rad, pixsize_rad, epsilon, False)
        with pytest.raises(RuntimeError, match = error_string):
            gridder = Gridder(uvw_gpu, vis_gpu, vis_gpu, weight_gpu, pixsize_rad, pixsize_rad, epsilon, False)
        with pytest.raises(RuntimeError, match = error_string):
            gridder = Gridder(uvw_gpu, freqs_gpu, uvw_gpu, weight_gpu, pixsize_rad, pixsize_rad, epsilon, False)
        with pytest.raises(RuntimeError, match = error_string):
            gridder = Gridder(uvw_gpu, freqs_gpu, vis_gpu, vis_gpu, pixsize_rad, pixsize_rad, epsilon, False)

        # test for wrong sizes/values on inputs
        error_string = "Invalid function argument"
        with pytest.raises(RuntimeError, match = error_string):
            gridder = Gridder(uvw_gpu[:, 0:2], freqs_gpu, vis_gpu, weight_gpu, pixsize_rad, pixsize_rad, epsilon, False)
        with pytest.raises(RuntimeError, match = error_string):
            gridder = Gridder(uvw_gpu, freqs_gpu[0:2], vis_gpu, weight_gpu, pixsize_rad, pixsize_rad, epsilon, False)
        with pytest.raises(RuntimeError, match = error_string):
            gridder = Gridder(uvw_gpu[0:-2, :], freqs_gpu, vis_gpu, weight_gpu, pixsize_rad, pixsize_rad, epsilon, False)
        with pytest.raises(RuntimeError, match = error_string):
            gridder = Gridder(uvw_gpu, freqs_gpu, vis_gpu, weight_gpu, pixsize_rad, pixsize_rad*2, epsilon, False)
        with pytest.raises(RuntimeError, match = error_string):
            gridder = Gridder(uvw_gpu, freqs_gpu, vis_gpu, weight_gpu, pixsize_rad, pixsize_rad*2, epsilon, False)
        # should test epsilon!!

        ## tests for exec()

        # Create gridder, need to create a valid gridder!
        gridder = Gridder(uvw_gpu, freqs_gpu, vis_gpu, weight_gpu, pixsize_rad, pixsize_rad, epsilon, False)

        gridder.exec(uvw_gpu, freqs_gpu, vis_gpu, weight_gpu, dirty_image_gpu)

        # this checks that sdp_gridder_check_inputs() is being called, but could do exhaustive checking like above...
        error_string = "Memory location mismatch"
        with pytest.raises(RuntimeError, match = error_string):
            gridder.exec(uvw_gpu, freqs, vis_gpu, weight_gpu, dirty_image_gpu)

        ## test dirty_image is correct

        error_string = "Memory location mismatch"
        with pytest.raises(RuntimeError, match = error_string):
            gridder.exec(uvw_gpu, freqs_gpu, vis_gpu, weight_gpu, dirty_image)

        error_string = "Unsupported data type\\(s\\)"
        with pytest.raises(RuntimeError, match = error_string):
            gridder.exec(uvw_gpu, freqs_gpu, vis_gpu, weight_gpu, vis_gpu)  

        error_string = "Invalid function argument"
        with pytest.raises(RuntimeError, match = error_string):
            gridder.exec(uvw_gpu, freqs_gpu, vis_gpu, weight_gpu, dirty_image_gpu[:, 0:-1])

        # don't know how to test contiguity
        # don't know how to test read-only from python


def test_get_w_range():

    print(" ")  # just for separation of debug output
    print(" ")

    # load dataset
    test_data = np.load("tests/test_data/vla_d_3_chan.npz")
    freq_hz = test_data["freqs"]
    uvw = test_data["uvw"]

    print(freq_hz)
    print(freq_hz.dtype)

    true_min_abs_w = np.amin(np.abs(uvw[:, 2]))*freq_hz[0]/299792458.0
    true_max_abs_w = np.amax(np.abs(uvw[:, 2]))*freq_hz[-1]/299792458.0

    print("min_abs_w is %.12e" % true_min_abs_w)
    print("max_abs_w is %.12e" % true_max_abs_w)

    # test with numpy arguments
    print("testing numpy arguments...")
    min_abs_w, max_abs_w = Gridder.get_w_range(uvw, freq_hz)
    # print(rrmse(min_abs_w, true_min_abs_w))
    # print(rrmse(max_abs_w, true_max_abs_w))
    assert(rrmse(min_abs_w, true_min_abs_w) < 1e-15)
    assert(rrmse(max_abs_w, true_max_abs_w) < 1e-15)

    # Run gridder test on GPU, using cupy arrays.
    if cupy:
        freq_hz_gpu = cupy.asarray(freq_hz)
        uvw_gpu = cupy.asarray(uvw)

        # print(type(uvw_gpu))
        # print(type(freq_hz_gpu))

        # test with cupy arguments
        print("testing cupy arguments...")
        min_abs_w, max_abs_w = Gridder.get_w_range(uvw_gpu, freq_hz_gpu)
        # print(rrmse(min_abs_w, true_min_abs_w))
        # print(rrmse(max_abs_w, true_max_abs_w))
        assert (rrmse(min_abs_w, true_min_abs_w) < 1e-15)
        assert (rrmse(max_abs_w, true_max_abs_w) < 1e-15)

    # test with bad arguments
    print("testing bad arguments...")
    min_abs_w, max_abs_w = Gridder.get_w_range(None, None)
    assert(min_abs_w == -1)
    assert(max_abs_w == -1)


def test_gridder():

    print(" ")  # just for separation of debug output
    print(" ")

    # load dataset
    test_data = np.load("tests/test_data/vla_d_3_chan.npz")
    vis = test_data["vis"]
    freqs = test_data["freqs"]
    uvw = test_data["uvw"]
    weight = np.ones(vis.shape)
    # parameters
    imSize = 1024
    pixsize_deg = 1.94322419749866394E-02
    pixsize_rad = pixsize_deg * np.pi / 180.0
    print(pixsize_rad)

    epsilon = 1e-5

    # Run gridder test on GPU, using cupy arrays.
    if cupy:
        vis_gpu = cupy.asarray(vis)
        freqs_gpu = cupy.asarray(freqs)
        uvw_gpu = cupy.asarray(uvw)
        weight_gpu = cupy.asarray(weight)
        dirty_image_gpu = cupy.zeros([imSize, imSize], dtype=np.float64)

        do_wstacking = False

        # print(vis_gpu.dtype)
        # print(freqs_gpu)
        # print(uvw_gpu)
        # print(dirty_image_gpu)

        # Create gridder
        gridder = Gridder(uvw_gpu, freqs_gpu, vis_gpu, weight_gpu, pixsize_rad, pixsize_rad, epsilon, do_wstacking, dirty_image_gpu)

        # Run gridder
        gridder.exec(uvw_gpu, freqs_gpu, vis_gpu, weight_gpu, dirty_image_gpu)