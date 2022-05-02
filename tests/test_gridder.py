# See the LICENSE file at the top-level directory of this distribution.

import numpy as np
from ska_sdp_func.utility import Error
import pytest

try:
    import cupy
    print("All good!")
except ImportError:
    cupy = None

from ska_sdp_func import Gridder

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

    epsilon = 1e-5

    # Run gridder test on GPU, using cupy arrays.
    if cupy:
        print("hi!!")
        vis_gpu = cupy.asarray(vis)
        freqs_gpu = cupy.asarray(freqs)
        uvw_gpu = cupy.asarray(uvw)
        weight_gpu = cupy.asarray(weight)

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

        dirty_image = np.zeros([imSize, imSize], dtype=np.float64)
        dirty_image_gpu = cupy.zeros([imSize, imSize], dtype=np.float64)
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
        print("hi!!")
        vis_gpu = cupy.asarray(vis)
        freqs_gpu = cupy.asarray(freqs)
        uvw_gpu = cupy.asarray(uvw)
        weight_gpu = cupy.asarray(weight)

        do_wstacking = False

        print(vis_gpu.dtype)
        print(freqs_gpu)
        print(uvw_gpu)

        # Create gridder
        gridder = Gridder(uvw_gpu, freqs_gpu, vis_gpu, weight_gpu, pixsize_rad, pixsize_rad, epsilon, do_wstacking)

        # Run gridder
        dirty_image_gpu = cupy.zeros([imSize, imSize], dtype=np.float64)
        print(dirty_image_gpu)
        gridder.exec(uvw_gpu, freqs_gpu, vis_gpu, weight_gpu, dirty_image_gpu)