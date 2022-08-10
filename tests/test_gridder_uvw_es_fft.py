# See the LICENSE file at the top-level directory of this distribution.
""" Module to test gridding functions. """

try:
    import cupy
except ImportError:
    cupy = None

import numpy as np
import pytest

from ska_sdp_func import GridderUvwEsFft


def rrmse(in_x, in_y):
    """Calculates the relative RMS error between the inputs."""
    return np.linalg.norm(in_x - in_y) / np.linalg.norm(in_y)


def atest_gridder_plan():
    """Test"""
    print(" ")  # just for separation of debug output
    print(" ")

    # load dataset
    test_data = np.load("tests/test_data/vla_d_3_chan.npz")
    vis = test_data["vis"]
    freqs = test_data["freqs"]
    uvw = test_data["uvw"]
    weight = np.ones(vis.shape)
    # parameters
    im_size = 1024
    pixsize_deg = 1.94322419749866394e-02
    pixsize_rad = pixsize_deg * np.pi / 180.0

    print(f"pixsize_rad is {pixsize_rad:.12e}")

    epsilon = 1e-5

    dirty_image = np.zeros([im_size, im_size], dtype=np.float64)

    # Run gridder test on GPU, using cupy arrays.
    if cupy:
        print("hi!!")
        vis_gpu = cupy.asarray(vis)
        freqs_gpu = cupy.asarray(freqs)
        uvw_gpu = cupy.asarray(uvw)
        weight_gpu = cupy.asarray(weight)
        dirty_image_gpu = cupy.zeros([im_size, im_size], dtype=np.float64)

        # # tests for plan creation

        # test for memory mismatch on inputs
        error_string = "Memory location mismatch"
        with pytest.raises(RuntimeError, match=error_string):
            gridder = GridderUvwEsFft(
                uvw,
                freqs_gpu,
                vis_gpu,
                weight_gpu,
                dirty_image_gpu,
                pixsize_rad,
                pixsize_rad,
                epsilon,
                False,
            )
        with pytest.raises(RuntimeError, match=error_string):
            gridder = GridderUvwEsFft(
                uvw_gpu,
                freqs,
                vis_gpu,
                weight_gpu,
                dirty_image_gpu,
                pixsize_rad,
                pixsize_rad,
                epsilon,
                False,
            )
        with pytest.raises(RuntimeError, match=error_string):
            gridder = GridderUvwEsFft(
                uvw_gpu,
                freqs_gpu,
                vis,
                weight_gpu,
                dirty_image_gpu,
                pixsize_rad,
                pixsize_rad,
                epsilon,
                False,
            )
        with pytest.raises(RuntimeError, match=error_string):
            gridder = GridderUvwEsFft(
                uvw_gpu,
                freqs_gpu,
                vis_gpu,
                weight,
                dirty_image_gpu,
                pixsize_rad,
                pixsize_rad,
                epsilon,
                False,
            )

        # test for wrong type on inputs
        error_string = "Unsupported data type\\(s\\)"
        with pytest.raises(RuntimeError, match=error_string):
            gridder = GridderUvwEsFft(
                vis_gpu,
                freqs_gpu,
                vis_gpu,
                weight_gpu,
                dirty_image_gpu,
                pixsize_rad,
                pixsize_rad,
                epsilon,
                False,
            )
        with pytest.raises(RuntimeError, match=error_string):
            gridder = GridderUvwEsFft(
                uvw_gpu,
                vis_gpu,
                vis_gpu,
                weight_gpu,
                dirty_image_gpu,
                pixsize_rad,
                pixsize_rad,
                epsilon,
                False,
            )
        with pytest.raises(RuntimeError, match=error_string):
            gridder = GridderUvwEsFft(
                uvw_gpu,
                freqs_gpu,
                uvw_gpu,
                weight_gpu,
                dirty_image_gpu,
                pixsize_rad,
                pixsize_rad,
                epsilon,
                False,
            )
        with pytest.raises(RuntimeError, match=error_string):
            gridder = GridderUvwEsFft(
                uvw_gpu,
                freqs_gpu,
                vis_gpu,
                vis_gpu,
                dirty_image_gpu,
                pixsize_rad,
                pixsize_rad,
                epsilon,
                False,
            )

        # test for wrong sizes/values on inputs
        error_string = "Invalid function argument"
        with pytest.raises(RuntimeError, match=error_string):
            gridder = GridderUvwEsFft(
                uvw_gpu[:, 0:2],
                freqs_gpu,
                vis_gpu,
                weight_gpu,
                dirty_image_gpu,
                pixsize_rad,
                pixsize_rad,
                epsilon,
                False,
            )
        with pytest.raises(RuntimeError, match=error_string):
            gridder = GridderUvwEsFft(
                uvw_gpu,
                freqs_gpu[0:2],
                vis_gpu,
                weight_gpu,
                dirty_image_gpu,
                pixsize_rad,
                pixsize_rad,
                epsilon,
                False,
            )
        with pytest.raises(RuntimeError, match=error_string):
            gridder = GridderUvwEsFft(
                uvw_gpu[0:-2, :],
                freqs_gpu,
                vis_gpu,
                weight_gpu,
                dirty_image_gpu,
                pixsize_rad,
                pixsize_rad,
                epsilon,
                False,
            )
        with pytest.raises(RuntimeError, match=error_string):
            gridder = GridderUvwEsFft(
                uvw_gpu,
                freqs_gpu,
                vis_gpu,
                weight_gpu,
                dirty_image_gpu,
                pixsize_rad,
                pixsize_rad * 2,
                epsilon,
                False,
            )
        with pytest.raises(RuntimeError, match=error_string):
            gridder = GridderUvwEsFft(
                uvw_gpu,
                freqs_gpu,
                vis_gpu,
                weight_gpu,
                dirty_image_gpu,
                pixsize_rad,
                pixsize_rad * 2,
                epsilon,
                False,
            )
        # should test epsilon!!

        # # tests for exec()

        # Create gridder, need to create a valid gridder!
        gridder = GridderUvwEsFft(
            uvw_gpu,
            freqs_gpu,
            vis_gpu,
            weight_gpu,
            dirty_image_gpu,
            pixsize_rad,
            pixsize_rad,
            epsilon,
            False,
        )

        gridder.ms2dirty(
            uvw_gpu, freqs_gpu, vis_gpu, weight_gpu, dirty_image_gpu
        )

        # this checks that sdp_gridder_check_inputs() is being called,
        # but could do exhaustive checking like above...
        error_string = "Memory location mismatch"
        with pytest.raises(RuntimeError, match=error_string):
            gridder.ms2dirty(
                uvw_gpu, freqs, vis_gpu, weight_gpu, dirty_image_gpu
            )

        # # test dirty_image is correct

        error_string = "Memory location mismatch"
        with pytest.raises(RuntimeError, match=error_string):
            gridder.ms2dirty(
                uvw_gpu, freqs_gpu, vis_gpu, weight_gpu, dirty_image
            )

        error_string = "Unsupported data type\\(s\\)"
        with pytest.raises(RuntimeError, match=error_string):
            gridder.ms2dirty(uvw_gpu, freqs_gpu, vis_gpu, weight_gpu, vis_gpu)

        error_string = "Invalid function argument"
        with pytest.raises(RuntimeError, match=error_string):
            gridder.ms2dirty(
                uvw_gpu,
                freqs_gpu,
                vis_gpu,
                weight_gpu,
                dirty_image_gpu[:, 0:-1],
            )

        # don't know how to test contiguity
        # don't know how to test read-only from python


def test_get_w_range():
    """Test."""
    print(" ")  # just for separation of debug output
    print(" ")

    # load dataset
    # test_data = np.load("tests/test_data/vla_d_3_chan.npz")
    # freq_hz = test_data["freqs"]
    # uvw = test_data["uvw"]
    num_vis = 1000
    num_chan = 10
    fov = 2  # degrees
    im_size = 1024

    speed_of_light = 299792458.0
    np.random.seed(40)
    pixel_size_rad = fov * np.pi / 180 / im_size
    f_0 = 1e9

    freq_hz = f_0 + np.arange(num_chan) * (f_0 / num_chan)
    uvw = (np.random.rand(num_vis, 3) - 0.5) / (
        pixel_size_rad * f_0 / speed_of_light
    )

    print(freq_hz)
    print(freq_hz.dtype)

    true_min_abs_w = np.amin(np.abs(uvw[:, 2])) * freq_hz[0] / speed_of_light
    true_max_abs_w = np.amax(np.abs(uvw[:, 2])) * freq_hz[-1] / speed_of_light

    print(f"min_abs_w is {true_min_abs_w:.12e}")
    print(f"max_abs_w is {true_max_abs_w:.12e}")

    # test with numpy arguments
    print("testing numpy arguments...")
    min_abs_w, max_abs_w = GridderUvwEsFft.get_w_range(uvw, freq_hz)
    # print(rrmse(min_abs_w, true_min_abs_w))
    # print(rrmse(max_abs_w, true_max_abs_w))
    assert rrmse(min_abs_w, true_min_abs_w) < 1e-15
    assert rrmse(max_abs_w, true_max_abs_w) < 1e-15

    # Run gridder test on GPU, using cupy arrays.
    if cupy:
        freq_hz_gpu = cupy.asarray(freq_hz)
        uvw_gpu = cupy.asarray(uvw)

        # print(type(uvw_gpu))
        # print(type(freq_hz_gpu))

        # test with cupy arguments
        print("testing cupy arguments...")
        min_abs_w, max_abs_w = GridderUvwEsFft.get_w_range(uvw_gpu, freq_hz_gpu)
        # print(rrmse(min_abs_w, true_min_abs_w))
        # print(rrmse(max_abs_w, true_max_abs_w))
        assert rrmse(min_abs_w, true_min_abs_w) < 1e-15
        assert rrmse(max_abs_w, true_max_abs_w) < 1e-15

    # test with bad arguments
    print("testing bad arguments...")
    min_abs_w, max_abs_w = GridderUvwEsFft.get_w_range(None, None)
    assert min_abs_w == -1
    assert max_abs_w == -1


def test_gridder_degridder_sp_2d():
    """Test."""
    adj_error, pass_threshold = run_gridder_adjointness_check(
        do_single=True, do_w_stacking=False, epsilon=1e-5
    )
    assert adj_error < pass_threshold


def test_gridder_degridder_sp_3d():
    """Test."""
    adj_error, pass_threshold = run_gridder_adjointness_check(
        do_single=True, do_w_stacking=True, epsilon=1e-5
    )
    assert adj_error < pass_threshold


def test_gridder_degridder_dp_2d():
    """Test."""
    adj_error, pass_threshold = run_gridder_adjointness_check(
        do_single=False, do_w_stacking=False, epsilon=1e-12
    )
    assert adj_error < pass_threshold


def test_gridder_degridder_dp_3d():
    """Test."""
    adj_error, pass_threshold = run_gridder_adjointness_check(
        do_single=False, do_w_stacking=True, epsilon=1e-12
    )
    assert adj_error < pass_threshold


def run_gridder_adjointness_check(do_single, do_w_stacking, epsilon=1e-5):
    """Run an adjointness test, which tests both gridding and degridding."""

    num_vis = 1000
    num_chan = 10
    nxydirty = 1024
    fov = 2  # degrees

    # print("\n\nTesting gridding/degridding with {} rows and {} " \
    #       "frequency channels".format(num_vis, num_chan))
    # print("Dirty image has {}x{} pixels, " \
    #       "FOV={} degrees".format(nxydirty, nxydirty, fov))
    # print("Requested accuracy: {}".format(epsilon))

    speed_of_light = 299792458.0
    np.random.seed(40)
    pixel_size_rad = fov * np.pi / 180 / nxydirty
    f_0 = 1e9

    freqs = f_0 + np.arange(num_chan) * (f_0 / num_chan)
    uvw = (np.random.rand(num_vis, 3) - 0.5) / (
        pixel_size_rad * f_0 / speed_of_light
    )
    test_vis = (
        np.random.rand(num_vis, num_chan)
        - 0.5
        + 1j * (np.random.rand(num_vis, num_chan) - 0.5)
    )
    test_dirty_image = np.random.rand(nxydirty, nxydirty) - 0.5
    weight = np.ones([num_vis, num_chan])

    # print(test_vis)
    # print(test_dirty_image)

    if do_single:
        test_vis = test_vis.astype(np.complex64)
        test_dirty_image = test_dirty_image.astype(np.float32)
        freqs = freqs.astype(np.float32)
        uvw = uvw.astype(np.float32)
        weight = weight.astype(np.float32)

    adj_error = -1

    # Run gridder test on GPU, using cupy arrays.
    if cupy:
        # run ms2dirty
        freqs_gpu = cupy.asarray(freqs)
        uvw_gpu = cupy.asarray(uvw)
        weight_gpu = cupy.asarray(weight)
        test_vis_gpu = cupy.asarray(test_vis)
        test_dirty_image_gpu = cupy.asarray(test_dirty_image)

        vis_gpu = cupy.zeros(
            [num_vis, num_chan],
            np.complex64 if do_single else np.complex128,
        )
        dirty_image_gpu = cupy.zeros(
            [nxydirty, nxydirty],
            np.float32 if do_single else np.float64,
        )

        # Create gridder
        gridder = GridderUvwEsFft(
            uvw_gpu,
            freqs_gpu,
            test_vis_gpu,
            weight_gpu,
            dirty_image_gpu,
            pixel_size_rad,
            pixel_size_rad,
            epsilon,
            do_w_stacking,
        )

        # Run gridder
        gridder.grid_uvw_es_fft(
            uvw_gpu, freqs_gpu, test_vis_gpu, weight_gpu, dirty_image_gpu
        )

        # Check output
        dirty_image = cupy.asnumpy(dirty_image_gpu)
        # print(test_dirty_image)
        # print(dirty_image)

        adj1 = np.vdot(dirty_image, test_dirty_image)

        # Run gridder
        gridder.ifft_grid_uvw_es(
            uvw_gpu, freqs_gpu, vis_gpu, weight_gpu, test_dirty_image_gpu
        )

        vis = cupy.asnumpy(vis_gpu)

        adj2 = np.vdot(vis, test_vis).real
        adj_error = np.abs(adj1 - adj2) / np.maximum(
            np.abs(adj1), np.abs(adj2)
        )

        print()
        print("************************************************************")
        print(f"adjointness test - adj1: {adj1}, adj2: {adj2}")
        print(f"adjointness test: {adj_error}")
        print("************************************************************")
        print()

    return adj_error, 1e-5 if do_single else 1e-12  # pass_threshold
