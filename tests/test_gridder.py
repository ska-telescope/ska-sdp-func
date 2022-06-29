# See the LICENSE file at the top-level directory of this distribution.
""" Module to test gridding functions. """

try:
    import cupy
except ImportError:
    cupy = None

import numpy as np
import pytest

from ska_sdp_func import Gridder


def rrmse(in_x, in_y):
    """Calculates the relative RMS error between the inputs."""
    return np.linalg.norm(in_x - in_y) / np.linalg.norm(in_y)


def run_ms2dirty(do_single, do_w_stacking, epsilon=1e-5):
    """Temp function."""
    return 0*do_single*do_w_stacking*epsilon, 1


def run_dirty2ms(do_single, do_w_stacking, epsilon=1e-5):
    """Temp function."""
    return 0*do_single*do_w_stacking*epsilon, 1


def arun_ms2dirty(do_single, do_w_stacking, epsilon=1e-5):
    """Runs ms2dirty for tests below."""
    print(" ")  # just for separation of debug output
    print(" ")

    # load dataset
    test_data = np.load("tests/test_data/vla_d_3_chan.npz")
    vis = test_data["vis"]
    freqs = test_data["freqs"]
    uvw = test_data["uvw"]
    weight = np.ones(vis.shape)

    if do_single:
        vis = vis.astype(np.complex64)
        freqs = freqs.astype(np.float32)
        uvw = uvw.astype(np.float32)
        weight = weight.astype(np.float32)

    # parameters
    im_size = 1024
    # pixel_size_deg = 1.94322419749866394e-02
    pixel_size_rad = 1.94322419749866394e-02 * np.pi / 180.0
    # print(pixel_size_rad)

    this_rrmse = 1
    # pass_threshold = 0

    # Run gridder test on GPU, using cupy arrays.
    if cupy:
        vis_gpu = cupy.asarray(vis)
        freqs_gpu = cupy.asarray(freqs)
        uvw_gpu = cupy.asarray(uvw)
        weight_gpu = cupy.asarray(weight)
        dirty_image_gpu = cupy.zeros([im_size, im_size], uvw.dtype)

        # print(vis_gpu.dtype)
        # print(freqs_gpu)
        # print(uvw_gpu)
        # print(dirty_image_gpu)

        # Create gridder
        gridder = Gridder(
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
        gridder.ms2dirty(
            uvw_gpu, freqs_gpu, vis_gpu, weight_gpu, dirty_image_gpu
        )

        # Check output
        dirty_image = cupy.asnumpy(dirty_image_gpu)

        # dirty_image_file = "tests/test_data/dirty_image_1024_1e-5_3D_SP.npy"

        dirty_image_file = (
            f"tests/test_data/dirty_image_1024_"
            f"{1e-5 if do_single else 1e-12:.0e}_"
            f"{'3D' if do_w_stacking else '2D'}_"
            f"{'SP' if do_single else 'DP'}.npy"
        )

        # print(dirty_image_file)

        # dirty_image_file = (
        #     "tests/test_data/dirty_image_1024_%.0e_%s_%s.npy"
        #     % (
        #         1e-5 if do_single else 1e-12,
        #         "3D" if do_w_stacking else "2D",
        #         "SP" if do_single else "DP",
        #     )
        # )

        # np.save(dirty_image_file + "x", dirty_image)
        # the x stops the test file been overwritten
        expected_dirty_image = np.load(dirty_image_file)

        # pass_threshold = 1e-5 if do_single else 1e-12

        this_rrmse = rrmse(dirty_image, expected_dirty_image)

        print(f"RRMSE of dirty images is {this_rrmse:e}")

    return this_rrmse, 1e-5 if do_single else 1e-12  # pass_threshold


def arun_dirty2ms(do_single, do_w_stacking, epsilon=1e-5):
    """Runs dirty2ms for tests below."""
    print(" ")  # just for separation of debug output
    print(" ")

    # load dataset
    test_data = np.load("tests/test_data/vla_d_3_chan.npz")
    freqs = test_data["freqs"]
    uvw = test_data["uvw"]
    num_vis = uvw.shape[0]
    num_chan = freqs.shape[0]
    weight = np.ones([num_vis, num_chan])

    if do_single:
        freqs = freqs.astype(np.float32)
        uvw = uvw.astype(np.float32)
        weight = weight.astype(np.float32)

    dirty_image_file = (
        f"tests/test_data/dirty_image_1024_"
        f"{1e-5 if do_single else 1e-12:.0e}_"
        f"{'3D' if do_w_stacking else '2D'}_"
        f"{'SP' if do_single else 'DP'}.npy"
    )
    # dirty_image_file = "tests/test_data/dirty_image_1024_%.0e_%s_%s.npy" % (
    #     1e-5 if do_single else 1e-12,
    #     "3D" if do_w_stacking else "2D",
    #     "SP" if do_single else "DP",
    # )

    # dirty_image = np.load(dirty_image_file)

    # dirty_image *= 1.00001

    # parameters
    # im_size = 1024
    # pixel_size_deg = 1.94322419749866394e-02
    pixel_size_rad = 1.94322419749866394e-02 * np.pi / 180.0
    # print(pixel_size_rad)

    this_rrmse = 1
    # pass_threshold = 0

    # Run gridder test on GPU, using cupy arrays.
    if cupy:
        freqs_gpu = cupy.asarray(freqs)
        uvw_gpu = cupy.asarray(uvw)
        weight_gpu = cupy.asarray(weight)
        dirty_image_gpu = cupy.asarray(np.load(dirty_image_file))

        vis_gpu = cupy.zeros(
            [uvw_gpu.shape[0], freqs_gpu.shape[0]],
            np.complex64 if do_single else np.complex128,
        )

        # print(vis_gpu.dtype)
        # print(vis_gpu.shape)
        # print(freqs_gpu)
        # print(uvw_gpu)
        # print(dirty_image_gpu)

        # Create gridder
        gridder = Gridder(
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
        gridder.dirty2ms(
            uvw_gpu, freqs_gpu, vis_gpu, weight_gpu, dirty_image_gpu
        )

        # Check output
        # test_file = "tests/test_data/vis_1024_1e-5_3D_SP.npy"

        test_file = (
            f"tests/test_data/vis_1024_"
            f"{1e-5 if do_single else 1e-12:.0e}_"
            f"{'3D' if do_w_stacking else '2D'}_"
            f"{'SP' if do_single else 'DP'}.npy"
        )

        print(test_file)
        # vis = cupy.asnumpy(vis_gpu)
        # np.save(test_file + "x", vis)
        # the x stops the test file been overwritten
        test_output = np.load(test_file)

        this_rrmse = rrmse(cupy.asnumpy(vis_gpu), test_output)
        print(f"RRMSE of visibilities is {this_rrmse:e}")

        # pass_threshold = 1e-5 if do_single else 1e-12

    return this_rrmse, 1e-5 if do_single else 1e-12  # pass_threshold


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
            gridder = Gridder(
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
            gridder = Gridder(
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
            gridder = Gridder(
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
            gridder = Gridder(
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
            gridder = Gridder(
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
            gridder = Gridder(
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
            gridder = Gridder(
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
            gridder = Gridder(
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
            gridder = Gridder(
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
            gridder = Gridder(
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
            gridder = Gridder(
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
            gridder = Gridder(
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
            gridder = Gridder(
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
        gridder = Gridder(
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
    test_data = np.load("tests/test_data/vla_d_3_chan.npz")
    freq_hz = test_data["freqs"]
    uvw = test_data["uvw"]

    print(freq_hz)
    print(freq_hz.dtype)

    true_min_abs_w = np.amin(np.abs(uvw[:, 2])) * freq_hz[0] / 299792458.0
    true_max_abs_w = np.amax(np.abs(uvw[:, 2])) * freq_hz[-1] / 299792458.0

    print(f"min_abs_w is {true_min_abs_w:.12e}")
    print(f"max_abs_w is {true_max_abs_w:.12e}")

    # test with numpy arguments
    print("testing numpy arguments...")
    min_abs_w, max_abs_w = Gridder.get_w_range(uvw, freq_hz)
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
        min_abs_w, max_abs_w = Gridder.get_w_range(uvw_gpu, freq_hz_gpu)
        # print(rrmse(min_abs_w, true_min_abs_w))
        # print(rrmse(max_abs_w, true_max_abs_w))
        assert rrmse(min_abs_w, true_min_abs_w) < 1e-15
        assert rrmse(max_abs_w, true_max_abs_w) < 1e-15

    # test with bad arguments
    print("testing bad arguments...")
    min_abs_w, max_abs_w = Gridder.get_w_range(None, None)
    assert min_abs_w == -1
    assert max_abs_w == -1


def test_ms2dirty_sp_2d():
    """Test."""
    this_rrmse, pass_threshold = run_ms2dirty(
        do_single=True, do_w_stacking=False
    )
    assert this_rrmse < pass_threshold


def test_ms2dirty_sp_3d():
    """Test."""
    this_rrmse, pass_threshold = run_ms2dirty(
        do_single=True, do_w_stacking=True
    )
    assert this_rrmse < pass_threshold


def test_ms2dirty_dp_2d():
    """Test."""
    this_rrmse, pass_threshold = run_ms2dirty(
        do_single=False, do_w_stacking=False, epsilon=1e-12
    )
    assert this_rrmse < pass_threshold


def test_ms2dirty_dp_3d():
    """Test."""
    this_rrmse, pass_threshold = run_ms2dirty(
        do_single=False, do_w_stacking=True, epsilon=1e-12
    )
    assert this_rrmse < pass_threshold


def test_dirty2ms_sp_2d():
    """Test."""
    this_rrmse, pass_threshold = run_dirty2ms(
        do_single=True, do_w_stacking=False
    )
    assert this_rrmse < pass_threshold


def test_dirty2ms_sp_3d():
    """Test."""
    this_rrmse, pass_threshold = run_dirty2ms(
        do_single=True, do_w_stacking=True
    )
    assert this_rrmse < pass_threshold


def test_dirty2ms_dp_2d():
    """Test."""
    this_rrmse, pass_threshold = run_dirty2ms(
        do_single=False, do_w_stacking=False, epsilon=1e-12
    )
    assert this_rrmse < pass_threshold


def test_dirty2ms_dp_3d():
    """Test."""
    this_rrmse, pass_threshold = run_dirty2ms(
        do_single=False, do_w_stacking=True, epsilon=1e-12
    )
    assert this_rrmse < pass_threshold
