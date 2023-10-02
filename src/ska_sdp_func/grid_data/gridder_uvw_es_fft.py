# See the LICENSE file at the top-level directory of this distribution.
""" Module for gridding functions. """

import ctypes

try:
    import cupy
except ImportError:
    cupy = None

import numpy as np

from ..utility import Lib, Mem, StructWrapper


class GridderUvwEsFft(StructWrapper):
    """Processing function GridderUvwEsFft."""

    def __init__(
        self,
        uvw,
        freq_hz,
        vis,
        weight,
        dirty_image,
        pixel_size_x_rad,
        pixel_size_y_rad,
        epsilon: float,
        do_w_stacking: bool,
    ):
        """Creates a plan for (de)gridding using the supplied parameters and
        input and output buffers.

        This currently only supports processing on a GPU.

        Parameters
        ==========
        uvw: cupy.ndarray((num_rows, 3), dtype=numpy.float32 or numpy.float64)
            (u,v,w) coordinates.
        freq_hz: cupy.ndarray((num_chan,), dtype=numpy.float32 or
            numpy.float64)
            Channel frequencies.
        vis: cupy.ndarray((num_rows, num_chan), dtype=numpy.complex64 or
            numpy.complex128)
            The input/output visibility data.
            Its data type determines the precision used for the (de)gridding.
        weight: cupy.ndarray((num_rows, num_chan), same precision as **vis**)
            Its values are used to multiply the input.
        dirty_image: cupy.ndarray((num_pix, num_pix), dtype=numpy.float32 or
            numpy.float64)
            The input/output dirty image, **must be square**.
        pixel_size_x_rad: float
            Angular x pixel size (in radians) of the dirty image.
        pixel_size_y_rad: float
            Angular y pixel size (in radians) of the dirty image (must be the
            same as pixel_size_x_rad).
        epsilon: float
            Accuracy at which the computation should be done.
            Must be larger than 2e-13.
            If **vis** has type numpy.complex64, it must be larger than 1e-5.
        do_w_stacking: bool
            If True, the full improved w-stacking algorithm is carried out,
            otherwise the w values are assumed to be zero.
        """
        if do_w_stacking:
            min_abs_w, max_abs_w = GridderUvwEsFft.get_w_range(uvw, freq_hz)
        else:
            min_abs_w = 0
            max_abs_w = 0

        create_args = (
            Mem(uvw),
            Mem(freq_hz),
            Mem(vis),
            Mem(weight),
            Mem(dirty_image),
            pixel_size_x_rad,
            pixel_size_y_rad,
            epsilon,
            min_abs_w,
            max_abs_w,  # 10
            do_w_stacking,
        )
        super().__init__(
            Lib.sdp_gridder_uvw_es_fft_create_plan,
            create_args,
            Lib.sdp_gridder_uvw_es_fft_free_plan,
        )

    @staticmethod
    def get_w_range(uvw, freq_hz):
        """Calculate w-range from UVW-coordinates."""
        if isinstance(uvw, np.ndarray):
            min_abs_w = np.amin(np.abs(uvw[:, 2]))
            max_abs_w = np.amax(np.abs(uvw[:, 2]))
        elif cupy and isinstance(uvw, cupy.ndarray):
            min_abs_w = cupy.amin(cupy.abs(uvw[:, 2]))
            max_abs_w = cupy.amax(cupy.abs(uvw[:, 2]))
        else:
            print(f"Unsupported uvw type of {type(uvw)}.")
            return -1, -1

        min_abs_w *= freq_hz[0] / 299792458.0
        max_abs_w *= freq_hz[-1] / 299792458.0

        return min_abs_w, max_abs_w

    def grid_uvw_es_fft(self, uvw, freq_hz, vis, weight, dirty_image):
        """Generate a dirty image from visibility data.

        Parameters
        ==========
        uvw: as above.
        freq_hz: as above.
        vis: as above.
        weight: as above.
        dirty_image: as above.
        """
        Lib.sdp_grid_uvw_es_fft(
            self,
            Mem(uvw),
            Mem(freq_hz),
            Mem(vis),
            Mem(weight),
            Mem(dirty_image),
        )

    def ifft_grid_uvw_es(self, uvw, freq_hz, vis, weight, dirty_image):
        """Generate visibility data from a dirty image.

        Parameters
        ==========
        uvw: as above.
        freq_hz: as above.
        vis: as above.
        weight: as above.
        dirty_image: as above.
        """
        Lib.sdp_ifft_degrid_uvw_es(
            self,
            Mem(uvw),
            Mem(freq_hz),
            Mem(vis),
            Mem(weight),
            Mem(dirty_image),
        )


Lib.wrap_func(
    "sdp_gridder_uvw_es_fft_create_plan",
    restype=GridderUvwEsFft.handle_type(),
    argtypes=[
        Mem.handle_type(),
        Mem.handle_type(),
        Mem.handle_type(),
        Mem.handle_type(),
        Mem.handle_type(),  # 5
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,  # 10
        ctypes.c_int,
    ],
    check_errcode=True,
)

Lib.wrap_func(
    "sdp_gridder_uvw_es_fft_free_plan",
    restype=None,
    argtypes=[GridderUvwEsFft.handle_type()],
)

Lib.wrap_func(
    "sdp_grid_uvw_es_fft",
    restype=None,
    argtypes=[
        GridderUvwEsFft.handle_type(),
        Mem.handle_type(),
        Mem.handle_type(),
        Mem.handle_type(),
        Mem.handle_type(),
        Mem.handle_type(),
    ],
    check_errcode=True,
)

Lib.wrap_func(
    "sdp_ifft_degrid_uvw_es",
    restype=None,
    argtypes=[
        GridderUvwEsFft.handle_type(),
        Mem.handle_type(),
        Mem.handle_type(),
        Mem.handle_type(),
        Mem.handle_type(),
        Mem.handle_type(),
    ],
    check_errcode=True,
)
