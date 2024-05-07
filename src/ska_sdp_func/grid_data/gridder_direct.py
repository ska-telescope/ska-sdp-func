# See the LICENSE file at the top-level directory of this distribution.
"""Module for (de)gridding functions."""

import ctypes

import numpy

from ..utility import Lib, Mem, StructWrapper


class GridderDirect(StructWrapper):
    """Uses discrete Fourier transformation and PSWF for subgrid (de)gridding.

    Very inefficient, but as accurate as one can be (by defintion).

    These functions are ported from the implementation provided
    in Peter Wortmann's subgrid_imaging Jupyter notebook.
    """

    def __init__(
        self, image_size: int, subgrid_size: int, theta: float, support: int
    ):
        """Creates a plan for (de)gridding using the supplied parameters.

        :param image_size: Total image size in pixels.
        :param subgrid_size: Sub-grid size in pixels.
        :param theta: Total image size in directional cosines.
        :param support: Support size.
        """
        create_args = (image_size, subgrid_size, theta, support)
        super().__init__(
            Lib.sdp_gridder_direct_create,
            create_args,
            Lib.sdp_gridder_direct_free,
        )

    def degrid(
        self,
        subgrid_image: numpy.ndarray,
        subgrid_offset_u: int,
        subgrid_offset_v: int,
        freq0_hz: float,
        dfreq_hz: float,
        uvw: numpy.ndarray,
        start_chs: numpy.ndarray,
        end_chs: numpy.ndarray,
        vis: numpy.ndarray,
    ):
        """Degrid visibilities using direct Fourier transformation.

        This is painfully slow, but as good as we can make it by definition.

        The caller must ensure the output visibility array is sized correctly.

        :param subgrid_image: Fourier transformed subgrid to degrid from.
            Note that the subgrid could especially span the entire grid,
            in which case this could simply be the entire (corrected) image.
        :param subgrid_offset_u, subgrid_offset_v:
            Offset of subgrid centre relative to grid centre
        :param freq0_hz: Frequency of first channel (Hz)
        :param dfreq_hz: Channel width (Hz)
        :param uvw: ``float[uvw_count, 3]``
            UVW coordinates of visibilities (in m)
        :param start_chs: ``int[uvw_count]``
            First channel to degrid for every uvw
        :param end_chs: ``int[uvw_count]``
            Channel at which to stop degridding for every uvw
        :param vis: ``complex[uvw_count, ch_count]`` Output visibilities
        """
        Lib.sdp_gridder_direct_degrid(
            self,
            Mem(subgrid_image),
            subgrid_offset_u,
            subgrid_offset_v,
            freq0_hz,
            dfreq_hz,
            Mem(uvw),
            Mem(start_chs),
            Mem(end_chs),
            Mem(vis),
        )

    def degrid_correct(
        self, facet: numpy.ndarray, facet_offset_l: int, facet_offset_m: int
    ):
        """Do degrid correction to enable degridding from the FT of the image.

        :param facet: ``complex[facet_size,facet_size]`` Facet.
        :param facet_offset_l, facet_offset_m:
            Offset of facet centre relative to image centre
        """
        Lib.sdp_gridder_direct_degrid_correct(
            self, Mem(facet), facet_offset_l, facet_offset_m
        )

    def grid(
        self,
        vis: numpy.ndarray,
        uvw: numpy.ndarray,
        start_chs: numpy.ndarray,
        end_chs: numpy.ndarray,
        freq0_hz: float,
        dfreq_hz: float,
        subgrid_image: numpy.ndarray,
        subgrid_offset_u: int,
        subgrid_offset_v: int,
    ):
        """Grid visibilities using direct Fourier transformation

        This is painfully slow, but as good as we can make it by definition

        The caller must ensure the output subgrid_image is sized correctly.

        :param vis: ``complex[uvw_count, ch_count]`` Input visibilities
        :param uvw: ``float[uvw_count, 3]``
            UVW coordinates of visibilities (in m)
        :param start_chs: ``int[uvw_count]``
            First channel to grid for every uvw
        :param end_chs: ``int[uvw_count]``
            Channel at which to stop gridding for every uvw
        :param freq0_hz: Frequency of first channel (Hz)
        :param dfreq_hz: Channel width (Hz)
        :param subgrid_image: Fourier transformed subgrid to be gridded to.
            Note that the subgrid could especially span the entire grid,
            in which case this could simply be the entire (corrected) image.
        :param subgrid_offset_u, subgrid_offset_v:
            Offset of subgrid relative to grid centre
        """
        Lib.sdp_gridder_direct_grid(
            self,
            Mem(vis),
            Mem(uvw),
            Mem(start_chs),
            Mem(end_chs),
            freq0_hz,
            dfreq_hz,
            Mem(subgrid_image),
            subgrid_offset_u,
            subgrid_offset_v,
        )

    def grid_correct(
        self, facet: numpy.ndarray, facet_offset_l: int, facet_offset_m: int
    ):
        """Do grid correction after gridding.

        :param facet: ``complex[facet_size,facet_size]`` Facet.
        :param facet_offset_l, facet_offset_m:
            Offset of facet centre relative to image centre
        """
        Lib.sdp_gridder_direct_grid_correct(
            self, Mem(facet), facet_offset_l, facet_offset_m
        )


Lib.wrap_func(
    "sdp_gridder_direct_create",
    restype=GridderDirect.handle_type(),
    argtypes=[
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_int,
    ],
    check_errcode=True,
)

Lib.wrap_func(
    "sdp_gridder_direct_degrid",
    restype=None,
    argtypes=[
        GridderDirect.handle_type(),
        Mem.handle_type(),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_double,
        Mem.handle_type(),
        Mem.handle_type(),
        Mem.handle_type(),
        Mem.handle_type(),
    ],
    check_errcode=True,
)

Lib.wrap_func(
    "sdp_gridder_direct_degrid_correct",
    restype=None,
    argtypes=[
        GridderDirect.handle_type(),
        Mem.handle_type(),
        ctypes.c_int,
        ctypes.c_int,
    ],
    check_errcode=True,
)

Lib.wrap_func(
    "sdp_gridder_direct_grid",
    restype=None,
    argtypes=[
        GridderDirect.handle_type(),
        Mem.handle_type(),
        Mem.handle_type(),
        Mem.handle_type(),
        Mem.handle_type(),
        ctypes.c_double,
        ctypes.c_double,
        Mem.handle_type(),
        ctypes.c_int,
        ctypes.c_int,
    ],
    check_errcode=True,
)

Lib.wrap_func(
    "sdp_gridder_direct_grid_correct",
    restype=None,
    argtypes=[
        GridderDirect.handle_type(),
        Mem.handle_type(),
        ctypes.c_int,
        ctypes.c_int,
    ],
    check_errcode=True,
)

Lib.wrap_func(
    "sdp_gridder_direct_free",
    restype=None,
    argtypes=[GridderDirect.handle_type()],
)
