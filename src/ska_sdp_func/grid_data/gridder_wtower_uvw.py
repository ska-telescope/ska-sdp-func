# See the LICENSE file at the top-level directory of this distribution.
"""Module for (de)gridding functions using w-towers."""

import ctypes

import numpy

from ..utility import Lib, Mem, StructWrapper


class GridderWtowerUVW(StructWrapper):
    """Uses w-towers / w-stacking for uvw (de)gridding.

    These functions are ported from the implementation provided
    in Peter Wortmann's subgrid_imaging Jupyter notebook.
    """

    def __init__(
        self,
        image_size: int,
        subgrid_size: int,
        theta: float,
        w_step: float,
        shear_u: float,
        shear_v: float,
        support: int,
        oversampling: int,
        w_support: int,
        w_oversampling: int,
    ):
        """Create plan for w-towers (de)gridder.

        :param image_size: Total image size in pixels.
        :param subgrid_size: Sub-grid size in pixels.
        :param theta: Total image size in direction cosines.
        :param w_step: Spacing between w-planes.
        :param shear_u: Shear parameter in u (use zero for no shear).
        :param shear_v: Shear parameter in v (use zero for no shear).
        :param support: Kernel support size in (u, v).
        :param oversampling: Oversampling factor for uv-kernel.
        :param w_support: Support size in w.
        :param w_oversampling: Oversampling factor for w-kernel.
        """
        create_args = (
            image_size,
            subgrid_size,
            theta,
            w_step,
            shear_u,
            shear_v,
            support,
            oversampling,
            w_support,
            w_oversampling,
        )
        super().__init__(
            Lib.sdp_gridder_wtower_uvw_create,
            create_args,
            Lib.sdp_gridder_wtower_uvw_free,
        )

    def degrid(
        self,
        subgrid_image: numpy.ndarray,
        subgrid_offset_u: int,
        subgrid_offset_v: int,
        subgrid_offset_w: int,
        freq0_hz: float,
        dfreq_hz: float,
        uvws: numpy.ndarray,
        start_chs: numpy.ndarray,
        end_chs: numpy.ndarray,
        vis: numpy.ndarray,
    ):
        """Degrid visibilities using w-stacking/towers.

        The caller must ensure the output visibility array is sized correctly.

        :param subgrid_image: Fourier transformed subgrid to degrid from.
            Note that the subgrid could especially span the entire grid,
            in which case this could simply be the entire (corrected) image.
        :param subgrid_offset_u:
            Offset of subgrid centre relative to grid centre.
        :param subgrid_offset_v:
            Offset of subgrid centre relative to grid centre.
        :param subgrid_offset_w:
            Offset of subgrid centre relative to grid centre.
        :param freq0_hz: Frequency of first channel (Hz)
        :param dfreq_hz: Channel width (Hz)
        :param uvws: ``float[uvw_count, 3]``
            UVW coordinates of visibilities (in m)
        :param start_chs: ``int[uvw_count]``
            First channel to degrid for every uvw
        :param end_chs: ``int[uvw_count]``
            Channel at which to stop degridding for every uvw
        :param vis: ``complex[uvw_count, ch_count]`` Output visibilities
        """
        Lib.sdp_gridder_wtower_uvw_degrid(
            self,
            Mem(subgrid_image),
            subgrid_offset_u,
            subgrid_offset_v,
            subgrid_offset_w,
            freq0_hz,
            dfreq_hz,
            Mem(uvws),
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
        Lib.sdp_gridder_wtower_uvw_degrid_correct(
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
        subgrid_offset_w: int,
    ):
        """Grid visibilities using w-stacking/towers.

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
        :param subgrid_offset_u:
            Offset of subgrid centre relative to grid centre.
        :param subgrid_offset_v:
            Offset of subgrid centre relative to grid centre.
        :param subgrid_offset_w:
            Offset of subgrid centre relative to grid centre.
        """
        Lib.sdp_gridder_wtower_uvw_grid(
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
            subgrid_offset_w,
        )

    def grid_correct(
        self, facet: numpy.ndarray, facet_offset_l: int, facet_offset_m: int
    ):
        """Do grid correction after gridding.

        :param facet: ``complex[facet_size,facet_size]`` Facet.
        :param facet_offset_l, facet_offset_m:
            Offset of facet centre relative to image centre
        """
        Lib.sdp_gridder_wtower_uvw_grid_correct(
            self, Mem(facet), facet_offset_l, facet_offset_m
        )


Lib.wrap_func(
    "sdp_gridder_wtower_uvw_create",
    restype=GridderWtowerUVW.handle_type(),
    argtypes=[
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ],
    check_errcode=True,
)

Lib.wrap_func(
    "sdp_gridder_wtower_uvw_degrid",
    restype=None,
    argtypes=[
        GridderWtowerUVW.handle_type(),
        Mem.handle_type(),
        ctypes.c_int,
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
    "sdp_gridder_wtower_uvw_degrid_correct",
    restype=None,
    argtypes=[
        GridderWtowerUVW.handle_type(),
        Mem.handle_type(),
        ctypes.c_int,
        ctypes.c_int,
    ],
    check_errcode=True,
)

Lib.wrap_func(
    "sdp_gridder_wtower_uvw_grid",
    restype=None,
    argtypes=[
        GridderWtowerUVW.handle_type(),
        Mem.handle_type(),
        Mem.handle_type(),
        Mem.handle_type(),
        Mem.handle_type(),
        ctypes.c_double,
        ctypes.c_double,
        Mem.handle_type(),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ],
    check_errcode=True,
)

Lib.wrap_func(
    "sdp_gridder_wtower_uvw_grid_correct",
    restype=None,
    argtypes=[
        GridderWtowerUVW.handle_type(),
        Mem.handle_type(),
        ctypes.c_int,
        ctypes.c_int,
    ],
    check_errcode=True,
)

Lib.wrap_func(
    "sdp_gridder_wtower_uvw_free",
    restype=None,
    argtypes=[GridderWtowerUVW.handle_type()],
)
