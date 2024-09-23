# See the LICENSE file at the top-level directory of this distribution.

"""Test (de)gridding functions for subgrid gridder and degridder."""

import numpy
import scipy

from ska_sdp_func.grid_data import GridderDirect

C_0 = 299792458.0


def dft(flmn, uvws):
    return numpy.array(
        [
            numpy.sum(
                flmn[:, 0]
                * numpy.exp((-2.0j * numpy.pi) * numpy.dot(flmn[:, 1:], uvw.T))
            )
            for uvw in uvws
        ]
    )


def idft(vis, uvws, lmns):
    return numpy.array(
        [
            numpy.sum(
                vis * numpy.exp((2.0j * numpy.pi) * numpy.dot(lmn, uvws.T))
            )
            for lmn in lmns
        ]
    )


def lm_to_n(l, m, h_u, h_v):  # noqa: E741
    """
    Find location on sky sphere

    Incoming coordinates are assumed to already be transformed
    :param l, m: Horizontal / vertical sky coordinates
    :param h_u, h_v: Horizontal / vertical shear factors
    :returns: n, the coordinate towards the phase centre
    """

    # Easy case
    if h_u == 0 and h_v == 0:
        return numpy.sqrt(1 - l * l - m * m) - 1

    # Sheared case
    hul_hvm_1 = h_u * l + h_v * m - 1  # = -1 with h_u=h_v=0
    hu2_hv2_1 = h_u * h_u + h_v * h_v + 1  # = 1 with h_u=h_v=0
    return (
        numpy.sqrt(hul_hvm_1 * hul_hvm_1 - hu2_hv2_1 * (l * l + m * m))
        + hul_hvm_1
    ) / hu2_hv2_1


def image_to_flmn(image, theta, h_u, h_v):
    """
    Convert image into list of sources

    :param image: Image, assumed to be at centre of sky sphere
    :param theta:
        Size of image in (l,m) coordinate system (i.e. directional cosines)
    :param h_u, h_v: Horizontal / vertical shear factors
    :returns: List of (flux, l, m, n) tuples
    """
    image_size_l = image.shape[0]
    image_size_m = image.shape[1]
    ils, ims = numpy.where(image != 0)
    ls = (ils - image_size_l // 2) * (theta / image_size_l)
    ms = (ims - image_size_m // 2) * (theta / image_size_m)
    return numpy.transpose(
        [image[ils, ims], ls, ms, lm_to_n(ls, ms, h_u, h_v)]
    )


def shift_uvw(uvw, offsets, theta, w_step=0):
    return uvw - numpy.array(offsets) * [1 / theta, 1 / theta, w_step]


def make_pswf(support, size):
    pswf = scipy.special.pro_ang1(
        0,
        0,
        numpy.pi * support / 2,
        numpy.arange(-size // 2, size // 2) / size * 2,
    )[0]
    if size % 2 == 0:
        pswf[0] = 1e-15
    return pswf


class DFTGridKernel:
    """
    Uses discrete Fourier transformation and PSWF for subgrid (de)gridding

    Very inefficient, but as accurate as one can be
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
    ):

        # Image / subgrid setup. We assume that some gridding kernels might
        # need to know the subgrid size in advance (e.g. to prepare cached
        # w-kernels)
        self.image_size = image_size
        self.subgrid_size = subgrid_size
        self.theta = theta
        self.w_step = w_step  # not currently used
        self.shear_u = shear_u
        self.shear_v = shear_v
        self.support = support

        # Processing function plan / common parameters
        self.pswf = make_pswf(support, image_size)
        self.pswf_sg = make_pswf(support, subgrid_size)

    def degrid_correct(
        self,
        facet: numpy.ndarray,
        facet_offset_l: int,
        facet_offset_m: int,
        w_offset: int = 0,
    ):
        """
        Do degrid correction to enable degridding from the FT of the image

        :param facet: ``complex[facet_size,facet_size]`` Input facet
        :param facet_offset_l, facet_offset_m:
            Offset of facet centre relative to image centre
        :returns: Corrected image facet
        """

        # Determine PSWF portions that apply to facet
        pswf_l = numpy.roll(self.pswf, -facet_offset_l)
        pswf_l = pswf_l[
            self.image_size // 2
            - facet.shape[0] // 2 : self.image_size // 2
            + facet.shape[0] // 2
        ]
        pswf_m = numpy.roll(self.pswf, -facet_offset_m)
        pswf_m = pswf_m[
            self.image_size // 2
            - facet.shape[1] // 2 : self.image_size // 2
            + facet.shape[1] // 2
        ]

        # Apply
        return facet / pswf_l[:, numpy.newaxis] / pswf_m[numpy.newaxis, :]

    def degrid_subgrid(
        self,
        subgrid_image: numpy.ndarray,
        subgrid_offsets: tuple[int, int, int],
        ch_count: int,
        freq0: float,
        dfreq: float,
        uvws: numpy.ndarray,
        start_chs: numpy.ndarray,
        end_chs: numpy.ndarray,
    ):
        """
        Degrid visibilities using direct Fourier transformation

        This is painfully slow, but as good as we can make it by definition

        :param subgrid_image: Fourier transformed subgrid to degrid from.
            Note that the subgrid could especially span the entire grid,
            in which case this could simply be the entire (corrected) image.
        :param subgrid_offsets:
            Offset of subgrid centre relative to grid centre,
            in pixels & wplanes
        :param ch_count: Channel count (determines size of array returned)
        :param freq0: Frequency of first channel (Hz)
        :param dfreq: Channel width (Hz)
        :param uvws: ``float[uvw_count, 3]``
            UVW coordinates of vibilities (in m)
        :param start_chs: ``int[uvw_count]``
            First channel to degrid for every uvw
        :param end_chs: ``int[uvw_count]``
            Channel at which to stop degridding for every uvw
        :returns: ``complex[uvw_count, ch_count]`` Visibilities
        """

        # Convert image into positions
        subgrid_image_uncorrected = (
            subgrid_image
            * self.pswf_sg[:, numpy.newaxis]
            * self.pswf_sg[numpy.newaxis, :]
        )
        image_flmns = image_to_flmn(
            subgrid_image_uncorrected, self.theta, self.shear_u, self.shear_v
        )

        # Create array to return
        uvw_count = uvws.shape[0]
        vis_out = numpy.zeros((uvw_count, ch_count), dtype=complex)
        for i, (uvw, start_ch, end_ch) in enumerate(
            zip(uvws, start_chs, end_chs)
        ):

            # Skip if there's no visibility to degrid
            if start_ch >= end_ch:
                continue
            assert start_ch >= 0
            assert end_ch <= ch_count

            # Scale + shift UVWs
            uvw_scaled = numpy.vstack(
                [uvw * ((freq0 + dfreq * ch) / C_0) for ch in range(ch_count)]
            )
            uwv_shifted = shift_uvw(
                uvw_scaled, subgrid_offsets, self.theta, self.w_step
            )

            # Degrid visibilities
            vis_out[i, start_ch:end_ch] = dft(
                image_flmns, uwv_shifted[start_ch:end_ch]
            )

        return vis_out

    def grid_correct(
        self,
        facet: numpy.ndarray,
        facet_offset_l: int,
        facet_offset_m: int,
        w_offset: int = 0,
    ):
        """
        Do grid correction after gridding

        :param facet: ``complex[facet_size,facet_size]``
            Facet data resulting from gridding
        :param facet_offset_l, facet_offset_m:
            Offset of facet centre relative to image centre
        :returns: Corrected image facet
        """

        # Determine PSWF portions that apply to facet
        pswf_l = numpy.roll(self.pswf, -facet_offset_l)
        pswf_l = pswf_l[
            self.image_size // 2
            - facet.shape[0] // 2 : self.image_size // 2
            + facet.shape[0] // 2
        ]
        pswf_m = numpy.roll(self.pswf, -facet_offset_m)
        pswf_m = pswf_m[
            self.image_size // 2
            - facet.shape[1] // 2 : self.image_size // 2
            + facet.shape[1] // 2
        ]

        # Apply
        return facet / pswf_l[:, numpy.newaxis] / pswf_m[numpy.newaxis, :]

    def grid_subgrid(
        self,
        vis: numpy.ndarray,
        uvws: numpy.ndarray,
        start_chs: numpy.ndarray,
        end_chs: numpy.ndarray,
        ch_count: int,
        freq0: float,
        dfreq: float,
        subgrid_image: numpy.ndarray,
        subgrid_offsets: tuple[int, int, int],
    ):
        """
        Grid visibilities using direct Fourier transformation

        This is painfully slow, but as good as we can make it by definition

        :param vis: ``complex[uvw_count, ch_count]`` Input visibilities
        :param uvws: ``float[uvw_count, 3]``
            UVW coordinates of vibilities (in m)
        :param start_chs: ``int[uvw_count]``
            First channel to degrid for every uvw
        :param end_chs: ``int[uvw_count]``
            Channel at which to stop degridding for every uvw
        :param ch_count: Channel count (determines size of array returned)
        :param freq0: Frequency of first channel (Hz)
        :param dfreq: Channel width (Hz)
        :param subgrid_image: Fourier transformed subgrid to be gridded to.
            Note that the subgrid could especially span the entire grid,
            in which case this could simply be the entire (corrected) image.
        :param subgrid_offset_u, subgrid_offset_v:
            Offset of subgrid relative to grid centre
        """

        # Generate lmns for subgrid image
        subgrid_image_lmns = image_to_flmn(
            numpy.ones_like(subgrid_image),
            self.theta,
            self.shear_u,
            self.shear_v,
        )[:, 1:]

        # Create array to return
        fluxes = numpy.zeros(subgrid_image.size, dtype=complex)
        for i, (uvw, start_ch, end_ch) in enumerate(
            zip(uvws, start_chs, end_chs)
        ):

            # Skip if there's no visibility to grid
            if start_ch >= end_ch:
                continue
            assert start_ch >= 0
            assert end_ch <= ch_count

            # Scale + shift UVWs
            uvw_scaled = numpy.vstack(
                [uvw * ((freq0 + dfreq * ch) / C_0) for ch in range(ch_count)]
            )
            uwv_shifted = shift_uvw(
                uvw_scaled, subgrid_offsets, self.theta, self.w_step
            )

            # Grid visibilities
            fluxes += idft(
                vis[i, start_ch:end_ch],
                uwv_shifted[start_ch:end_ch],
                subgrid_image_lmns,
            )

        # Reshape, convolve
        subgrid_image += (
            fluxes.reshape(subgrid_image.shape)
            * self.pswf_sg[:, numpy.newaxis]
            * self.pswf_sg[numpy.newaxis, :]
        )


def test_gridder_direct():
    # Common parameters
    image_size = 128  # Total image size in pixels
    theta = 0.1  # Total image size in directional cosines.
    w_step = 100.5
    shear_u = 0.1
    shear_v = -0.4
    support = 10
    print("Grid size: ", image_size / theta, "wavelengths")
    subgrid_size = image_size // 4
    idu = 90
    idv = 90
    idw = 50
    ch_count = 100
    freq0_hz = 1e6
    dfreq_hz = 1e3

    # Create an image for input to degridding.
    image = numpy.zeros((subgrid_size, subgrid_size))
    image[subgrid_size // 4, subgrid_size // 4] = 1.0
    image[5 * subgrid_size // 6, 2 * subgrid_size // 6] = 0.5
    num_uvw = 300
    uvw = numpy.random.random_sample((num_uvw, 3)) * 100
    start_chs = numpy.zeros((num_uvw), dtype=numpy.int32)
    end_chs = numpy.ones((num_uvw), dtype=numpy.int32) * (ch_count)

    # Generate reference data set.
    gridder_ref = DFTGridKernel(
        image_size, subgrid_size, theta, w_step, shear_u, shear_v, support
    )
    vis_ref = gridder_ref.degrid_subgrid(
        image,
        (idu, idv, idw),
        ch_count,
        freq0_hz,
        dfreq_hz,
        uvw,
        start_chs,
        end_chs,
    )

    # Call the degridder in PFL.
    gridder = GridderDirect(
        image_size, subgrid_size, theta, w_step, shear_u, shear_v, support
    )
    vis = gridder.degrid_subgrid(
        image,
        (idu, idv, idw),
        ch_count,
        freq0_hz,
        dfreq_hz,
        uvw,
        start_chs,
        end_chs,
    )

    # Check they are the same.
    numpy.testing.assert_allclose(vis, vis_ref)

    # Generate reference subgrid (image, really).
    img_ref = numpy.zeros((subgrid_size, subgrid_size), dtype=complex)
    gridder_ref.grid_subgrid(
        vis_ref,
        uvw,
        start_chs,
        end_chs,
        ch_count,
        freq0_hz,
        dfreq_hz,
        img_ref,
        (idu, idv, idw),
    )

    # Call the gridder in PFL.
    img_tst = numpy.zeros_like(img_ref)
    gridder.grid_subgrid(
        vis_ref,
        uvw,
        start_chs,
        end_chs,
        ch_count,
        freq0_hz,
        dfreq_hz,
        img_tst,
        (idu, idv, idw),
    )

    # Check they are the same.
    numpy.testing.assert_allclose(img_tst, img_ref)


def test_gridder_direct_degrid_correct():
    image_size = 128  # Total image size in pixels
    theta = 0.1  # Total image size in directional cosines.
    w_step = 100.5
    shear_u = 0.1
    shear_v = -0.4
    support = 10
    subgrid_size = image_size // 4

    # Create a test image.
    image = numpy.random.random_sample((subgrid_size, subgrid_size))
    facet_offset_l = 5
    facet_offset_m = 15

    # Generate reference data.
    gridder_ref = DFTGridKernel(
        image_size, subgrid_size, theta, w_step, shear_u, shear_v, support
    )
    img_ref = gridder_ref.degrid_correct(image, facet_offset_l, facet_offset_m)

    # Call the degrid correction function in PFL.
    gridder = GridderDirect(
        image_size, subgrid_size, theta, w_step, shear_u, shear_v, support
    )
    img_tst = numpy.array(image)
    gridder.degrid_correct(img_tst, facet_offset_l, facet_offset_m)

    # Check they are the same.
    numpy.testing.assert_allclose(img_tst, img_ref)
