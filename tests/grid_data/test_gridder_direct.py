# See the LICENSE file at the top-level directory of this distribution.

"""Test degridding functions."""

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


def image_to_flmn(image, theta):
    """
    Convert image into list of sources

    :param image: Image, assumed to be at centre of sky sphere
    :param theta:
        Size of image in (l,m) coordinate system (i.e. directional cosines)
    :returns: List of (flux, l, m, n) tuples
    """
    result = []
    image_size = image.shape[0]
    for il, im in zip(*numpy.where(image != 0)):
        dir_l = (il - image_size // 2) * theta / image_size
        dir_m = (im - image_size // 2) * theta / image_size
        dir_n = numpy.sqrt(1 - dir_l * dir_l - dir_m * dir_m) - 1
        assert image[il, im] != 0
        result.append((image[il, im], dir_l, dir_m, dir_n))
    return numpy.array(result)


def shift_uvw(uvw, idu, idv, theta):
    return uvw - [idu / theta, idv / theta, 0]


class DFTGridKernel:
    """
    Uses discrete Fourier transformation and PSWF for subgrid (de)gridding

    Very inefficient, but as accurate as one can be
    """

    def __init__(
        self, image_size: int, subgrid_size: int, theta: float, support: int
    ):

        # Image / subgrid setup. We assume that some gridding kernels might
        # need to know the subgrid size in advance (e.g. to prepare cached
        # w-kernels)
        self.image_size = image_size
        self.subgrid_size = subgrid_size
        self.theta = theta

        # Processing function plan / common parameters
        image_coords = (
            numpy.arange(-image_size // 2, image_size // 2) / image_size * 2
        )
        self.pswf = scipy.special.pro_ang1(
            0, 0, numpy.pi * support / 2, image_coords
        )[0]
        self.pswf[0] = 1e-15
        subgrid_image_coords = (
            numpy.arange(-subgrid_size // 2, subgrid_size // 2)
            / subgrid_size
            * 2
        )
        self.pswf_sg = scipy.special.pro_ang1(
            0, 0, numpy.pi * support / 2, subgrid_image_coords
        )[0]
        self.pswf_sg[0] = 1e-15

    def degrid_correct(
        self, facet: numpy.ndarray, facet_offset_l: int, facet_offset_m: int
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
            - (facet.shape[0] // 2) : (self.image_size // 2)
            + facet.shape[0] // 2
        ]
        pswf_m = numpy.roll(self.pswf, -facet_offset_m)
        pswf_m = pswf_m[
            self.image_size // 2
            - (facet.shape[1] // 2) : (self.image_size // 2)
            + facet.shape[1] // 2
        ]

        # Apply
        return facet / pswf_l[:, numpy.newaxis] / pswf_m[numpy.newaxis, :]

    def degrid_subgrid(
        self,
        subgrid_image: numpy.ndarray,
        subgrid_offset_u: int,
        subgrid_offset_v: int,
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
        :param subgrid_offset_u, subgrid_offset_v:
          Offset of subgrid centre relative to grid centre
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
        image_flmns = image_to_flmn(subgrid_image_uncorrected, self.theta)

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
                uvw_scaled, subgrid_offset_u, subgrid_offset_v, self.theta
            )

            # Degrid visibilities
            vis_out[i, start_ch:end_ch] = dft(
                image_flmns, uwv_shifted[start_ch:end_ch]
            )

        return vis_out


def test_gridder_direct():
    # Common parameters
    image_size = 512  # Total image size in pixels
    theta = 0.1  # Total image size in directional cosines.
    support = 10
    print("Grid size: ", image_size / theta, "wavelengths")
    subgrid_size = image_size // 4
    idu = 90
    idv = 90
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
    gridder_ref = DFTGridKernel(image_size, subgrid_size, theta, support)
    vis_ref = gridder_ref.degrid_subgrid(
        image, idu, idv, ch_count, freq0_hz, dfreq_hz, uvw, start_chs, end_chs
    )

    # Call the degridder in PFL.
    vis = numpy.zeros_like(vis_ref)
    gridder = GridderDirect(image_size, subgrid_size, theta, support)
    gridder.degrid(
        image, idu, idv, freq0_hz, dfreq_hz, uvw, start_chs, end_chs, vis
    )

    # Check they are the same.
    numpy.testing.assert_allclose(vis, vis_ref)
