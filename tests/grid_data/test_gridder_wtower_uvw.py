# See the LICENSE file at the top-level directory of this distribution.

"""Test w-towers (de)gridding functions for subgrid gridder and degridder."""

import math
import time

import numpy
import scipy

try:
    import cupy
except ImportError:
    cupy = None

import ska_sdp_func.grid_data as sdp_grid_func

# from ska_sdp_func.utility import Lib, Mem

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


def fft(a):
    if len(a.shape) == 2:
        return numpy.fft.fftshift(numpy.fft.fft2(numpy.fft.ifftshift(a)))
    else:
        return numpy.fft.fftshift(numpy.fft.fft(numpy.fft.ifftshift(a)))


def ifft(a):
    if len(a.shape) == 2:
        return numpy.fft.fftshift(numpy.fft.ifft2(numpy.fft.ifftshift(a)))
    else:
        return numpy.fft.fftshift(numpy.fft.ifft(numpy.fft.ifftshift(a)))


def lm_to_n(d_l, d_m, h_u, h_v):
    """
    Find location on sky sphere

    Incoming coordinates are assumed to already be transformed
    :param d_l, d_m: Horizontal / vertical sky coordinates
    :param h_u, h_v: Horizontal / vertical shear factors
    :returns: d_n, the coordinate towards the phase centre
    """

    # Easy case
    if h_u == 0 and h_v == 0:
        return numpy.sqrt(1 - d_l * d_l - d_m * d_m) - 1

    # Sheared case
    hul_hvm_1 = h_u * d_l + h_v * d_m - 1  # = -1 with h_u=h_v=0
    hu2_hv2_1 = h_u * h_u + h_v * h_v + 1  # = 1 with h_u=h_v=0
    return (
        numpy.sqrt(hul_hvm_1 * hul_hvm_1 - hu2_hv2_1 * (d_l * d_l + d_m * d_m))
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
    result = []
    image_size_l = image.shape[0]
    image_size_m = image.shape[1]
    for il, im in zip(*numpy.where(image != 0)):
        d_l = (il - image_size_l // 2) * theta / image_size_l
        d_m = (im - image_size_m // 2) * theta / image_size_m
        assert image[il, im] != 0
        result.append((image[il, im], d_l, d_m, lm_to_n(d_l, d_m, h_u, h_v)))
    return numpy.array(result)


def shift_uvw(uvw, offsets, theta, w_step=0):
    return uvw - numpy.array(offsets) * [1 / theta, 1 / theta, w_step]


def clamp_channels(uvw, freq0, dfreq, start_ch, end_ch, min_uvw, max_uvw):
    """
    Clamp channels for a particular uvw position

    Restricts a channel range such that all visibilities lie in the given
    uvw bounding box. Adapted from:
    https://gitlab.com/ska-telescope/sdp/ska-sdp-exec-iotest/-/blob/proc-func-refactor/src/grid.c?ref_type=heads#L464

    :param uvw: UVW position (in meters)
    :param freq0: Frequency of first channel
    :param dfreq: Channel width
    :param start_ch, end_ch: Channel range to clamp (excluding end!)
    :param min_uvw: Minimum values for u,v,w (inclusive)
    :param max_uvw: Maximum values for u,v,w (exclusive)
    :returns: Clamped (start_ch, end_ch) or (0,0) if no channels overlap
    """

    # We have to be slightly careful about degenerate cases in the
    # division below - not only can we have divisions by zero,
    # but also channel numbers that go over the integer range.
    # So it is safer to round these coordinates to zero for the
    # purpose of the bounds check.
    eta = 1e-3
    for _u, _min, _max in zip(uvw, min_uvw, max_uvw):
        u0 = freq0 * _u / C_0
        du = dfreq * _u / C_0
        # Note the symmetry below: we get precisely the same expression
        # for maximum and minimum, however start_ch is inclusive but
        # end_ch is exclusive. This means that two calls to
        # clamp_channels where any min_uvw is equal to any max_uvw will
        # never return overlapping channel ranges.
        try:
            if _u > eta:
                start_ch = max(start_ch, int(math.ceil((_min - u0) / du)))
                end_ch = min(end_ch, int(math.ceil((_max - u0) / du)))
            elif _u < -eta:
                start_ch = max(start_ch, int(math.ceil((_max - u0) / du)))
                end_ch = min(end_ch, int(math.ceil((_min - u0) / du)))
            else:
                # Assume _u = 0, which makes this a binary decision:
                # Does the range include 0 or not? Also let's be careful
                # just in case somebody puts a subgrid boundary right at zero.
                if _min > 0 or _max <= 0:
                    return (0, 0)
        except OverflowError:
            print(_u, _max, _min, du)
            raise

    if end_ch <= start_ch:
        return (0, 0)
    return (start_ch, end_ch)


def uvw_bounds(uvw, freq0, dfreq, start_ch, end_ch):
    """
    Determine UVW bound

    "Inverse" to clamp_channels, which would return start_ch/end_ch such
    that uvw_bounds are strictly smaller than provided

    :param uvw: UVW position (in meters)
    :param freq0: Frequency of first channel
    :param dfreq: Channel width
    :param start_ch, end_ch: Channel range
    :returns: (min_uvw, max_uvw) tuple
    """

    # Meaninless if all channels are "masked"
    if start_ch >= end_ch:
        return (
            [math.inf, math.inf, math.inf],
            [-math.inf, -math.inf, -math.inf],
        )

    # Go through u/v/w coordinates sequentially (slow!)
    uvw_min = []
    uvw_max = []
    for _u in uvw:
        u0 = freq0 * _u / C_0
        du = dfreq * _u / C_0
        if _u >= 0:
            uvw_min.append(u0 + start_ch * du)
            uvw_max.append(u0 + (end_ch - 1) * du)
        else:
            uvw_max.append(u0 + start_ch * du)
            uvw_min.append(u0 + (end_ch - 1) * du)

    return (uvw_min, uvw_max)


def uvw_bounds_all(uvws, freq0, dfreq, start_chs, end_chs):

    uvw_min = numpy.array([math.inf, math.inf, math.inf])
    uvw_max = numpy.array([-math.inf, -math.inf, -math.inf])

    for uvw, start_ch, end_ch in zip(uvws, start_chs, end_chs):
        if start_ch >= end_ch:
            continue
        uvw_min2, uvw_max2 = uvw_bounds(uvw, freq0, dfreq, start_ch, end_ch)
        uvw_min = numpy.minimum(uvw_min, uvw_min2)
        uvw_max = numpy.maximum(uvw_max, uvw_max2)

    return (uvw_min, uvw_max)


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


def make_pswf_n(w_support, size, theta, w_step, h_u, h_v):
    ns = image_to_flmn(numpy.ones((size, size)), theta, h_u, h_v)[:, 3]
    pswf_n = scipy.special.pro_ang1(
        0, 0, numpy.pi * w_support / 2, ns * w_step * 2
    )[0].reshape(size, size)
    pswf_n[numpy.isnan(pswf_n)] = 1
    return pswf_n


def make_kernel(window, support, oversampling):
    """
    Convert image-space window function to oversampled kernel.

    The painfully inefficient way. This could be done much better using FFT.
    """

    # We don't actually use n (w=0), so we can use theta=1 and h_u=h_v=0
    flmns = image_to_flmn(window[:, numpy.newaxis], 1, 0, 0)
    vr_us = numpy.array(
        [(u, 0, 0) for u in numpy.arange(-support // 2, support // 2, 1)]
    )
    return [
        dft(flmns, vr_us + [-du / oversampling, 0, 0]).real.reshape(support)
        / support
        for du in range(-oversampling, 1)
    ]


def make_pswf_kernel(support, vr_size, oversampling):
    return make_kernel(make_pswf(support, vr_size), vr_size, oversampling)


def make_w_pattern(subgrid_size, theta, shear_u, shear_v, w_step):
    subgrid_pattern = numpy.ones((subgrid_size, subgrid_size), dtype=complex)
    subgrid_image_flmns = image_to_flmn(
        subgrid_pattern, theta, shear_u, shear_v
    )
    w_pattern = idft(
        numpy.array([1]),
        numpy.array([[0, 0, w_step]]),
        subgrid_image_flmns[:, 1:],
    )
    return w_pattern.reshape(subgrid_size, subgrid_size)


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


class LocalGridKernel(DFTGridKernel):
    """
    Grids from local grid regions instead of the entire subgrid

    Somewhat more efficient than full DFT, but still needs to do
    a local FFT + DFT per visibility.
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
        vr_size: int,
    ):
        super().__init__(
            image_size, subgrid_size, theta, w_step, shear_u, shear_v, support
        )
        self.vr_size = vr_size


class WtowerGridKernel(LocalGridKernel):
    """
    Uses w-towers / w-stacking for gridding
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
        vr_size: int,
    ):

        super().__init__(
            image_size,
            subgrid_size,
            theta,
            w_step,
            shear_u,
            shear_v,
            support,
            vr_size,
        )
        self.w_step = w_step

        # Generate w-pattern. This is the iDFT of a sole visibility at (0,0,w).
        # Our plan is roughly to convolve in uvw space by a delta function
        # to move the grid in w.
        self.w_pattern = make_w_pattern(
            subgrid_size, theta, shear_u, shear_v, w_step
        )


class WtowerUVGridKernel(WtowerGridKernel):
    """
    Uses w-towers and separable UV kernel.
    Remaining w is applied using small-scale FFT.
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
        vr_size: int,
        oversampling: int,
    ):

        super().__init__(
            image_size,
            subgrid_size,
            theta,
            w_step,
            shear_u,
            shear_v,
            support,
            vr_size,
        )
        self.oversampling = oversampling

        # Generate convolution kernel
        self._uv_kernel = make_pswf_kernel(support, vr_size, oversampling)


class WtowerUVWGridKernel(WtowerUVGridKernel):
    """
    Uses w-towers / w-stacking for uvw gridding
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

        # We now hard-code the visibility region size to the (uv) support
        # size - no need to increase it any more for dealing with w
        super().__init__(
            image_size,
            subgrid_size,
            theta,
            w_step,
            shear_u,
            shear_v,
            support,
            support,
            oversampling,
        )
        self.w_oversampling = w_oversampling
        self.w_support = w_support

        # Generate oversampled w convolution kernel
        self._w_kernel = make_pswf_kernel(w_support, w_support, w_oversampling)

        # Generate PSWF window function on n-axis (projected on lm)
        self.pswf_n = make_pswf_n(
            w_support, image_size, theta, w_step, shear_u, shear_v
        )

    def degrid_correct(
        self,
        facet: numpy.ndarray,
        facet_offset_l: int,
        facet_offset_m: int,
    ):
        """
        Do degrid correction to enable degridding from the FT of the image

        :param facet: ``complex[facet_size,facet_size]`` Input facet
        :param facet_offset_l, facet_offset_m:
            Offset of facet centre relative to image centre
        :returns: Corrected image facet
        """

        # Determine PSWF portions that apply to facet
        left = self.image_size // 2 - facet.shape[0] // 2
        right = self.image_size // 2 + facet.shape[0] // 2
        pswf_l = numpy.roll(self.pswf, -facet_offset_l)
        pswf_l = pswf_l[left:right]
        pswf_m = numpy.roll(self.pswf, -facet_offset_m)
        pswf_m = pswf_m[left:right]
        pswf_n = numpy.roll(
            self.pswf_n, (-facet_offset_l, -facet_offset_m), axis=(0, 1)
        )
        pswf_n = pswf_n[left:right, left:right]

        # Apply
        return (
            facet
            / pswf_l[:, numpy.newaxis]
            / pswf_m[numpy.newaxis, :]
            / pswf_n
        )

    def grid_correct(
        self,
        facet: numpy.ndarray,
        facet_offset_l: int,
        facet_offset_m: int,
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
        left = self.image_size // 2 - facet.shape[0] // 2
        right = self.image_size // 2 + facet.shape[0] // 2
        pswf_l = numpy.roll(self.pswf, -facet_offset_l)
        pswf_l = pswf_l[left:right]
        pswf_m = numpy.roll(self.pswf, -facet_offset_m)
        pswf_m = pswf_m[left:right]
        pswf_n = numpy.roll(
            self.pswf_n, (-facet_offset_l, -facet_offset_m), axis=(0, 1)
        )
        pswf_n = pswf_n[left:right, left:right]

        # Apply
        return (
            facet
            / pswf_l[:, numpy.newaxis]
            / pswf_m[numpy.newaxis, :]
            / pswf_n
        )

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
        Degrid visibilities

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

        # Determine w-range
        uvw_min, uvw_max = uvw_bounds_all(
            uvws, freq0, dfreq, start_chs, end_chs
        )

        # Get subgrid at first w-plane
        eta = 1e-5
        first_w_plane = (
            int(numpy.floor(uvw_min[2] / self.w_step - eta))
            - subgrid_offsets[2]
        )
        last_w_plane = (
            int(numpy.floor(uvw_max[2] / self.w_step + eta))
            - subgrid_offsets[2]
            + 1
        )

        # First w-plane we need to generate is support/2 below the first one
        # with visibilities.
        w_subgrid_image = subgrid_image / self.w_pattern ** (
            first_w_plane - self.w_support // 2
        )
        subgrids = numpy.empty(
            (self.w_support, self.subgrid_size, self.subgrid_size),
            dtype=complex,
        )
        for i in range(self.w_support):
            subgrids[i] = fft(w_subgrid_image)
            w_subgrid_image /= self.w_pattern

        # Create array to return
        uvw_count = uvws.shape[0]
        vis_out = numpy.zeros((uvw_count, ch_count), dtype=complex)
        for w_plane in range(first_w_plane, last_w_plane + 1):

            # Move to next w-plane
            if w_plane != first_w_plane:
                # Shift subgrids, add new w-plane
                subgrids[:-1] = subgrids[1:]
                subgrids[-1] = fft(w_subgrid_image)
                w_subgrid_image /= self.w_pattern

            for i, (uvw, start_ch, end_ch) in enumerate(
                zip(uvws, start_chs, end_chs)
            ):

                # Skip if there's no visibility to degrid
                if start_ch >= end_ch:
                    continue
                assert start_ch >= 0
                assert end_ch <= ch_count

                # Select only visibilities on this w-plane. Add a bit of
                # margin in u/v to prevent single visibilities falling off.
                min_uvw = [
                    uvw_min[0] - 1,
                    uvw_min[1] - 1,
                    (w_plane + subgrid_offsets[2] - 1) * self.w_step,
                ]
                max_uvw = [
                    uvw_max[0] + 1,
                    uvw_max[1] + 1,
                    (w_plane + subgrid_offsets[2]) * self.w_step,
                ]
                start_ch, end_ch = clamp_channels(
                    uvw, freq0, dfreq, start_ch, end_ch, min_uvw, max_uvw
                )
                if start_ch >= end_ch:
                    continue

                # Scale + shift UVWs
                uvw_stretched = numpy.vstack(
                    [
                        uvw * ((freq0 + dfreq * ch) / C_0)
                        for ch in range(ch_count)
                    ]
                )
                uvw_shifted = shift_uvw(
                    uvw_stretched, subgrid_offsets, self.theta, self.w_step
                )
                uvw_shifted -= [0, 0, w_plane * self.w_step]

                self._degrid_vis_uvw(
                    vis_out[i][start_ch:end_ch],
                    uvw_shifted[start_ch:end_ch],
                    subgrids,
                )

        return vis_out

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
        Grid visibilities using w-stacking/towers.

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

        # Determine w-range
        uvw_min, uvw_max = uvw_bounds_all(
            uvws, freq0, dfreq, start_chs, end_chs
        )
        eta = 1e-5
        first_w_plane = (
            int(numpy.floor(uvw_min[2] / self.w_step - eta))
            - subgrid_offsets[2]
        )
        last_w_plane = (
            int(numpy.floor(uvw_max[2] / self.w_step + eta))
            - subgrid_offsets[2]
            + 1
        )

        # Create subgrid image and subgrids to accumulate on
        w_subgrid_image = numpy.zeros_like(subgrid_image)
        subgrids = numpy.zeros(
            (self.w_support, self.subgrid_size, self.subgrid_size),
            dtype=complex,
        )

        # Iterate over w-planes
        for w_plane in range(first_w_plane, last_w_plane + 1):

            # Move to next w-plane
            if w_plane != first_w_plane:

                # Accumulate zero-th subgrid, shift, clear upper subgrid
                w_subgrid_image /= self.w_pattern
                w_subgrid_image += ifft(subgrids[0])
                subgrids[:-1] = subgrids[1:]
                subgrids[-1] = 0

            # Make subgrid for gridding visibilities to
            for i, (uvw, start_ch, end_ch) in enumerate(
                zip(uvws, start_chs, end_chs)
            ):

                # Skip if there's no visibility to degrid
                if start_ch >= end_ch:
                    continue
                assert start_ch >= 0
                assert end_ch <= ch_count

                # Select only visibilities on this w-plane. Add a bit of
                # margin in u/v to prevent single visibilities falling off.
                min_uvw = [
                    uvw_min[0] - 1,
                    uvw_min[1] - 1,
                    (w_plane + subgrid_offsets[2] - 1) * self.w_step,
                ]
                max_uvw = [
                    uvw_max[0] + 1,
                    uvw_max[1] + 1,
                    (w_plane + subgrid_offsets[2]) * self.w_step,
                ]
                start_ch, end_ch = clamp_channels(
                    uvw, freq0, dfreq, start_ch, end_ch, min_uvw, max_uvw
                )
                if start_ch >= end_ch:
                    continue

                # Scale + shift UVWs
                uvw_scaled = numpy.vstack(
                    [
                        uvw * ((freq0 + dfreq * ch) / C_0)
                        for ch in range(ch_count)
                    ]
                )
                uvw_shifted = shift_uvw(
                    uvw_scaled, subgrid_offsets, self.theta, self.w_step
                )
                uvw_shifted -= [0, 0, w_plane * self.w_step]

                # Grid local visibilities
                self._grid_vis_uvw(
                    vis[i][start_ch:end_ch],
                    uvw_shifted[start_ch:end_ch],
                    subgrids,
                )

        # Accumulate remaining data from subgrids
        for i in range(self.w_support):
            w_subgrid_image /= self.w_pattern
            w_subgrid_image += ifft(subgrids[i])

        # Return updated subgrid image
        subgrid_image += (
            w_subgrid_image
            * self.w_pattern ** (last_w_plane + self.w_support // 2 - 1)
            * self.subgrid_size**2
        )

    def _degrid_vis_uvw(self, vis_out, uvw_shifted, subgrids):

        # Degrid visibilities
        for j, (u, v, w) in enumerate(uvw_shifted):

            # Determine top-left corner of grid region centered
            # approximately on visibility
            iu0 = (
                int(round(self.theta * u - (self.vr_size - 1) / 2))
                + self.subgrid_size // 2
            )
            iv0 = (
                int(round(self.theta * v - (self.vr_size - 1) / 2))
                + self.subgrid_size // 2
            )
            iu_shift = iu0 + self.vr_size // 2 - self.subgrid_size // 2
            iv_shift = iv0 + self.vr_size // 2 - self.subgrid_size // 2

            # Get grid region in image space, convolve
            vis_region = subgrids[
                :, iu0 : iu0 + self.vr_size, iv0 : iv0 + self.vr_size
            ]
            u_factor = self._uv_kernel[
                int(
                    numpy.round(
                        (u * self.theta - iu_shift + 1) * self.oversampling
                    )
                )
            ]
            v_factor = self._uv_kernel[
                int(
                    numpy.round(
                        (v * self.theta - iv_shift + 1) * self.oversampling
                    )
                )
            ]
            w_factor = self._w_kernel[
                int(numpy.round((w / self.w_step + 1) * self.w_oversampling))
            ]
            u_factor = u_factor[numpy.newaxis, :, numpy.newaxis]
            v_factor = v_factor[numpy.newaxis, numpy.newaxis, :]
            w_factor = w_factor[
                :, numpy.newaxis, numpy.newaxis
            ]  # w is outermost axis here!

            # Degrid visibility - without additional FFTs this time!
            vis_out[j] = numpy.sum(vis_region * u_factor * v_factor * w_factor)

    def _grid_vis_uvw(self, vis, uvw_shifted, subgrids):

        # Grid visibilities
        for j, (u, v, w) in enumerate(uvw_shifted):

            # Determine top-left corner of grid region centered
            # approximately on visibility
            iu0 = (
                int(round(self.theta * u - (self.vr_size - 1) / 2))
                + self.subgrid_size // 2
            )
            iv0 = (
                int(round(self.theta * v - (self.vr_size - 1) / 2))
                + self.subgrid_size // 2
            )
            iu_shift = iu0 + self.vr_size // 2 - self.subgrid_size // 2
            iv_shift = iv0 + self.vr_size // 2 - self.subgrid_size // 2

            # Determine convolution
            u_factor = self._uv_kernel[
                int(
                    numpy.round(
                        (u * self.theta - iu_shift + 1) * self.oversampling
                    )
                )
            ]
            v_factor = self._uv_kernel[
                int(
                    numpy.round(
                        (v * self.theta - iv_shift + 1) * self.oversampling
                    )
                )
            ]
            w_factor = self._w_kernel[
                int(numpy.round((w / self.w_step + 1) * self.w_oversampling))
            ]
            u_factor = u_factor[numpy.newaxis, :, numpy.newaxis]
            v_factor = v_factor[numpy.newaxis, numpy.newaxis, :]
            w_factor = w_factor[
                :, numpy.newaxis, numpy.newaxis
            ]  # w is outermost axis here!

            # Grid visibility. The convolution above now covers
            # all required offsets
            subgrids[
                :, iu0 : iu0 + self.vr_size, iv0 : iv0 + self.vr_size
            ] += (vis[j] * u_factor * v_factor * w_factor)


def test_gridder_wtower_uvw():
    # Common parameters
    image_size = 256  # Total image size in pixels
    subgrid_size = image_size // 4  # Needs to be even.
    theta = 0.02  # Total image size in directional cosines.
    shear_u = 0.2
    shear_v = 0.1
    support = 10
    oversampling = 16 * 1024
    w_step = 280
    w_support = 10
    w_oversampling = 16 * 1024
    print("Grid size: ", image_size / theta, "wavelengths")
    idu = 80
    idv = 90
    idw = 12
    ch_count = 100
    freq0_hz = 1e6
    dfreq_hz = 1e3

    # Create an image for input to degridding.
    numpy.random.seed(123)
    image = numpy.zeros((subgrid_size, subgrid_size))
    image[subgrid_size // 4, subgrid_size // 4] = 1.0
    image[5 * subgrid_size // 6, 2 * subgrid_size // 6] = 0.5
    num_uvw = 300
    uvw = numpy.random.random_sample((num_uvw, 3)) * 100
    start_chs = numpy.zeros((num_uvw), dtype=numpy.int32)
    end_chs = numpy.ones((num_uvw), dtype=numpy.int32) * (ch_count)

    # Create the reference (de)gridder.
    t0 = time.time()
    gridder_ref = WtowerUVWGridKernel(
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
    t1_r = time.time() - t0
    print(f"Reference uvw (de)gridder creation took {t1_r:.4f} s.")

    # Create the PFL (de)gridder.
    t0 = time.time()
    gridder = sdp_grid_func.GridderWtowerUVW(
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
    t1 = time.time() - t0
    print(
        f"PFL uvw (de)gridder creation took {t1:.4f} s. "
        f"(speed-up: {t1_r / t1:.0f})"
    )

    # Call the reference degridder.
    t0 = time.time()
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
    t1_r = time.time() - t0
    print(f"Reference uvw degrid_subgrid took {t1_r:.4f} s.")

    # Call the degridder in PFL.
    vis = numpy.zeros((num_uvw, ch_count), dtype=numpy.complex128)
    t0 = time.time()
    gridder.degrid(
        image, idu, idv, idw, freq0_hz, dfreq_hz, uvw, start_chs, end_chs, vis
    )
    t1 = time.time() - t0
    print(f"PFL uvw degrid took {t1:.4f} s. (speed-up: {t1_r / t1:.0f})")

    # Check results from both are the same.
    for r in range(num_uvw):
        numpy.testing.assert_allclose(
            vis[r, :],
            vis_ref[r, :],
            atol=1e-14,
            rtol=1e-13,
            err_msg=f"degridded data for row {r} is not consistent: uvw={uvw[r,:]}",
        )

    # Generate reference subgrid.
    img_ref = numpy.zeros((subgrid_size, subgrid_size), dtype=complex)
    t0 = time.time()
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
    t1_r = time.time() - t0
    print(f"Reference uvw grid_subgrid took {t1_r:.4f} s.")

    # Call the gridder in PFL.
    img_tst = numpy.zeros_like(img_ref)
    t0 = time.time()
    gridder.grid(
        vis_ref,
        uvw,
        start_chs,
        end_chs,
        freq0_hz,
        dfreq_hz,
        img_tst,
        idu,
        idv,
        idw,
    )
    t1 = time.time() - t0
    print(f"PFL uvw grid took {t1:.4f} s. (speed-up: {t1_r / t1:.0f})")

    # Check results from both are the same.
    assert numpy.max(numpy.abs(img_tst - img_ref)) < 1e-10

    # Check make_kernel for consistency.
    window = numpy.random.random_sample((support))
    t0 = time.time()
    kernel_ref = numpy.array(make_kernel(window, support, oversampling))
    t1_r = time.time() - t0
    print(f"Reference make_kernel took {t1_r:.4f} s.")
    kernel_pfl = numpy.zeros_like(kernel_ref)
    t0 = time.time()
    sdp_grid_func.make_kernel(window, kernel_pfl)
    t1 = time.time() - t0
    print(f"PFL make_kernel took {t1:.4f} s. (speed-up: {t1_r / t1:.0f})")
    numpy.testing.assert_allclose(kernel_ref, kernel_pfl)

    # Check make_kernel_pswf for consistency.
    t0 = time.time()
    pswf_kernel_ref = numpy.array(
        make_pswf_kernel(support, support, oversampling)
    )
    t1_r = time.time() - t0
    print(f"Reference make_pswf_kernel took {t1_r:.4f} s.")
    pswf_kernel_pfl = numpy.zeros_like(pswf_kernel_ref)
    t0 = time.time()
    sdp_grid_func.make_pswf_kernel(support, pswf_kernel_pfl)
    t1 = time.time() - t0
    print(f"PFL make_pswf_kernel took {t1:.4f} s. (speed-up: {t1_r / t1:.0f})")
    numpy.testing.assert_allclose(pswf_kernel_ref, pswf_kernel_pfl)

    # Check make_w_pattern for consistency.
    t0 = time.time()
    w_pattern_ref = make_w_pattern(
        subgrid_size, theta, shear_u, shear_v, w_step
    )
    t1_r = time.time() - t0
    print(f"Reference make_w_pattern took {t1_r:.4f} s.")
    w_pattern_pfl = numpy.zeros_like(w_pattern_ref)
    t0 = time.time()
    sdp_grid_func.make_w_pattern(
        subgrid_size, theta, shear_u, shear_v, w_step, w_pattern_pfl
    )
    t1 = time.time() - t0
    print(f"PFL make_w_pattern took {t1:.4f} s. (speed-up: {t1_r / t1:.0f})")
    numpy.testing.assert_allclose(w_pattern_ref, w_pattern_pfl)


def test_gridder_wtower_uvw_gpu():
    # Common parameters
    image_size = 2048  # Total image size in pixels
    subgrid_size = image_size // 4  # Needs to be even.
    theta = 0.02  # Total image size in directional cosines.
    shear_u = 0.2
    shear_v = 0.1
    support = 10
    oversampling = 16 * 1024
    w_step = 280
    w_support = 10
    w_oversampling = 16 * 1024
    print("Grid size: ", image_size / theta, "wavelengths")
    idu = 80
    idv = 90
    idw = 0
    ch_count = 100
    freq0_hz = 1e6
    dfreq_hz = 1e3

    # Create an image for input to degridding.
    numpy.random.seed(123)
    image = numpy.zeros((subgrid_size, subgrid_size), dtype=numpy.float32)
    image[subgrid_size // 4, subgrid_size // 4] = 1.0
    image[5 * subgrid_size // 6, 2 * subgrid_size // 6] = 0.5
    num_uvw = 30000
    uvw = numpy.random.random_sample((num_uvw, 3)) * 100
    uvw = uvw.astype(numpy.float32)
    start_chs = numpy.zeros((num_uvw), dtype=numpy.int32)
    end_chs = numpy.ones((num_uvw), dtype=numpy.int32) * (ch_count)

    # Create the PFL (de)gridder.
    gridder = sdp_grid_func.GridderWtowerUVW(
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

    # Call the degridder with data in CPU memory.
    vis0 = numpy.zeros((num_uvw, ch_count), dtype=numpy.complex64)
    t0 = time.time()
    gridder.degrid(
        image, idu, idv, idw, freq0_hz, dfreq_hz, uvw, start_chs, end_chs, vis0
    )
    t1_r = time.time() - t0
    print(f"PFL CPU degrid took {t1_r:.4f} s.")

    if cupy:
        # Copy data to GPU memory.
        image_gpu = cupy.asarray(image)
        uvw_gpu = cupy.asarray(uvw)
        start_chs_gpu = cupy.asarray(start_chs)
        end_chs_gpu = cupy.asarray(end_chs)
        vis_gpu = cupy.zeros((num_uvw, ch_count), dtype=cupy.complex64)

        # Call the degridder with data in GPU memory.
        t0 = time.time()
        gridder.degrid(
            image_gpu,
            idu,
            idv,
            idw,
            freq0_hz,
            dfreq_hz,
            uvw_gpu,
            start_chs_gpu,
            end_chs_gpu,
            vis_gpu,
        )
        t1 = time.time() - t0
        print(f"PFL GPU degrid took {t1:.4f} s.")

        # Check results from both are the same.
        vis = cupy.asnumpy(vis_gpu)
        numpy.testing.assert_allclose(vis, vis0, atol=1e-6, rtol=1e-5)


def test_gridder_wtower_uvw_degrid_correct():
    # Common parameters
    image_size = 256  # Total image size in pixels
    subgrid_size = image_size // 4
    theta = 0.1  # Total image size in directional cosines.
    shear_u = 0.2
    shear_v = 0.1
    support = 10
    oversampling = 16 * 1024
    w_step = 280
    w_support = 10
    w_oversampling = 16 * 1024

    # Create a test image.
    image = numpy.random.random_sample((subgrid_size, subgrid_size))
    facet_offset_l = 5
    facet_offset_m = 15

    # Generate reference data.
    gridder_ref = WtowerUVWGridKernel(
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
    img_ref = gridder_ref.degrid_correct(image, facet_offset_l, facet_offset_m)

    # Call the degrid correction function in PFL.
    gridder = sdp_grid_func.GridderWtowerUVW(
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
    img_tst = numpy.array(image)
    gridder.degrid_correct(img_tst, facet_offset_l, facet_offset_m)

    # Check they are the same.
    numpy.testing.assert_allclose(img_tst, img_ref)
