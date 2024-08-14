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

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

import ska_sdp_func.grid_data as sdp_grid_func

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
        result.append(
            (numpy.real(image[il, im]), d_l, d_m, lm_to_n(d_l, d_m, h_u, h_v))
        )
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


def make_wstacking_pattern(
    image_size: int,
    theta: float,
    w_step: float,
    shear_u: float,
    shear_v: float,
):
    # Generate w-pattern for whole image. In a "real" implementation we would
    # likely not do this (do not want to keep image-sized objects in memory!)
    # and instead generate the pattern on-the-fly in (de)grid_correct
    # (we should assume that grid correction is applied rarely enough that it
    #  won't matter too much for performance there)
    grid_pattern = numpy.ones((image_size, image_size))
    grid_image_flmns = image_to_flmn(grid_pattern, theta, shear_u, shear_v)
    img_w_pattern = idft(
        numpy.array([1]),
        numpy.array([[0, 0, w_step]]),
        grid_image_flmns[:, 1:],
    )
    return img_w_pattern.reshape(image_size, image_size)


def w_stacking_correct(
    facet: numpy.ndarray,
    img_w_pattern: numpy.ndarray,
    image_size: int,
    facet_offset_l: int,
    facet_offset_m: int,
    w_offset: int,
    inverse: bool,
):
    # Apply usual correction
    if w_offset == 0:
        return facet

    # Determine w-pattern portion that applies to facet
    left = image_size // 2 - facet.shape[0] // 2
    right = image_size // 2 + facet.shape[0] // 2
    img_w = numpy.roll(
        img_w_pattern**w_offset,
        (-facet_offset_l, -facet_offset_m),
        axis=(0, 1),
    )
    if not inverse:
        return facet / img_w[left:right, left:right]
    else:
        return facet * img_w[left:right, left:right]


def baselines(ants_uvw):
    res = []
    for i in range(ants_uvw.shape[0]):
        for j in range(i + 1, ants_uvw.shape[0]):
            res.append(ants_uvw[j] - ants_uvw[i])
    return numpy.array(res)


def xyz_to_uvw(xyz, ha, dec):
    x, y, z = numpy.hsplit(xyz, 3)
    u = x * numpy.cos(ha) - y * numpy.sin(ha)
    v0 = x * numpy.sin(ha) + y * numpy.cos(ha)
    w = z * numpy.sin(dec) - v0 * numpy.cos(dec)
    v = z * numpy.cos(dec) + v0 * numpy.sin(dec)
    return numpy.hstack([u, v, w])


def xyz_to_baselines(ants_xyz, ha_range, dec):
    return numpy.concatenate(
        [baselines(xyz_to_uvw(ants_xyz, hax, dec)) for hax in ha_range]
    )


def generate_uvw():
    ha_range = numpy.arange(
        numpy.radians(0), numpy.radians(90), numpy.radians(90 / 32)
    )
    dec = numpy.radians(40)  # 50 degrees from zenith!
    vlas = numpy.array(
        [
            (-401.2842, -270.6395, 1.3345),
            (-1317.9926, -889.0279, 2.0336),
            (-2642.9943, -1782.7459, 7.8328),
            (-4329.9414, -2920.6298, 4.217),
            (-6350.012, -4283.1247, -6.0779),
            (-8682.4872, -5856.4585, -7.3861),
            (-11311.4962, -7629.385, -19.3219),
            (-14224.3397, -9594.0268, -32.2199),
            (-17410.1952, -11742.6658, -52.5716),
            (438.6953, -204.4971, -0.1949),
            (1440.9974, -671.8529, 0.6199),
            (2889.4597, -1347.2324, 12.4453),
            (4733.627, -2207.126, 19.9349),
            (6942.0661, -3236.8423, 28.0543),
            (9491.9269, -4425.5098, 19.3104),
            (12366.0731, -5765.3061, 13.8351),
            (15550.4596, -7249.6904, 25.3408),
            (19090.2771, -8748.4418, -53.2768),
            (-38.0377, 434.7135, -0.026),
            (-124.9775, 1428.1567, -1.4012),
            (-259.3684, 2963.3547, -0.0815),
            (-410.6587, 4691.5051, -0.3722),
            (-602.292, 6880.1408, 0.5885),
            (-823.5569, 9407.5172, 0.0647),
            (-1072.9272, 12255.8935, -4.2741),
            (-1349.2489, 15411.7447, -7.7693),
            (-1651.4637, 18863.4683, -9.2248),
        ]
    )
    # Ignoring frequency, so wavelength=1m
    return xyz_to_baselines(vlas, ha_range, dec)


def flatten_uvws_wl(
    ch_count: int,
    freq0: float,
    dfreq: float,
    uvws: numpy.ndarray,
    start_chs: numpy.ndarray,
    end_chs: numpy.ndarray,
):
    """
    Flatten UVWs, convert into wavelengths and put relative to subgrid centre.

    Useful for Python implementations, not a good idea for performance.
    """
    uvws_count = numpy.sum(end_chs - start_chs)
    uvws_out = numpy.empty((uvws_count, 3))

    pos = 0
    for uvw, start_ch, end_ch in zip(uvws, start_chs, end_chs):
        if start_ch >= end_ch:
            continue
        assert start_ch >= 0
        assert end_ch <= ch_count

        # Scale + shift UVWs
        count = end_ch - start_ch
        uvw_scaled = numpy.vstack(
            [
                uvw * ((freq0 + dfreq * ch) / C_0)
                for ch in range(start_ch, end_ch)
            ]
        )
        uvws_out[pos : pos + count] = uvw_scaled
        pos += count

    assert pos == uvws_count
    return uvws_out


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
        subgrid_offsets,
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
        :param subgrid_offsets:
            Tuple of integers containing offset of subgrid in (u, v, w)
            relative to grid centre.
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
        uvw_min2, uvw_max2 = sdp_grid_func.uvw_bounds_all(
            uvws, freq0, dfreq, start_chs, end_chs
        )
        numpy.testing.assert_allclose(uvw_min2, uvw_min)
        numpy.testing.assert_allclose(uvw_max2, uvw_max)

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

                # Bounds check.
                duvw = uvw * dfreq / C_0
                half_subgrid = self.subgrid_size // 2
                u_min = numpy.floor(
                    self.theta * (uvw_shifted[0][0] + start_ch * duvw[0])
                )
                u_max = numpy.ceil(
                    self.theta * (uvw_shifted[0][0] + (end_ch - 1) * duvw[0])
                )
                v_min = numpy.floor(
                    self.theta * (uvw_shifted[0][1] + start_ch * duvw[1])
                )
                v_max = numpy.ceil(
                    self.theta * (uvw_shifted[0][1] + (end_ch - 1) * duvw[1])
                )
                if (
                    u_min < -half_subgrid
                    or u_max >= half_subgrid
                    or v_min < -half_subgrid
                    or v_max >= half_subgrid
                ):
                    continue

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
        subgrid_offsets,
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
        :param subgrid_offsets:
            Tuple of integers containing offset of subgrid in (u, v, w)
            relative to grid centre.
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

                # Bounds check.
                duvw = uvw * dfreq / C_0
                half_subgrid = self.subgrid_size // 2
                u_min = numpy.floor(
                    self.theta * (uvw_shifted[0][0] + start_ch * duvw[0])
                )
                u_max = numpy.ceil(
                    self.theta * (uvw_shifted[0][0] + (end_ch - 1) * duvw[0])
                )
                v_min = numpy.floor(
                    self.theta * (uvw_shifted[0][1] + start_ch * duvw[1])
                )
                v_max = numpy.ceil(
                    self.theta * (uvw_shifted[0][1] + (end_ch - 1) * duvw[1])
                )
                if (
                    u_min < -half_subgrid
                    or u_max >= half_subgrid
                    or v_min < -half_subgrid
                    or v_max >= half_subgrid
                ):
                    continue

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


class WtowerUVWGridKernelWStack(WtowerUVWGridKernel):
    """
    Uses w-towers / w-stacking for uvw gridding
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.img_w_pattern = make_wstacking_pattern(
            self.image_size,
            self.theta,
            self.w_step,
            self.shear_u,
            self.shear_v,
        )

    def degrid_correct(
        self,
        facet: numpy.ndarray,
        facet_offset_l: int,
        facet_offset_m: int,
        w_offset: int = 0,
    ):
        return w_stacking_correct(
            super().degrid_correct(facet, facet_offset_l, facet_offset_m),
            self.img_w_pattern,
            self.image_size,
            facet_offset_l,
            facet_offset_m,
            w_offset,
            False,
        )

    def grid_correct(
        self,
        facet: numpy.ndarray,
        facet_offset_l: int,
        facet_offset_m: int,
        w_offset: int = 0,
    ):
        return w_stacking_correct(
            super().grid_correct(facet, facet_offset_l, facet_offset_m),
            self.img_w_pattern,
            self.image_size,
            facet_offset_l,
            facet_offset_m,
            w_offset,
            True,
        )


def clamp_channels_single(us, freq0, dfreq, start_chs, end_chs, _min, _max):
    """
    Clamp channels for a array of positions

    Restricts a channel range such that all visibilities lie in the given
    bounding

    :param us: positions (in meters)
    :param freq0: Frequency of first channel
    :param dfreq: Channel width
    :param start_chs, end_chs: Channel range to clamp (excluding end!)
    :param _min: Minimum values for position (inclusive)
    :param _max: Maximum values for position (exclusive)
    :returns: Clamped (start_chs, end_chs) or (0,0) if no channels overlap
    """

    # Determine positions far away from zero
    eta = 1e-2
    mask = numpy.abs(us) > eta * C_0 / dfreq

    # Clamp non-zero positions
    u0 = us[mask] * (freq0 / C_0)
    du = us[mask] * (dfreq / C_0)
    mins = numpy.ceil((_min - u0) / du).astype(int)
    maxs = numpy.ceil((_max - u0) / du).astype(int)
    positive_mask = du > 0
    start_chs = numpy.array(start_chs)
    end_chs = numpy.array(end_chs)
    start_chs[mask] = numpy.maximum(
        start_chs[mask], numpy.where(positive_mask, mins, maxs)
    )
    end_chs[mask] = numpy.minimum(
        end_chs[mask], numpy.where(positive_mask, maxs, mins)
    )

    # Clamp zero positions if range doesn't include them
    if _min > 0 or _max <= 0:
        start_chs[~mask] = 0
        end_chs[~mask] = 0

    # Normalise, return
    end_chs = numpy.maximum(end_chs, start_chs)
    return (start_chs, end_chs)


def shift_grid(grid, idu, idv):
    return numpy.roll(numpy.roll(grid, -idu, 0), -idv, 1)


def subgrid_cut_out(grid, subgrid_size):
    image_size = grid.shape[0]
    assert grid.shape[1] == image_size
    start = image_size // 2 - subgrid_size // 2
    end = image_size // 2 + (subgrid_size + 1) // 2
    return grid[start:end, start:end]


def grid_pad(subgrid, image_size):  # inverse to subgrid_cut_out
    grid = numpy.zeros((image_size, image_size), dtype=complex)
    # slightly tricky - we are using the fact that subgrid_cut_out returns
    # a slice of "grid"
    factor = (image_size / subgrid.shape[0]) ** 2
    subgrid_cut_out(grid, subgrid.shape[0])[:] = subgrid * factor
    return grid


def worst_case_image(image_size: int, theta: float, fov: float):
    # Make sources / image with source in corners of fov. Make sure it doesn't
    # divide the subgrid size equally (then there's a good chance we're
    # operating on-grid, and therefore not actually testing the window
    # function)
    fov_edge = int(image_size / theta * fov / 2)
    while image_size % fov_edge == 0:
        fov_edge -= 1
    image = numpy.zeros((image_size, image_size), dtype=float)

    # Put sources into corners, careful not to generate actual
    # symmetries
    image[image_size // 2 + fov_edge, image_size // 2 + fov_edge] = 0.3
    image[image_size // 2 - fov_edge, image_size // 2 - fov_edge] = 0.2
    image[image_size // 2 + fov_edge, image_size // 2 - fov_edge - 1] = 0.3
    image[image_size // 2 - fov_edge - 1, image_size // 2 + fov_edge] = 0.2

    return image


def find_gridder_accuracy(
    grid_kernel,
    fov: float,
    subgrid_frac: float = 2 / 3,
    num_samples: int = 3,
    w: float = 0,
):

    image_size = grid_kernel.image_size
    subgrid_size = grid_kernel.subgrid_size
    theta = grid_kernel.theta
    shear_u = grid_kernel.shear_u
    shear_v = grid_kernel.shear_v

    # Make image
    image = worst_case_image(image_size, theta, fov)
    sources_lmn = image_to_flmn(image, theta, shear_u, shear_v)

    # Apply correction, extract subgrid
    image = grid_kernel.degrid_correct(image, 0, 0)
    subgrid_image = ifft(subgrid_cut_out(fft(image), subgrid_size))

    # Determine error at random points with w=0 and u,v less than
    # subgrid_size/3 distance from centre (optimal w-tower size -
    # we assume that this is always greater than any support the
    # gridding kernel might need, so we can see it as included)
    uvs = numpy.linspace(
        -subgrid_size * subgrid_frac / theta / 2,
        subgrid_size * subgrid_frac / theta / 2,
        num_samples,
    )
    us, vs = numpy.meshgrid(uvs, uvs)
    uvws = numpy.array([(u, v, w) for u, v in zip(us.flatten(), vs.flatten())])

    # Test gridder at these points
    result = grid_kernel.degrid_subgrid(
        subgrid_image,
        (0, 0, 0),
        1,
        C_0,
        C_0,
        uvws,
        numpy.zeros(len(uvws), dtype=numpy.int32),
        numpy.ones(len(uvws), dtype=numpy.int32),
    )
    ref = dft(sources_lmn, uvws)

    # Calculate root mean square error
    return numpy.sqrt(numpy.mean(numpy.abs(result[:, 0] - ref) ** 2))


def find_max_w_tower_height(
    grid_kernel,
    fov: float,
    subgrid_frac: float = 2 / 3,
    num_samples: int = 3,
    target_err: float = None,
):

    # If no target error is specified, default to twice the error at w=0
    if target_err is None:
        target_err = (
            find_gridder_accuracy(
                grid_kernel,
                fov,
                subgrid_frac=subgrid_frac,
                num_samples=num_samples,
                w=0,
            )
            * 2
        )

    # Start a simple binary search
    iw = 1
    diw = 1
    accelerate = True
    while True:

        # Determine error
        err = find_gridder_accuracy(
            grid_kernel,
            fov,
            subgrid_frac=subgrid_frac,
            num_samples=num_samples,
            w=iw * grid_kernel.w_step,
        )

        # Below? Advance. Above? Go back
        if err < target_err:
            if accelerate:
                diw *= 2
            elif diw > 1:
                diw //= 2
            else:
                return 2 * iw
            iw += diw
        elif diw > 1:
            diw //= 2
            iw -= diw
            accelerate = False
        else:
            return 2 * (iw - 1)


def degrid_all(
    image,
    ch_count,
    freq0,
    dfreq,
    uvw,
    kernel,
    subgrid_frac,
    w_tower_height,
    verbose: bool = False,
):
    # Assume we're using all visibilities
    start_chs = numpy.zeros(len(uvw), dtype=numpy.int32)
    end_chs = ch_count * numpy.ones(len(uvw), dtype=numpy.int32)

    # Determine effective size of subgrids and w-tower height
    # (both depend critically on how much we want to "use" of the subgrid)
    eff_sg_size = int(numpy.floor(kernel.subgrid_size * subgrid_frac))
    eff_sg_distance = eff_sg_size / kernel.theta
    w_stack_distance = w_tower_height * kernel.w_step

    # Determine (maximum) number of subgrids and w-stacking planes needed
    eta = 1e-5
    uvw_min, uvw_max = uvw_bounds_all(uvw, freq0, dfreq, start_chs, end_chs)
    min_iu = int(numpy.floor(uvw_min[0] / eff_sg_distance + 0.5 - eta))
    max_iu = int(numpy.floor(uvw_max[0] / eff_sg_distance + 0.5 + eta))
    min_iv = int(numpy.floor(uvw_min[1] / eff_sg_distance + 0.5 - eta))
    max_iv = int(numpy.floor(uvw_max[1] / eff_sg_distance + 0.5 + eta))
    min_iw = int(numpy.floor(uvw_min[2] / w_stack_distance + 0.5 - eta))
    max_iw = int(numpy.floor(uvw_max[2] / w_stack_distance + 0.5 + eta))

    vis = numpy.zeros((len(uvw), ch_count), dtype=complex)
    vis_count = 0
    for iw in range(min_iw, max_iw + 1):

        # Select visibilities on w-plane
        start_chs_w, end_chs_w = clamp_channels_single(
            uvw[:, 2],
            freq0,
            dfreq,
            start_chs,
            end_chs,
            iw * w_stack_distance - w_stack_distance / 2,
            (iw + 1) * w_stack_distance - w_stack_distance / 2,
        )
        if numpy.sum(end_chs_w - start_chs_w) == 0:
            continue

        # Do image correction / w-stacking
        image_corrected = kernel.degrid_correct(
            image, 0, 0, iw * w_tower_height
        )
        grid = fft(image_corrected)

        for iu in range(min_iu, max_iu + 1):

            # Select visibilities in column
            start_chs_u, end_chs_u = clamp_channels_single(
                uvw[:, 0],
                freq0,
                dfreq,
                start_chs_w,
                end_chs_w,
                iu * eff_sg_distance - eff_sg_distance / 2,
                (iu + 1) * eff_sg_distance - eff_sg_distance / 2,
            )
            if numpy.sum(end_chs_u - start_chs_u) == 0:
                continue
            expected_vis_count = vis_count + numpy.sum(end_chs_u - start_chs_u)
            for iv in range(min_iv, max_iv + 1):

                # Select visibilities in subgrid
                start_chs_v, end_chs_v = clamp_channels_single(
                    uvw[:, 1],
                    freq0,
                    dfreq,
                    start_chs_u,
                    end_chs_u,
                    iv * eff_sg_distance - eff_sg_distance / 2,
                    (iv + 1) * eff_sg_distance - eff_sg_distance / 2,
                )
                num_vis = numpy.sum(end_chs_v - start_chs_v)
                if num_vis == 0:
                    continue
                if verbose:
                    print(f"subgrid {iu}/{iv}/{iw}: {num_vis} visibilities")

                # Prepare subgrid
                subgrid_image = ifft(
                    subgrid_cut_out(
                        shift_grid(grid, iu * eff_sg_size, iv * eff_sg_size),
                        kernel.subgrid_size,
                    )
                )

                # Degrid visibilities. Small optimisation: Only pass in
                # baselines and time steps where there's actually visibilities
                # (reduces overhead)
                mask = end_chs_v > start_chs_v
                vis[mask] += kernel.degrid_subgrid(
                    subgrid_image,
                    (iu * eff_sg_size, iv * eff_sg_size, iw * w_tower_height),
                    ch_count,
                    freq0,
                    dfreq,
                    uvw[mask],
                    start_chs_v[mask],
                    end_chs_v[mask],
                )

                vis_count += numpy.sum(end_chs_v - start_chs_v)
            assert vis_count == expected_vis_count
    assert vis_count == len(uvw) * ch_count
    return vis


def grid_all(
    vis,
    ch_count,
    freq0,
    dfreq,
    uvw,
    kernel,
    subgrid_frac,
    w_tower_height,
    verbose: bool = False,
):
    # Assume we're using all visibilities
    start_chs = numpy.zeros(len(uvw), dtype=numpy.int32)
    end_chs = ch_count * numpy.ones(len(uvw), dtype=numpy.int32)

    # Determine effective size of subgrids and w-tower height
    # (both depend critically on how much we want to "use" of the subgrid)
    eff_sg_size = int(numpy.floor(kernel.subgrid_size * subgrid_frac))
    eff_sg_distance = eff_sg_size / kernel.theta
    w_stack_distance = w_tower_height * kernel.w_step

    # Determine (maximum) number of subgrids and w-stacking planes needed
    eta = 1e-5
    uvw_min, uvw_max = uvw_bounds_all(uvw, freq0, dfreq, start_chs, end_chs)
    min_iu = int(numpy.floor(uvw_min[0] / eff_sg_distance + 0.5 - eta))
    max_iu = int(numpy.floor(uvw_max[0] / eff_sg_distance + 0.5 + eta))
    min_iv = int(numpy.floor(uvw_min[1] / eff_sg_distance + 0.5 - eta))
    max_iv = int(numpy.floor(uvw_max[1] / eff_sg_distance + 0.5 + eta))
    min_iw = int(numpy.floor(uvw_min[2] / w_stack_distance + 0.5 - eta))
    max_iw = int(numpy.floor(uvw_max[2] / w_stack_distance + 0.5 + eta))

    image = numpy.zeros((kernel.image_size, kernel.image_size), dtype=complex)
    vis_count = 0
    for iw in range(min_iw, max_iw + 1):

        # Select visibilities on w-plane
        start_chs_w, end_chs_w = clamp_channels_single(
            uvw[:, 2],
            freq0,
            dfreq,
            start_chs,
            end_chs,
            iw * w_stack_distance - w_stack_distance / 2,
            (iw + 1) * w_stack_distance - w_stack_distance / 2,
        )
        if numpy.sum(end_chs_w - start_chs_w) == 0:
            continue

        grid = numpy.zeros(
            (kernel.image_size, kernel.image_size), dtype=complex
        )
        for iu in range(min_iu, max_iu + 1):

            # Select visibilities in column
            start_chs_u, end_chs_u = clamp_channels_single(
                uvw[:, 0],
                freq0,
                dfreq,
                start_chs_w,
                end_chs_w,
                iu * eff_sg_distance - eff_sg_distance / 2,
                (iu + 1) * eff_sg_distance - eff_sg_distance / 2,
            )
            if numpy.sum(end_chs_u - start_chs_u) == 0:
                continue
            expected_vis_count = vis_count + numpy.sum(end_chs_u - start_chs_u)
            for iv in range(min_iv, max_iv + 1):

                # Select visibilities in subgrid
                start_chs_v, end_chs_v = clamp_channels_single(
                    uvw[:, 1],
                    freq0,
                    dfreq,
                    start_chs_u,
                    end_chs_u,
                    iv * eff_sg_distance - eff_sg_distance / 2,
                    (iv + 1) * eff_sg_distance - eff_sg_distance / 2,
                )
                num_vis = numpy.sum(end_chs_v - start_chs_v)
                if num_vis == 0:
                    continue
                if verbose:
                    print(f"subgrid {iu}/{iv}/{iw}: {num_vis} visibilities")

                # Grid visibilities. Again small optimisation: Only pass in
                # baselines and time steps where there's actually visibilities
                # (reduces overhead)
                mask = end_chs_v > start_chs_v
                subgrid_image = numpy.zeros(
                    (kernel.subgrid_size, kernel.subgrid_size), dtype=complex
                )
                kernel.grid_subgrid(
                    vis[mask],
                    uvw[mask],
                    start_chs_v[mask],
                    end_chs_v[mask],
                    ch_count,
                    freq0,
                    dfreq,
                    subgrid_image,
                    (iu * eff_sg_size, iv * eff_sg_size, iw * w_tower_height),
                )

                # Add to grid
                grid += shift_grid(
                    grid_pad(fft(subgrid_image), kernel.image_size),
                    -iu * eff_sg_size,
                    -iv * eff_sg_size,
                )
                vis_count += numpy.sum(end_chs_v - start_chs_v)
            assert vis_count == expected_vis_count

        # Do image correction / w-stacking
        image += kernel.grid_correct(ifft(grid), 0, 0, iw * w_tower_height)

    assert vis_count == len(uvw) * ch_count
    return image


##############################################################################
# Actual tests start here.
##############################################################################


def test_gridder_wtower_uvw():
    # Common parameters
    image_size = 256  # Total image size in pixels
    subgrid_size = image_size // 4  # Needs to be even.
    theta = 0.0008  # Total image size in directional cosines.
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

    # Create an image for input to degridding.
    numpy.random.seed(123)
    image = numpy.zeros((subgrid_size, subgrid_size))
    image[subgrid_size // 4, subgrid_size // 4] = 1.0
    image[5 * subgrid_size // 6, 2 * subgrid_size // 6] = 0.5

    # Create some (u,v,w) coordinates.
    # ch_count = 100
    # freq0_hz = 1e6
    # dfreq_hz = 1e3
    # num_uvw = 300
    # uvw = numpy.random.random_sample((num_uvw, 3)) * 100
    ch_count = 2
    freq0_hz = C_0
    dfreq_hz = C_0 / 100
    uvw = generate_uvw()
    num_uvw = uvw.shape[0]
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
            err_msg=(
                f"degridded data for row {r} is not consistent: "
                f"uvw={uvw[r,:]}"
            ),
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
    theta = 0.002  # Total image size in directional cosines.
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

    # Create an image for input to degridding.
    float_type = numpy.float64
    complex_type = numpy.complex128
    numpy.random.seed(123)
    image = numpy.zeros((subgrid_size, subgrid_size), dtype=float_type)
    image[subgrid_size // 4, subgrid_size // 4] = 1.0
    image[5 * subgrid_size // 6, 2 * subgrid_size // 6] = 0.5

    # Create some (u,v,w) coordinates.
    # ch_count = 100
    # freq0_hz = 1e6
    # dfreq_hz = 1e3
    # num_uvw = 3000
    # uvw = numpy.random.random_sample((num_uvw, 3)) * 100
    ch_count = 10
    freq0_hz = C_0
    dfreq_hz = C_0 / 100
    uvw = generate_uvw()
    num_uvw = uvw.shape[0]
    uvw = uvw.astype(float_type)
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
    vis0 = numpy.zeros((num_uvw, ch_count), dtype=complex_type)
    t0 = time.time()
    gridder.degrid(
        image, idu, idv, idw, freq0_hz, dfreq_hz, uvw, start_chs, end_chs, vis0
    )
    t1_r = time.time() - t0
    print(f"PFL CPU degrid took {t1_r:.4f} s.")

    # Call the gridder with data in CPU memory.
    img0 = numpy.zeros((subgrid_size, subgrid_size), dtype=complex_type)
    t0 = time.time()
    gridder.grid(
        vis0,
        uvw,
        start_chs,
        end_chs,
        freq0_hz,
        dfreq_hz,
        img0,
        idu,
        idv,
        idw,
    )
    t1_r = time.time() - t0
    print(f"PFL CPU grid took {t1_r:.4f} s.")

    if cupy:
        # Copy data to GPU memory.
        image_gpu = cupy.asarray(image)
        uvw_gpu = cupy.asarray(uvw)
        start_chs_gpu = cupy.asarray(start_chs)
        end_chs_gpu = cupy.asarray(end_chs)

        # Call the degridder with data in GPU memory.
        vis_gpu = cupy.zeros((num_uvw, ch_count), dtype=complex_type)
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
        numpy.testing.assert_allclose(vis, vis0, atol=1e-7)

        # Call the gridder with data in GPU memory.
        img_gpu = cupy.zeros((subgrid_size, subgrid_size), dtype=complex_type)
        t0 = time.time()
        gridder.grid(
            vis_gpu,
            uvw_gpu,
            start_chs_gpu,
            end_chs_gpu,
            freq0_hz,
            dfreq_hz,
            img_gpu,
            idu,
            idv,
            idw,
        )
        t1 = time.time() - t0
        print(f"PFL GPU grid took {t1:.4f} s.")

        # Check results from both are the same.
        img = cupy.asnumpy(img_gpu)
        numpy.testing.assert_allclose(img, img0, atol=1e-7)


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
    facet_offset_m = -15

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


def test_gridder_degrid_correct_wstack():
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
    w_offset = 50

    # Create a test image.
    image = numpy.random.random_sample((subgrid_size, subgrid_size)) + 0j
    facet_offset_l = 5
    facet_offset_m = -15

    # Generate reference data.
    gridder_ref = WtowerUVWGridKernelWStack(
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
    img_ref = gridder_ref.degrid_correct(
        image, facet_offset_l, facet_offset_m, w_offset
    )

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
    gridder.degrid_correct(img_tst, facet_offset_l, facet_offset_m, w_offset)

    # Check they are the same.
    numpy.testing.assert_allclose(img_tst, img_ref)


def test_subgrid_cut_out():
    image_size = 512
    subgrid_size = image_size // 4
    offset_u = -255
    offset_v = -170
    numpy.random.seed(123)
    grid = numpy.random.random_sample((image_size, image_size)) + 0j
    subgrid = numpy.zeros((subgrid_size, subgrid_size), dtype=numpy.complex128)
    subgrid_ref = subgrid_cut_out(
        shift_grid(grid, offset_u, offset_v), subgrid_size
    )
    sdp_grid_func.subgrid_cut_out(grid, offset_u, offset_v, subgrid)

    # Check they are the same.
    numpy.testing.assert_allclose(subgrid, subgrid_ref)

    # Test GPU version.
    if cupy:
        grid_gpu = cupy.asarray(grid)
        subgrid_gpu = cupy.zeros(subgrid.shape, dtype=cupy.complex128)
        sdp_grid_func.subgrid_cut_out(
            grid_gpu, offset_u, offset_v, subgrid_gpu
        )
        subgrid_gpu_copy = cupy.asnumpy(subgrid_gpu)

        # Check they are the same.
        numpy.testing.assert_allclose(subgrid_gpu_copy, subgrid_ref)


def test_subgrid_add():
    image_size = 512
    subgrid_size = image_size // 4
    offset_u = 255
    offset_v = 170
    numpy.random.seed(123)
    subgrid = numpy.random.random_sample((subgrid_size, subgrid_size)) + 0j

    grid_ref = numpy.zeros((image_size, image_size), dtype=numpy.complex128)
    grid_ref += shift_grid(grid_pad(subgrid, image_size), -offset_u, -offset_v)

    grid = numpy.zeros((image_size, image_size), dtype=numpy.complex128)
    factor = (image_size / subgrid.shape[0]) ** 2
    sdp_grid_func.subgrid_add(grid, -offset_u, -offset_v, subgrid, factor)

    # Check they are the same.
    numpy.testing.assert_allclose(grid, grid_ref)

    # Test GPU version.
    if cupy:
        grid_gpu = cupy.zeros(grid.shape, dtype=cupy.complex128)
        subgrid_gpu = cupy.asarray(subgrid)
        sdp_grid_func.subgrid_add(
            grid_gpu, -offset_u, -offset_v, subgrid_gpu, factor
        )
        grid_gpu_copy = cupy.asnumpy(grid_gpu)

        # Check they are the same.
        numpy.testing.assert_allclose(grid_gpu_copy, grid_ref)


def test_gridder_wstack():
    # Common parameters
    image_size = 512  # Total image size in pixels
    subgrid_size = image_size // 4
    theta = 0.01  # Total image size in directional cosines.
    fov = 2 * numpy.arcsin(theta / 2)
    shear_u = 0.0
    shear_v = 0.0
    support = 10
    oversampling = 16 * 1024
    w_step = 100
    w_support = 10
    w_oversampling = 16 * 1024

    # Create an image for input to degridding.
    numpy.random.seed(123)
    image = numpy.zeros((image_size, image_size), dtype=numpy.complex128)
    sources = [
        (2, image_size // 4, 2),
        (1, -image_size // 4 + 2, image_size // 4 - 12),
    ]
    for flux, il, im in sources:
        image[il + image_size // 2, im + image_size // 2] += flux

    # Define a visibility set to degrid.
    num_chan = 16
    freq0_hz = C_0
    dfreq_hz = C_0 / 100
    uvw = generate_uvw()
    num_rows = uvw.shape[0]
    # plt.scatter(uvw[:,0], uvw[:,1])
    # plt.show()
    start_chs = numpy.zeros((num_rows), dtype=numpy.int32)
    end_chs = numpy.ones((num_rows), dtype=numpy.int32) * num_chan
    flmn = image_to_flmn(image, theta, shear_u, shear_v)
    vis_dft = dft(
        flmn,
        flatten_uvws_wl(num_chan, freq0_hz, dfreq_hz, uvw, start_chs, end_chs),
    ).reshape(len(uvw), num_chan)

    # Create the reference kernel.
    t0 = time.time()
    gridder_ref = WtowerUVWGridKernelWStack(
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
    print(f"Creating reference gridder kernel took {t1:.4f} s.")
    subgrid_frac = 2 / 3
    t0 = time.time()
    w_tower_height = find_max_w_tower_height(gridder_ref, fov, subgrid_frac)
    t1 = time.time() - t0
    print(f"find_max_w_tower_height took {t1:.4f} s.")

    # Call the reference degridding function.
    t0 = time.time()
    vis_ref = degrid_all(
        image,
        num_chan,
        freq0_hz,
        dfreq_hz,
        uvw,
        gridder_ref,
        subgrid_frac,
        w_tower_height,
        True,
    )
    t1 = time.time() - t0
    print(f"REF degrid_all took {t1:.4f} s.")

    # Call the PFL degridding function.
    vis = numpy.zeros_like(vis_dft)
    t0 = time.time()
    sdp_grid_func.wstack_wtower_degrid_all(
        image,
        freq0_hz,
        dfreq_hz,
        uvw,
        subgrid_size,
        theta,
        w_step,
        shear_u,
        shear_v,
        support,
        oversampling,
        w_support,
        w_oversampling,
        subgrid_frac,
        w_tower_height,
        1,
        vis,
    )
    t1 = time.time() - t0
    print(f"PFL wstack_wtower_degrid_all took {t1:.4f} s.")

    # Check they are the same.
    numpy.testing.assert_allclose(vis, vis_ref)

    # Call the reference gridding function.
    t0 = time.time()
    image_ref = grid_all(
        vis_dft,
        num_chan,
        freq0_hz,
        dfreq_hz,
        uvw,
        gridder_ref,
        subgrid_frac,
        w_tower_height,
        True,
    )
    t1 = time.time() - t0
    print(f"REF grid_all took {t1:.4f} s.")

    # Call the PFL gridding function.
    image_out = numpy.zeros((image_size, image_size), dtype=numpy.complex128)
    t0 = time.time()
    sdp_grid_func.wstack_wtower_grid_all(
        vis_dft,
        freq0_hz,
        dfreq_hz,
        uvw,
        subgrid_size,
        theta,
        w_step,
        shear_u,
        shear_v,
        support,
        oversampling,
        w_support,
        w_oversampling,
        subgrid_frac,
        w_tower_height,
        1,
        image_out,
    )
    t1 = time.time() - t0
    print(f"PFL wstack_wtower_grid_all took {t1:.4f} s.")

    # Check they are the same, but ignore the pixels around the border.
    left = 30
    right = -30
    im_ref = numpy.real(image_ref[left:right, left:right])
    im_pfl = numpy.real(image_out[left:right, left:right])
    numpy.testing.assert_allclose(im_pfl, im_ref, atol=1e-5)

    if plt:
        im_ratio = im_pfl / im_ref
        im_diff = im_pfl - im_ref
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
            2, 2, figsize=(12, 9), dpi=300
        )
        im = ax1.imshow(im_ref)
        ax1.set_title("Reference w-towers Python gridder")
        fig.colorbar(im, ax=ax1)
        im = ax2.imshow(im_pfl)
        ax2.set_title("PFL w-towers gridder (CPU version)")
        fig.colorbar(im, ax=ax2)
        im = ax3.imshow(im_ratio)
        ax3.set_title("Ratio (PFL / ref)")
        fig.colorbar(im, ax=ax3)
        im = ax4.imshow(im_diff)
        ax4.set_title("Diff (PFL - ref)")
        fig.colorbar(im, ax=ax4)
        plt.savefig("test_verify_gridder_cpu.png")


def test_gpu_gridder_wstack():
    if not cupy:
        return

    # Common parameters
    image_size = 512  # Total image size in pixels
    subgrid_size = image_size // 4
    theta = 0.01  # Total image size in directional cosines.
    fov = 2 * numpy.arcsin(theta / 2)
    shear_u = 0.0
    shear_v = 0.0
    support = 10
    oversampling = 16 * 1024
    w_step = 100
    w_support = 10
    w_oversampling = 16 * 1024

    # Create an image for input to degridding.
    numpy.random.seed(123)
    image = numpy.zeros((image_size, image_size), dtype=numpy.complex128)
    sources = [
        (2, image_size // 4, 2),
        (1, -image_size // 4 + 2, image_size // 4 - 12),
    ]
    for flux, il, im in sources:
        image[il + image_size // 2, im + image_size // 2] += flux

    # Define a visibility set to degrid.
    num_chan = 8
    freq0_hz = C_0
    dfreq_hz = C_0 / 100
    uvw = generate_uvw()
    num_rows = uvw.shape[0]
    # plt.scatter(uvw[:,0], uvw[:,1])
    # plt.show()
    start_chs = numpy.zeros((num_rows), dtype=numpy.int32)
    end_chs = numpy.ones((num_rows), dtype=numpy.int32) * num_chan
    flmn = image_to_flmn(image, theta, shear_u, shear_v)
    vis_dft = dft(
        flmn,
        flatten_uvws_wl(num_chan, freq0_hz, dfreq_hz, uvw, start_chs, end_chs),
    ).reshape(len(uvw), num_chan)

    # Create the reference kernel (needed for find_max_w_tower_height).
    t0 = time.time()
    gridder_ref = WtowerUVWGridKernelWStack(
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
    print(f"Creating reference gridder kernel took {t1:.4f} s.")
    subgrid_frac = 2 / 3
    t0 = time.time()
    w_tower_height = find_max_w_tower_height(gridder_ref, fov, subgrid_frac)
    t1 = time.time() - t0
    print(f"find_max_w_tower_height took {t1:.4f} s.")

    # Call the CPU PFL degridding function.
    vis_ref = numpy.zeros_like(vis_dft)
    t0 = time.time()
    sdp_grid_func.wstack_wtower_degrid_all(
        image,
        freq0_hz,
        dfreq_hz,
        uvw,
        subgrid_size,
        theta,
        w_step,
        shear_u,
        shear_v,
        support,
        oversampling,
        w_support,
        w_oversampling,
        subgrid_frac,
        w_tower_height,
        1,
        vis_ref,
    )
    t1 = time.time() - t0
    print(f"PFL wstack_wtower_degrid_all (CPU) took {t1:.4f} s.")

    # Call the CPU PFL gridding function.
    img_ref = numpy.zeros((image_size, image_size), dtype=numpy.complex128)
    t0 = time.time()
    sdp_grid_func.wstack_wtower_grid_all(
        vis_dft,
        freq0_hz,
        dfreq_hz,
        uvw,
        subgrid_size,
        theta,
        w_step,
        shear_u,
        shear_v,
        support,
        oversampling,
        w_support,
        w_oversampling,
        subgrid_frac,
        w_tower_height,
        1,
        img_ref,
    )
    t1 = time.time() - t0
    print(f"PFL wstack_wtower_grid_all (CPU) took {t1:.4f} s.")
    left = 30
    right = -30
    im_pfl_ref = numpy.real(img_ref[left:right, left:right])

    # Copy data to GPU memory.
    vis_gpu = cupy.zeros(vis_dft.shape, dtype=numpy.complex128)
    image_gpu = cupy.asarray(image)
    uvw_gpu = cupy.asarray(uvw)

    # Call the GPU PFL degridding function.
    t0 = time.time()
    sdp_grid_func.wstack_wtower_degrid_all(
        image_gpu,
        freq0_hz,
        dfreq_hz,
        uvw_gpu,
        subgrid_size,
        theta,
        w_step,
        shear_u,
        shear_v,
        support,
        oversampling,
        w_support,
        w_oversampling,
        subgrid_frac,
        w_tower_height,
        1,
        vis_gpu,
    )
    t1 = time.time() - t0
    print(f"PFL wstack_wtower_degrid_all (GPU) took {t1:.4f} s.")

    # Check they are the same.
    vis_gpu_copy = cupy.asnumpy(vis_gpu)
    numpy.testing.assert_allclose(vis_gpu_copy, vis_ref, atol=1e-7)

    # Call the GPU PFL gridding function.
    vis_dft_gpu = cupy.asarray(vis_dft)
    img_gpu = cupy.zeros_like(image_gpu)
    t0 = time.time()
    sdp_grid_func.wstack_wtower_grid_all(
        vis_dft_gpu,
        freq0_hz,
        dfreq_hz,
        uvw_gpu,
        subgrid_size,
        theta,
        w_step,
        shear_u,
        shear_v,
        support,
        oversampling,
        w_support,
        w_oversampling,
        subgrid_frac,
        w_tower_height,
        1,
        img_gpu,
    )
    t1 = time.time() - t0
    print(f"PFL wstack_wtower_grid_all (GPU) took {t1:.4f} s.")

    # Check they are the same, but ignore the pixels around the border.
    img_gpu_copy = cupy.asnumpy(img_gpu)
    left = 30
    right = -30
    im_pfl_gpu = numpy.real(img_gpu_copy[left:right, left:right])
    numpy.testing.assert_allclose(im_pfl_gpu, im_pfl_ref, atol=1e-6)

    if plt:
        im_ratio = im_pfl_gpu / im_pfl_ref
        im_diff = im_pfl_gpu - im_pfl_ref
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
            2, 2, figsize=(12, 9), dpi=300
        )
        im = ax1.imshow(im_pfl_ref)
        ax1.set_title("PFL w-towers gridder (CPU version)")
        fig.colorbar(im, ax=ax1)
        im = ax2.imshow(im_pfl_gpu)
        ax2.set_title("PFL w-towers gridder (GPU version)")
        fig.colorbar(im, ax=ax2)
        im = ax3.imshow(im_ratio)
        ax3.set_title("Ratio (PFL(GPU) / PFL(CPU))")
        fig.colorbar(im, ax=ax3)
        im = ax4.imshow(im_diff)
        ax4.set_title("Diff (PFL(GPU) - PFL(CPU))")
        fig.colorbar(im, ax=ax4)
        plt.savefig("test_verify_gridder_gpu.png")
