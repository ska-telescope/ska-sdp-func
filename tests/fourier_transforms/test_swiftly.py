
"""
Unit tests for swiftly.py functions.

Adapted from ska-sdp-distributed-fourier-transform
"""

import itertools

import numpy
import pytest

from ska_sdp_func.fourier_transforms.swiftly import Swiftly

TEST_PARAMS = {
    "W": 13.5625,
    "N": 1024,
    "yB_size": 416,
    "yN_size": 512,
    "xA_size": 228,
    "xM_size": 256,
}

# Helper routines, stolen from ska-sdp-distributed-fourier-transform

def create_slice(fill_val, axis_val, dims, axis):
    """
    Create a tuple of length = dims.

    Elements of the tuple:
        fill_val if axis != dim_index;
        axis_val if axis == dim_index,
        where dim_index is each value in range(dims)

    See test for examples.

    :param fill_val: value to use for dimensions where dim != axis
    :param axis_val: value to use for dimensions where dim == axis
    :param dims: length of tuple to be produced
                 (i.e. number of dimensions); int
    :param axis: axis (index) along which axis_val to be used; int

    :return: tuple of length dims
    """

    if not isinstance(axis, int) or not isinstance(dims, int):
        raise ValueError(
            "create_slice: axis and dims values have to be integers."
        )

    return tuple(axis_val if i == axis else fill_val for i in range(dims))

def pad_mid(a, n, axis):
    """
    Pad an array to a desired size with zeros at a given axis.
    (Surround the middle with zeros till it reaches the given size)

    :param a: numpy array to be padded
    :param n: size to be padded to (desired size)
    :param axis: axis along which to pad

    :return: padded numpy array
    """
    n0 = a.shape[axis]
    if n == n0:
        return a
    pad = create_slice(
        (0, 0),
        (n // 2 - n0 // 2, (n + 1) // 2 - (n0 + 1) // 2),
        len(a.shape),
        axis,
    )
    return numpy.pad(a, pad, mode="constant", constant_values=0.0)

def make_facet_from_sources(
    sources: list[tuple[float, int]],
    image_size: int,
    facet_size: int,
    facet_offsets: list[int],
    facet_masks: list[numpy.ndarray] = None,
):
    """
    Generates a facet from a source list

    This basically boils down to adding pixels on a grid, taking into account
    that coordinates might wrap around. Length of facet_offsets tuple decides
    how many dimensions the result has.

    :param sources: List of (intensity, *coords) tuples, all image
        coordinates integer and relative to image centre
    :param image_size: All coordinates and offset are
        interpreted as modulo this size
    :param facet_size: Desired size of facet
    :param facet_offsets: Offset tuple of facet mid-point
    :param facet_masks: Mask expressions (optional)
    :returns: Numpy array with facet data
    """

    # Allocate facet
    dims = len(facet_offsets)
    facet = numpy.zeros(dims * [facet_size], dtype=complex)

    # Set indicated pixels on facet
    offs = numpy.array(facet_offsets, dtype=int) - dims * [facet_size // 2]
    for intensity, *coord in sources:
        # Determine position relative to facet centre
        coord = numpy.mod(coord - offs, image_size)

        # Is the source within boundaries?
        if any((coord < 0) | (coord >= facet_size)):
            continue

        # Set pixel
        facet[tuple(coord)] += intensity

    # Apply facet mask
    for axis, mask in enumerate(facet_masks or []):
        facet *= broadcast(numpy.array(mask), dims, axis)

    return facet


def make_subgrid_from_sources(
    sources: list[tuple[float, int]],
    image_size: int,
    subgrid_size: int,
    subgrid_offsets: list[int],
    subgrid_masks: list[numpy.ndarray] = None,
):
    """
    Generates a subgrid from a source list

    This solves a direct Fourier transformation for the given sources.
    Note that in contrast to make_facet_from_sources this can get fairly
    expensive. Length of subgrid_offsets tuple decides how many dimensions
    the result has.

    :param sources: List of (intensity, *coords) tuples, all image
        coordinates integer and relative to image centre
    :param image_size: Image size. Determines grid resolution and
        normalisation.
    :param subgrid_size: Desired size of subgrid
    :param subgrid_offsets: Offset tuple of subgrid mid-point
    :param subgrid_masks: Mask expressions (optional)
    :returns: Numpy array with subgrid data
    """

    # Allocate subgrid
    dims = len(subgrid_offsets)
    subgrid = numpy.zeros(dims * [subgrid_size], dtype=complex)

    # Determine subgrid data via DFT
    uvs = numpy.transpose(
        numpy.mgrid[
            tuple(
                slice(off - subgrid_size // 2, off + (subgrid_size + 1) // 2)
                for off in reversed(subgrid_offsets)
            )
        ][::-1]
    )
    for intensity, *coords in sources:
        norm_int = intensity / image_size**dims
        subgrid += norm_int * numpy.exp(
            (2j * numpy.pi / image_size) * numpy.dot(uvs, coords)
        )

    # Apply subgrid masks
    for axis, mask in enumerate(subgrid_masks or []):
        subgrid *= broadcast(numpy.array(mask), dims, axis)

    return subgrid


@pytest.mark.parametrize(
    "xA_size", [TEST_PARAMS["xA_size"], TEST_PARAMS["xA_size"] - 1]
)
@pytest.mark.parametrize(
    "yB_size", [TEST_PARAMS["yB_size"], TEST_PARAMS["yB_size"] - 1]
)
def test_facet_to_subgrid_basic(xA_size, yB_size):
    """Test basic properties of 1D facet to subgrid distributed FT
    primitives for cases where the subgrids are expected to be a
    constant value.
    """

    # Basic layout parameters
    N = TEST_PARAMS["N"]
    yN_size = TEST_PARAMS["yN_size"]
    xM_size = TEST_PARAMS["xM_size"]
    W = TEST_PARAMS['W']
    xM_yN_size = xM_size * yN_size // N
    Nx = N // yN_size
    Ny = N // xM_size

    # Instantiate
    swiftly = Swiftly(
        N, yN_size, xM_size, W
    )

    # Test with different values and facet offsets
    for val, facet_off in itertools.product(
        [0, 1, 0.1], numpy.arange(-5 * Ny, 5 * Ny // 2, Ny)
    ):
        # Set value at centre of image (might be off-centre for
        # the facet depending on offset)
        facet = numpy.zeros(yB_size, dtype=complex)
        facet[yB_size // 2 - facet_off] = val
        prepped = numpy.empty(yN_size, dtype=complex)
        swiftly.prepare_facet(facet[numpy.newaxis], prepped[numpy.newaxis], facet_off)

        # Now generate subgrids at different (valid) subgrid offsets.
        for sg_off in numpy.arange(0, 10 * Nx, Nx):
            subgrid_contrib = numpy.empty(xM_yN_size, dtype=complex)
            swiftly.extract_from_facet(
                prepped[numpy.newaxis], subgrid_contrib[numpy.newaxis], sg_off
            )
            subgrid_acc = numpy.zeros(xM_size, dtype=complex)
            swiftly.add_to_subgrid(
                subgrid_contrib[numpy.newaxis], subgrid_acc[numpy.newaxis], facet_off
            )
            subgrid = numpy.empty(xA_size, dtype=complex)
            swiftly.finish_subgrid(subgrid_acc[numpy.newaxis], subgrid[numpy.newaxis], sg_off)

            # Now the entire subgrid should have (close to) a
            # constant value
            numpy.testing.assert_array_almost_equal(subgrid, val / N)


@pytest.mark.parametrize(
    "xA_size", [TEST_PARAMS["xA_size"], TEST_PARAMS["xA_size"] - 1]
)
@pytest.mark.parametrize(
    "yB_size", [TEST_PARAMS["yB_size"], TEST_PARAMS["yB_size"] - 1]
)
def test_facet_to_subgrid_dft_1d(xA_size, yB_size):
    """Test facet to subgrid distributed FT primitives against direct
    Fourier transformation implementation.
    """

    # Basic layout parameters
    N = TEST_PARAMS["N"]
    yN_size = TEST_PARAMS["yN_size"]
    xM_size = TEST_PARAMS["xM_size"]
    W = TEST_PARAMS['W']
    xM_yN_size = xM_size * yN_size // N
    Nx = N // yN_size
    Ny = N // xM_size

    # Instantiate
    swiftly = Swiftly(
        N, yN_size, xM_size, W
    )

    # Test with different values and facet offsets
    for sources, facet_off in itertools.product(
        [
            [(1, 0)],
            [(2, 1)],
            [(1, -3)],
            [(-0.1, 5)],
            [(1, 20), (2, 5), (3, -4)],
            [(1, -yB_size)],  # border - clamped below
            [(1, yB_size)],  # border - clamped below
            [(1, i) for i in range(-10, 10)],
        ],
        numpy.arange(-100 * Ny, 100 * Ny, 10 * Ny),
    ):
        # Clamp coordinate(s) to facet size
        min_x = -(yB_size - 1) // 2 + facet_off
        max_x = min_x + yB_size - 1
        sources = [(i, min(max(x, min_x), max_x)) for i, x in sources]

        # Set sources in facet
        facet = make_facet_from_sources(sources, N, yB_size, [facet_off])

        # We assume all sources are on the facet
        assert numpy.sum(facet) == sum(
            src[0] for src in sources
        ), f"{sources} {yB_size} {facet_off}"
        prepped = numpy.empty(yN_size, dtype=complex)
        swiftly.prepare_facet(facet[numpy.newaxis], prepped[numpy.newaxis], facet_off)

        # Now generate subgrids at different (valid) subgrid offsets.
        for sg_off in [0, Nx, -Nx, N]:
            subgrid_contrib = numpy.empty(xM_yN_size, dtype=complex)
            swiftly.extract_from_facet(
                prepped[numpy.newaxis], subgrid_contrib[numpy.newaxis], sg_off
            )
            subgrid_acc = numpy.zeros(xM_size, dtype=complex)
            swiftly.add_to_subgrid(
                subgrid_contrib[numpy.newaxis], subgrid_acc[numpy.newaxis], facet_off
            )
            subgrid = numpy.empty(xA_size, dtype=complex)
            swiftly.finish_subgrid(subgrid_acc[numpy.newaxis], subgrid[numpy.newaxis], sg_off)

            # Now check against DFT
            expected = make_subgrid_from_sources(sources, N, xA_size, [sg_off])
            if not numpy.allclose(subgrid, expected):
                print(sources, facet_off)
                numpy.save("facet.npy", facet)
                numpy.save("subgrid.npy", subgrid)
                numpy.save("expected.npy", expected)

            numpy.testing.assert_array_almost_equal(subgrid, expected)

def extract_mid(a, n, axis):
    """
    Extract a section from middle of a map (array) along a given axis.
    This is the reverse operation to pad.

    :param a: numpy array from which to extract
    :param n: size of section
    :param axis: axis along which to extract (int: 0, 1)

    :return: extracted numpy array
    """
    assert n <= a.shape[axis]
    cx = a.shape[axis] // 2
    if n % 2 != 0:
        slc = slice(cx - n // 2, cx + n // 2 + 1)
    else:
        slc = slice(cx - n // 2, cx + n // 2)
    return a[create_slice(slice(None), slc, len(a.shape), axis)]

def test_facet_to_subgrid_dft_2d():
    """Test facet to subgrid distributed FT primitives against direct
    Fourier transformation implementation -- 2D version
    """

    # Basic layout parameters
    N = TEST_PARAMS["N"]
    xA_size = TEST_PARAMS["xA_size"]
    xM_size = TEST_PARAMS["xM_size"]
    yB_size = TEST_PARAMS["yB_size"]
    yN_size = TEST_PARAMS["yN_size"]
    W = TEST_PARAMS['W']
    xM_yN_size = xM_size * yN_size // N
    Nx = N // yN_size
    Ny = N // xM_size

    # Instantiate
    swiftly = Swiftly(
        N, yN_size, xM_size, W
    )

    # Test with different values and facet offsets
    for sources, facet_offs in itertools.product(
        [
            [(1, 0, 0)],
            [(1, 20, 4), (2, 2, 5), (3, -5, -4)],
        ],
        [[0, 0], [Ny, Ny], [-Ny, Ny], [0, -Ny]],
    ):
        # Set sources in facet
        facet = make_facet_from_sources(sources, N, yB_size, facet_offs)

        # We assume all sources are on the facet
        assert numpy.sum(facet) == sum(src[0] for src in sources)
        prepped0 = numpy.empty((yN_size, yB_size), dtype=complex)
        swiftly.prepare_facet(facet.T, prepped0.T, facet_offs[0])
        prepped = numpy.empty((yN_size, yN_size), dtype=complex)
        swiftly.prepare_facet(prepped0, prepped, facet_offs[1])

        # Now generate subgrids at different (valid) subgrid offsets.
        for sg_offs in [[0, 0], [0, Nx], [Nx, 0], [-Nx, -Nx]]:
            subgrid_contrib0 = numpy.empty((xM_yN_size, yN_size), dtype=complex)
            swiftly.extract_from_facet(
                prepped.T, subgrid_contrib0.T, sg_offs[0]
            )
            subgrid_contrib = numpy.empty((xM_yN_size, xM_yN_size), dtype=complex)
            swiftly.extract_from_facet(
                subgrid_contrib0, subgrid_contrib, sg_offs[1]
            )

            subgrid_acc = numpy.zeros((xM_size, xM_size), dtype=complex)
            swiftly.add_to_subgrid_2d(
                subgrid_contrib, subgrid_acc, facet_offs[0], facet_offs[1]
            )

            swiftly.finish_subgrid_inplace_2d(
                subgrid_acc, sg_offs[0], sg_offs[1])
            subgrid = extract_mid(extract_mid(subgrid_acc, xA_size, 0), xA_size, 1)

            # Now check against DFT
            expected = make_subgrid_from_sources(sources, N, xA_size, sg_offs)
            numpy.testing.assert_array_almost_equal(subgrid, expected)


@pytest.mark.parametrize(
    "xA_size", [TEST_PARAMS["xA_size"], TEST_PARAMS["xA_size"] - 1]
)
@pytest.mark.parametrize(
    "yB_size", [TEST_PARAMS["yB_size"], TEST_PARAMS["yB_size"] - 1]
)
def test_subgrid_to_facet_basic(xA_size, yB_size):
    """Test basic properties of 1D subgrid to facet distributed FT
    primitives for cases where a subgrid is set to a constant value.
    """

    # Basic layout parameters
    N = TEST_PARAMS["N"]
    xA_size = TEST_PARAMS["xA_size"]
    xM_size = TEST_PARAMS["xM_size"]
    yB_size = TEST_PARAMS["yB_size"]
    yN_size = TEST_PARAMS["yN_size"]
    W = TEST_PARAMS['W']
    xM_yN_size = xM_size * yN_size // N
    Nx = N // yN_size
    Ny = N // xM_size

    # Instantiate
    swiftly = Swiftly(
        N, yN_size, xM_size, W
    )
    sg_offs = Nx * numpy.arange(-9, 8)
    facet_offs = Ny * numpy.arange(-9, 8)

    # Test linearity with different values, and start with subgrids at
    # different (valid) subgrid offsets.
    for val, sg_off in itertools.product([0, 1, 0.1], sg_offs):
        # Constant-value subgrid
        subgrid = (val / xA_size) * numpy.ones(xA_size, dtype=complex)
        prepped = pad_mid(subgrid, xM_size, 0)
        swiftly.prepare_subgrid_inplace(prepped[numpy.newaxis], sg_off)

        # Check different facet offsets
        for facet_off in facet_offs:
            extracted = numpy.empty(xM_yN_size, dtype=complex)
            swiftly.extract_from_subgrid(
                prepped[numpy.newaxis], extracted[numpy.newaxis], facet_off)
            accumulated = numpy.zeros(yN_size, dtype=complex)
            swiftly.add_to_facet(
                extracted[numpy.newaxis], accumulated[numpy.newaxis], sg_off
            )
            facet = numpy.empty(yB_size, dtype=complex)
            swiftly.finish_facet(
                accumulated[numpy.newaxis], facet[numpy.newaxis], facet_off
            )

            # Check that we have value at centre of image
            numpy.testing.assert_array_almost_equal(
                facet[yB_size // 2 - facet_off], val
            )


@pytest.mark.parametrize(
    "xA_size", [TEST_PARAMS["xA_size"], TEST_PARAMS["xA_size"] - 1]
)
@pytest.mark.parametrize(
    "yB_size", [TEST_PARAMS["yB_size"], TEST_PARAMS["yB_size"] - 1]
)
def test_subgrid_to_facet_dft(xA_size, yB_size):
    """Test basic properties of 1D subgrid to facet distributed FT
    primitives for cases where a subgrid is set to a constant value.
    """

    # Basic layout parameters
    N = TEST_PARAMS["N"]
    xA_size = TEST_PARAMS["xA_size"]
    xM_size = TEST_PARAMS["xM_size"]
    yB_size = TEST_PARAMS["yB_size"]
    yN_size = TEST_PARAMS["yN_size"]
    W = TEST_PARAMS['W']
    xM_yN_size = xM_size * yN_size // N
    Nx = N // yN_size
    Ny = N // xM_size

    # Instantiate
    swiftly = Swiftly(
        N, yN_size, xM_size, W
    )
    
    source_lists = [
        [(1, 0)],
        [(2, 1)],
        [(1, -3)],
        [(-0.1, 5)],
    ]
    sg_offs = Nx * numpy.arange(-9, 8)
    facet_offs = Ny * numpy.arange(-9, 8)

    # Test linearity with different values, and start with subgrids at
    # different (valid) subgrid offsets.
    for sources, sg_off in itertools.product(source_lists, sg_offs):
        # Generate subgrid. As we are only filling the grid partially
        # here, we have to scale it.
        subgrid = (
            make_subgrid_from_sources(sources, N, xA_size, [sg_off])
            / xA_size
            * N
        )
        prepped = pad_mid(subgrid, xM_size, 0)
        swiftly.prepare_subgrid_inplace(prepped[numpy.newaxis], sg_off)

        # Check different facet offsets
        for facet_off in facet_offs:
            extracted = numpy.empty(xM_yN_size, dtype=complex)
            swiftly.extract_from_subgrid(
                prepped[numpy.newaxis], extracted[numpy.newaxis], facet_off)
            accumulated = numpy.zeros(yN_size, dtype=complex)
            swiftly.add_to_facet(
                extracted[numpy.newaxis], accumulated[numpy.newaxis], sg_off
            )
            facet = numpy.empty(yB_size, dtype=complex)
            swiftly.finish_facet(
                accumulated[numpy.newaxis], facet[numpy.newaxis], facet_off
            )

            # Check that pixels in questions have correct value. As -
            # again - we have only partially filled the grid, the only
            # value we can really check is the (singular) one we set
            # previously.
            expected = make_facet_from_sources(
                sources, N, yB_size, [facet_off]
            )
            numpy.testing.assert_array_almost_equal(
                facet[expected != 0], expected[expected != 0]
            )
            if sources[0][0] > 0:
                numpy.testing.assert_array_less(
                    facet[expected == 0], numpy.max(expected)
                )
            else:
                numpy.testing.assert_array_less(
                    -facet[expected == 0], numpy.max(-expected)
                )


def test_subgrid_to_facet_dft_2d():
    """Test basic properties of 1D subgrid to facet distributed FT
    primitives for cases where a subgrid is set to a constant value.
    """

    # Basic layout parameters
    N = TEST_PARAMS["N"]
    xA_size = TEST_PARAMS["xA_size"]
    xM_size = TEST_PARAMS["xM_size"]
    yB_size = TEST_PARAMS["yB_size"]
    yN_size = TEST_PARAMS["yN_size"]
    W = TEST_PARAMS['W']
    xM_yN_size = xM_size * yN_size // N
    Nx = N // yN_size
    Ny = N // xM_size

    # Instantiate
    swiftly = Swiftly(
        N, yN_size, xM_size, W
    )

    # Parameters to try
    source_lists = [
        [(1, 0, 0)],
        [(1, 20, 4)],
        [(2, 2, 5)],
        [(3, -5, 4)],
    ]
    sg_offs = [[0, 0], [0, Nx], [Nx, 0], [-Nx, -Nx]]
    facet_offs = [[0, 0], [Ny, Ny], [-Ny, Ny], [0, -Ny]]

    # Test linearity with different values, and start with subgrids at
    # different (valid) subgrid offsets.
    for sources, sg_off in itertools.product(source_lists, sg_offs):
        # Generate subgrid. As we are only filling the grid partially
        # here, we have to scale it.
        subgrid = (
            make_subgrid_from_sources(sources, N, xA_size, sg_off)
            / xA_size
            / xA_size
            * N
            * N
        )
        prepped = pad_mid(pad_mid(subgrid, xM_size, 0), xM_size, 1)
        swiftly.prepare_subgrid_inplace_2d(prepped, sg_off[0], sg_off[1])

        # Check different facet offsets
        for facet_off in facet_offs:
            extracted0 = numpy.empty((xM_yN_size, xM_size), dtype=complex)
            swiftly.extract_from_subgrid(
                prepped.T, extracted0.T, facet_off[0])
            extracted1 = numpy.empty((xM_yN_size, xM_yN_size), dtype=complex)
            swiftly.extract_from_subgrid(
                extracted0, extracted1, facet_off[1])
            
            accumulated0 = numpy.zeros((yN_size, xM_yN_size), dtype=complex)
            swiftly.add_to_facet(
                extracted1.T, accumulated0.T, sg_off[0]
            )
            accumulated1 = numpy.zeros((yN_size, yN_size), dtype=complex)
            swiftly.add_to_facet(
                accumulated0, accumulated1, sg_off[1]
            )
            facet0 = numpy.empty((yB_size, yN_size), dtype=complex)
            swiftly.finish_facet(
                accumulated1.T, facet0.T, facet_off[0]
            )
            facet1 = numpy.empty((yB_size, yB_size), dtype=complex)
            swiftly.finish_facet(
                facet0, facet1, facet_off[1]
            )

            # Check that pixels in questions have correct value. As -
            # again - we have only partially filled the grid, the only
            # value we can really check is the (singular) one we set
            # previously.
            expected = make_facet_from_sources(sources, N, yB_size, facet_off)
            numpy.testing.assert_array_almost_equal(
                facet1[expected != 0], expected[expected != 0]
            )
            numpy.testing.assert_array_less(
                facet1[expected == 0], numpy.max(expected)
            )
