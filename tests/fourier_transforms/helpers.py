"""
Helper routines, stolen from ska-sdp-distributed-fourier-transform
"""

from __future__ import annotations

from typing import List, Tuple

import numpy


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


def broadcast(a, dims, axis):
    """
    Stretch input array to shape determined by the dims and axis values.
    See tests for examples of how the shape of the input array will change
    depending on what dims-axis combination is given

    :param a: input numpy ndarray
    :param dims: dimensions to broadcast ("stretch") input array to; int
    :param axis: axis along which the new dimension(s) should be added; int

    :return: array with new shape
    """
    return a[create_slice(numpy.newaxis, slice(None), dims, axis)]


def make_facet_from_sources(
    sources: List[Tuple[float, int]],
    image_size: int,
    facet_size: int,
    facet_offsets: List[int],
    facet_masks: List[numpy.ndarray] = None,
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
    sources: List[tuple[float, int]],
    image_size: int,
    subgrid_size: int,
    subgrid_offsets: List[int],
    subgrid_masks: List[numpy.ndarray] = None,
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
