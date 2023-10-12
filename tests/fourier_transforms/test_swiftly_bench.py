"""
Benchmarks for swiftly.py functions.

Adapted from ska-sdp-distributed-fourier-transform
"""

import itertools

import numpy
import pytest
from helpers import (
    extract_mid,
    make_facet_from_sources,
    make_subgrid_from_sources,
    pad_mid,
)

from ska_sdp_func.fourier_transforms.swiftly import Swiftly

TEST_PARAMS = {
    "20k[1]-n5k-1k": dict(
        W=13.5625,
        fov=1,
        N=20480,
        Nx=128,
        yB_size=4160,
        yN_size=5120,
        yP_size=5120,
        xA_size=896,
        xM_size=1024,
    ),
    "16k[1]-n4k-1k": dict(
        W=13.5625,
        fov=1,
        N=16384,
        Nx=128,
        yB_size=3328,
        yN_size=4096,
        yP_size=4096,
        xA_size=896,
        xM_size=1024,
    ),
}


@pytest.mark.parametrize(
    "fn_to_bench",
    [
        "prepare_facet0",
        "prepare_facet1",
        "extract_from_facet0",
        "extract_from_facet1",
        "add_to_subgrid_2d",
        "finish_subgrid_inplace_2d",
    ],
)
@pytest.mark.parametrize("config", TEST_PARAMS.keys())
def test_facet_to_subgrid_2d_bench(fn_to_bench, config, benchmark):
    """Test facet to subgrid distributed FT primitives against direct
    Fourier transformation implementation -- 2D version
    """

    # Basic layout parameters
    N = TEST_PARAMS[config]["N"]
    xA_size = TEST_PARAMS[config]["xA_size"]
    xM_size = TEST_PARAMS[config]["xM_size"]
    yB_size = TEST_PARAMS[config]["yB_size"]
    yN_size = TEST_PARAMS[config]["yN_size"]
    W = TEST_PARAMS[config]["W"]
    xM_yN_size = xM_size * yN_size // N
    Nx = N // yN_size
    Ny = N // xM_size

    # Instantiate
    swiftly = Swiftly(N, yN_size, xM_size, W)

    def maybe_benchmark(fn_name, fn, *args, **kwargs):
        if fn_name == fn_to_bench:
            benchmark(fn, *args, **kwargs)
        else:
            fn(*args, **kwargs)

    # Test with different values and facet offsets
    sources = [(1, 20, 4), (2, 2, 5), (3, -5, -4)]
    facet_offs = [-Ny, Ny]

    # Set sources in facet
    facet = make_facet_from_sources(sources, N, yB_size, facet_offs)

    # We assume all sources are on the facet
    assert numpy.sum(facet) == sum(src[0] for src in sources)
    prepped0 = numpy.empty((yN_size, yB_size), dtype=complex)
    maybe_benchmark(
        "prepare_facet0",
        swiftly.prepare_facet,
        facet.T,
        prepped0.T,
        facet_offs[0],
    )
    prepped = numpy.empty((yN_size, yN_size), dtype=complex)
    maybe_benchmark(
        "prepare_facet1",
        swiftly.prepare_facet,
        prepped0,
        prepped,
        facet_offs[1],
    )

    # Now generate subgrids at different (valid) subgrid offsets.
    sg_offs = [-Nx, -Nx]
    subgrid_contrib0 = numpy.empty((xM_yN_size, yN_size), dtype=complex)
    maybe_benchmark(
        "extract_from_facet0",
        swiftly.extract_from_facet,
        prepped.T,
        subgrid_contrib0.T,
        sg_offs[0],
    )
    subgrid_contrib = numpy.empty((xM_yN_size, xM_yN_size), dtype=complex)
    maybe_benchmark(
        "extract_from_facet1",
        swiftly.extract_from_facet,
        subgrid_contrib0,
        subgrid_contrib,
        sg_offs[1],
    )

    subgrid_acc = numpy.zeros((xM_size, xM_size), dtype=complex)
    maybe_benchmark(
        "add_to_subgrid_2d",
        swiftly.add_to_subgrid_2d,
        subgrid_contrib,
        subgrid_acc,
        facet_offs[0],
        facet_offs[1],
    )

    maybe_benchmark(
        "finish_subgrid_inplace_2d",
        swiftly.finish_subgrid_inplace_2d,
        subgrid_acc,
        sg_offs[0],
        sg_offs[1],
    )
    subgrid = extract_mid(extract_mid(subgrid_acc, xA_size, 0), xA_size, 1)

    # Now check against DFT - except for the addition function, because that
    # one getting called multiple times won't produce the same result
    if fn_to_bench not in ["add_to_subgrid_2d", "finish_subgrid_inplace_2d"]:
        expected = make_subgrid_from_sources(sources, N, xA_size, sg_offs)
        numpy.testing.assert_array_almost_equal(subgrid, expected)
