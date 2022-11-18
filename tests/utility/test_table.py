# See the LICENSE file at the top-level directory of this distribution.

"""Test Table functions."""

import numpy

from ska_sdp_func.utility import Lib, Table

try:
    import xarray
except ImportError:
    xarray = None


def test_dataset():
    """Test use of xarray Dataset."""

    # Define dimension sizes.
    num_times = 10
    num_baselines = 5
    num_channels = 3
    num_pols = 4
    vis_shape = (num_times, num_baselines, num_channels, num_pols)
    uvw_shape = (num_times, num_baselines, 3)

    # Create empty numpy arrays.
    vis = numpy.zeros(vis_shape, dtype=numpy.complex128)
    uvw = numpy.zeros(uvw_shape, dtype=numpy.float64)

    # Create a xarray Dataset to wrap the arrays.
    if xarray:
        data_vars = {
            "vis": (["time", "baseline", "frequency", "polarisation"], vis),
            "uvw": (["time", "baseline", "spatial"], uvw)
        }
        dataset = xarray.Dataset(data_vars=data_vars)

        # Print dataset before and after function call.
        print("Before function call:")
        print(dataset.uvw)

        Lib.sdp_function_example_table(Table(dataset))

        print("After function call:")
        print(dataset.uvw)



Lib.wrap_func(
    "sdp_function_example_table",
    restype=None,
    argtypes=[Table.handle_type()],
    check_errcode=True,
)
