# See the LICENSE file at the top-level directory of this distribution.

"""Test example function."""

import numpy

try:
    import cupy
except ImportError:
    cupy = None

from utility import SkyCoord
from ska_sdp_func import sky_function


def test_sky_coord():
    """Test vector addition function."""
    # Run vector add test on CPU, using numpy arrays.
    sky_coordinate = SkyCoord("Type01", 1.0, 0.0, 1.0, 2.0)
    sky_function(sky_coordinate)
    

