# See the LICENSE file at the top-level directory of this distribution.

"""Module to describe sky coordinates for passing to processing functions."""

import ctypes

from .error import Error
from .lib import Lib

class SkyCoord:
    class Handle(ctypes.Structure):
        """Class handle for use by ctypes."""
    
    def __init__(self, coordinate_type, epoch, coord_0, coord_1, coord_2):
        """Create a new sky coordinates object.

        The arguments are coordinate type as string, epoch, and three coordinate values coord_0, coord_1, coord_2.
        """
        self._handle = None
        error_status = Error()
        sky_coord_create = Lib.handle().sdp_sky_coord_create
        sky_coord_create.restype = SkyCoord.handle_type()
        sky_coord_create.argtypes = [
            ctypes.c_char_p,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            Error.handle_type(),
        ]
        self._handle = sky_coord_create(
            coordinate_type.encode('ascii'),
            epoch,
            coord_0,
            coord_1,
            coord_2,
            error_status.handle()
        )
        
    def __del__(self):
        """Called when the handle is destroyed."""
        if self._handle:
            sky_coord_free = Lib.handle().sdp_sky_coord_free
            sky_coord_free.argtypes = [SkyCoord.handle_type()]
            sky_coord_free(self._handle)

    def handle(self):
        """Return a handle for use by ctypes in a function call."""
        return self._handle

    @staticmethod
    def handle_type():
        """Return the type of the handle for use in the argtypes list."""
        return ctypes.POINTER(SkyCoord.Handle)