# See the LICENSE file at the top-level directory of this distribution.

"""Module to wrap sky coordinates for passing to processing functions."""

import ctypes

try:
    import astropy.coordinates
except:
    astropy = None

from .error import Error
from .lib import Lib


class SkyCoord:
    """Class to wrap sky coordinates for passing to processing functions."""

    class Handle(ctypes.Structure):
        """Class handle for use by ctypes."""

    def __init__(self, *args):
        """Create a new sky coordinate object.

        The arguments are the coordinate type as a string,
        and up to three coordinate values coord0, coord1, coord2.

        Alternatively, an astropy SkyCoord object can be passed instead.

        The default epoch value is 2000.0, but can be set using
        :meth:`set_epoch`.
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
            Error.handle_type(),
        ]
        if len(args) == 1 and astropy:
            if isinstance(args[0], astropy.coordinates.SkyCoord):
                astropy_coord = args[0]
                if astropy_coord.frame.name == "icrs":
                    self._handle = sky_coord_create(
                        "icrs".encode("ascii"),
                        astropy_coord.ra.rad,
                        astropy_coord.dec.rad,
                        0.0,
                        error_status.handle()
                    )
                else:
                    raise RuntimeError("Unknown astropy coordinate frame")
            else:
                raise RuntimeError(
                        "Object is not of type astropy.coordinates.SkyCoord")
        elif len(args) >= 3:
            self._handle = sky_coord_create(
                args[0].encode("ascii"),
                args[1],
                args[2],
                args[3] if len(args) >= 4 else 0.0,
                error_status.handle(),
            )
        else:
            raise RuntimeError("Unknown construction method for SkyCoord")

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

    def epoch(self):
        """Returns the value of the coordinate epoch.

        :return: Value of the coordinate epoch.
        :rtype: float
        """
        if self._handle:
            sky_coord_epoch = Lib.handle().sdp_sky_coord_epoch
            sky_coord_epoch.argtypes = [SkyCoord.handle_type()]
            sky_coord_epoch.restype = ctypes.c_double
            return float(sky_coord_epoch(self._handle))
        return 0

    def value(self, dim):
        """Returns the value of the selected coordinate.

        :param dim: Coordinate dimension index (starting 0; max 2).
        :type dim: int

        :return: Value of specified coordinate.
        :rtype: float
        """
        if self._handle:
            sky_coord_value = Lib.handle().sdp_sky_coord_value
            sky_coord_value.argtypes = [SkyCoord.handle_type(), ctypes.c_int32]
            sky_coord_value.restype = ctypes.c_double
            return float(sky_coord_value(self._handle, ctypes.c_int32(dim)))
        return 0

    def set_epoch(self, epoch):
        """Sets the coordinate epoch value.

        :param epoch: Value of coordinate epoch.
        :type epoch: float
        """
        if self._handle:
            sky_coord_set_epoch = Lib.handle().sdp_sky_coord_set_epoch
            sky_coord_set_epoch.argtypes = [
                SkyCoord.handle_type(),
                ctypes.c_double,
            ]
            sky_coord_set_epoch(self._handle, ctypes.c_double(epoch))

    def type(self):
        """Returns the coordinate type string.

        :return: String describing coordinate type.
        :rtype: str
        """
        if self._handle:
            sky_coord_type = Lib.handle().sdp_sky_coord_type
            sky_coord_type.argtypes = [SkyCoord.handle_type()]
            sky_coord_type.restype = ctypes.c_char_p
            return sky_coord_type(self._handle).decode()
        return ""
