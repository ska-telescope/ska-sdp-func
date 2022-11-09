# See the LICENSE file at the top-level directory of this distribution.

"""Module to wrap sky coordinates for passing to processing functions."""

import ctypes

try:
    import astropy
    import astropy.coordinates
except ImportError:
    astropy = None

from .lib import Lib
from .struct_wrapper import StructWrapper


class SkyCoord(StructWrapper):
    """Class to wrap sky coordinates for passing to processing functions."""

    def __init__(self, *args):
        """Create a new sky coordinate object.

        The arguments are the coordinate type as a string,
        and up to three coordinate values coord0, coord1, coord2.

        Alternatively, an existing SkyCoord or an astropy SkyCoord object
        can be passed instead.

        The default epoch value is 2000.0, but can be set using
        :meth:`set_epoch`.
        """
        create_args = tuple()
        if len(args) == 1:
            if isinstance(args[0], SkyCoord):
                other = args[0]
                # Copy of an existing SkyCoord.
                create_args = (
                    other.type().encode("ascii"),
                    other.value(0),
                    other.value(1),
                    other.value(2),
                )
            elif astropy:
                if isinstance(args[0], astropy.coordinates.SkyCoord):
                    astropy_coord = args[0]
                    if astropy_coord.frame.name == "icrs":
                        create_args = (
                            "icrs".encode("ascii"),
                            astropy_coord.ra.rad,
                            astropy_coord.dec.rad,
                            0.0,
                        )
                    else:
                        raise RuntimeError("Unknown astropy coordinate frame")
            else:
                raise RuntimeError("Unknown object passed to SkyCoord constructor")
        elif len(args) >= 3:
            create_args = (
                args[0].encode("ascii"),
                args[1],
                args[2],
                args[3] if len(args) >= 4 else 0.0,
            )
        else:
            raise RuntimeError("Unknown construction method for SkyCoord")
        super().__init__(Lib.sdp_sky_coord_create, create_args, Lib.sdp_sky_coord_free)

    def epoch(self) -> float:
        """Returns the value of the coordinate epoch"""
        return Lib.sdp_sky_coord_epoch(self)

    def value(self, dim: int) -> float:
        """Returns the value of the selected coordinate.

        Args:
            dim: Coordinate dimension index (starting 0; max 2).

        Returns:
            Value of specified coordinate, or 0 if 'dim' is out of bounds.
        """
        return Lib.sdp_sky_coord_value(self, dim)

    def set_epoch(self, epoch: float) -> None:
        """Sets the coordinate epoch value.

        Args:
            epoch: Value of coordinate epoch.
        """
        Lib.sdp_sky_coord_set_epoch(self, epoch)

    def type(self) -> str:
        """Returns the coordinate type string"""
        return Lib.sdp_sky_coord_type(self).decode()  # convert bytes to str


Lib.wrap_func(
    "sdp_sky_coord_create",
    restype=SkyCoord.handle_type(),
    argtypes=[
        ctypes.c_char_p,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
    ],
)

Lib.wrap_func(
    "sdp_sky_coord_free",
    restype=None,
    argtypes=[SkyCoord.handle_type()],
)

Lib.wrap_func(
    "sdp_sky_coord_epoch",
    restype=ctypes.c_double,
    argtypes=[SkyCoord.handle_type()],
)

Lib.wrap_func(
    "sdp_sky_coord_value",
    restype=ctypes.c_double,
    argtypes=[SkyCoord.handle_type(), ctypes.c_int32],
)

Lib.wrap_func(
    "sdp_sky_coord_set_epoch",
    restype=None,
    argtypes=[
        SkyCoord.handle_type(),
        ctypes.c_double,
    ],
)

Lib.wrap_func(
    "sdp_sky_coord_type",
    restype=ctypes.c_char_p,
    argtypes=[SkyCoord.handle_type()],
)
