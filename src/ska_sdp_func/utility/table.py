# See the LICENSE file at the top-level directory of this distribution.

"""Module to wrap xarray Datasets for passing to processing functions."""

import ctypes

try:
    import xarray
except ImportError:
    xarray = None

from ..utility import Lib, Mem, StructWrapper


class Table(StructWrapper):
    """Class to wrap an xarray Dataset for passing to processing functions."""

    def __init__(self, *args):
        """Create a new wrapper for a structure of arrays."""
        super().__init__(Lib.sdp_table_create, (), Lib.sdp_table_free)
        obj = args[0] if len(args) == 1 else None
        if xarray and isinstance(obj, xarray.Dataset):
            for name, array in obj.data_vars.items():
                Lib.sdp_table_set_column(
                    self, name.encode("ascii"), Mem(array)
                )
            for name, array in obj.coords.items():
                Lib.sdp_table_set_column(
                    self, name.encode("ascii"), Mem(array)
                )


Lib.wrap_func(
    "sdp_table_create",
    restype=Table.handle_type(),
    argtypes=[],
)


Lib.wrap_func(
    "sdp_table_set_column",
    restype=None,
    argtypes=[Table.handle_type(), ctypes.c_char_p, Mem.handle_type()],
)


Lib.wrap_func("sdp_table_free", restype=None, argtypes=[Table.handle_type()])
