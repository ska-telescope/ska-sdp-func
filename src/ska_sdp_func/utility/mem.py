# See the LICENSE file at the top-level directory of this distribution.

"""Module to wrap arrays for passing to processing functions."""

import ctypes

import numpy

try:
    import cupy
except ImportError:
    cupy = None

from .lib import Lib
from .struct_wrapper import StructWrapper


class Mem(StructWrapper):
    """Class to wrap arrays for passing to processing functions"""

    class MemType:
        """Enumerator to hold memory element type."""

        SDP_MEM_VOID = 0
        SDP_MEM_CHAR = 1
        SDP_MEM_INT = 2
        SDP_MEM_FLOAT = 4
        SDP_MEM_DOUBLE = 8
        SDP_MEM_COMPLEX_FLOAT = 36
        SDP_MEM_COMPLEX_DOUBLE = 40

    class MemLocation:
        """Enumerator to hold memory location."""

        SDP_MEM_CPU = 0
        SDP_MEM_GPU = 1

    def __init__(self, *args):
        """Create a new wrapper for an array.

        The array to wrap (either a numpy or cupy array) should be
        passed as the first argument.
        """
        obj = args[0] if len(args) == 1 else None
        if isinstance(obj, numpy.ndarray):
            if obj.dtype in (numpy.int8, numpy.byte):
                mem_type = self.MemType.SDP_MEM_CHAR
            elif obj.dtype == numpy.int32:
                mem_type = self.MemType.SDP_MEM_INT
            elif obj.dtype == numpy.float32:
                mem_type = self.MemType.SDP_MEM_FLOAT
            elif obj.dtype == numpy.float64:
                mem_type = self.MemType.SDP_MEM_DOUBLE
            elif obj.dtype == numpy.complex64:
                mem_type = self.MemType.SDP_MEM_COMPLEX_FLOAT
            elif obj.dtype == numpy.complex128:
                mem_type = self.MemType.SDP_MEM_COMPLEX_DOUBLE
            else:
                raise TypeError("Unsupported type of numpy array")
            shape = (ctypes.c_int64 * obj.ndim)(*obj.shape)
            strides = (ctypes.c_int64 * obj.ndim)(*obj.strides)
            create_args = (
                obj.ctypes.data_as(ctypes.POINTER(ctypes.c_void_p)),
                mem_type,
                self.MemLocation.SDP_MEM_CPU,
                obj.ndim,
                shape,
                strides,
            )
            super().__init__(
                Lib.sdp_mem_create_wrapper, create_args, Lib.sdp_mem_free
            )
            Lib.sdp_mem_set_read_only(self, not obj.flags.writeable)

        elif cupy and isinstance(obj, cupy.ndarray):
            if obj.dtype in (cupy.int8, cupy.byte):
                mem_type = self.MemType.SDP_MEM_CHAR
            elif obj.dtype == cupy.int32:
                mem_type = self.MemType.SDP_MEM_INT
            elif obj.dtype == cupy.float32:
                mem_type = self.MemType.SDP_MEM_FLOAT
            elif obj.dtype == cupy.float64:
                mem_type = self.MemType.SDP_MEM_DOUBLE
            elif obj.dtype == cupy.complex64:
                mem_type = self.MemType.SDP_MEM_COMPLEX_FLOAT
            elif obj.dtype == cupy.complex128:
                mem_type = self.MemType.SDP_MEM_COMPLEX_DOUBLE
            else:
                raise TypeError("Unsupported type of cupy array")
            shape = (ctypes.c_int64 * obj.ndim)(*obj.shape)
            strides = (ctypes.c_int64 * obj.ndim)(*obj.strides)
            create_args = (
                ctypes.cast(obj.data.ptr, ctypes.POINTER(ctypes.c_void_p)),
                mem_type,
                self.MemLocation.SDP_MEM_GPU,
                obj.ndim,
                shape,
                strides,
            )
            super().__init__(
                Lib.sdp_mem_create_wrapper, create_args, Lib.sdp_mem_free
            )
            # cupy doesn't appear to have a "writeable" flag.
            Lib.sdp_mem_set_read_only(self, 0)

        elif obj is None:
            shape = (ctypes.c_int64)(1)
            strides = (ctypes.c_int64)(1)
            create_args = (
                ctypes.c_void_p(),
                self.MemType.SDP_MEM_VOID,
                self.MemLocation.SDP_MEM_CPU,
                0,
                shape,
                strides,
            )
            super().__init__(
                Lib.sdp_mem_create_wrapper, create_args, Lib.sdp_mem_free
            )
        else:
            raise TypeError("Unsupported argument type")


Lib.wrap_func(
    "sdp_mem_create_wrapper",
    restype=Mem.handle_type(),
    argtypes=[
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_int64),
    ],
    check_errcode=True,
)


Lib.wrap_func(
    "sdp_mem_set_read_only",
    restype=None,
    argtypes=[Mem.handle_type(), ctypes.c_int32],
)


Lib.wrap_func("sdp_mem_free", restype=None, argtypes=[Mem.handle_type()])
