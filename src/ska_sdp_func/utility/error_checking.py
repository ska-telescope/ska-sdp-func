# See the LICENSE file at the top-level directory of this distribution.

"""Utilities to automatically check whether wrapped C functions set a
non-zero error code, and raise a custom exception if they do."""

import ctypes
from typing import Callable, Dict

ERROR_CODE_ARGTYPE: type = ctypes.POINTER(ctypes.c_int)

ERROR_CODE_MEANING: Dict[int, str] = {
    0: "No error",
    1: "Generic runtime error",
    2: "Invalid function argument",
    3: "Unsupported data type(s)",
    4: "Memory allocation failure",
    5: "Memory copy failure",
    6: "Memory location mismatch",
}

DEFAULT_ERROR_MSG = "Unknown error"


class CError(Exception):
    """
    Raised when a wrapped C function sets an error code.
    """


def error_checking(lib_func: Callable) -> Callable:
    """
    Decorator to enable error checking on a C function of the library.
    The error code (c_int) is expected to be passed by pointer as the
    *last* argument of the C function.
    """

    def wrapped(*args):
        code = ctypes.c_int(0)
        result = lib_func(*args, ctypes.byref(code))
        if code:
            val = code.value
            msg = ERROR_CODE_MEANING.get(val, DEFAULT_ERROR_MSG)
            raise CError(f"Error {val}: {msg}")
        return result

    return wrapped
