# See the LICENSE file at the top-level directory of this distribution.

"""Support error codes returned by processing functions."""

import ctypes


class Error:
    """Wrapper for error codes."""

    error_codes = {
        0: "No error",
        1: "Generic runtime error",
        2: "Invalid function argument",
        3: "Unsupported data type(s)",
        4: "Memory allocation failure",
        5: "Memory copy failure",
        6: "Memory location mismatch",
    }

    def __init__(self):
        """Create an error code wrapper for passing to a processing function."""
        self._error = ctypes.c_int(0)

    def check(self):
        """Check if an error occurred and raise a Python exception if needed."""
        if self._error.value != 0:
            raise RuntimeError(Error.error_codes[self._error.value])

    def handle(self):
        """Return a handle for use by ctypes in a function call."""
        return ctypes.byref(self._error)

    @staticmethod
    def handle_type():
        """Return the type of the handle for use in the argtypes list."""
        return ctypes.POINTER(ctypes.c_int)
