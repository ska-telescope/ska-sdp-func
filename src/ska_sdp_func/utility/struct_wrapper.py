# See the LICENSE file at the top-level directory of this distribution.

"""Base class for any object that wraps a C struct."""

import ctypes
from typing import Any, Callable, Optional


class StructWrapper:
    """
    Base class for any object that wraps a C struct. Its role consists of:
    * Wrapping and managing the lifetime of a handle to a C struct
    * Defining a distinct ctypes.Structure subclass to represent said handle
        when passing it to wrapped C functions. That way, it's not possible
        to accidentally pass the wrong sub-type of StructWrapper to a C
        function without ctypes raising an Exception.

    Properties:
        _as_parameter_: hook used by ctypes internally (see ctypes docs).
            Returns the wrapped handle. This allows to directly pass any
            instance of this class to wrapped C functions.

    Classmethods:
        handle_type: the type of the handle, to be specified when defining the
            argtypes of a function that takes this class as an argument.
    """

    _HANDLE_CLASS: Optional[type] = None
    """
    Trivial ctypes.Structure subclass that will be used to represent
    any derived class when calling wrapped C functions. Each derived
    class gets assigned a distinct handle class.
    """

    def __init_subclass__(cls) -> None:
        cls._HANDLE_CLASS = type(
            f"{cls.__name__}Handle", (ctypes.Structure,), {}
        )

    def __init__(
        self,
        create_func: Callable,
        create_args: tuple,
        free_func: Callable[[Any], None],
    ) -> None:
        """
        Create a StructWrapper instance.

        Args:
            create_func: Function to call to create the underlying C struct
            create_args: Arguments to pass to create_func
            free_func: Function to be call when freeing the memory allocated
                to the underlying C struct when this object gets deleted.
                Must take the created handle as its single argument.

        Raises:
            RuntimeError: if the allocation of the C struct failed returns a
                null pointer.
        """
        if not callable(free_func):
            raise ValueError("free_func must be callable")

        self._handle = None
        self._free_func = free_func

        handle = create_func(*create_args)

        # If the creation failed silently, this could result in a null pointer,
        # in which case handle=None
        if not handle:
            raise RuntimeError(
                "Cannot initialise struct wrapper: creation function for "
                f"{type(self).__name__} handle returned a null pointer"
            )
        self._handle = handle

    def __del__(self):
        if self._handle:
            self._free_func(self._handle)
        self._handle = None

    @property
    def _as_parameter_(self):
        return self._handle

    @classmethod
    def handle_type(cls) -> type:
        """Return the type of the handle for use in the argtypes list."""
        return ctypes.POINTER(cls._HANDLE_CLASS)
