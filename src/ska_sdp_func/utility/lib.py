# See the LICENSE file at the top-level directory of this distribution.

"""Private module to find and return a handle to the compiled library."""

import ctypes
import glob
import os
import threading
from typing import Any, Callable

from .error_checking import error_checking, ERROR_CODE_ARGTYPE

# We don't want a module-level variable for the library handle,
# as that would mean the documentation doesn't build properly if the library
# can't be found when it gets imported - and unfortunately nested packages
# trigger a bug with mock imports.
# We use the 'Lib' class instead to hold the library handle as a static member


class LibMeta(type):
    """
    Metaclass for Lib, its only purpose it to customise the behaviour of
    __getattr__ on Lib. We want to wrap C functions as follows:

    >>> Lib.wrap_func(func_name, restype=..., argtypes=[...])

    We then want to access wrapped C functions as follows:

    >>> Lib.func_name

    Also, we want to delay the C function wrapping to the moment when the user
    tries to access it as above. That is to work around an issue with the docs
    build, during which the library may not exist yet. We thus cannot refer to
    Lib.handle() at import time.

    If the attribute lookup on 'Lib.func_name' fails, it means that the C
    function 'func_name' has not been wrapped yet. __getattr__ gets called by
    Python when an attribute lookup fails, and we make sure it triggers the
    function wrapping.
    """

    def __getattr__(cls, name: str) -> Callable:
        return cls._get_func(name)


class Lib(metaclass=LibMeta):
    """Class that conveniently exposes the C functions of the library.

    To make a given function available, it must first be wrapped explicitly by
    calling Lib.wrap_func() as follows:

    >>> Lib.wrap_func(my_function, restype=..., argtypes=[...])

    This has to be done just once. Then one may call the function as follows:

    >>> Lib.my_function(...)
    """

    name = "libska_sdp_func"
    env_name = "SKA_SDP_FUNC_LIB_DIR"
    this_dir = os.path.dirname(os.path.abspath(__file__))
    package_root = os.path.join(this_dir, "..", "..")
    search_dirs = [".", package_root, "/usr/local/lib"]
    lib = None
    mutex = threading.Lock()

    # Used to delay the wrapping of a C function until the first
    # time it is called.
    _wrap_function_args: dict[str, tuple] = {}

    @staticmethod
    def handle():
        """Return a handle to the library for use by ctypes."""
        if not Lib.lib:
            lib_path = Lib.find_lib(Lib.search_dirs)
            with Lib.mutex:
                if not Lib.lib:
                    try:
                        Lib.lib = ctypes.CDLL(lib_path)
                    except OSError:
                        pass
            if not Lib.lib:
                raise RuntimeError(
                    f"Cannot find {Lib.name} in {Lib.search_dirs}. "
                    f"Try setting the environment variable {Lib.env_name}"
                )
        return Lib.lib

    @staticmethod
    def find_lib(lib_search_dirs):
        """Try to find the shared library in the listed directories."""
        lib_path = ""
        env_dir = os.environ.get(Lib.env_name)
        if env_dir and env_dir not in Lib.search_dirs:
            lib_search_dirs.insert(0, env_dir)
        for test_dir in lib_search_dirs:
            lib_list = glob.glob(os.path.join(test_dir, Lib.name) + "*")
            if lib_list:
                lib_path = os.path.abspath(lib_list[0])
                break
        return lib_path

    @staticmethod
    def wrap_func(func_name: str, *, restype: Any, argtypes: list, check_errcode=False) -> None:
        """Convenience function to wrap a C function from the library and make
        it callable from Python.

        Args:
            func_name: The name of the C function to be wrapped
            restype: The type of the result returned by the C function.
                Must be a single ctypes type (e.g. ctypes.c_double),
                or None if the C function returns void.
            argtypes: list of the the argument types taken by the C function.
                See the ctypes docs or the developer guide for what is acceptable as an argument
                type (this includes, but is not limited to ctypes types such as ctypes.c_double).
                NOTE: for functions that expect an error code pointer as their last argument,
                    do NOT specify its type in argtypes. Use check_errcode=True instead.
            check_errcode: if True, it will be assumed that the C function has an additional
                argument in last place used to pass a c_int pointer that represents an error code.
                The C function will be wrapped in an extra layer that does error code checking
                automatically, and raise a Python exception if it is non-zero.
        """
        # Store params to perform the actual wrapping later, when the user
        # requests the function for the first time.
        Lib._wrap_function_args[func_name] = (restype, argtypes, check_errcode)

    @staticmethod
    def _wrap_func(func_name: str) -> None:
        """Does the actual wrapping, makes the function accessible via:
        >>> Lib.func_name
        """
        try:
            func = getattr(Lib.handle(), func_name)
        except AttributeError as err:
            msg = f"The C library does not expose a function named {func_name!r}"
            raise AttributeError(msg) from err

        try:
            restype, argtypes, check_errcode = Lib._wrap_function_args[func_name]
        except KeyError as err:
            msg = f"The wrapping details for {func_name!r} have not been defined"
            raise KeyError(msg) from err

        func.restype = restype
        func.argtypes = argtypes
        if check_errcode:
            func.argtypes.append(ERROR_CODE_ARGTYPE)
            func = error_checking(func)
        setattr(Lib, func_name, func)

    @staticmethod
    def _get_func(func_name: str) -> Callable:
        # We get here only when the attribute lookup 'Lib.func_name' failed
        Lib._wrap_func(func_name)
        return getattr(Lib, func_name)
