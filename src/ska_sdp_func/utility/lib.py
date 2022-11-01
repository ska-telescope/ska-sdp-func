# See the LICENSE file at the top-level directory of this distribution.

"""Private module to find and return a handle to the compiled library."""

import ctypes
import glob
import os
import threading

# We don't want a module-level variable for the library handle,
# as that would mean the documentation doesn't build properly if the library
# can't be found when it gets imported - and unfortunately nested packages
# trigger a bug with mock imports. We use this static class instead
# to hold the library handle.


class Lib:
    """Class to hold a handle to the compiled library."""

    name = "libska_sdp_func"
    env_name = "SKA_SDP_FUNC_LIB_DIR"
    this_dir = os.path.dirname(os.path.abspath(__file__))
    package_root = os.path.join(this_dir, "..", "..")
    search_dirs = [".", package_root, "/usr/local/lib"]
    lib = None
    mutex = threading.Lock()

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
