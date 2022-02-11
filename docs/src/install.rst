.. _install_guide:

******************
Installation Guide
******************

The C Library
=============

The processing function library should be compiled from source using CMake.
If GPU acceleration is required, make sure the CUDA toolkit is installed first.

From the top-level directory, run the following commands to compile and
install the library:

  .. code-block:: bash

     mkdir build
     cd build
     cmake ..
     make -j8
     make install

The C unit tests can then be run from the same build directory:

  .. code-block:: bash

     ctest

The Python Bindings
===================

The Python bindings are implemented using ctypes to call the compiled C
functions - this module is pure Python, so no compilation or external
packages are needed to install it.

After compiling the C library (above), from the top-level directory, run:

  .. code-block:: bash

     pip3 install .

to install the Python package.

If the C library is not installed into ``/usr/local/lib``, then its location
must be specified by setting the environment variable ``SKA_SDP_FUNC_LIB_DIR``.

The Python unit tests can then be run using `pytest <https://pytest.org>`_,
from the top-level directory:

  .. code-block:: bash

     pytest

Uninstalling
------------

The Python package can be uninstalled using:

  .. code-block:: bash

     pip3 uninstall ska-sdp-func
