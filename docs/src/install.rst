******************
Installation guide
******************

If GPU acceleration is required, make sure the CUDA toolkit is installed first.

The C library
=============

The processing function library is compiled from source using CMake.

From the top-level directory, run the following commands to compile and
install the library:

  .. code-block:: bash

     mkdir build
     cd build
     cmake .. [OPTIONS]
     make -j16
     make install

The CMake options are as follows:

- Use ``-DFIND_CUDA=OFF|ON`` to specify whether or not CUDA should be used.
  The default value for this is ``ON``.

- Use ``-DCUDA_ARCH="x.y"`` to compile CUDA code for the specified GPU
  architecture(s). The default value for this is all architectures
  from 6.0 to 8.6 (Pascal to Ampere). Multiple architectures should be
  separated by semi-colons.

The C unit tests can then be run from the same build directory:

  .. code-block:: bash

     ctest

The Python library
==================

To control the build, the following environment variables can be used.

- Use ``CMAKE_BUILD_PARALLEL_LEVEL`` to specify the maximum number of
  simultaneous jobs to launch during the build.
  To utilise all cores of a 16-core CPU when building the package, use:

  .. code-block:: bash

     export CMAKE_BUILD_PARALLEL_LEVEL=16

- Use ``CMAKE_ARGS`` to pass down extra arguments to the CMake step,
  like the CUDA architecture as described above.
  For example, to build GPU code only for the Ampere A100 architecture, use:

  .. code-block:: bash

     export CMAKE_ARGS="-DCUDA_ARCH=8.0"

From the top-level directory, run the following commands to install
the Python package:

  .. code-block:: bash

     pip3 install .

The compiled library will be built as part of this step, so it does not need to
be installed separately.

The Python unit tests can then be run using `pytest <https://pytest.org>`_,
from the top-level directory:

  .. code-block:: bash

     pytest

Uninstalling
------------

The Python package can be uninstalled using:

  .. code-block:: bash

     pip3 uninstall ska-sdp-func
