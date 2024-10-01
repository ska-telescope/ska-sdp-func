******************
Installation guide
******************

Python wheel
============

Pre-built Python wheels are available for Linux x86_64-based systems,
which include MKL as well as cuFFT.
To install the wheel from the SKA central artefact repository, use:

  .. code-block:: bash

     pip3 install --extra-index-url https://artefact.skao.int/repository/pypi-internal/simple ska-sdp-func

Uninstalling
------------

The Python package can be uninstalled using:

  .. code-block:: bash

     pip3 uninstall ska-sdp-func


Building from source
====================

If GPU acceleration is required, make sure the CUDA toolkit is installed first,
from https://developer.nvidia.com/cuda-downloads.

Similarly, if the Intel MKL library should be used to perform FFTs, install the
Intel oneAPI toolkit, from
https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html

The C library
-------------

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

- Use ``-DCMAKE_BUILD_TYPE=Release|Debug`` to specify whether the library
  should be built in release or debug mode.
  Release mode turns on all optimisations, while debug mode turns off
  optimisations and includes debugging symbols instead.
  The default value for this is ``Release``.

- Use ``-DFIND_CUDA=OFF|ON`` to specify whether or not CUDA should be used, if
  it is found at compilation time. The default value for this is ``ON``.

- Use ``-DFIND_MKL=OFF|ON`` to specify whether or not the Intel MKL library
  should be used (for FFTs), if it is found at compilation time.
  The default value for this is ``ON``.

- Use ``-DCUDA_ARCH="x.y"`` to compile CUDA code for the specified GPU
  architecture(s). The default value for this is all architectures
  from 6.0 to 8.6 (Pascal to Ampere). Multiple architectures should be
  separated by semi-colons.

- Use ``-DBUILD_INFO=ON`` to display information about the compiler
  and compilation flags used in the build.
  The default value for this is ``OFF``.

If using the Intel compiler, the following options will also need to be set:

- Use ``-DCMAKE_C_COMPILER=icx`` to specify that the Intel compiler
  should be used to compile C code.

- Use ``-DCMAKE_CXX_COMPILER=icpx`` to specify that the Intel compiler
  should be used to compile C++ code.

- Use ``-DNVCC_COMPILER_BINDIR=/usr/bin/g++`` to specify that NVCC should use
  ``g++`` for host code, since NVCC does not work with the Intel compiler.

The C unit tests can then be run from the same build directory:

  .. code-block:: bash

     ctest

The Python library
------------------

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

  If using the Intel compiler as well as CUDA, the following could be used:

  .. code-block:: bash

     export CMAKE_ARGS="-DCUDA_ARCH=8.0 -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx -DNVCC_COMPILER_BINDIR=/usr/bin/g++ -DBUILD_INFO=ON"

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
