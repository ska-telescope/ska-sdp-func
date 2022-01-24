.. _install_guide:

******************
Installation Guide
******************

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
