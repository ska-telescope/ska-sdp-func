
.. |br| raw:: html

   <br /><br />


********************
Adding new functions
********************

To add a new function to the processing function library, the
following notes can be used as a guide.

In C or C++
===========

1. Create a new file and write the function in C or C++.

   - The file should be saved in a suitable place in the repository
     under ``src/ska-sdp-func`` - please create a new subdirectory here if
     necessary.
     The naming convention is ``sdp_<function_name>.cpp`` or
     ``sdp_<function_name>.c`` (where ``<function_name>`` should be replaced
     with something appropriate).

   - Processing functions should not normally allocate memory themselves,
     so all input and output arrays should be passed as
     wrapped :cpp:struct:`sdp_Mem` pointers via the function arguments.
     This will allow the function to be used in a variety of contexts.

   - Do not use global variables to implement the function.

   - Array properties should be checked to make sure they are as expected -
     the following utility functions can be used for this:

      - :cpp:func:`sdp_mem_type`
      - :cpp:func:`sdp_mem_location`
      - :cpp:func:`sdp_mem_is_c_contiguous`
      - :cpp:func:`sdp_mem_is_complex`
      - :cpp:func:`sdp_mem_is_read_only`
      - :cpp:func:`sdp_mem_num_dims`
      - :cpp:func:`sdp_mem_shape_dim`
      - :cpp:func:`sdp_mem_stride_bytes_dim`
      - :cpp:func:`sdp_mem_stride_elements_dim`

   - If an :cpp:enum:`sdp_Error` error code is passed to the function,
     check it first, and only proceed if it is zero.

     .. warning::
      The :cpp:enum:`sdp_Error` argument, if any, must be passed *in last place*
      by convention; the Python wrapping code depends on this.

   - If a problem is encountered while making the checks, set the error
     code passed to the function, report a suitable message, and return.
     Errors can be reported using :any:`SDP_LOG_ERROR`, which takes a
     printf-style format string and arguments, and automatically adds the
     required fields needed to comply with the SKA logging standard.

   - After making the necessary checks on the function arguments, pull the
     pointer to the start of each array out of each wrapper using
     :cpp:func:`sdp_mem_data`, and cast to the appropriate pointer type.

   - It may be convenient to call other private functions in the file to
     implement the algorithm, which could (for example) use C++ templates
     to work with different floating-point data types.

   - In the local ``CMakeLists.txt`` file, add the name of the new source
     file to the list of sources used to build the library.
     |br|

2. Write a header file to expose the public function prototype.

   - Save the header in the same location in the repository as the source file,
     and remember to ``#include`` it there.
     The ``#include`` should use the relative path to the header in quotes,
     but omit the top-level ``src/`` prefix.

   - Document the function and its arguments in the header,
     using Doxygen-style comments.

   - A template header for a function that takes one input and one output array
     might look as follows:

   .. code-block:: C

      /* See the LICENSE file at the top-level directory of this distribution. */

      #ifndef SKA_SDP_PROC_FUNC_NAME_H_  /* (Use the function name here) */
      #define SKA_SDP_PROC_FUNC_NAME_H_

      /**
       * @file sdp_function_name.h
       *       (Change this to match the name of the header file)
       */

      #include "ska-sdp-func/utility/sdp_mem.h"

      #ifdef __cplusplus
      extern "C" {
      #endif

      /**
       * @brief Brief description of the function.
       *
       * Detailed description of the function, and its inputs and outputs.
       *
       * @param input Description of input array.
       * @param output Description of output array.
       * @param status Error status.
       */
      void sdp_function_name(
              const sdp_Mem* input,
              sdp_Mem* output,
              sdp_Error* status);

      #ifdef __cplusplus
      }
      #endif

      #endif /* include guard */

3. (Optional) If implementing a GPU version of the function, write the
   required CUDA kernel(s) in another new file.

   - The file name should be based on that used for the C/C++ code, but end in
     ``.cu`` (instead of ``.cpp`` or ``.c``).
     Save the CUDA kernels in the same directory as the other source files
     used to implement the function.

   - Use the :any:`SDP_CUDA_KERNEL` macro at the end of the ``.cu`` file
     to make the name(s) of the kernel(s) known to the library.

   - In the C/C++ code, use :cpp:func:`sdp_mem_location` to check if the
     arrays passed to the function are in GPU memory.
     If they are, launch the CUDA kernel(s) using
     :cpp:func:`sdp_launch_cuda_kernel`, specifying the name of the kernel
     given to :any:`SDP_CUDA_KERNEL`, pointers to the kernel arguments,
     and its launch configuration.
     For arrays in GPU memory, use :cpp:func:`sdp_mem_gpu_buffer` to get
     a pointer to the start of the array for the kernel argument list.

   - In the local ``CMakeLists.txt`` file, add the name of the new ``.cu``
     file to the list of CUDA kernel sources used to build the library.
     |br|

4. Write a unit test to exercise the new function.

   - The source file for the test should be called ``test_<function_name>.cpp``
     and placed in an appropriate subdirectory of the ``tests`` directory.
     This will be used to build a self-contained test executable for that
     function.

   - Test the new function as much as possible. If it supports multiple
     data types and data locations, test all options which are expected to
     work.
     Try to test the unhappy paths as well, to check that they fail as
     expected.

   - In the local ``CMakeLists.txt`` file, add the root name of the test file
     (without the directory name or ``.cpp`` extension) to the list of tests.
     |br|

5. Re-build, re-test, and re-install the library. From the build directory:

   .. code-block:: bash

      make
      make test
      make install


In Python
=========

The compiled function should usually be exposed in a Python module to allow
it to be used easily from Python scripts. A utility class is provided which can
wrap either numpy arrays or cupy arrays, passing them directly to our
processing functions without needlessly copying data.

1. Inside the directory ``src/ska_sdp_func/``, find an appropriate place
   to add the Python function. In many cases you may want to simply create a
   new Python source file.

   - At the top of the file, import the Python utility classes.
     It may be necessary to ``import ctypes`` as well, depending on the
     parameters needed by the function.

     .. code-block:: Python

        import ctypes
        from ..utility import Lib, Mem

   - Declare a Python function, giving it a suitable name and specifying
     parameters in the usual way.
     Remember to pass output arrays as parameters, too.

   - Add a Python docstring to describe the function, and its
     inputs and outputs.

   - In the Python function, the first thing we need to do is wrap the
     arrays, storing the pointer to the underlying buffer so we can pass
     this to our C or C++ processing function.
     To do this simply construct new ``Mem`` wrappers, giving each one the
     array as its only constructor argument.
     For numpy or cupy arrays called ``input_a`` and ``output``, this might
     look like:

     .. code-block:: Python

        mem_input_a = Mem(input_a)
        mem_output = Mem(output)

   - Functions in the library are exposed as members of ``Lib``, i.e. to call
     a function named ``sdp_func``, one only has to write:

     .. code-block:: Python

        Lib.sdp_func(...)

     However, **before** the function can be called, ``ctypes`` needs to know the type of each
     function argument we're about to pass, and the type of the result it returns.
     This must be specified using the ``Lib.wrap_func()`` convenience function, which
     requires the following arguments:

      - The name of the library function, i.e. ``"sdp_func"`` here.

      - ``restype``: The type of the result, use ``None`` if the function returns ``void``

      - ``argtypes``: A list containing the type of every argument of the function. 
        However, if the function takes an ``sdp_Error`` argument (in last place, by convention),
        **omit it**, use the option below instead.
        
      - ``check_errcode``: Boolean value (``False`` by default). If ``True``,
        the wrapping code will automatically extend ``argtypes`` to pass an error code.
        Furthermore, the function will be wrapped so that it automatically checks whether
        a non-zero error code has been set. If this happens, a special ``CError`` exception
        will be raised.

     The Python ``Mem`` class (like other classes that wrap a C struct under the hood) has 
     a convenience classmethod to return what type should be specified in ``argtypes``
     for them: ``handle_type()``.
    
     Therefore, if our library function that we wish to call takes an integer,
     two :cpp:struct:`sdp_Mem` handles and a ``sdp_Error`` argument, we would
     specify this as follows:

     .. code-block:: Python

        Lib.wrap_func(
          "sdp_func",
          restype=None,  # returns 'void'
          argtypes=[
            ctypes.c_int,
            Mem.handle_type(),
            Mem.handle_type(),  # we omit the 'sdp_Error' argument from the list ...
          ],
          check_error=True  # ... and instead turn on automatic error checking here
        )

     .. warning::

      ``ctypes`` cannot infer the signature of a C function by itself,
      and will blindly trust whatever was specified via ``Lib.wrap_func()``.
      Exert caution here, as unexpected or undefined behaviour will ensue if the 
      number of arguments is incorrect, or if you forget to set ``check_error=True``
      on a function that expects an ``sdp_Error`` argument.

      The ``wrap_func`` call only needs to be made once; place it
      outside of Python functions directly at the module level.

   - The function can then be called directly, and it is available as a data member
     of the ``Lib`` class. You may directly pass ``Mem`` and ``Error`` objects to it.
     Conversions from Python fundamental types (e.g. ``int`` and ``float``)
     to C types are natively handled by ``ctypes``; no need to explicitly cast
     ``42`` to ``ctypes.c_int(42)`` for example.

     .. code-block:: Python

        Lib.sdp_func(
            42,
            mem_input_a,
            mem_output,
        )


2. If you want to expose the function directly under the Python module
   ``ska_sdp_func.<module_name>``, use a local import in the file
   ``src/ska_sdp_func/<module_name>/__init__.py`` - the function can
   then be used by importing it as follows:

   .. code-block:: Python

      from ska_sdp_func.<module_name> import <function_name>

   Otherwise, the name of the file will need to be specified as well:

   .. code-block:: Python

      from ska_sdp_func.<module_name>.<file_name> import <function_name>

3. Write a Python unit test to check the operation of the Python function.

   - For it to be found by ``pytest``, the test file should be named
     ``test_<function_name>.py``, and placed somewhere in the ``tests``
     directory.
     Inside the file, create a Python function with a name starting
     with ``test_``, which will be found automatically by ``pytest``.
     |br|

4. Re-install and re-test the library. From the repository root:

   .. code-block:: bash

      pip3 install .
      pytest


Updating documentation
======================

Descriptions from the Doxygen comments and Python docstrings should be
included in the Sphinx documentation, so they can be found easily.

1. Find (or create) an appropriate reStructuredText file inside
   the ``docs/src/`` directory.
   Processing functions are currently documented under top-level
   groups in directory names starting with ``module_``.

2. In the file, use the Sphinx directives from Breathe
   (e.g. ``doxygenfunction``) to document the C function using the
   Doxygen comments, and ``autofunction`` to document the Python function
   using the Python docstring.
   As an example, the source of the :ref:`vector_functions` page currently
   looks like this:

   .. code-block:: rst

      .. _vector_functions:

      ****************
      Vector functions
      ****************

      C/C++
      =====

      .. doxygenfunction:: sdp_vector_add


      Python
      ======

      .. autofunction:: ska_sdp_func.examples.vector_add

   - Remember to update the ``index.rst`` files to add the page to the table
     of contents, if necessary.


Worked example
==============

For a very simple example of how to implement a function both in C++ and call
it from Python, see the code for the ``sdp_vector_add`` function and its
wrapper:

1. The C++ implementation is at ``src/ska-sdp-func/examples/sdp_vector_add.cpp``
2. The C header is at ``src/ska-sdp-func/examples/sdp_vector_add.h``
3. The CUDA kernel is at ``src/ska-sdp-func/examples/sdp_vector_add.cu``
4. The C++ unit test is at ``tests/examples/test_vector_add.cpp``

For the Python wrapper:

1. The wrapper function is in ``src/ska_sdp_func/examples/vector.py``
2. The Python test is in ``tests/examples/test_vector_add.py``

For the documentation:

1. The reStructuredText markup is in ``docs/src/module_examples/vector.rst``
