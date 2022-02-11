
*****************
Launching Kernels
*****************

CUDA kernels should be defined in their own source files with a ``.cu``
file extension, and registered with the library in the same file as they are
defined by using the SDP_CUDA_KERNEL macro. This is done to avoid the need to
compile the calling code as well as the kernels with ``nvcc``, since CUDA is an
optional dependency.

Once registered, a kernel can be launched by calling
:cpp:func:`sdp_launch_cuda_kernel`, providing the name of the kernel
as a string, and an array for its function arguments.
(This is essentially a thin wrapper around ``cudaLaunchKernel``).

The following example registers two versions of a simple templated kernel:

.. code-block:: CUDA

   #include "utility/sdp_device_wrapper.h"

   template<typename T>
   __global__
   void vector_add (
       const int64_t num_elements,
       const T *const __restrict__ input_a,
       const T *const __restrict__ input_b,
       T *__restrict__ output)
   {
       const int64_t i = blockDim.x * blockIdx.x + threadIdx.x;
       if (i < num_elements)
       {
           output[i] = input_a[i] + input_b[i];
       }
   }

   SDP_CUDA_KERNEL(vector_add<float>)
   SDP_CUDA_KERNEL(vector_add<double>)


Include the header *"utility/sdp_device_wrapper.h"* to use these functions.

.. doxygengroup:: device_func
   :content-only:

.. doxygendefine:: SDP_CUDA_KERNEL
