***************
Memory handling
***************

In general, processing functions should not allocate their own memory.
Instead, they should be able to work with externally-managed memory that
could be under the control of an execution engine or another library, and to
maintain flexibility, we should make minimal assumptions about how this
will be done.

For these reasons, we need to be able to supply our functions with pointers
to externally-managed memory: we do this by wrapping the raw pointer in
a simple structure, together with some metadata that describes the memory
at that location. When passing them to a processing function, this provides
the information we need to know how these wrapped pointers should be handled.

The pointer and its metadata are encapsulated in the :any:`sdp_Mem`
structure, which is an opaque type in order to keep all its members private.
The functions are all exported using a simple C interface so they can be used
from any environment (and further wrapped, if deemed necessary).

Include the header *"ska-sdp-func/utility/sdp_mem.h"* to use these functions.

.. doxygengroup:: Mem_struct
   :content-only:

.. doxygengroup:: Mem_enum
   :content-only:

.. doxygengroup:: Mem_func
   :content-only:
