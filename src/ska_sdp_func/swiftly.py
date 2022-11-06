# See the LICENSE file at the top-level directory of this distribution.

"""Module for FFT functions."""

import ctypes
import functools
import inspect

import numpy

from .utility import Error, Lib, Mem

WRAPPER_FNS = {
    int: (ctypes.c_int64, lambda v: v),
    float: (ctypes.c_double, lambda v: v),
    numpy.ndarray: (Mem.handle_type(), Mem),
}


def auto_wrap_method(c_fn_name, add_handle=True, add_error_status=True):
    """
    Generates a wrapper method for the specified C function based on
    the decorated function's parameter annotations.

    :param add_handle: Specifies whether a handle associated with the
      containing class should be passed to the C function as the
      first parameter. The containing class must define the class
      method ``handle_type`` (returning the handle's ctype) and the
      method ``handle`` for this to work.
    :param add_error_status: Append a handle to an :py:class:`Error`
      object to the parameter list, and check it after the call
    :returns: Wrapped function
    """

    def wrapper(orig_fn):

        # Load function handle function signature. Ensure all
        # parameters are annotated
        sig = inspect.signature(orig_fn)

        # Add remaining parameters
        anns = inspect.get_annotations(orig_fn)
        argtypes = []
        for par in list(sig.parameters)[1:]:
            if par not in anns:
                raise ValueError(
                    f"Parameter {par} lacks type annotation! Cannot auto-wrap!"
                )
            if anns[par] not in WRAPPER_FNS:
                raise ValueError(
                    f"Parameter {par} has unknown type annotation {anns[par]}!"
                    " Cannot auto-wrap!"
                )
            argtypes.append(WRAPPER_FNS[anns[par]][0])

        # Make actual implementation
        # pylint: disable=unsubscriptable-object
        function = Lib.handle()[c_fn_name]

        @functools.wraps(orig_fn)
        def wrapped_fn(self, *args, **kwargs):

            # Construct function handle. We can only meaningfully do
            # this on the first call, as we cannot easily get to
            # handle_type() otherwise.
            if function.argtypes is None:
                print("Set argtypes")
                if add_handle:
                    function.argtypes = [self.handle_type()] + argtypes
                else:
                    function.argtypes = argtypes

            # Gather + wrap parameters
            bound = sig.bind(self, *args, **kwargs)
            args = []
            if add_handle:
                args.append(self.handle())
            for par in list(sig.parameters)[1:]:
                wrap_fn = WRAPPER_FNS[anns[par]][1]
                args.append(wrap_fn(bound.arguments[par]))

            # Call, passing an additional error status if requested
            if add_error_status:
                error_status = Error()
                function(*args, error_status.handle())
                error_status.check()
            else:
                function(*args, error_status.handle())

            # Finally call original function, in case there's any
            # additional action to take.
            return orig_fn(*args, **kwargs)

        return wrapped_fn

    return wrapper


class Swiftly:
    """Creates a plan for the SwiFTly algorithm (streaming
    widefield Fourier transform for large-scale interferometry)

    Image, facet and subgrid sizes must be compatible. Facets and
    subgrids must be appropriately padded for algorithm to work at any
    level of precision, this is the responsibility of the user.

    :param image_size: Size of entire (virtual) image in pixels
    :param xM_size: Internal padded subgrid size
    :param yN_size: Internal padded facet size
    :param W: Parameter for PSWF window function
    """

    class Handle(ctypes.Structure):
        """Class handle for use by ctypes."""

    # pylint: disable=invalid-name
    def __init__(self, image_size: int, yN_size: int, xM_size: int, W: float):
        """Creates a plan for Swiftly"""

        # Safe parameters
        self._image_size = image_size
        self._xM_size = xM_size
        self._yN_size = yN_size
        self._W = W

        self._handle = None
        error_status = Error()
        function_create = Lib.handle().sdp_swiftly_create
        function_create.restype = Swiftly.handle_type()
        function_create.argtypes = [
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_double,
            Error.handle_type(),
        ]
        self._handle = function_create(
            image_size, yN_size, xM_size, W, error_status.handle()
        )
        error_status.check()

    def __del__(self):
        """Releases handle to the processing function."""
        if self._handle:
            function_free = Lib.handle().sdp_swiftly_free
            function_free.argtypes = [Swiftly.handle_type()]
            function_free(self._handle)

    @property
    def image_size(self):
        """Returns image size"""
        return self._image_size

    @property
    def facet_size(self):
        """Returns padded facet size"""
        return self._yN_size

    @property
    def subgrid_size(self):
        """Returns padded subgrid size"""
        return self._xM_size

    @property
    def contribution_size(self):
        """Returns size of subgrid <> facet contribution"""
        return (self._xM_size * self._yN_size) / self._image_size

    @property
    def pswf_parameter(self):
        """Returns parameter used for PSWF"""
        return self._W

    def handle(self):
        """Returns a handle to the wrapped processing function.

        Use this handle when calling the function in the compiled library.

        :return: Handle to wrapped function.
        :rtype: ctypes.POINTER(Swiftly.Handle)
        """
        return self._handle

    @staticmethod
    def handle_type():
        """Static convenience method to return the ctypes handle type.

        Use this when defining the list of argument types.

        :return: Type of the function handle.
        :rtype: ctypes.POINTER(Swiftly.Handle)
        """
        return ctypes.POINTER(Swiftly.Handle)

    @auto_wrap_method("sdp_swiftly_prepare_facet")
    def prepare_facet(
        self,
        facet: numpy.ndarray,
        prep_facet_out: numpy.ndarray,
        facet_offset: int,
    ):
        """
        Performs SwiFTly facet preparation

        This multiplies in ``Fb`` and does Fourier transformation. Common work
        to be done for each facet before calling :py:meth:`extract_from_facet`.

        :param facet: ``[*, <facet_size]`` Facet data
        :param prep_facet_out: ``[*, facet_size]`` Prepared facet output
        :param facet_offset: Facet mid-point offset relative to image mid-point
        """

    @auto_wrap_method("sdp_swiftly_extract_from_facet")
    def extract_from_facet(
        self,
        prep_facet: numpy.ndarray,
        contribution_out: numpy.ndarray,
        subgrid_offset: int,
    ):
        """
        Extract facet contribution to a subgrid

        Copies out all data from prepared facet data that relates to a
        subgrid at a particular offset. The returned representation is
        optimised for representing this data in a compact way, and should
        be used for distribution.

        :param prep_facet: ``[*, facet_size]`` Prepared facet output
        :param contribution_out: ``[*, contribution_size]``
             Facet contribution to subgrid
        :param subgrid_offset: Subgrid mid-point relative to grid mid-point
        """

    @auto_wrap_method("sdp_swiftly_add_to_subgrid")
    def add_to_subgrid(
        self,
        contribution: numpy.ndarray,
        subgrid_image_inout: numpy.ndarray,
        facet_offset: int,
    ):
        """
        Add facet contribution to a subgrid image

        Accumulates a facet contribution in subgrid image passed. Subgrid
        image should be filled with zeros when passed to function
        initially. Use sdp_finish_subgrid to obtain subgrid data.

        :param contribution: ``[*, contribution_size]``
           Facet contribution to subgrid.
        :param subgrid_image_inout: ``[*, subgrid_size]``
           Subgrid image for accumulation.
        :param facet_offset: Facet mid-point offset relative to image mid-point
        """

    @auto_wrap_method("sdp_swiftly_finish_subgrid_inplace")
    def finish_subgrid_inplace(
        self, subgrid_inout: numpy.ndarray, subgrid_offset: int
    ):
        """
        Finish subgrid after contribution accumulation

        Performs the final Fourier Transformation to obtain the subgrid
        from the subgrid image sum.

        :subgrid_inout: ``[*, subgrid_size]``
           Subgrid / subgrid image for transform.
        :subgrid_offset: Subgrid mid-point offset relative to grid mid-point
        """
