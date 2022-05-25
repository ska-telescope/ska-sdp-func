# See the LICENSE file at the top-level directory of this distribution.

from .utility import Error, Lib, Mem


def dft_point_v00(source_directions, source_fluxes, uvw_lambda, vis):
    """Basic prediction of visibilities from point sources using a DFT.

    This version of the function is compatible with the memory layout of
    arrays used by RASCIL.

    Parameters can be either numpy or cupy arrays, but must all be consistent.
    Computation is performed either on the CPU or GPU as appropriate.

    Input parameters ``source_directions`` and ``uvw_lambda`` are arrays
    of packed 3D coordinates.

    Array dimensions must be as follows:

    * ``source_directions`` is 2D and real-valued, with shape:

      * [ num_components, 3 ]

    * ``source_fluxes`` is 3D and complex-valued, with shape:

      * [ num_components, num_channels, num_pols ]

    * ``uvw_lambda`` is 4D and real-valued, with shape:

      * [ num_times, num_baselines, num_channels, 3 ]

    * ``vis`` is 4D and complex-valued, with shape:

      * [ num_times, num_baselines, num_channels, num_pols ]

    :param source_directions: Source direction cosines.
    :type source_directions: numpy.ndarray or cupy.ndarray

    :param source_fluxes: Complex source flux values.
    :type source_fluxes: numpy.ndarray or cupy.ndarray

    :param uvw_lambda: Baseline (u,v,w) coordinates, in wavelengths.
    :type uvw_lambda: numpy.ndarray or cupy.ndarray

    :param vis: Output complex visibilities.
    :type vis: numpy.ndarray or cupy.ndarray
    """
    mem_source_directions = Mem(source_directions)
    mem_source_fluxes = Mem(source_fluxes)
    mem_uvw_lambda = Mem(uvw_lambda)
    mem_vis = Mem(vis)
    error_status = Error()
    lib_dft = Lib.handle().sdp_dft_point_v00
    lib_dft.argtypes = [
        Mem.handle_type(),
        Mem.handle_type(),
        Mem.handle_type(),
        Mem.handle_type(),
        Error.handle_type(),
    ]
    lib_dft(
        mem_source_directions.handle(),
        mem_source_fluxes.handle(),
        mem_uvw_lambda.handle(),
        mem_vis.handle(),
        error_status.handle(),
    )
    error_status.check()
