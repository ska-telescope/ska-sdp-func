*************************
Gridder utility functions
*************************

These functions are mainly internal utilities for the w-towers
sub-grid gridder and degridder. The most useful ones are likely to be
:func:`ska_sdp_func.grid_data.determine_w_step` and
:func:`ska_sdp_func.grid_data.find_max_w_tower_height`, which can
be used to evaluate those parameters which the gridder requires.

All the Python functions call the underlying C versions, which
should be faster than pure Python implementations.

C/C++
=====

.. doxygengroup:: gridder_clamp_chan_func
   :content-only:

.. doxygengroup:: gridder_util_func
   :content-only:

Python
======

.. autofunction:: ska_sdp_func.grid_data.clamp_channels_single

.. autofunction:: ska_sdp_func.grid_data.clamp_channels_uv

.. autofunction:: ska_sdp_func.grid_data.determine_w_step

.. autofunction:: ska_sdp_func.grid_data.find_max_w_tower_height

.. autofunction:: ska_sdp_func.grid_data.make_kernel

.. autofunction:: ska_sdp_func.grid_data.make_pswf_kernel

.. autofunction:: ska_sdp_func.grid_data.make_w_pattern

.. autofunction:: ska_sdp_func.grid_data.rms_diff

.. autofunction:: ska_sdp_func.grid_data.subgrid_add

.. autofunction:: ska_sdp_func.grid_data.subgrid_cut_out

.. autofunction:: ska_sdp_func.grid_data.uvw_bounds_all
