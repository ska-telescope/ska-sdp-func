**********************************
W-stacking with w-towers functions
**********************************

These functions wrap the gridding kernel functions described on the
page :ref:`w_tower_gridder_kernel`.
They provide a convenient way to call the w-towers gridder, where sub-grids
are handled internally.

Some notes on the parameters passed to these functions:

- The spacing (in wavelengths) between small w-layers within each tower is
  given by the ``w_step`` parameter, while the spacing between the large
  w-layers used for w-stacking is given by ``w_tower_height``
  multiplied by ``w_step``.

- The oversampling factors ``oversampling`` and ``w_oversampling``, which are
  used to tabulate the convolution kernels, should be large enough.
  Values of 16384 for both are sensible.

- The kernel support sizes ``support`` and ``w_support`` should be even.
  Values of 8 or 10 for both are sensible.

- The fraction of the sub-grid that should actually be used for (de)gridding
  is given by the ``subgrid_frac`` parameter, which should typically be set
  to 2/3 (or 0.66). The remaining 1/3 is the border, which will overlap with
  neighbouring sub-grids.

C/C++
=====

.. doxygengroup:: grid_wstack_wtower_func
   :content-only:

Python
======

.. autofunction:: ska_sdp_func.grid_data.wstack_wtower_degrid_all

.. autofunction:: ska_sdp_func.grid_data.wstack_wtower_grid_all
