/* See the LICENSE file at the top-level directory of this distribution. */

#ifndef SKA_SDP_PROC_FUNC_MEM_UTILS_H_
#define SKA_SDP_PROC_FUNC_MEM_UTILS_H_

/**
 * @file sdp_mem_utils.h
 *
 * Contains C++ utilities for accessing ::sdp_Mem objects
 */

#ifndef __cplusplus
#error Memory views require C++ language features!
#endif

/**
 * @defgroup Mem_view_struct
 * @{
 */

#include "ska-sdp-func/utility/sdp_logging.h"
#include "ska-sdp-func/utility/sdp_mem.h"

#include <complex>
#include <assert.h>

/**
 * @brief Utility class for accessing simple strided arrays
 *
 * Use #sdp_mem_check_and_view to safely obtain views from ::sdp_Mem
 * object, then sdp_MemView::operator()() to address memory contents:
 *
 * @code{.cpp}
 *    sdp_MemViewCpu<double, 1> view; sdp_Error status;
 *    sdp_mem_check_and_view(mem, &view, &status);
 *    if (*status) return;
 *
 *    double sum = 0;
 *    for (int i = 0; i < view.shape[0]; i++) {
 *        sum += view(i);
 *    }
 * @endcode
 *
 * Note that sdp_MemView::operator()() will perform bounds checks
 * unless ``NDEBUG`` is defined.
 */
template <typename num_t, int32_t num_dims, sdp_MemLocation loc>
struct sdp_MemView
{
    /** Default constructor */
    sdp_MemView() : ptr(NULL) {}

    num_t *ptr; /**< Pointer to first array element */
    int64_t shape[num_dims]; /**< Size of array in every dimension */
    int64_t stride[num_dims]; /**< Memory stride in dimension (in units of element type size) */

    /**
     * @brief Operator for accessing data contained in 0-dimensional
     * array views
     */
    inline num_t &operator ()()
    {
#ifndef NDEBUG
        static_assert(num_dims == 0, "Wrong number of indices passed to operator ()!");
#endif
        return *ptr;
    }

    /**
     * @brief Operator for accessing data contained in 1-dimensional
     * array views
     */
    inline num_t &operator ()(int64_t i0)
    {
#ifndef NDEBUG
        static_assert(num_dims == 1, "Wrong number of indices passed to operator ()!");
        assert(i0 >= 0 && i0 < shape[0]);
#endif
        return ptr[stride[0] * i0];
    }

    /**
     * @brief Operator for accessing data contained in 2-dimensional
     * array views
     */
    inline num_t &operator ()(int64_t i0, int64_t i1)
    {
#ifndef NDEBUG
        static_assert(num_dims == 2, "Wrong number of indices passed to operator ()!");
        assert(i0 >= 0 && i0 < shape[0]);
        assert(i1 >= 0 && i1 < shape[1]);
#endif
        return ptr[stride[0] * i0 + stride[1] * i1];
    }

    /**
     * @brief Operator for accessing data contained in 3-dimensional
     * array views
     */
    inline num_t &operator ()(int64_t i0, int64_t i1, int64_t i2)
    {
#ifndef NDEBUG
        static_assert(num_dims == 3, "Wrong number of indices passed to operator ()!");
        assert(i0 >= 0 && i0 < shape[0]);
        assert(i1 >= 0 && i1 < shape[1]);
        assert(i2 >= 0 && i2 < shape[2]);
#endif
        return ptr[stride[0] * i0 + stride[1] * i1 + stride[2] * i2];
    }

    /**
     * @brief Operator for accessing data contained in 4-dimensional
     * array views
     */
    inline num_t &operator ()(int64_t i0, int64_t i1, int64_t i2, int64_t i3)
    {
#ifndef NDEBUG
        static_assert(num_dims == 4, "Wrong number of indices passed to operator ()!");
        assert(i0 >= 0 && i0 < shape[0]);
        assert(i1 >= 0 && i1 < shape[1]);
        assert(i2 >= 0 && i2 < shape[2]);
        assert(i3 >= 0 && i3 < shape[3]);
#endif
        return ptr[stride[0] * i0 + stride[1] * i1 + stride[2] * i2 + stride[3] * i3];
    }

    /**
     * @brief Operator for accessing data contained in 5-dimensional
     * array views
     */
    inline num_t &operator ()(int64_t i0, int64_t i1, int64_t i2, int64_t i3, int64_t i4)
    {
#ifndef NDEBUG
        static_assert(num_dims == 5, "Wrong number of indices passed to operator ()!");
        assert(i0 >= 0 && i0 < shape[0]);
        assert(i1 >= 0 && i1 < shape[1]);
        assert(i2 >= 0 && i2 < shape[2]);
        assert(i3 >= 0 && i3 < shape[3]);
        assert(i4 >= 0 && i4 < shape[4]);
#endif
        return ptr[stride[0] * i0 + stride[1] * i1 + stride[2] * i2 + stride[3] * i3 + stride[4] * i4];
    }

};

template <typename num_t, int32_t num_dims>
using sdp_MemViewCpu = sdp_MemView<num_t, num_dims, SDP_MEM_CPU>;
template <typename num_t, int32_t num_dims>
using sdp_MemViewGpu = sdp_MemView<num_t, num_dims, SDP_MEM_GPU>;

/** @} */

template <typename num_t> sdp_MemType sdp_mem_lift_type();
template<> sdp_MemType sdp_mem_lift_type<char>()
{
    return SDP_MEM_CHAR;
}
template<> sdp_MemType sdp_mem_lift_type<int32_t>()
{
    return SDP_MEM_INT;
}
template<> sdp_MemType sdp_mem_lift_type<float>()
{
    return SDP_MEM_FLOAT;
}
template<> sdp_MemType sdp_mem_lift_type<double>()
{
    return SDP_MEM_DOUBLE;
}
template<> sdp_MemType sdp_mem_lift_type<std::complex<float> >()
{
    return SDP_MEM_COMPLEX_FLOAT;
}
template<> sdp_MemType sdp_mem_lift_type<std::complex<double> >()
{
    return SDP_MEM_COMPLEX_DOUBLE;
}

/**
 * @defgroup Mem_view
 * @{
 */

/**
 * @brief Attempts to generate view of memory object
 *
 * This checks that the memory:
 *  * has the expected type
 *  * has the expected number of dimension
 *  * is located in the right memory space
 *  * is writeable
 *
 * ``status`` will be set if a check fails. Use
 * #sdp_mem_check_and_view macro to automatically fill ``func``,
 * ``expr``, ``file`` and ``line`` by call location.
 *
 * @param mem Handle to memory block to view
 * @param view Handle to memory block to view
 * @param status Output error status.
 * @param func Function to report in error message
 * @param expr Expression string to report in error message
 * @param file File name to report in error message
 * @param line Line to report in error message
 */
template <typename num_t, int32_t num_dims, sdp_MemLocation loc>
void sdp_mem_check_and_view_at(
    sdp_Mem *mem,
    sdp_MemView<num_t, num_dims, loc> *view,
    sdp_Error *status, const char *func, const char *expr, const char *file, int line)
{

    // Do type & shape check
    sdp_mem_check_type_at(mem, sdp_mem_lift_type<num_t>(), status,
                          func, expr, file, line);
    sdp_mem_check_num_dims_at(mem, num_dims, status,
                              func, expr, file, line);
    sdp_mem_check_location_at(mem, loc, status,
                              func, expr, file, line);
    sdp_mem_check_writeable_at(mem, status, func, expr, file, line);
    if (*status) return;

    // Set pointer, shape & strides
    view->ptr = static_cast<num_t *>(sdp_mem_data(mem));
    int32_t dim;
    for (dim = 0; dim < num_dims; dim++) {
        view->shape[dim] = sdp_mem_shape_dim(mem, dim);
        view->stride[dim] = sdp_mem_stride_elements_dim(mem, dim);
    }
}

/**
 * @brief Attempts to generate constant view of memory object
 *
 * This checks that the memory:
 *  * has the expected type
 *  * has the expected number of dimension
 *  * is located in the right memory space
 *
 * ``status`` will be set if a check fails. Use
 * #sdp_mem_check_and_view macro to automatically fill ``func``,
 * ``expr``, ``file`` and ``line`` by call location.
 *
 * @param mem Handle to memory block to view
 * @param view Handle to memory block to view
 * @param status Output error status.
 * @param func Function to report in error message
 * @param expr Expression string to report in error message
 * @param file File name to report in error message
 * @param line Line to report in error message
 */
template <typename num_t, int32_t num_dims, sdp_MemLocation loc>
void sdp_mem_check_and_view_at(
    sdp_Mem *mem,
    sdp_MemView<const num_t, num_dims, loc> *view,
    sdp_Error *status, const char *func, const char *expr, const char *file, int line)
{

    // Do type & shape check
    sdp_mem_check_type_at(mem, sdp_mem_lift_type<num_t>(), status,
                          func, expr, file, line);
    sdp_mem_check_num_dims_at(mem, num_dims, status,
                              func, expr, file, line);
    sdp_mem_check_location_at(mem, loc, status,
                              func, expr, file, line);
    if (*status) return;

    // Set pointer, shape & strides
    view->ptr = static_cast<const num_t *>(sdp_mem_data_const(mem));
    int32_t dim;
    for (dim = 0; dim < num_dims; dim++) {
        view->shape[dim] = sdp_mem_shape_dim(mem, dim);
        view->stride[dim] = sdp_mem_stride_elements_dim(mem, dim);
    }
}

/**
 * @brief Attempts to generate view of memory object
 *
 * This checks that the memory:
 *  * has the expected type
 *  * has the expected number of dimension
 *  * is located in the right memory space
 *  * is writeable (if non-``const`` view type)
 *
 * ``status`` will be set if a check fails.
 *
 * @param mem Handle to memory block to view
 * @param view Handle to memory block to view
 * @param status Output error status.
 */
#define sdp_mem_check_and_view(mem, view, status)                       \
    sdp_mem_check_and_view_at(mem, view, status, __func__, #mem, __FILE__, __LINE__)

/** @} */ /* End group Mem_view. */

#endif
