/* See the LICENSE file at the top-level directory of this distribution. */

#ifndef SKA_SDP_PROC_FUNC_MEM_VIEW_H_
#define SKA_SDP_PROC_FUNC_MEM_VIEW_H_

/**
 * @file sdp_mem_view.h
 *
 * Contains C++ utilities for accessing ::sdp_Mem objects
 */

#ifndef __cplusplus
#error Memory views require C++ language features!
#endif

#include "ska-sdp-func/utility/sdp_logging.h"
#include "ska-sdp-func/utility/sdp_mem.h"

#include <assert.h>
#include <complex>

#ifdef __CUDACC__
#define SDP_MV_INLINE __host__ __device__ __forceinline__
#else
#define SDP_MV_INLINE inline
#endif

// Check if static_assert is available before trying to use it.
#if __cpp_static_assert >= 200410L
#define SDP_MV_STATIC_ASSERT_NUM_DIMS(NUM) \
    static_assert(num_dims == NUM, \
        "Wrong number of indices passed to operator ()!" \
    );
#else
#define SDP_MV_STATIC_ASSERT_NUM_DIMS(NUM)
#endif

#ifndef NDEBUG
#define SDP_MV_CHECK_DIM(NUM) \
    assert(i ## NUM >= 0); assert(i ## NUM < shape[NUM]);
#else
#define SDP_MV_CHECK_DIM(NUM)
#endif

// Helper macros for repetition, used for dimension bounds check.
#define SDP_MV_REPEAT_0(MACRO)
#define SDP_MV_REPEAT_1(MACRO) MACRO(0)
#define SDP_MV_REPEAT_2(MACRO) SDP_MV_REPEAT_1(MACRO) MACRO(1)
#define SDP_MV_REPEAT_3(MACRO) SDP_MV_REPEAT_2(MACRO) MACRO(2)
#define SDP_MV_REPEAT_4(MACRO) SDP_MV_REPEAT_3(MACRO) MACRO(3)
#define SDP_MV_REPEAT_5(MACRO) SDP_MV_REPEAT_4(MACRO) MACRO(4)
#define SDP_MV_GET_REPEAT_MACRO(NUM) SDP_MV_REPEAT_ ## NUM

// Call the appropriate SDP_MV_REPEAT_X macro.
#define SDP_MV_REPEAT(NUM, MACRO) SDP_MV_GET_REPEAT_MACRO(NUM)(MACRO)

#define SDP_MV_CHECK_DIMS(NUM) \
    SDP_MV_STATIC_ASSERT_NUM_DIMS(NUM) \
    SDP_MV_REPEAT(NUM, SDP_MV_CHECK_DIM)

/**
 * @defgroup Mem_view_struct
 * @{
 */

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
template<typename num_t, int32_t num_dims, sdp_MemLocation loc>
struct sdp_MemView
{
    /** Default constructor */
    sdp_MemView() : ptr(NULL)
    {
    }

    num_t* ptr; /**< Pointer to first array element */
    int64_t shape[num_dims]; /**< Size of array in every dimension */
    int64_t stride[num_dims]; /**< Memory stride in dimension (in units of element type size) */

    /**
     * @brief Operator for accessing const data via 0-dimensional array views.
     */
    SDP_MV_INLINE const num_t &operator ()() const
    {
        SDP_MV_CHECK_DIMS(0)
        return *ptr;
    }

    /**
     * @brief Operator for accessing data via 0-dimensional array views.
     */
    SDP_MV_INLINE num_t &operator ()()
    {
        SDP_MV_CHECK_DIMS(0)
        return *ptr;
    }

    /**
     * @brief Operator for accessing const data via 1-dimensional array views.
     */
    SDP_MV_INLINE const num_t &operator ()(int64_t i0) const
    {
        SDP_MV_CHECK_DIMS(1)
        return ptr[stride[0] * i0];
    }

    /**
     * @brief Operator for accessing data via 1-dimensional array views.
     */
    SDP_MV_INLINE num_t &operator ()(int64_t i0)
    {
        SDP_MV_CHECK_DIMS(1)
        return ptr[stride[0] * i0];
    }

    /**
     * @brief Operator for accessing const data via 2-dimensional array views.
     */
    SDP_MV_INLINE const num_t &operator ()(int64_t i0, int64_t i1) const
    {
        SDP_MV_CHECK_DIMS(2)
        return ptr[stride[0] * i0 + stride[1] * i1];
    }

    /**
     * @brief Operator for accessing data via 2-dimensional array views.
     */
    SDP_MV_INLINE num_t &operator ()(int64_t i0, int64_t i1)
    {
        SDP_MV_CHECK_DIMS(2)
        return ptr[stride[0] * i0 + stride[1] * i1];
    }

    /**
     * @brief Operator for accessing const data via 3-dimensional array views.
     */
    SDP_MV_INLINE
    const num_t &operator ()(int64_t i0, int64_t i1, int64_t i2) const
    {
        SDP_MV_CHECK_DIMS(3)
        return ptr[stride[0] * i0 + stride[1] * i1 + stride[2] * i2];
    }

    /**
     * @brief Operator for accessing data via 3-dimensional array views.
     */
    SDP_MV_INLINE num_t &operator ()(int64_t i0, int64_t i1, int64_t i2)
    {
        SDP_MV_CHECK_DIMS(3)
        return ptr[stride[0] * i0 + stride[1] * i1 + stride[2] * i2];
    }

    /**
     * @brief Operator for accessing const data via 4-dimensional array views.
     */
    SDP_MV_INLINE const num_t &operator ()(
            int64_t i0,
            int64_t i1,
            int64_t i2,
            int64_t i3
    ) const
    {
        SDP_MV_CHECK_DIMS(4)
        return ptr[stride[0] * i0 + stride[1] * i1 + stride[2] * i2 +
                       stride[3] * i3];
    }

    /**
     * @brief Operator for accessing data via 4-dimensional array views.
     */
    SDP_MV_INLINE
    num_t &operator ()(int64_t i0, int64_t i1, int64_t i2, int64_t i3)
    {
        SDP_MV_CHECK_DIMS(4)
        return ptr[stride[0] * i0 + stride[1] * i1 + stride[2] * i2 +
                       stride[3] * i3];
    }

    /**
     * @brief Operator for accessing const data via 5-dimensional array views.
     */
    SDP_MV_INLINE const num_t &operator ()(
            int64_t i0,
            int64_t i1,
            int64_t i2,
            int64_t i3,
            int64_t i4
    ) const
    {
        SDP_MV_CHECK_DIMS(5)
        return ptr[stride[0] * i0 + stride[1] * i1 + stride[2] * i2 +
                       stride[3] * i3 + stride[4] * i4];
    }

    /**
     * @brief Operator for accessing data via 5-dimensional array views.
     */
    SDP_MV_INLINE num_t &operator ()(
            int64_t i0,
            int64_t i1,
            int64_t i2,
            int64_t i3,
            int64_t i4
)
    {
        SDP_MV_CHECK_DIMS(5)
        return ptr[stride[0] * i0 + stride[1] * i1 + stride[2] * i2 +
                       stride[3] * i3 + stride[4] * i4];
    }
};

template<typename num_t, int32_t num_dims>
using sdp_MemViewCpu = sdp_MemView<num_t, num_dims, SDP_MEM_CPU>;
template<typename num_t, int32_t num_dims>
using sdp_MemViewGpu = sdp_MemView<num_t, num_dims, SDP_MEM_GPU>;

/** @} */

template<typename num_t>
sdp_MemType sdp_mem_lift_type();


template<>
inline sdp_MemType sdp_mem_lift_type<char>()
{
    return SDP_MEM_CHAR;
}


template<>
inline sdp_MemType sdp_mem_lift_type<int32_t>()
{
    return SDP_MEM_INT;
}


template<>
inline sdp_MemType sdp_mem_lift_type<float>()
{
    return SDP_MEM_FLOAT;
}


template<>
inline sdp_MemType sdp_mem_lift_type<const float>()
{
    return SDP_MEM_FLOAT;
}


template<>
inline sdp_MemType sdp_mem_lift_type<double>()
{
    return SDP_MEM_DOUBLE;
}


template<>
inline sdp_MemType sdp_mem_lift_type<const double>()
{
    return SDP_MEM_DOUBLE;
}


template<>
inline sdp_MemType sdp_mem_lift_type<std::complex<float> >()
{
    return SDP_MEM_COMPLEX_FLOAT;
}


template<>
inline sdp_MemType sdp_mem_lift_type<const std::complex<float> >()
{
    return SDP_MEM_COMPLEX_FLOAT;
}


template<>
inline sdp_MemType sdp_mem_lift_type<std::complex<double> >()
{
    return SDP_MEM_COMPLEX_DOUBLE;
}


template<>
inline sdp_MemType sdp_mem_lift_type<const std::complex<double> >()
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
template<typename num_t, int32_t num_dims, sdp_MemLocation loc>
void sdp_mem_check_and_view_at(
        sdp_Mem* mem,
        sdp_MemView<num_t, num_dims, loc>* view,
        sdp_Error* status,
        const char* func,
        const char* expr,
        const char* file,
        int line
)
{
    // Do type & shape check
    sdp_mem_check_type_at(mem, sdp_mem_lift_type<num_t>(), status,
            func, expr, file, line
    );
    sdp_mem_check_num_dims_at(mem, num_dims, status,
            func, expr, file, line
    );
    sdp_mem_check_location_at(mem, loc, status,
            func, expr, file, line
    );
    sdp_mem_check_writeable_at(mem, status, func, expr, file, line);

    // Set pointer, shape & strides
    view->ptr = static_cast<num_t*>(sdp_mem_data(mem));
    int32_t dim;
    for (dim = 0; dim < num_dims; dim++)
    {
        view->shape[dim] = *status ? 0 : sdp_mem_shape_dim(mem, dim);
        view->stride[dim] = *status ? 0 : sdp_mem_stride_elements_dim(mem, dim);
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
template<typename num_t, int32_t num_dims, sdp_MemLocation loc>
void sdp_mem_check_and_view_at(
        const sdp_Mem* mem,
        sdp_MemView<const num_t, num_dims, loc>* view,
        sdp_Error* status,
        const char* func,
        const char* expr,
        const char* file,
        int line
)
{
    // Do type & shape check
    sdp_mem_check_type_at(mem, sdp_mem_lift_type<num_t>(), status,
            func, expr, file, line
    );
    sdp_mem_check_num_dims_at(mem, num_dims, status,
            func, expr, file, line
    );
    sdp_mem_check_location_at(mem, loc, status,
            func, expr, file, line
    );

    // Set pointer, shape & strides
    view->ptr = static_cast<const num_t*>(sdp_mem_data_const(mem));
    for (int32_t dim = 0; dim < num_dims; dim++)
    {
        view->shape[dim] = *status ? 0 : sdp_mem_shape_dim(mem, dim);
        view->stride[dim] = *status ? 0 : sdp_mem_stride_elements_dim(mem, dim);
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
#define sdp_mem_check_and_view(mem, view, status) \
    sdp_mem_check_and_view_at(mem, \
        view, \
        status, \
        __func__, \
        #mem, \
        __FILE__, \
        __LINE__ \
    )

/** @} */ /* End group Mem_view. */

#endif
