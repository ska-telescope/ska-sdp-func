
#include "ska-sdp-func/utility/sdp_logging.h"

#include <complex>
#include <assert.h>

void sdp_mem_check_writeable_(
    sdp_Mem *mem,
    sdp_Error *status,
    const char *func,
    const char *expr,
    const char *file,
    int line)
{
    if (sdp_mem_is_read_only(mem)) {

        // Log & set status
        sdp_log_message(SDP_LOG_LEVEL_ERROR, stderr, func, file, line,
                        "%s: Expected '%s' not to be read-only!",
                        func, expr);
        *status = SDP_ERR_INVALID_ARGUMENT;
    }
}
#define sdp_mem_check_writeable(mem, status)                      \
    sdp_mem_check_writeable_(mem, status, #mem, __func__, __FILE__, __LINE__)

const char *sdp_mem_location_name(sdp_MemLocation loc)
{
    switch(loc) {
    case SDP_MEM_CPU: return "CPU";
    case SDP_MEM_GPU: return "GPU";
    default: return "???";
    }
}

void sdp_mem_check_location_(
    sdp_Mem *mem,
    sdp_MemLocation loc,
    sdp_Error *status,
    const char *func,
    const char *expr,
    const char *file,
    int line)
{
    if (sdp_mem_location(mem) != loc) {

        // Log & set status
        sdp_log_message(SDP_LOG_LEVEL_ERROR, stderr, func, file, line,
                        "%s: Expected '%s' to be in %s memory (found %s)!",
                        func, expr,
                        sdp_mem_location_name(loc),
                        sdp_mem_location_name(sdp_mem_location(mem)));
        *status = SDP_ERR_MEM_LOCATION;
    }
}
#define sdp_mem_check_location(mem, loc, status)                      \
    sdp_mem_check_location_(mem, loc, status, #mem, __func__, __FILE__, __LINE__)

void sdp_mem_check_num_dims_(
    sdp_Mem *mem,
    int64_t ndims,
    sdp_Error *status,
    const char *func,
    const char *expr,
    const char *file,
    int line)
{
    if (sdp_mem_num_dims(mem) != ndims) {

        // Log & set status
        sdp_log_message(
            SDP_LOG_LEVEL_ERROR, stderr, func, file, line,
            "%s: Expected '%s' shape to have %d dimensions (found %d)!",
            func, expr, ndims, sdp_mem_num_dims(mem));
        *status = SDP_ERR_INVALID_ARGUMENT;
    }
}
#define sdp_mem_check_num_dims(mem, ndims, status)                      \
    sdp_mem_check_num_dims_(mem, ndims, status, #mem, __func__, __FILE__, __LINE__)

const char *sdp_mem_type_name(sdp_MemType typ)
{
    switch(typ) {
    case SDP_MEM_VOID: return "void";
    case SDP_MEM_CHAR: return "char";
    case SDP_MEM_INT: return "int";
    case SDP_MEM_FLOAT: return "float";
    case SDP_MEM_DOUBLE: return "double";
    case SDP_MEM_COMPLEX_FLOAT: return "complex float";
    case SDP_MEM_COMPLEX_DOUBLE: return "complex double";
    default: return "???";
    }
}

void sdp_mem_check_type_(
    sdp_Mem *mem,
    sdp_MemType typ,
    sdp_Error *status,
    const char *func,
    const char *expr,
    const char *file,
    int line)
{
    if (sdp_mem_type(mem) != typ) {

        // Log & set status
        sdp_log_message(
            SDP_LOG_LEVEL_ERROR, stderr, func, file, line,
            "%s: Expected '%s' to have type %s (found %s)!",
            func, expr,
            sdp_mem_type_name(typ),
            sdp_mem_type_name(sdp_mem_type(mem)));
        *status = SDP_ERR_DATA_TYPE;
    }
}
#define sdp_mem_check_type(mem, typ, status)                      \
    sdp_mem_check_type_(mem, typ, status, #mem, __func__, __FILE__, __LINE__)

void sdp_mem_check_shape_dim_(
    sdp_Mem *mem,
    int32_t dim,
    int64_t size,
    sdp_Error *status,
    const char *func,
    const char *expr,
    const char *file,
    int line)
{
    if (sdp_mem_shape_dim(mem, dim) != size) {

        // Log & set status
        sdp_log_message(SDP_LOG_LEVEL_ERROR, stderr, func, file, line,
                        "%s: Expected '%s' to have size %d in dimension %d!",
                        func, expr, size, dim);
        *status = SDP_ERR_INVALID_ARGUMENT;
    }
}
#define sdp_mem_check_shape_dim(mem, dim, size, status)                 \
    sdp_mem_check_shape_dim_(mem, dim, size, status, #mem, __func__, __FILE__, __LINE__)


template <typename num_t, int32_t num_dims>
struct sdp_CpuVec
{
    sdp_CpuVec() : ptr(NULL) {}
    num_t *ptr;
    int64_t shape[num_dims];
    int64_t stride[num_dims];

    // Experimental way to make "vec[a][b][c]" work. Theoretically
    // this should all inline and the loop should unroll, so it will
    // all compile into nothing at all - without any "higher" template
    // metaprogramming.
    inline sdp_CpuVec<num_t, num_dims-1> operator [] (int64_t i0) const {
        sdp_CpuVec<num_t, num_dims-1> tmp;
#ifndef NDEBUG
        assert(i0 >= 0 && i0 < shape[0]);
#endif
        tmp.ptr = ptr + i0 * shape[0];
        for (int32_t dim = 0; dim < num_dims; dim++) {
            tmp.shape[dim] = shape[dim+1];
            tmp.stride[dim] = stride[dim+1];
        }
        return tmp;
    }
};

template <typename num_t>
struct sdp_CpuVec<num_t, 1>
{
    sdp_CpuVec() : ptr(NULL) {}
    num_t *ptr;
    int64_t shape[1];
    int64_t stride[1];

    inline num_t &operator [] (int64_t i0) const {
#ifndef NDEBUG
        assert(i0 >= 0 && i0 < shape[0]);
#endif
        return ptr[i0 * stride[0]];
    }
};

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

template <typename num_t>
inline num_t &V0(const sdp_CpuVec<num_t, 0> &v)
{
    return *v.ptr;
}

template <typename num_t>
inline num_t &V1(const sdp_CpuVec<num_t, 1> &v, int64_t i0)
{
#ifndef NDEBUG
    assert(i0 >= 0 && i0 < v.shape[0]);
#endif
    return v.ptr[v.stride[0] * i0];
}

template <typename num_t>
inline num_t &V2(const sdp_CpuVec<num_t, 2> &v, int64_t i0, int64_t i1)
{
#ifndef NDEBUG
    assert(i0 >= 0 && i0 < v.size[0]);
    assert(i1 >= 0 && i1 < v.size[0]);
#endif
    return v.ptr[v.stride[0] * i0 + v.stride[1] * i1];
}

template <typename num_t>
inline num_t &V3(const sdp_CpuVec<num_t, 3> &v, int64_t i0, int64_t i1, int64_t i2)
{
#ifndef NDEBUG
    assert(i0 >= 0 && i0 < v.size[0]);
    assert(i1 >= 0 && i1 < v.size[1]);
    assert(i2 >= 0 && i2 < v.size[2]);
#endif
    return v.ptr[v.stride[0] * i0 + v.stride[1] * i1 + v.stride[2] * i2];
}

template <typename num_t>
inline num_t &V4(const sdp_CpuVec<num_t, 4> &v,
                 int64_t i0, int64_t i1,
                 int64_t i2, int64_t i3)
{
#ifndef NDEBUG
    assert(i0 >= 0 && i0 < v.size[0]);
    assert(i1 >= 0 && i1 < v.size[1]);
    assert(i2 >= 0 && i2 < v.size[2]);
    assert(i3 >= 0 && i3 < v.size[3]);
#endif
    return v.ptr[v.stride[0] * i0 + v.stride[1] * i1 +
                 v.stride[2] * i2 + v.stride[3] * i3];
}

template <typename num_t, int32_t num_dims>
void sdp_mem_checked_get_vec_(
    sdp_Mem *mem,
    sdp_CpuVec<num_t, num_dims> *vec,
    sdp_Error *status,
    const char *func,
    const char *expr,
    const char *file,
    int line)
{

    // Do type & shape check
    sdp_mem_check_type_(mem, sdp_mem_lift_type<num_t>(), status,
                        func, expr, file, line);
    sdp_mem_check_num_dims_(mem, num_dims, status,
                            func, expr, file, line);
    sdp_mem_check_location_(mem, SDP_MEM_CPU, status,
                            func, expr, file, line);
    sdp_mem_check_writeable_(mem, status, func, expr, file, line);
    if (*status) return;

    // Set pointer, shape & strides
    vec->ptr = static_cast<num_t *>(sdp_mem_data(mem));
    int32_t dim;
    for (dim = 0; dim < num_dims; dim++) {
        vec->shape[dim] = sdp_mem_shape_dim(mem, dim);
        vec->stride[dim] = sdp_mem_stride_elements_dim(mem, dim);
    }
}
template <typename num_t, int32_t num_dims>
void sdp_mem_checked_get_vec_(
    sdp_Mem *mem,
    sdp_CpuVec<const num_t, num_dims> *vec,
    sdp_Error *status,
    const char *func,
    const char *expr,
    const char *file,
    int line)
{

    // Do type & shape check
    sdp_mem_check_type_(mem, sdp_mem_lift_type<num_t>(), status,
                        func, expr, file, line);
    sdp_mem_check_num_dims_(mem, num_dims, status,
                            func, expr, file, line);
    sdp_mem_check_location_(mem, SDP_MEM_CPU, status,
                        func, expr, file, line);
    if (*status) return;

    // Set pointer, shape & strides
    vec->ptr = static_cast<const num_t *>(sdp_mem_data_const(mem));
    int32_t dim;
    for (dim = 0; dim < num_dims; dim++) {
        vec->shape[dim] = sdp_mem_shape_dim(mem, dim);
        vec->stride[dim] = sdp_mem_stride_elements_dim(mem, dim);
    }
}
#define sdp_mem_checked_get_vec(mem, vec, status)                              \
    sdp_mem_checked_get_vec_(mem, vec, status, #mem, __func__, __FILE__, __LINE__)

