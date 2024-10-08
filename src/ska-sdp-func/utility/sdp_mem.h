/* See the LICENSE file at the top-level directory of this distribution. */

#ifndef SKA_SDP_PROC_FUNC_MEM_H_
#define SKA_SDP_PROC_FUNC_MEM_H_

/**
 * @file sdp_mem.h
 */

#include <stdint.h>

#include "ska-sdp-func/utility/sdp_device_wrapper.h"
#include "ska-sdp-func/utility/sdp_errors.h"
#include "ska-sdp-func/utility/sdp_logging.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup Mem_struct
 * @{
 */

/**
 * @struct sdp_Mem
 *
 * @brief Simple structure to wrap a pointer to allocated memory.
 *
 * This structure is an opaque type to keep implementation details private.
 * It exists as a way to allow processing functions to work with memory
 * that may be allocated and owned by someone else.
 *
 * Call ::sdp_mem_create_wrapper() to create a wrapper for memory
 * which has already been allocated, and is therefore assumed to be owned
 * by someone else.
 *
 * Call ::sdp_mem_create() to allocate memory which is owned by the wrapper
 * itself.
 *
 * We wrap a raw pointer since this is a universally-supported way of
 * accessing memory. The pointer is stored together with metadata that
 * describes the memory at that location, namely:
 *
 * - The data type of each element (see ::sdp_MemType);
 * - The location of the memory (whether in system RAM or on a GPU;
 *   see ::sdp_MemLocation);
 * - The total number of elements allocated;
 * - Whether the wrapper owns the memory to which it points.
 *   This determines whether the wrapped memory will be deallocated when
 *   calling ::sdp_mem_free() and the reference count reaches zero.
 * - Whether the memory should be considered read-only.
 *
 * The following are also stored, for working with tensors:
 * - The number of dimensions.
 * - The size of each dimension.
 * - The stride in each dimension, in bytes (for Python compatibility).
 */
struct sdp_Mem;

/** @} */ /* End group Mem_struct. */

/**
 * @defgroup Mem_enum
 * @{
 */

/**
 * @enum sdp_MemType
 *
 * @brief Enumerator to specify the data type of memory elements.
 */
enum sdp_MemType
{
    /// @cond EXCLUDE
    SDP_MEM_VOID = 0,
    /// @endcond

    //! Char type (1 byte).
    SDP_MEM_CHAR = 1,

    //! Integer type (4 bytes).
    SDP_MEM_INT = 2,

    //! Single precision floating point type (4 bytes).
    SDP_MEM_FLOAT = 4,

    //! Double precision floating point type (8 bytes).
    SDP_MEM_DOUBLE = 8,

    //! Complex flag.
    SDP_MEM_COMPLEX = 32,

    //! Single precision complex floating point type (8 bytes).
    SDP_MEM_COMPLEX_FLOAT = SDP_MEM_FLOAT | SDP_MEM_COMPLEX,

    //! Double precision complex floating point type (16 bytes).
    SDP_MEM_COMPLEX_DOUBLE = SDP_MEM_DOUBLE | SDP_MEM_COMPLEX
};

/**
 * @enum sdp_MemLocation
 *
 * @brief Enumerator to specify the location of allocated memory.
 */
enum sdp_MemLocation
{
    //! Memory is on the host.
    SDP_MEM_CPU,

    //! Memory is on the GPU.
    SDP_MEM_GPU
};

/** @} */ /* End group Mem_enum. */

/* Typedefs. */
typedef struct sdp_Mem sdp_Mem;
typedef enum sdp_MemType sdp_MemType;
typedef enum sdp_MemLocation sdp_MemLocation;

/**
 * @defgroup Mem_func
 * @{
 */

/**
 * @brief Allocate a multi-dimensional block of memory.
 *
 * The shape of the memory block is given by the @p shape parameter, which is
 * an array of length @p num_dims elements.
 *
 * @param type Enumerated element data type of memory to allocate.
 * @param location Enumerated memory location.
 * @param num_dims Number of dimensions.
 * @param shape Size of each dimenion, in elements.
 * @param status Error status.
 * @return ::sdp_Mem* Handle to allocated memory.
 */
sdp_Mem* sdp_mem_create(
        sdp_MemType type,
        sdp_MemLocation location,
        int32_t num_dims,
        const int64_t* shape,
        sdp_Error* status
);

/**
 * @brief Wraps a pointer to a multi-dimensional array which is owned elsewhere.
 *
 * Since it is owned by someone else, the memory will not be deallocated
 * when the handle is freed.
 *
 * The shape of the memory block is given by the @p shape parameter, which is
 * an array of length @p num_dims elements. The stride (in bytes) for each
 * dimension can also be optionally specified using the @p stride parameter -
 * if NULL, then this will be computed from the shape.
 *
 * @param data Raw pointer to wrap.
 * @param type Enumerated element data type of memory at @p data.
 * @param location Enumerated memory location of memory at @p data.
 * @param num_dims Number of dimensions.
 * @param shape Size of each dimenion, in elements.
 * @param stride Stride of each dimension, in bytes. May be NULL.
 * @param status Error status.
 * @return ::sdp_Mem* Handle to wrapped memory.
 */
sdp_Mem* sdp_mem_create_wrapper(
        void* data,
        sdp_MemType type,
        sdp_MemLocation location,
        int32_t num_dims,
        const int64_t* shape,
        const int64_t* stride,
        sdp_Error* status
);

/**
 * @brief Creates a wrapper for a slice in a multidimensional array.
 *
 * This provides a way to wrap a portion (a slice) of an existing array.
 * Both the offset to the slice and the shape of the slice are passed
 * as parameters.
 *
 * @param src Handle to source memory block.
 * @param slice_offsets Indices to specify offsets to the slice.
 *     Must have the same number of indices as the dimensions of @p src
 * @param num_dims_slice Number of dimensions of the returned slice wrapper.
 * @param slice_shape Size of each dimension of the slice, in elements.
 * @param status Error status.
 * @return ::sdp_Mem* Handle to wrapped memory.
 */
sdp_Mem* sdp_mem_create_wrapper_for_slice(
        const sdp_Mem* src,
        const int64_t* slice_offsets,
        const int32_t num_dims_slice,
        const int64_t* slice_shape,
        sdp_Error* status
);

/**
 * @brief Create a shallow copy, or an alias, of a block's metadata.
 *
 * @param src Handle to source memory block.
 * @return ::sdp_Mem* Handle to aliased memory.
 */
sdp_Mem* sdp_mem_create_alias(const sdp_Mem* src);

/**
 * @brief Create a deep copy of a memory block in the specified location.
 *
 * @param src Handle to source memory block.
 * @param location Enumerated memory location for destination.
 * @param status Error status.
 * @return ::sdp_Mem* Handle to copied memory.
 */
sdp_Mem* sdp_mem_create_copy(
        const sdp_Mem* src,
        sdp_MemLocation location,
        sdp_Error* status
);

/**
 * @brief Clears contents of a memory block by setting all its elements to zero.
 *
 * @param mem Handle to memory to clear.
 * @param status Error status.
 */
void sdp_mem_clear_contents(sdp_Mem* mem, sdp_Error* status);

/**
 * @brief Clears the specified elements of a memory block.
 *
 * @param mem Handle to memory block.
 * @param start_index Start index to clear.
 * @param num_elements Number of elements to clear.
 * @param status Error status.
 */
void sdp_mem_clear_portion(
        sdp_Mem* mem,
        int64_t start_index,
        int64_t num_elements,
        sdp_Error* status
);

/**
 * @brief Converts double to single precision, or vice versa.
 *
 * This function is mainly used in tests.
 *
 * @param src Source data to convert.
 * @param output_type Required type of output data to return.
 * @param status Error status.
 */
sdp_Mem* sdp_mem_convert_precision(
        const sdp_Mem* src,
        sdp_MemType output_type,
        sdp_Error* status
);

/**
 * @brief Copies memory contents from one block to another.
 *
 * @param dst Handle to destination memory block.
 * @param src Handle to source memory block.
 * @param offset_dst Start offset (number of elements) into destination block.
 * @param offset_src Start offset (number of elements) from source block.
 * @param num_elements Number of elements to copy.
 * @param status Error status.
 */
void sdp_mem_copy_contents(
        sdp_Mem* dst,
        const sdp_Mem* src,
        int64_t offset_dst,
        int64_t offset_src,
        int64_t num_elements,
        sdp_Error* status
);

/**
 * @brief Copies memory contents from one block to another.
 *
 * This function uses cudaMemcpyAsync for transfers that involve GPU memory.
 *
 * @param dst Handle to destination memory block.
 * @param src Handle to source memory block.
 * @param offset_dst Start offset (number of elements) into destination block.
 * @param offset_src Start offset (number of elements) from source block.
 * @param num_elements Number of elements to copy.
 * @param stream Handle to CUDA stream to use.
 * @param status Error status.
 */
void sdp_mem_copy_contents_async(
        sdp_Mem* dst,
        const sdp_Mem* src,
        int64_t offset_dst,
        int64_t offset_src,
        int64_t num_elements,
        sdp_CudaStream* stream,
        sdp_Error* status
);

/**
 * @brief Returns a raw pointer to the memory wrapped by the handle.
 *
 * @param mem Handle to memory block.
 * @return void* Raw pointer to allocated memory.
 */
void* sdp_mem_data(sdp_Mem* mem);

/**
 * @brief Returns a raw const pointer to the memory wrapped by the handle.
 *
 * @param mem Handle to memory block.
 * @return void* Raw const pointer to allocated memory.
 */
const void* sdp_mem_data_const(const sdp_Mem* mem);

/**
 * @brief Returns a pointer to the GPU buffer, if memory is on the GPU.
 *
 * This is needed when launching a GPU kernel with ::sdp_launch_cuda_kernel().
 *
 * The status flag is set and a null pointer is returned if the wrapped memory
 * is not on the GPU.
 *
 * @param mem Handle to memory block.
 * @param status Error status.
 * @return void* Pointer used for GPU buffer when launching a kernel.
 */
void* sdp_mem_gpu_buffer(sdp_Mem* mem, sdp_Error* status);

/**
 * @brief Returns a pointer to the GPU buffer, if memory is on the GPU.
 *
 * This is needed when launching a GPU kernel with ::sdp_launch_cuda_kernel().
 *
 * The status flag is set and a null pointer is returned if the wrapped memory
 * is not on the GPU.
 *
 * @param mem Handle to memory block.
 * @param status Error status.
 * @return void* Const pointer used for GPU buffer when launching a kernel.
 */
const void* sdp_mem_gpu_buffer_const(const sdp_Mem* mem, sdp_Error* status);

/**
 * @brief Decrements the reference counter.
 *
 * If the reference counter reaches zero, any memory owned by the handle
 * will be released, and the handle destroyed.
 *
 * This is an alias for ::sdp_mem_ref_dec().
 *
 * @param mem Handle to memory block.
 */
void sdp_mem_free(sdp_Mem* mem);

/**
 * @brief Returns true if the dimension strides are C contiguous.
 *
 * @param mem Handle to memory block.
 * @return True if dimension strides are C contiguous.
 */
int32_t sdp_mem_is_c_contiguous(const sdp_Mem* mem);

/**
 * @brief Returns true if data element are in floating point format.
 *
 * @param mem Handle to memory block.
 * @return True if data elements are in floating point.
 */
int32_t sdp_mem_is_floating_point(const sdp_Mem* mem);

/**
 * @brief Returns true if data elements are of complex type.
 *
 * @param mem Handle to memory block.
 * @return True if data type is complex.
 */
int32_t sdp_mem_is_complex(const sdp_Mem* mem);

/**
 * @brief Returns true if data are complex 4-vectors, or 2-by-2 matrices.
 *
 * @param mem Handle to memory block.
 * @return True if data type is complex-4.
 */
int32_t sdp_mem_is_complex4(const sdp_Mem* mem);

/**
 * @brief Returns true if data elements are double precision.
 *
 * @param mem Handle to memory block.
 * @return True if data elements are double precision.
 */
int32_t sdp_mem_is_double(const sdp_Mem* mem);

/**
 * @brief Returns true if the metadata between two objects is consistent.
 *
 * Checks are done on the data type, number of dimensions,
 * strides, and dimension sizes.
 *
 * @param mem1 Handle to first memory block.
 * @param mem2 Handle to second memory block.
 * @param check_location If true, check the data locations also match.
 * @return True if memory blocks have matching meta-data.
 */
int32_t sdp_mem_is_matching(
        const sdp_Mem* mem1,
        const sdp_Mem* mem2,
        int32_t check_location
);

/**
 * @brief Returns true if the read-only flag is set.
 *
 * @param mem Handle to memory block.
 * @return True if memory should be considered to be read-only.
 */
int32_t sdp_mem_is_read_only(const sdp_Mem* mem);

/**
 * @brief Returns the enumerated location of the memory.
 *
 * @param mem Handle to memory block.
 * @return ::sdp_MemLocation The enumerated memory location.
 */
sdp_MemLocation sdp_mem_location(const sdp_Mem* mem);

/**
 * @brief Returns the number of dimensions in the memory block.
 *
 * @param mem Handle to memory block.
 * @return int32_t The number of dimensions in the block.
 */
int32_t sdp_mem_num_dims(const sdp_Mem* mem);

/**
 * @brief Returns the total number of elements in the memory block.
 *
 * @param mem Handle to memory block.
 * @return int64_t The total number of data elements in the block.
 */
int64_t sdp_mem_num_elements(const sdp_Mem* mem);

/**
 * @brief Fills memory with random values between 0 and 1. Useful for testing.
 *
 * @param mem Handle to memory block.
 * @param status Error status.
 */
void sdp_mem_random_fill(sdp_Mem* mem, sdp_Error* status);

/**
 * @brief Decrement the reference counter.
 *
 * If the reference counter reaches zero, any memory owned by the handle
 * will be released, and the handle destroyed.
 *
 * This is an alias for ::sdp_mem_free().
 *
 * @param mem Handle to memory block.
 */
void sdp_mem_ref_dec(sdp_Mem* mem);

/**
 * @brief Increment the reference counter.
 *
 * Call if ownership of the handle needs to be transferred without incurring
 * the cost of an actual copy.
 *
 * @param mem Handle to memory block.
 * @return ::sdp_Mem* Handle to memory block (same as @p mem).
 */
sdp_Mem* sdp_mem_ref_inc(sdp_Mem* mem);

/**
 * @brief Multiplies all elements in an array by the given value.
 *
 * @param mem Array to be updated.
 * @param value Value by which to multiply each element.
 * @param status Error status.
 */
void sdp_mem_scale_real(sdp_Mem* mem, double value, sdp_Error* status);

/**
 * @brief Set the flag specifying whether the memory should be read-only.
 *
 * @param mem Handle to memory block.
 * @param value True or false value of the read-only flag to set.
 */
void sdp_mem_set_read_only(sdp_Mem* mem, int32_t value);

/**
 * @brief Sets values in an array.
 *
 * For large arrays, this is very much faster than calling memset().
 *
 * @param mem Array to be set.
 * @param value Value to set in each element.
 * @param status Error status.
 */
void sdp_mem_set_value(sdp_Mem* mem, int value, sdp_Error* status);

/**
 * @brief Returns the number of elements in the specified dimension.
 *
 * The slowest varying dimension is the first index (0), and
 * the fastest varying dimension is the last index (num_dims - 1).
 *
 * @param mem Handle to memory block.
 * @param dim The dimension index to return.
 * @return int64_t Number of elements in the specified dimension.
 */
int64_t sdp_mem_shape_dim(const sdp_Mem* mem, int32_t dim);

/**
 * @brief Returns the stride (in bytes) of the specified dimension.
 *
 * The slowest varying dimension is the first index (0), and
 * the fastest varying dimension is the last index (num_dims - 1).
 *
 * This is in bytes for compatibility with Python.
 *
 * @param mem Handle to memory block.
 * @param dim The dimension index to return.
 * @return int64_t Stride in bytes for the specified dimension.
 */
int64_t sdp_mem_stride_bytes_dim(const sdp_Mem* mem, int32_t dim);

/**
 * @brief Returns the stride (in elements) of the specified dimension.
 *
 * The slowest varying dimension is the first index (0), and
 * the fastest varying dimension is the last index (num_dims - 1).
 *
 * @param mem Handle to memory block.
 * @param dim The dimension index to return.
 * @return int64_t Stride in elements for the specified dimension.
 */
int64_t sdp_mem_stride_elements_dim(const sdp_Mem* mem, int32_t dim);

/**
 * @brief Returns the enumerated data type of the memory.
 *
 * @param mem Handle to memory block.
 * @return ::sdp_MemType The enumerated element data type.
 */
sdp_MemType sdp_mem_type(const sdp_Mem* mem);

/**
 * @brief Returns the size of one element of a data type, in bytes.
 *
 * @param type Enumerated data type.
 * @return int64_t Size of data type in bytes.
 */
int64_t sdp_mem_type_size(sdp_MemType type);

/**
 * @brief Returns a string representation of memory location.
 *
 * @param location Enumerated memory location.
 * @return String representation.
 */
const char* sdp_mem_location_name(sdp_MemLocation location);

/**
 * @brief Returns a string representation of data type.
 *
 * @param type Enumerated memory data type.
 * @return String representation.
 */
const char* sdp_mem_type_name(sdp_MemType type);

/**
 * @brief Checks that an array is writeable.
 *
 * Use the ::sdp_mem_check_writeable macro to automatically
 * fill @p expr, @p func, @p file and @p line by call location.
 * @p status will be set if the check fails.
 *
 * @param mem Handle to memory block to check.
 * @param status Output error status.
 * @param expr Expression string to report in error message.
 * @param func Function to report in error message.
 * @param file File name to report in error message.
 * @param line Line to report in error message.
 */
void sdp_mem_check_writeable_at(
        const sdp_Mem* mem,
        sdp_Error* status,
        const char* expr,
        const char* func,
        const char* file,
        int line
);

/**
 * @brief Checks that an array is writeable.
 *
 * @p status will be set if the check fails.
 *
 * @param mem Handle to memory block to check.
 * @param status Output error status.
 */
#define sdp_mem_check_writeable(mem, status) \
    sdp_mem_check_writeable_at(mem, \
        status, \
        #mem, \
        __func__, \
        FILENAME, \
        __LINE__ \
    )

/**
 * @brief Checks that an array is C contiguous.
 *
 * Use the ::sdp_mem_check_c_contiguity macro to automatically
 * fill @p expr, @p func, @p file and @p line by call location.
 * @p status will be set if the check fails.
 *
 * @param mem Handle to memory block to check.
 * @param status Output error status.
 * @param expr Expression string to report in error message.
 * @param func Function to report in error message.
 * @param file File name to report in error message.
 * @param line Line to report in error message.
 */
void sdp_mem_check_c_contiguity_at(
        const sdp_Mem* mem,
        sdp_Error* status,
        const char* expr,
        const char* func,
        const char* file,
        int line
);

/**
 * @brief Checks that an array is C contiguous.
 *
 * @p status will be set if the check fails.
 *
 * @param mem Handle to memory block to check.
 * @param status Output error status.
 */
#define sdp_mem_check_c_contiguity(mem, status) \
    sdp_mem_check_c_contiguity_at(mem, \
        status, \
        #mem, \
        __func__, \
        FILENAME, \
        __LINE__ \
    )

/**
 * @brief Checks that an array resides in the expected memory space.
 *
 * Use the ::sdp_mem_check_location macro to automatically
 * fill @p expr, @p func, @p file and @p line by call location.
 * @p status will be set if the check fails.
 *
 * @param mem Handle to memory block to check.
 * @param expected_location Expected memory location.
 * @param status Output error status.
 * @param expr Expression string to report in error message.
 * @param func Function to report in error message.
 * @param file File name to report in error message.
 * @param line Line to report in error message.
 */
void sdp_mem_check_location_at(
        const sdp_Mem* mem,
        sdp_MemLocation expected_location,
        sdp_Error* status,
        const char* expr,
        const char* func,
        const char* file,
        int line
);

/**
 * @brief Checks that an array resides in the expected memory space
 *
 * @p status will be set if the check fails.
 *
 * @param mem Handle to memory block to check.
 * @param expected_location Expected memory location.
 * @param status Output error status.
 */
#define sdp_mem_check_location(mem, expected_location, status) \
    sdp_mem_check_location_at(mem, \
        expected_location, \
        status, \
        #mem, \
        __func__, \
        FILENAME, \
        __LINE__ \
    )

/**
 * @brief Checks that an array has expected number of dimensions.
 *
 * Use the ::sdp_mem_check_num_dims macro to automatically
 * fill @p expr, @p func, @p file and @p line by call location.
 * @p status will be set if the check fails.
 *
 * @param mem Handle to memory block to check.
 * @param expected_num_dims Expected number of dimensions.
 * @param status Output error status.
 * @param expr Expression string to report in error message.
 * @param func Function to report in error message.
 * @param file File name to report in error message.
 * @param line Line to report in error message.
 */
void sdp_mem_check_num_dims_at(
        const sdp_Mem* mem,
        int64_t expected_num_dims,
        sdp_Error* status,
        const char* expr,
        const char* func,
        const char* file,
        int line
);

/**
 * @brief Checks that an array has expected number of dimensions.
 *
 * @p status will be set if the check fails.
 *
 * @param mem Handle to memory block to check.
 * @param expected_num_dims Expected number of dimensions.
 * @param status Output error status.
 */
#define sdp_mem_check_num_dims(mem, expected_num_dims, status) \
    sdp_mem_check_num_dims_at(mem, \
        expected_num_dims, \
        status, \
        #mem, \
        __func__, \
        FILENAME, \
        __LINE__ \
    )

/**
 * @brief Checks that a given dimension of an array has the expected size.
 *
 * Use the ::sdp_mem_check_dim_size macro to automatically
 * fill @p expr, @p func, @p file and @p line by call location.
 * @p status will be set if the check fails.
 *
 * @param mem Handle to memory block to check.
 * @param dim Dimension index to check.
 * @param size Expected size of specified dimension.
 * @param status Output error status.
 * @param expr Expression string to report in error message.
 * @param func Function to report in error message.
 * @param file File name to report in error message.
 * @param line Line to report in error message.
 */
void sdp_mem_check_dim_size_at(
        const sdp_Mem* mem,
        int32_t dim,
        int64_t size,
        sdp_Error* status,
        const char* expr,
        const char* func,
        const char* file,
        int line
);

/**
 * @brief Checks that a given dimension of an array has the expected size.
 *
 * @p status will be set if the check fails.
 *
 * @param mem Handle to memory block to check.
 * @param dim Dimension index to check.
 * @param size Expected size of specified dimension.
 * @param status Output error status.
 */
#define sdp_mem_check_dim_size(mem, dim, size, status) \
    sdp_mem_check_dim_size_at(mem, \
        dim, \
        size, \
        status, \
        #mem, \
        __func__, \
        FILENAME, \
        __LINE__ \
    )

/**
 * @brief Checks that an array has the expected shape.
 *
 * Use the ::sdp_mem_check_shape macro to automatically
 * fill @p expr, @p func, @p file and @p line by call location.
 * @p status will be set if the check fails.
 *
 * @param mem Handle to memory block to check.
 * @param expected_num_dims Expected number of dimensions.
 * @param expected_shape Expected size of each dimension.
 * @param status Output error status.
 * @param expr Expression string to report in error message.
 * @param func Function to report in error message.
 * @param file File name to report in error message.
 * @param line Line to report in error message.
 */
void sdp_mem_check_shape_at(
        const sdp_Mem* mem,
        int32_t expected_num_dims,
        const int64_t* expected_shape,
        sdp_Error* status,
        const char* expr,
        const char* func,
        const char* file,
        int line
);

/**
 * @brief Checks that an array has the expected shape.
 *
 * @p status will be set if the check fails.
 *
 * @param mem Handle to memory block to check.
 * @param expected_num_dims Expected number of dimensions.
 * @param expected_shape Expected size of each dimension.
 * @param status Output error status.
 */
#define sdp_mem_check_shape(mem, expected_num_dims, expected_shape, status) \
    sdp_mem_check_shape_at(mem, \
        expected_num_dims, \
        expected_shape, \
        status, \
        #mem, \
        __func__, \
        FILENAME, \
        __LINE__ \
    )

/**
 * @brief Checks that an array has the expected shape for
 * a certain dimension.
 *
 * Use the ::sdp_mem_check_shape_dim macro to automatically
 * fill @p expr, @p func, @p file and @p line by call location.
 * @p status will be set if the check fails.
 *
 * @param mem Handle to memory block to check.
 * @param expected_num_dims Expected number of dimensions.
 * @param expected_shape Expected size of each dimension.
 * @param status Output error status.
 * @param expr Expression string to report in error message.
 * @param func Function to report in error message.
 * @param file File name to report in error message.
 * @param line Line to report in error message.
 */
void sdp_mem_check_shape_dim_at(
        const sdp_Mem* mem,
        int32_t dim,
        const int64_t expected_shape,
        sdp_Error* status,
        const char* expr,
        const char* func,
        const char* file,
        int line
);

/**
 * @brief Checks that an array has the expected shape for
 * a certain dimension.
 *
 * @p status will be set if the check fails.
 *
 * @param mem Handle to memory block to check.
 * @param expected_num_dims Expected number of dimensions.
 * @param expected_shape Expected size of each dimension.
 * @param status Output error status.
 */
#define sdp_mem_check_shape_dim(mem, expected_num_dims, expected_shape, status) \
    sdp_mem_check_shape_dim_at(mem, \
        expected_num_dims, \
        expected_shape, \
        status, \
        #mem, \
        __func__, \
        FILENAME, \
        __LINE__ \
    )

/**
 * @brief Checks that dimensions of two given memory objects have the
 * same size
 *
 * ``status`` will be set if the check fails. Note that this function
 * will *not* fail if the dimension in question does not exist on
 * either memory object, use #sdp_mem_check_num_dims additionally. Use
 * #sdp_mem_check_same_shape macro to automatically fill ``func``,
 * ``expr``, ``file`` and ``line`` by call location.
 *
 * @param mem Handle to memory block to check
 * @param dim Dimension to check
 * @param mem2 Handle to memory block to check against
 * @param dim2 Dimension to check against
 * @param status Output error status
 * @param func Function to report in error message
 * @param expr Expression string to report in error message
 * @param expr2 Second expression string to report in error message
 * @param file File name to report in error message
 * @param line Line to report in error message
 */
void sdp_mem_check_same_shape_at(
        sdp_Mem* mem,
        int32_t dim,
        sdp_Mem* mem2,
        int32_t dim2,
        sdp_Error* status,
        const char* func,
        const char* expr,
        const char* expr2,
        const char* file,
        int line
);

/**
 * @brief Checks that a dimension of given memory has expected size
 *
 * ``status`` will be set if the check fails. Note that this function
 * will *not* fail if the dimension in question does not exist, use
 * #sdp_mem_check_num_dims additionally.
 *
 * @param mem Handle to memory block to check
 * @param dim Dimension to check
 * @param mem2 Handle to memory block to check against
 * @param dim2 Dimension to check against
 * @param status Output error status
 */
#define sdp_mem_check_same_shape(mem, dim, mem2, dim2, status) \
    sdp_mem_check_same_shape_at(mem, dim, mem2, dim2, status, __func__, \
        #mem, #mem2, __FILE__, __LINE__ \
    )

/**
 * @brief Checks that an array has the expected data type.
 *
 * Use the ::sdp_mem_check_type macro to automatically
 * fill @p expr, @p func, @p file and @p line by call location.
 * @p status will be set if the check fails.
 *
 * @param mem Handle to memory block to check.
 * @param expected_type Expected data type.
 * @param status Output error status.
 * @param expr Expression string to report in error message.
 * @param func Function to report in error message.
 * @param file File name to report in error message.
 * @param line Line to report in error message.
 */
void sdp_mem_check_type_at(
        const sdp_Mem* mem,
        sdp_MemType expected_type,
        sdp_Error* status,
        const char* expr,
        const char* func,
        const char* file,
        int line
);

/**
 * @brief Checks that an array has the expected data type.
 *
 * @p status will be set if the check fails.
 *
 * @param mem Handle to memory block to check.
 * @param expected_type Expected data type.
 * @param status Output error status.
 */
#define sdp_mem_check_type(mem, expected_type, status) \
    sdp_mem_check_type_at(mem, \
        expected_type, \
        status, \
        #mem, \
        __func__, \
        FILENAME, \
        __LINE__ \
    )

/** @} */ /* End group Mem_func. */

#ifdef __cplusplus
}
#endif

#endif /* include guard */
