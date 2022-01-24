/* See the LICENSE file at the top-level directory of this distribution. */

#ifndef SKA_SDP_PROC_FUNC_ERRORS_H_
#define SKA_SDP_PROC_FUNC_ERRORS_H_

/**
 * @brief Processing function library error codes.
 */
enum sdp_Error
{
    //! No error.
    SDP_SUCCESS = 0,

    //! Generic runtime error.
    SDP_ERR_RUNTIME,

    //! Invalid function argument.
    SDP_ERR_INVALID_ARGUMENT,

    //! Unsupported data type.
    SDP_ERR_DATA_TYPE,

    //! Memory allocation failure.
    SDP_ERR_MEM_ALLOC_FAILURE,

    //! Memory copy failure.
    SDP_ERR_MEM_COPY_FAILURE,

    //! Unsupported memory location.
    SDP_ERR_MEM_LOCATION
};

typedef enum sdp_Error sdp_Error;

#endif /* include guard */
