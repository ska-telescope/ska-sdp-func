/* See the LICENSE file at the top-level directory of this distribution. */

#include <complex>

#include "ska-sdp-func/rfi_flagger/sdp_rfi_flagger.h"
#include "ska-sdp-func/utility/sdp_device_wrapper.h"
#include "ska-sdp-func/utility/sdp_logging.h"

static void check_params(
        const sdp_Mem* vis,
        const sdp_Mem* thresholds,
        sdp_Mem*  flags,
        sdp_Error* status)
{
    if (*status) return;

    if (sdp_mem_is_read_only(flags)) {
        *status = SDP_ERR_DATA_TYPE;
        SDP_LOG_ERROR("Output flags must be writable.");
        return;
    }

    if (!sdp_mem_is_c_contiguous(vis) || !sdp_mem_is_c_contiguous(thresholds) || !sdp_mem_is_c_contiguous(flags)) {
        *status = SDP_ERR_DATA_TYPE;
        SDP_LOG_ERROR("All arrays must be C contiguous.");
        return;
    }
    
    if (sdp_mem_num_dims(vis) != 4) {
        *status = SDP_ERR_DATA_TYPE;
        SDP_LOG_ERROR("Visibility array must by 4D.");
        return;
    }
    
    if( !sdp_mem_is_complex(vis) ) {
        *status = SDP_ERR_DATA_TYPE;
        SDP_LOG_ERROR("Visibilities must be complex.");
        return;
    }
    
    if (sdp_mem_num_dims(flags) != 4) {
        *status = SDP_ERR_DATA_TYPE;
        SDP_LOG_ERROR("Flag array must by 4D.");
        return;
    }
    
    if(sdp_mem_type(flags) != SDP_MEM_INT) {
        *status = SDP_ERR_DATA_TYPE;
        SDP_LOG_ERROR("Flags must be integers.");
        return;
    }
    
    if (
        (sdp_mem_type(vis) == SDP_MEM_COMPLEX_FLOAT && sdp_mem_type(thresholds) != SDP_MEM_FLOAT)
        ||
        (sdp_mem_type(vis) == SDP_MEM_COMPLEX_DOUBLE && sdp_mem_type(thresholds) != SDP_MEM_DOUBLE)
    ) {
        *status = SDP_ERR_DATA_TYPE;
        SDP_LOG_ERROR("Visibilities and thresholds arrays must have same precision.");
        return;
    }
    
    if (sdp_mem_location(vis) == SDP_MEM_GPU) {
        *status = SDP_ERR_MEM_LOCATION;
        SDP_LOG_ERROR("GPU is currently not supported.");
        return;
    }
    
    if (
        ( sdp_mem_location(vis) == SDP_MEM_CPU && (sdp_mem_location(thresholds)!=SDP_MEM_CPU || sdp_mem_location(flags) != SDP_MEM_CPU) ) 
    ) {
        *status = SDP_ERR_MEM_LOCATION;
        SDP_LOG_ERROR("All array's must be in the same memory location.");
        return;
    }
}

template<typename TCPU>
void  sum_threshold_rfi_flagger(
        int*  flags,
        const std::complex<TCPU>* const __restrict__ visibilities,
        const TCPU* const __restrict__ thresholds,
        const uint64_t num_timesamples,
        const uint64_t num_baselines,
        const uint64_t num_channels,
        const uint64_t num_polarisations,
        const uint64_t seqlen)
{

}


void sdp_sum_threshold_rfi_flagger(
        const sdp_Mem* vis,
        const sdp_Mem* thresholds,
        sdp_Mem* flags,
        const int64_t max_sequence_length,
        sdp_Error* status)
{
    check_params(vis, thresholds, flags, status);
    if (*status) return;

    const uint64_t num_timesamples   = (uint64_t) sdp_mem_shape_dim(vis, 0);
    const uint64_t num_baselines     = (uint64_t) sdp_mem_shape_dim(vis, 1);
    const uint64_t num_channels      = (uint64_t) sdp_mem_shape_dim(vis, 2);
    const uint64_t num_polarisations = (uint64_t) sdp_mem_shape_dim(vis, 3);
    
    if (sdp_mem_location(vis) == SDP_MEM_CPU) {
        if (sdp_mem_type(vis) == SDP_MEM_COMPLEX_FLOAT){
            sum_threshold_rfi_flagger(
                (int*) sdp_mem_data(flags),
                (const std::complex<float>*) sdp_mem_data_const(vis),
                (const float*) sdp_mem_data_const(thresholds),
                num_timesamples,
                num_baselines,
                num_channels,
                num_polarisations,
                max_sequence_length
            );
        }
        else if (sdp_mem_type(vis) == SDP_MEM_COMPLEX_DOUBLE) {
            sum_threshold_rfi_flagger(
                (int*) sdp_mem_data(flags),
                (const std::complex<double>*) sdp_mem_data_const(vis),
                (const double*) sdp_mem_data_const(thresholds),
                num_timesamples,
                num_baselines,
                num_channels,
                num_polarisations,
                max_sequence_length
            );
        }
        else {
            *status = SDP_ERR_DATA_TYPE;
            SDP_LOG_ERROR("Unsupported data type");
        }
    }
    //else if (sdp_mem_location(vis) == SDP_MEM_GPU) {
    //    const char* kernel_name = 0;
    //    if (sdp_mem_type(vis) == SDP_MEM_COMPLEX_DOUBLE) {
    //        kernel_name = "rfi_flagger<double,double2>";
    //    }
    //    else if (sdp_mem_type(vis) == SDP_MEM_COMPLEX_FLOAT) {
    //        kernel_name = "rfi_flagger<float,float2>";
    //    }
    //    else {
    //        *status = SDP_ERR_DATA_TYPE;
    //        SDP_LOG_ERROR("Unsupported data type");
    //    }
    //    
    //    
    //}
    else {
        *status = SDP_ERR_RUNTIME;
        SDP_LOG_ERROR("Unknown memory location for visibility data.");
        return;
    }
}
