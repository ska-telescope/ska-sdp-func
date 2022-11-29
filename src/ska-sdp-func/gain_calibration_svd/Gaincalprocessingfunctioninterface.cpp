/*
 * Gaincalprocessingfunctioninterface.cpp
 * Andrew Ensor
 * C with C++ templates/CUDA program providing an SDP processing function interface steps for the Gain Calibration algorithm
*/

#include "Gaincalprocessingfunctioninterface.h"
#include "Gaincallogger.h"
#include "Gaincalfunctionshost.h"
#include "Gaincalsimpletest.h"

/*
 * C (untemplated) version of the function which performs the entire gain calibration using sdp_Mem handles
 */
void perform_gaincalibration
    (
    sdp_Mem *vis_measured_device, // input data of measured visibilities
    sdp_Mem *vis_predicted_device, // input data of preducted visibilities
    sdp_Mem *receiver_pairs_device, // input data giving receiver pair for each baseline
    const unsigned int num_receivers, // number of receivers
    const unsigned int num_baselines, // number of baselines
    const unsigned int max_calibration_cycles, // maximum number of calibration cycles to perform
    sdp_Mem *gains_device // output data of calculated complex gains
    )
{
    if (sdp_mem_location(vis_measured_device)!=SDP_MEM_GPU || sdp_mem_location(vis_predicted_device)!=SDP_MEM_GPU
        || sdp_mem_location(receiver_pairs_device)!=SDP_MEM_GPU || sdp_mem_location(gains_device)!=SDP_MEM_GPU)
    {
        logger(LOG_CRIT, "Measured visibilities, predicted visibilities, receiver pairs, and gains must all be held in GPU device memory");
    }
    else if (sdp_mem_type(receiver_pairs_device)!=SDP_MEM_INT)
    {
        logger(LOG_CRIT, "Receiver pairs must be pairs of (unsigned) integer values");
    }
    else if (sdp_mem_type(vis_measured_device)==SDP_MEM_COMPLEX_FLOAT && sdp_mem_type(vis_predicted_device)==SDP_MEM_COMPLEX_FLOAT
        && sdp_mem_type(gains_device)==SDP_MEM_COMPLEX_FLOAT)
    {
        // calculate suitable cuda block size in 1D and number of available cuda threads
        int cuda_block_size;
        int cuda_num_threads;
        calculate_cuda_configs(&cuda_block_size, &cuda_num_threads);

        Jacobian_SVD_matrices<float> jacobian_svd_matrices = allocate_jacobian_svd_matrices<float>(num_receivers);

        perform_gain_calibration<float2, float2, float>
            ((float2 *)sdp_mem_data(vis_measured_device), (float2 *)sdp_mem_data(vis_predicted_device),
            (uint2 *)sdp_mem_data(receiver_pairs_device), num_receivers, num_baselines, max_calibration_cycles,
            jacobian_svd_matrices, (float2 *)sdp_mem_data(gains_device), true, cuda_block_size);

        free_jacobian_svd_matrices(jacobian_svd_matrices);
    }
    else if (sdp_mem_type(vis_measured_device)==SDP_MEM_COMPLEX_DOUBLE && sdp_mem_type(vis_predicted_device)==SDP_MEM_COMPLEX_DOUBLE
        && sdp_mem_type(gains_device)==SDP_MEM_COMPLEX_DOUBLE)
    {
        // calculate suitable cuda block size in 1D and number of available cuda threads
        int cuda_block_size;
        int cuda_num_threads;
        calculate_cuda_configs(&cuda_block_size, &cuda_num_threads);

        Jacobian_SVD_matrices<double> jacobian_svd_matrices = allocate_jacobian_svd_matrices<double>(num_receivers);

        perform_gain_calibration<double2, double2, double>
            ((double2 *)sdp_mem_data(vis_measured_device), (double2 *)sdp_mem_data(vis_predicted_device),
            (uint2 *)sdp_mem_data(receiver_pairs_device), num_receivers, num_baselines, max_calibration_cycles,
            jacobian_svd_matrices, (double2 *)sdp_mem_data(gains_device), true, cuda_block_size);

        free_jacobian_svd_matrices(jacobian_svd_matrices);
    }
    else
    {
        logger(LOG_CRIT, "Measured and predicted visibilities and gains must all be either SDP_MEM_COMPLEX_FLOAT or else SDP_MEM_COMPLEX_DOUBLE");
    }
}
