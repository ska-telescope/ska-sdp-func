// Copyright 2022 High Performance Computing Research Laboratory, Auckland University of Technology (AUT)

// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:

// 1. Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimer.

// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.

// 3. Neither the name of the copyright holder nor the names of its
// contributors may be used to endorse or promote products derived from this
// software without specific prior written permission.

// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

/*****************************************************************************
 * Gaincalsimpletest.cu
 * Andrew Ensor
 * C with C++ templates/CUDA code for preparing simple test measured and predicted visibility data sets
 *****************************************************************************/

#include "Gaincalsimpletest.h"

/*****************************************************************************
 * Gain calibration function that allocates and clears the data structures that will
 * hold the visibility data set on the host.
 * Note should be paired with a later call to free_visibilities_host
 *****************************************************************************/
template<typename VIS_PRECISION2>
VIS_PRECISION2* allocate_visibilities_host
    (const unsigned int num_baselines)
{
    VIS_PRECISION2 *visibilities_host;
    CUDA_CHECK_RETURN(cudaHostAlloc(&visibilities_host, num_baselines*sizeof(VIS_PRECISION2), 0));
    memset(visibilities_host, 0, num_baselines*sizeof(VIS_PRECISION2));
    return visibilities_host;
}

template float2* allocate_visibilities_host<float2>(const unsigned int);
template double2* allocate_visibilities_host<double2>(const unsigned int);


/*****************************************************************************
 * Gain calibration function that generates some simple sample visibilities on the host
 *****************************************************************************/
template<typename VIS_PRECISION2>
void generate_sample_visibilities_host
    (
    VIS_PRECISION2 *visibilities_host, // output array of predicted visibilities
    const unsigned int num_baselines // number of baselines
    )
{
    for (unsigned int baseline=0; baseline<num_baselines; baseline++)
    {
        visibilities_host[baseline].x = 1; // let compiler typecast to VIS_PRECISION
        visibilities_host[baseline].y = 0; // let compiler typecast to VIS_PRECISION
    }
}

template void generate_sample_visibilities_host<float2>(float2*, const unsigned int);
template void generate_sample_visibilities_host<double2>(double2*, const unsigned int);


/*****************************************************************************
 * Gain calibration function that deallocates host data structure that was used to
 * hold a visibility data set on the host.
 * Note should be paired with an earlier call to allocate_visibilities_host
 *****************************************************************************/
template<typename VIS_PRECISION2>
void free_visibilities_host(VIS_PRECISION2 *visibilities_host)
{
    CUDA_CHECK_RETURN(cudaFreeHost(visibilities_host));
}

template void free_visibilities_host<float2>(float2*);
template void free_visibilities_host<double2>(double2*);


/*****************************************************************************
 * Gain calibration function that allocates and clears the data structure that will
 * hold the receiver pairs for each baseline on the host.
 * Note should be paired with a later call to free_receiver_pairs_host
 *****************************************************************************/
uint2* allocate_receiver_pairs_host
    (
    const unsigned int num_baselines // number of baselines
    )
{
    uint2 *receiver_pairs_host;
    CUDA_CHECK_RETURN(cudaHostAlloc(&receiver_pairs_host, num_baselines*sizeof(uint2), 0));
    memset(receiver_pairs_host, 0, num_baselines*sizeof(uint2));
    return receiver_pairs_host;
}


/*****************************************************************************
 * Gain calibration function that generates some simple sample visibilities on the host
 *****************************************************************************/
void generate_sample_receiver_pairs_host
    (
    uint2 *receiver_pairs_host, // output array giving receiver pair for each baseline
    const unsigned int num_baselines, // number of baselines
    const unsigned int num_receivers // number of receivers
    )
{
    unsigned int first_receiver = 0;
    unsigned int second_receiver = 1;
    for (unsigned int baseline=0; baseline<num_baselines; baseline++)
    {
        receiver_pairs_host[baseline].x = first_receiver;
        receiver_pairs_host[baseline].y = second_receiver;
        second_receiver++;
        if (second_receiver >= num_receivers)
        {
            first_receiver++;
            second_receiver = first_receiver+1;
        }
    }
}


/*****************************************************************************
 * Gain calibration function that deallocates host data structure that was used to
 * hold the receiver pairs for each baseline on the host.
 * Note should be paired with an earlier call to allocate_receiver_pairs_host
 *****************************************************************************/
void free_receiver_pairs_host(uint2 *receiver_pairs_host)
{
    CUDA_CHECK_RETURN(cudaFreeHost(receiver_pairs_host));
}


/*****************************************************************************
 * Gain calibration function that allocates and clears the data structure that will
 * hold a visibility data set on the device.
 * Note should be paired with a later call to free_visibilities_device
 *****************************************************************************/
template<typename VIS_PRECISION2>
VIS_PRECISION2* allocate_visibilities_device
    (const unsigned int num_baselines)
{
    VIS_PRECISION2 *visibilities_device;
    CUDA_CHECK_RETURN(cudaMalloc((void**)&visibilities_device, num_baselines*sizeof(VIS_PRECISION2)));
    CUDA_CHECK_RETURN(cudaMemset(visibilities_device, 0, num_baselines*sizeof(VIS_PRECISION2))); // clear the data to zero
    return visibilities_device;
}

template float2* allocate_visibilities_device<float2>(const unsigned int);
template double2* allocate_visibilities_device<double2>(const unsigned int);

/*****************************************************************************
 * Temporary utility function that calculates some measured and predicted visibilities for given gains
 *****************************************************************************/
template<typename VIS_PRECISION2, typename VIS_PRECISION, typename PRECISION2>
void calculate_measured_and_predicted_visibilities_device
    (
    VIS_PRECISION2 *vis_predicted_host, // input array of predicted visibilities
    uint2 *receiver_pairs_host, // input array giving receiver pair for each baseline
    const unsigned int num_baselines, // number of baselines
    PRECISION2 *actual_gains_host, // actual complex gains for each receiver
    const unsigned int num_receivers, // number of receivers
    VIS_PRECISION2 *vis_measured_device, // output array of measured visibilities
    VIS_PRECISION2 *vis_predicted_device // output array of preducted visibilities
    )
{
    VIS_PRECISION2 *vis_measured_host;
    CUDA_CHECK_RETURN(cudaHostAlloc(&vis_measured_host, num_baselines*sizeof(VIS_PRECISION2), 0));
    memset(vis_measured_host, 0, num_baselines*sizeof(VIS_PRECISION2));

    // calculate the measured visibilities from the predicted visibilities and the gains
    for (unsigned int baseline=0; baseline<num_baselines; baseline++)
    {
        // multiply predicted visibilities by conjugate of first receiver gain and then multiply by second receiver gain
        PRECISION2 first_gain = actual_gains_host[receiver_pairs_host[baseline].x];
        PRECISION2 second_gain = actual_gains_host[receiver_pairs_host[baseline].y];
        PRECISION2 temp_product;
        temp_product.x = vis_predicted_host[baseline].x*first_gain.x + vis_predicted_host[baseline].y*first_gain.y;
        temp_product.y = vis_predicted_host[baseline].x*first_gain.y - vis_predicted_host[baseline].y*first_gain.x;
        vis_measured_host[baseline].x = (VIS_PRECISION)(temp_product.x*second_gain.x - temp_product.y*second_gain.y);
        vis_measured_host[baseline].y = (VIS_PRECISION)(temp_product.x*second_gain.y + temp_product.y*second_gain.x);
    }

    // copy the test visibilities to the device
    CUDA_CHECK_RETURN(cudaMemcpy(vis_predicted_device, vis_predicted_host, num_baselines*sizeof(VIS_PRECISION2), cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(vis_measured_device, vis_measured_host, num_baselines*sizeof(VIS_PRECISION2), cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaFreeHost(vis_measured_host));
}

template void calculate_measured_and_predicted_visibilities_device<float2, float, float2>(float2*, uint2*, const unsigned int, float2*, const unsigned int, float2*, float2*);
template void calculate_measured_and_predicted_visibilities_device<double2, double, double2>(double2*, uint2*, const unsigned int, double2*, const unsigned int, double2*, double2*);


/*****************************************************************************
 * Gain calibration function that deallocates device data structure that was used to
 * hold a visibility data set on the device.
 * Note should be paired with an earlier call to allocate_visibilities_device
 *****************************************************************************/
template<typename VIS_PRECISION2>
void free_visibilities_device(VIS_PRECISION2 *visibilities_device)
{
    CUDA_CHECK_RETURN(cudaFree(visibilities_device));
}

template void free_visibilities_device<float2>(float2*);
template void free_visibilities_device<double2>(double2*);


/*****************************************************************************
 * Gain calibration function that allocates and clears the data structure that will
 * hold the receiver pairs for each baseline on the device.
 * Note should be paired with a later call to free_receiver_pairs_device
 *****************************************************************************/
uint2* allocate_receiver_pairs_device
    (
    const unsigned int num_baselines // number of baselines
    )
{
    uint2 *receiver_pairs_device;
    CUDA_CHECK_RETURN(cudaMalloc((void**)&receiver_pairs_device, num_baselines*sizeof(uint2)));
    CUDA_CHECK_RETURN(cudaMemset(receiver_pairs_device, 0, num_baselines*sizeof(uint2))); // clear the data to zero
    return receiver_pairs_device;
}


/*****************************************************************************
 * Gain calibration function that copies receiver pairs for each baseline from host to the device.
 *****************************************************************************/
void set_receiver_pairs_device
    (
    uint2 *receiver_pairs_host, // input array of receiver pairs for each baseline
    uint2 *receiver_pairs_device, // inout array of receiver pairs for each baseline
    const unsigned int num_baselines // number of baselines
    )
{
    CUDA_CHECK_RETURN(cudaMemcpy(receiver_pairs_device, receiver_pairs_host, num_baselines*sizeof(uint2), cudaMemcpyHostToDevice));
}


/*****************************************************************************
 * Gain calibration function that deallocates device data structure that was used to
 * hold the receiver pairs for each baseline on the device.
 * Note should be paired with an earlier call to allocate_receiver_pairs_device
 *****************************************************************************/
void free_receiver_pairs_device(uint2 *receiver_pairs_device)
{
    CUDA_CHECK_RETURN(cudaFree(receiver_pairs_device));
}


/*****************************************************************************
 * Gain calibration function that allocates the data structure that will
 * hold the calculated gains on the device and initialises each gain to 1+0i.
 * Note should be paired with a later call to free_gains_device
 *****************************************************************************/
template<typename PRECISION2>
PRECISION2* allocate_gains_device
    (const unsigned int num_receivers)
{
    PRECISION2 *gains_device;
    CUDA_CHECK_RETURN(cudaMalloc((void**)&gains_device, num_receivers*sizeof(PRECISION2)));
    // create the initial gains to be each 1+0i and copy to device
    PRECISION2 *initial_gains_host;
    CUDA_CHECK_RETURN(cudaHostAlloc(&initial_gains_host, num_receivers*sizeof(PRECISION2), 0));
    for (unsigned int receiver=0; receiver<num_receivers; receiver++)
    {
        initial_gains_host[receiver].x = 1; // let compiler typecast to PRECISION
        initial_gains_host[receiver].y = 0; // let compiler typecast to PRECISION
    }
    CUDA_CHECK_RETURN(cudaMemcpy(gains_device, initial_gains_host, num_receivers*sizeof(PRECISION2), cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaFreeHost(initial_gains_host));
    return gains_device;
}

template float2* allocate_gains_device<float2>(const unsigned int);
template double2* allocate_gains_device<double2>(const unsigned int);


/*****************************************************************************
 * Gain calibration function that displays the actual and calculated gains with
 * all the calculated gains rotated so receiver 0 has zero phase.
 *****************************************************************************/
template<typename PRECISION2>
void display_gains_actual_and_calculated
    (
    PRECISION2 *actual_gains_host, // actual complex gains for each receiver
    PRECISION2 *gains_device, // calculated gains
    const unsigned int num_receivers
    )
{
    PRECISION2 *calculated_gains_host;
    CUDA_CHECK_RETURN(cudaHostAlloc(&calculated_gains_host, num_receivers*sizeof(PRECISION2), 0));
    memset(calculated_gains_host, 0, num_receivers*sizeof(PRECISION2));
    CUDA_CHECK_RETURN(cudaMemcpy(calculated_gains_host, gains_device, num_receivers*sizeof(PRECISION2), cudaMemcpyDeviceToHost));
    // calculate the phase rotation required to align phase for receiver 0, just in float precision okay for display
    float rotationActualReal = (float)actual_gains_host[0].x
        / sqrt((float)(actual_gains_host[0].x*actual_gains_host[0].x+actual_gains_host[0].y*actual_gains_host[0].y));
    float rotationActualImag = (float)-actual_gains_host[0].y
        / sqrt((float)(actual_gains_host[0].x*actual_gains_host[0].x+actual_gains_host[0].y*actual_gains_host[0].y));
    float rotationCalculatedReal = (float)calculated_gains_host[0].x
        / sqrt((float)(calculated_gains_host[0].x*calculated_gains_host[0].x+calculated_gains_host[0].y*calculated_gains_host[0].y));
    float rotationCalculatedImag = (float)-calculated_gains_host[0].y
        / sqrt((float)(calculated_gains_host[0].x*calculated_gains_host[0].x+calculated_gains_host[0].y*calculated_gains_host[0].y));
    for (unsigned int receiver=0; receiver<num_receivers; receiver++)
    {
        float rotatedActualReal = (float)(actual_gains_host[receiver].x*rotationActualReal - actual_gains_host[receiver].y*rotationActualImag);
        float rotatedActualImag = (float)(actual_gains_host[receiver].x*rotationActualImag + actual_gains_host[receiver].y*rotationActualReal);
        float rotatedCalculatedReal = (float)(calculated_gains_host[receiver].x*rotationCalculatedReal - calculated_gains_host[receiver].y*rotationCalculatedImag);
        float rotatedCalculatedImag = (float)(calculated_gains_host[receiver].x*rotationCalculatedImag + calculated_gains_host[receiver].y*rotationCalculatedReal);
        printf("Receiver %u has actual gain (%+9.4lf,%9.4lf) and rotated calculated gain (%+9.4lf,%9.4lf)\n",
            receiver, rotatedActualReal, rotatedActualImag, rotatedCalculatedReal, rotatedCalculatedImag);
    }
    CUDA_CHECK_RETURN(cudaFreeHost(calculated_gains_host));
}

template void display_gains_actual_and_calculated<float2>(float2*, float2*, const unsigned int);
template void display_gains_actual_and_calculated<double2>(double2*, double2*, const unsigned int);


/*****************************************************************************
 * Gain calibration function that deallocates device data structure that was used to
 * hold the calculated gains on the device.
 * Note should be paired with an earlier call to allocate_visibilities_device
 *****************************************************************************/
template<typename PRECISION2>
void free_gains_device(PRECISION2 *gains_device)
{
    CUDA_CHECK_RETURN(cudaFree(gains_device));
}

template void free_gains_device<float2>(float2*);
template void free_gains_device<double2>(double2*);

