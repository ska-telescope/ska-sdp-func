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
 * Gaincalfunctionsdevice.cu
 * Andrew Ensor
 * CUDA device functions for the Gain calibration algorithm
 *****************************************************************************/

#include "Gaincalfunctionsdevice.h"


/*****************************************************************************
 * Device function that returns the complex conjugate of a complex number
 *****************************************************************************/
template<typename PRECISION2>
__device__ PRECISION2 complex_conjugate(const PRECISION2 z)
{
    PRECISION2 z_conjugate;
    z_conjugate.x = z.x;
    z_conjugate.y = -z.y;
    return z_conjugate;
}


/*****************************************************************************
 * Device function that returns the complex product of two complex numbers
 *****************************************************************************/
template<typename PRECISION2>
__device__ PRECISION2 complex_multiply(const PRECISION2 z1, const PRECISION2 z2)
{
    PRECISION2 complex_product;
    complex_product.x = z1.x * z2.x - z1.y * z2.y;
    complex_product.y = z1.x * z2.y + z1.y * z2.x;
    return complex_product;
}


/*****************************************************************************
 * Device function that returns the complex subtraction of two complex numbers
 *****************************************************************************/
template<typename PRECISION2>
__device__ PRECISION2 complex_subtract(const PRECISION2 z1, const PRECISION2 z2)
{
    PRECISION2 complex_difference;
    complex_difference.x = z1.x - z2.x;
    complex_difference.y = z1.y - z2.y;
    return complex_difference;
}


/*****************************************************************************
 * Device function that returns the complex product of a complex number by positive i
 *****************************************************************************/
template<typename PRECISION2>
__device__ PRECISION2 complex_multiply_by_pos_i(const PRECISION2 z)
{
    PRECISION2 complex_product;
    complex_product.x = -z.y;
    complex_product.y = z.x;
    return complex_product;
}


/*****************************************************************************
 * Device function that returns the complex product of a complex number by negative i
 *****************************************************************************/
template<typename PRECISION2>
__device__ PRECISION2 complex_multiply_by_neg_i(const PRECISION2 z)
{
    PRECISION2 complex_product;
    complex_product.x = z.y;
    complex_product.y = -z.x;
    return complex_product;
}


/*****************************************************************************
 * Resets the gain for each receiver to default 1+0i 
 * Parallelised so each CUDA thread resets a single receiver gain
 *****************************************************************************/
template<typename PRECISION2>
__global__ void reset_gains
    (
    PRECISION2 *gains, // output complex gain for each receiver
    const unsigned int num_receivers // total number of receivers
    )
{
    const unsigned int receiver = blockIdx.x*blockDim.x + threadIdx.x;
    if (receiver < num_receivers)
    {
        gains[receiver].x = 1;
        gains[receiver].y = 0;
    }
}

template __global__ void reset_gains<float2>(float2*, const unsigned int);
template __global__ void reset_gains<double2>(double2*, const unsigned int);


/*****************************************************************************
 * Updates the gain calibrations though one gain calibration cycle
 * Note it is presumed there are num_baselines measured and predicted visibilities
 * Parallelised so each CUDA thread handles a single baseline/visibility
 *****************************************************************************/
template<typename VIS_PRECISION2, typename PRECISION2, typename PRECISION>
__global__ void update_gain_calibration
    (
    const VIS_PRECISION2 *vis_measured_device, // input array of measured visibilities
    const VIS_PRECISION2 *vis_predicted_device, // input array of preducted visibilities
    const PRECISION2 *gains_device, // output array of calculated complex gains
    const uint2 *receiver_pairs_device, // input array giving receiver pair for each baseline
    PRECISION *jacobtjacob, // inout 2Nx2N Jacobian^transpose * Jacobian matrix 
    PRECISION *jacobtresidual, // inout 2Nx1 Jacobian^transpose * Residuals matrix
    const unsigned int num_receivers, // number of receivers
    const unsigned int num_baselines // number of baselines
    )
{
    const int index = threadIdx.x + blockDim.x * blockIdx.x;
    if(index >= num_baselines)
        return;
    PRECISION2 vis_measured;
    vis_measured.x = (PRECISION)vis_measured_device[index].x; vis_measured.y = (PRECISION)vis_measured_device[index].y;
    PRECISION2 vis_predicted;
    vis_predicted.x = (PRECISION)vis_predicted_device[index].x; vis_predicted.y = (PRECISION)vis_predicted_device[index].y;
    uint2 antennas = receiver_pairs_device[index];
    PRECISION2 gainA = gains_device[antennas.x];
    PRECISION2 gainB_conjugate = complex_conjugate(gains_device[antennas.y]);

    // note do not treat residual as a complex but as two reals
    PRECISION2 residual = complex_subtract(vis_measured, complex_multiply(vis_predicted,complex_multiply(gainA, gainB_conjugate)));

    // calculate partial derivatives
    PRECISION2 part_respect_to_real_gain_a = complex_multiply(vis_predicted, gainB_conjugate);
    PRECISION2 part_respect_to_imag_gain_a = complex_multiply_by_pos_i(complex_multiply(vis_predicted, gainB_conjugate));
    PRECISION2 part_respect_to_real_gain_b = complex_multiply(vis_predicted,gainA);
    PRECISION2 part_respect_to_imag_gain_b = complex_multiply_by_neg_i(complex_multiply(vis_predicted, gainA));

    // calculate Q[2a],Q[2a+1],Q[2b],Q[2b+1] arrays in this order and note need atomic update 
    double qValue = part_respect_to_real_gain_a.x * residual.x + part_respect_to_real_gain_a.y * residual.y;
    atomicAdd(&(jacobtresidual[2*antennas.x]), qValue);
    qValue = part_respect_to_imag_gain_a.x * residual.x + part_respect_to_imag_gain_a.y * residual.y;
    atomicAdd(&(jacobtresidual[2*antennas.x+1]), qValue);
    qValue = part_respect_to_real_gain_b.x * residual.x + part_respect_to_real_gain_b.y * residual.y;
    atomicAdd(&(jacobtresidual[2*antennas.y]), qValue);
    qValue = part_respect_to_imag_gain_b.x * residual.x + part_respect_to_imag_gain_b.y * residual.y;
    atomicAdd(&(jacobtresidual[2*antennas.y+1]), qValue);

    // calculate Jacobian product on A matrix at [2a,2a], [2a,2a+1], [2a,2b], [2a,2b+1]
    uint num_cols = 2*num_receivers;
    // 2a,2a
    double aValue = part_respect_to_real_gain_a.x * part_respect_to_real_gain_a.x
        + part_respect_to_real_gain_a.y * part_respect_to_real_gain_a.y;
    uint aIndex = 2*antennas.x*num_cols + 2*antennas.x;
    atomicAdd(&(jacobtjacob[aIndex]), aValue);

    // 2a,2a+1
    aValue = part_respect_to_real_gain_a.x * part_respect_to_imag_gain_a.x
        + part_respect_to_real_gain_a.y * part_respect_to_imag_gain_a.y; 
    aIndex = 2*antennas.x*num_cols + 2*antennas.x+1;
    atomicAdd(&(jacobtjacob[aIndex]), aValue);

    // 2a,2b
    aValue = part_respect_to_real_gain_a.x * part_respect_to_real_gain_b.x
        + part_respect_to_real_gain_a.y * part_respect_to_real_gain_b.y;
    aIndex = 2*antennas.x*num_cols + 2*antennas.y;
    atomicAdd(&(jacobtjacob[aIndex]), aValue);

    // 2a,2b+1
    aValue = part_respect_to_real_gain_a.x * part_respect_to_imag_gain_b.x
        + part_respect_to_real_gain_a.y * part_respect_to_imag_gain_b.y;
    aIndex = 2*antennas.x*num_cols + 2*antennas.y+1;
    atomicAdd(&(jacobtjacob[aIndex]), aValue);

    // calculate Jacobian product on A matrix at [2a+1,2a], [2a+1,2a+1], [2a+1,2b], [2a+1,2b+1]
    // 2a+1, 2a
    aValue = part_respect_to_imag_gain_a.x * part_respect_to_real_gain_a.x
        + part_respect_to_imag_gain_a.y * part_respect_to_real_gain_a.y; 
    aIndex = (2*antennas.x+1)*num_cols + 2*antennas.x;
    atomicAdd(&(jacobtjacob[aIndex]), aValue);

    // 2a+1, 2a+1
    aValue = part_respect_to_imag_gain_a.x * part_respect_to_imag_gain_a.x
        + part_respect_to_imag_gain_a.y * part_respect_to_imag_gain_a.y; 
    aIndex = (2*antennas.x+1)*num_cols + 2*antennas.x+1;
    atomicAdd(&(jacobtjacob[aIndex]), aValue);

    // 2a+1, 2b
    aValue = part_respect_to_imag_gain_a.x * part_respect_to_real_gain_b.x
        + part_respect_to_imag_gain_a.y * part_respect_to_real_gain_b.y;
    aIndex = (2*antennas.x+1)*num_cols + 2*antennas.y;
    atomicAdd(&(jacobtjacob[aIndex]), aValue);

    // 2a+1, 2b+1
    aValue = part_respect_to_imag_gain_a.x * part_respect_to_imag_gain_b.x
        + part_respect_to_imag_gain_a.y * part_respect_to_imag_gain_b.y;
    aIndex = (2*antennas.x+1)*num_cols + 2*antennas.y+1;
    atomicAdd(&(jacobtjacob[aIndex]), aValue);

    // calculate Jacobian product on A matrix at [2b,2a], [2b,2a+1], [2b,2b], [2b,2b+1]
    // 2b, 2a
    aValue = part_respect_to_real_gain_b.x * part_respect_to_real_gain_a.x
        + part_respect_to_real_gain_b.y * part_respect_to_real_gain_a.y; 
    aIndex = 2*antennas.y*num_cols + 2*antennas.x;
    atomicAdd(&(jacobtjacob[aIndex]), aValue);

    // 2b, 2a+1
    aValue = part_respect_to_real_gain_b.x * part_respect_to_imag_gain_a.x
        + part_respect_to_real_gain_b.y * part_respect_to_imag_gain_a.y;
    aIndex = 2*antennas.y*num_cols + 2*antennas.x+1;
    atomicAdd(&(jacobtjacob[aIndex]), aValue);
	
    // 2b, 2b
    aValue = part_respect_to_real_gain_b.x * part_respect_to_real_gain_b.x
        + part_respect_to_real_gain_b.y * part_respect_to_real_gain_b.y;
    aIndex = 2*antennas.y*num_cols + 2*antennas.y;
    atomicAdd(&(jacobtjacob[aIndex]), aValue);

    // 2b, 2b+1
    aValue = part_respect_to_real_gain_b.x * part_respect_to_imag_gain_b.x
        + part_respect_to_real_gain_b.y * part_respect_to_imag_gain_b.y;
    aIndex = 2*antennas.y*num_cols + 2*antennas.y+1;
    atomicAdd(&(jacobtjacob[aIndex]), aValue);

    // calculate Jacobian product on A matrix at [2b+1,2a], [2b+1,2a+1], [2b+1,2b], [2b+1,2b+1]
    // 2b+1, 2a
    aValue =  part_respect_to_imag_gain_b.x * part_respect_to_real_gain_a.x
        + part_respect_to_imag_gain_b.y * part_respect_to_real_gain_a.y; 
    aIndex = (2*antennas.y+1)*num_cols + 2*antennas.x;
    atomicAdd(&(jacobtjacob[aIndex]), aValue);

    // 2b+1, 2a+1
    aValue =  part_respect_to_imag_gain_b.x * part_respect_to_imag_gain_a.x
        + part_respect_to_imag_gain_b.y * part_respect_to_imag_gain_a.y;
    aIndex = (2*antennas.y+1)*num_cols + 2*antennas.x+1;
    atomicAdd(&(jacobtjacob[aIndex]), aValue);

    // 2b+1, 2b
    aValue =  part_respect_to_imag_gain_b.x * part_respect_to_real_gain_b.x
        + part_respect_to_imag_gain_b.y * part_respect_to_real_gain_b.y;
    aIndex = (2*antennas.y+1)*num_cols + 2*antennas.y;
    atomicAdd(&(jacobtjacob[aIndex]), aValue);

    // 2b+1, 2b+1
    aValue = part_respect_to_imag_gain_b.x * part_respect_to_imag_gain_b.x
        + part_respect_to_imag_gain_b.y * part_respect_to_imag_gain_b.y; 
    aIndex = (2*antennas.y+1)*num_cols + 2*antennas.y+1;
    atomicAdd(&(jacobtjacob[aIndex]), aValue);
}

template __global__ void update_gain_calibration<half2, float2, float>
    (const half2*, const half2*, const float2*, const uint2*, float*, float*, const unsigned int, const unsigned int);
template __global__ void update_gain_calibration<float2, float2, float>
    (const float2*, const float2*, const float2*, const uint2*, float*, float*, const unsigned int, const unsigned int);
template __global__ void update_gain_calibration<float2, double2, double>
    (const float2*, const float2*, const double2*, const uint2*, double*, double*, const unsigned int, const unsigned int);
template __global__ void update_gain_calibration<double2, double2, double>
    (const double2*, const double2*, const double2*, const uint2*, double*, double*, const unsigned int, const unsigned int);


/*****************************************************************************
 * Calculates the matrix product diagonalS inverse * unitaryU transpose * jacobresidual
 * Parallelised so each CUDA thread handles one of the 2N entries in the product
 * where N is the number of receivers
 *****************************************************************************/
template<typename PRECISION>
__global__ void calculate_product_sujr
    (
    const PRECISION *diagonalS, // input 2N array for the diagonal matrix S in the SVD of Jacobian^transpose * Jacobian matrix
    const PRECISION *unitaryU, // input 2Nx2N array for the unitary matrix U in the SVD of Jacobian^transpose * Jacobian matrix
    const PRECISION *jacobtresidual, // input 2Nx1 array Q for the 2Nx1 Jacobian^transpose * Residuals matrix
    PRECISION *productSUJR, // output 2Nx1 array for the resulting SUQ product
    const unsigned int num_entries // total number 2N of entries in the product
    )
{
    const int index = threadIdx.x + blockDim.x * blockIdx.x;
    if(index >= num_entries)
        return;
    PRECISION s_inverse = (abs(diagonalS[index]) > 1E-6) ? 1.0/diagonalS[index] : 0.0;
    // form the sum of products of entries down column index of unitaryU (along row of transpose) with entries down column of jacobtresidual
    PRECISION sum_product = 0;
    for (int i=0; i<num_entries; i++)
    {
        sum_product += unitaryU[index*num_entries + i] * jacobtresidual[i];
    }
    productSUJR[index] = s_inverse * sum_product;
}

template __global__ void calculate_product_sujr<float>
    (const float*, const float*, const float*, float*, const unsigned int);
template __global__ void calculate_product_sujr<double>
    (const double*, const double*, const double*, double*, const unsigned int);


/*****************************************************************************
 * Calculates the matrix Delta as the product of d_V transpose * productSUJR
 * and adds it to the gains
 * Parallelised so each CUDA thread handles the two gains for one of the N antennas
 *****************************************************************************/
template<typename PRECISION, typename PRECISION2>
__global__ void calculate_delta_update_gains
    (
    const PRECISION *unitaryV, // input array for the unitary V matrix in the SVD of Jacobian^transpose * Jacobian matrix
    const PRECISION *productSUJR, // input 2Nx1 array for the SUQ matrix product
    PRECISION2 *gains_device, // inout array of antenna gains
    const unsigned int num_receivers, // number N of receivers
    const unsigned int num_entries // total number 2N of (real valued) gain entries
    )
{
    const int index = threadIdx.x + blockDim.x * blockIdx.x;
    if(index >= num_receivers)
        return;
    PRECISION delta_top = 0;
    PRECISION delta_bottom = 0;
    int vindex = index * 2;
    for (int i=0;i<num_entries; i++)
    {
        delta_top += productSUJR[i] * unitaryV[vindex*num_entries + i];
        delta_bottom += productSUJR[i] * unitaryV[(vindex+1)*num_entries + i];
    }
    gains_device[index].x += delta_top;
    gains_device[index].y += delta_bottom; 
}

template __global__ void calculate_delta_update_gains<float, float2>
    (const float*, const float*, float2*, const unsigned int, const unsigned int);
template __global__ void calculate_delta_update_gains<double, double2>
    (const double*, const double*, double2*, const unsigned int, const unsigned int);

