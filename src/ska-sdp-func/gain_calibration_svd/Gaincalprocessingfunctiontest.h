/**
 * @file Gaincalprocessingfunctiontest.h
 * @author Andrew Ensor
 * @brief C program for testing the SDP processing function interface steps for the Gain Calibration algorithm
 */

#ifndef GAINCAL_PROCESSING_FUNCTIONS_TEST_H
#define GAINCAL_PROCESSING_FUNCTIONS_TEST_H

#include <stdarg.h>
#include <stdbool.h>
#include <stdio.h>
#include <math.h>

#include "Gaincalprocessingfunctioninterface.h"

#include "ska-sdp-func/utility/sdp_mem.h"

/**
 * @brief Main function to execute the interface test.
 */
int gain_calibration_interface_test(void);

#endif /* include guard */