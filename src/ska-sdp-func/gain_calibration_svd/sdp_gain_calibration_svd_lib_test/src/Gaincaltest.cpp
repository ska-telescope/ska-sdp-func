/*
 * Gaincaltest.cpp
 * Andrew Ensor
 * C with C++ templates/CUDA program for testing steps of the Gain calibration algorithm
*/

#include "Gaincaltest.h"

#define GAINCAL_PRECISION_SINGLE 1

/**********************************************************************
 * Main method to execute
 **********************************************************************/
int main(int argc, char *argv[])
{
    printf("Gain calibration test starting");
    #ifdef GAINCAL_PRECISION_SINGLE
        printf(" using single precision\n");
        #define PRECISION float
    #else
        printf(" using double precision\n");
        #define PRECISION double
    #endif

    setLogLevel(LOG_INFO);

    printf("Gain calibration test ending\n");
    return 0;
}