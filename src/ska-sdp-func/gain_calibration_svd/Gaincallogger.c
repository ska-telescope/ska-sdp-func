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

/*
 * Gaincallogger.c
 * Andrew Ensor
 * C API for a basic logger used with the Gain calibration code
 *
*/

#include "Gaincallogger.h"

enum logType logLevel; /* please only change via setLogLevel */

/*****************************************************************************
 * Provides a basic logger via printf statements
 *****************************************************************************/
void logger(enum logType messageLevel, const char *message, ...)
{
    if (messageLevel <= logLevel)
    {
        va_list args;
        va_start(args, message);
        vprintf(message, args);
        printf("\n");
        va_end(args);
    }
}


/*****************************************************************************
 * Sets the logging level
 *****************************************************************************/
void setLogLevel(enum logType newLevel)
{
    if (newLevel < LOG_EMERG)
        newLevel = LOG_EMERG;
    else if (newLevel > LOG_DEBUG)
        newLevel = LOG_DEBUG;
    logLevel = newLevel;
}
