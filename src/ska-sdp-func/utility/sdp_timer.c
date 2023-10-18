/* See the LICENSE file at the top-level directory of this distribution. */

#include "ska-sdp-func/utility/sdp_timer.h"
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#ifndef SDP_OS_WIN
#include <sys/time.h>
#include <unistd.h>
#else
#include <windows.h>
#endif

#ifdef SDP_HAVE_CUDA
#include <cuda_runtime_api.h>
#endif


#ifdef __cplusplus
extern "C" {
#endif

struct sdp_Timer
{
#ifdef SDP_HAVE_CUDA
    cudaEvent_t start_cuda, end_cuda;
#endif
    double start, elapsed;
#ifdef SDP_OS_WIN
    double freq;
#endif
    sdp_TimerType type;
    int paused;
};

static double sdp_get_wtime(sdp_Timer* timer)
{
#if defined(SDP_OS_WIN)
    /* Windows-specific version. */
    LARGE_INTEGER cntr;
    QueryPerformanceCounter(&cntr);
    return (double)(cntr.QuadPart) / timer->freq;
#elif _POSIX_MONOTONIC_CLOCK > 0
    /* Use monotonic clock if available. */
    struct timespec ts;
    (void)timer;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
#else
    /* Use gettimeofday() as fallback. */
    struct timeval tv;
    (void)timer;
    gettimeofday(&tv, 0);
    return tv.tv_sec + tv.tv_usec / 1e6;
#endif
}

sdp_Timer* sdp_timer_create(sdp_TimerType type)
{
    sdp_Timer* timer = 0;
#ifdef SDP_OS_WIN
    LARGE_INTEGER freq;
#endif
    timer = (sdp_Timer*) calloc(1, sizeof(sdp_Timer));
#ifdef SDP_OS_WIN
    QueryPerformanceFrequency(&freq);
    timer->freq = (double)(freq.QuadPart);
#endif
    timer->type = type;
    timer->paused = 1;
#ifdef SDP_HAVE_CUDA
    if (timer->type == SDP_TIMER_CUDA)
    {
        cudaEventCreate(&timer->start_cuda);
        cudaEventCreate(&timer->end_cuda);
    }
#endif
    return timer;
}

void sdp_timer_free(sdp_Timer* timer)
{
    if (!timer) return;
#ifdef SDP_HAVE_CUDA
    if (timer->type == SDP_TIMER_CUDA)
    {
        cudaEventDestroy(timer->start_cuda);
        cudaEventDestroy(timer->end_cuda);
    }
#endif
    free(timer);
}

double sdp_timer_elapsed(sdp_Timer* timer)
{
    /* If timer is paused, return immediately with current elapsed time. */
    if (timer->paused) return timer->elapsed;

#ifdef SDP_HAVE_CUDA
    if (timer->type == SDP_TIMER_CUDA)
    {
        float millisec = 0.0f;

        /* Get elapsed time since start. */
        cudaEventRecord(timer->end_cuda, 0);
        cudaEventSynchronize(timer->end_cuda);
        cudaEventElapsedTime(&millisec, timer->start_cuda, timer->end_cuda);

        /* Increment elapsed time and restart. */
        timer->elapsed += millisec / 1000.0;
        cudaEventRecord(timer->start_cuda, 0);
    }
    else
#endif
    if (timer->type == SDP_TIMER_NATIVE)
    {
        const double now = sdp_get_wtime(timer);

        /* Increment elapsed time and restart. */
        timer->elapsed += (now - timer->start);
        timer->start = now;
    }
    return timer->elapsed;
}

void sdp_timer_pause(sdp_Timer* timer)
{
    if (timer->paused) return;
    (void)sdp_timer_elapsed(timer);
    timer->paused = 1;
}

void sdp_timer_reset(sdp_Timer* timer)
{
    timer->paused = 1;
    timer->start = 0.0;
    timer->elapsed = 0.0;
}

void sdp_timer_resume(sdp_Timer* timer)
{
    if (!timer->paused) return;
    sdp_timer_restart(timer);
}

void sdp_timer_restart(sdp_Timer* timer)
{
    timer->paused = 0;
#ifdef SDP_HAVE_CUDA
    if (timer->type == SDP_TIMER_CUDA)
    {
        cudaEventRecord(timer->start_cuda, 0);
        return;
    }
#endif
    timer->start = sdp_get_wtime(timer);
}

void sdp_timer_start(sdp_Timer* timer)
{
    timer->elapsed = 0.0;
    sdp_timer_restart(timer);
}

#ifdef __cplusplus
}
#endif
