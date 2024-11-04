/* See the LICENSE file at the top-level directory of this distribution. */

#include <cstdlib>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "ska-sdp-func/utility/sdp_thread_support.h"

struct sdp_Mutex
{
#ifdef _OPENMP
    omp_lock_t lock_;
#else
    int dummy; // Avoid an empty struct.
#endif
};


sdp_Mutex* sdp_mutex_create()
{
    sdp_Mutex* mutex = (sdp_Mutex*) calloc(1, sizeof(sdp_Mutex));
#ifdef _OPENMP
    omp_init_lock(&mutex->lock_);
#endif
    return mutex;
}


void sdp_mutex_free(sdp_Mutex* mutex)
{
    if (!mutex) return;
#ifdef _OPENMP
    omp_destroy_lock(&mutex->lock_);
#endif
    free(mutex);
}


void sdp_mutex_lock(sdp_Mutex* mutex)
{
#ifdef _OPENMP
    omp_set_lock(&mutex->lock_);
#endif
}


void sdp_mutex_unlock(sdp_Mutex* mutex)
{
#ifdef _OPENMP
    omp_unset_lock(&mutex->lock_);
#endif
}
