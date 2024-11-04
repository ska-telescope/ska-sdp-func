/* See the LICENSE file at the top-level directory of this distribution. */

#ifndef SDP_THREAD_SUPPORT_H_
#define SDP_THREAD_SUPPORT_H_

/**
 * @file sdp_thread_support.h
 */

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup Mutex_struct
 * @{
 */

/**
 * @struct sdp_Mutex
 *
 * @brief Provides a mutex.
 *
 * Call ::sdp_mutex_create() to create a mutex.
 * Call ::sdp_mutex_lock() and ::sdp_mutex_unlock() to respectively lock
 * and unlock the mutex.
 * Release memory used by the mutex by calling ::sdp_mutex_free().
 */
struct sdp_Mutex;

/** @} */ /* End group Mutex_struct. */

/* Typedefs. */
typedef struct sdp_Mutex sdp_Mutex;

/**
 * @defgroup Mutex_func
 * @{
 */

/**
 * @brief Creates a mutex.
 *
 * @return Pointer to created mutex.
 */
sdp_Mutex* sdp_mutex_create();

/**
 * @brief Destroys the mutex.
 *
 * @param mutex Pointer to mutex.
 */
void sdp_mutex_free(sdp_Mutex* mutex);

/**
 * @brief Locks the mutex.
 *
 * @param mutex Pointer to mutex.
 */
void sdp_mutex_lock(sdp_Mutex* mutex);

/**
 * @brief Unlocks the mutex.
 *
 * @param mutex Pointer to mutex.
 */
void sdp_mutex_unlock(sdp_Mutex* mutex);

/** @} */ /* End group Mutex_func. */

#ifdef _OPENMP
#define SDP_GET_THREAD_NUM omp_get_thread_num()
#else
#define SDP_GET_THREAD_NUM 0
#endif

#ifdef __cplusplus
}
#endif

#endif /* include guard */
