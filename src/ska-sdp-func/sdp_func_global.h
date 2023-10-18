/* See the LICENSE file at the top-level directory of this distribution. */

#ifndef SKA_SDP_FUNC_GLOBAL_H_
#define SKA_SDP_FUNC_GLOBAL_H_

/*
 * Only for things which are truly global,
 * e.g. macros for platform detection and function export.
 */

/* Platform detection. */
#if (defined(WIN32) || defined(_WIN32) || defined(__WIN32__))
    #define SDP_OS_WIN32
#endif
#if (defined(WIN64) || defined(_WIN64) || defined(__WIN64__))
    #define SDP_OS_WIN64
#endif

/* http://goo.gl/OUEZfb */
#if (defined(SDP_OS_WIN32) || defined(SDP_OS_WIN64))
    #define SDP_OS_WIN
#elif defined __APPLE__
    #include "TargetConditionals.h"
    #if TARGET_OS_MAC
        #define SDP_OS_MAC
    #endif
#elif (defined(__linux__) || defined(__linux))
    #define SDP_OS_LINUX
#elif (defined(__unix__) || defined(__unix))
    #define SDP_OS_UNIX
#else
    #error Could not detect OS type!
#endif

/*
 * Macro used to export public functions (only needed in header files).
 *
 * The modifier enables the function to be exported by the library so that
 * it can be used by other applications.
 *
 * For more information see:
 *   http://msdn.microsoft.com/en-us/library/a90k134d(v=VS.90).aspx
 */
#ifndef SDP_DECL_EXPORT
    #ifdef SDP_OS_WIN
        #define SDP_DECL_EXPORT __declspec(dllexport)
    #elif __GNUC__ >= 4
        #define SDP_DECL_EXPORT __attribute__((visibility ("default")))
    #else
        #define SDP_DECL_EXPORT
    #endif
#endif
#ifndef SDP_DECL_IMPORT
    #ifdef SDP_OS_WIN
        #define SDP_DECL_IMPORT __declspec(dllimport)
    #elif __GNUC__ >= 4
        #define SDP_DECL_IMPORT __attribute__((visibility ("default")))
    #else
        #define SDP_DECL_IMPORT
    #endif
#endif

#ifdef ska_sdp_func_EXPORTS
    #define SDP_EXPORT SDP_DECL_EXPORT
#else
    #define SDP_EXPORT SDP_DECL_IMPORT
#endif

/* RESTRICT macro. */
#if defined(__cplusplus) && defined(__GNUC__)
    #define RESTRICT __restrict__
#elif defined(_MSC_VER)
    #define RESTRICT __restrict
#elif !defined(__STDC_VERSION__) || __STDC_VERSION__ < 199901L
    #define RESTRICT
#else
    #define RESTRICT restrict
#endif

#endif /* include guard */
