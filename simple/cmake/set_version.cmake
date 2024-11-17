set(SDP_FUNC_VERSION "${SDP_FUNC_VERSION_MAJOR}.${SDP_FUNC_VERSION_MINOR}.${SDP_FUNC_VERSION_PATCH}")
set(SDP_FUNC_VERSION_STR "${SDP_FUNC_VERSION}")
if (SDP_FUNC_VERSION_SUFFIX AND NOT SDP_FUNC_VERSION_SUFFIX STREQUAL "")
    find_package(Git QUIET)
    if (GIT_FOUND)
        execute_process(
            COMMAND git log -1 --format=%h
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
            OUTPUT_VARIABLE GIT_COMMIT_HASH
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        execute_process(
            COMMAND git log -1 --format=%cd --date=short
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
            OUTPUT_VARIABLE GIT_COMMIT_DATE
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        set(GIT_COMMIT_INFO "${GIT_COMMIT_DATE} ${GIT_COMMIT_HASH}")
    endif()
    set(SDP_FUNC_VERSION_STR "${SDP_FUNC_VERSION}-${SDP_FUNC_VERSION_SUFFIX}")
    if (GIT_COMMIT_INFO)
        set(SDP_FUNC_VERSION_STR "${SDP_FUNC_VERSION_STR} ${GIT_COMMIT_INFO}")
    endif()
    if (CMAKE_BUILD_TYPE MATCHES [dD]ebug)
        set(SDP_FUNC_VERSION_STR "${SDP_FUNC_VERSION_STR} -debug-")
    endif()
    set(SDP_FUNC_VERSION_SHORT "${SDP_FUNC_VERSION}-${SDP_FUNC_VERSION_SUFFIX}"
        CACHE STRING "Short version"
    )
else()
    set(SDP_FUNC_VERSION_SHORT "${SDP_FUNC_VERSION}"
        CACHE STRING "Short version"
    )
endif()

# Add the short Git hash for the long version string.
if (GIT_COMMIT_HASH)
    set(SDP_FUNC_VERSION_LONG "${SDP_FUNC_VERSION_SHORT}-${GIT_COMMIT_HASH}"
        CACHE STRING "Long version"
    )
else()
    set(SDP_FUNC_VERSION_LONG "${SDP_FUNC_VERSION_SHORT}"
        CACHE STRING "Long version"
    )
endif()

configure_file(${PROJECT_SOURCE_DIR}/cmake/sdp_func_version.h.in
    ${PROJECT_BINARY_DIR}/src/sdp_func_version.h @ONLY)
