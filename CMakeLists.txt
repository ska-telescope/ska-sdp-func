cmake_minimum_required(VERSION 3.9 FATAL_ERROR)
if (APPLE)
    set(CMAKE_OSX_DEPLOYMENT_TARGET "10.13" CACHE STRING "Minimum macOS version")
endif()
project(ska_sdp_func CXX C)

# Project version.
set(SDP_FUNC_VERSION_ID "0x000006")
set(SDP_FUNC_VERSION_MAJOR 0)
set(SDP_FUNC_VERSION_MINOR 0)
set(SDP_FUNC_VERSION_PATCH 6)
set(SDP_FUNC_VERSION_SUFFIX "")

# Add local CMake module path.
list(INSERT CMAKE_MODULE_PATH 0 ${PROJECT_SOURCE_DIR}/cmake)
list(INSERT CMAKE_MODULE_PATH 0 ${PROJECT_SOURCE_DIR}/cmake/modules)

# Get string length of current source directory for log messages.
string(LENGTH "${CMAKE_SOURCE_DIR}/" SOURCE_PATH_SIZE)
add_definitions(-DSOURCE_PATH_SIZE=${SOURCE_PATH_SIZE})

# Set install directory paths.
if (NOT SDP_FUNC_LIB_INSTALL_DIR)
    set(SDP_FUNC_LIB_INSTALL_DIR "lib")
endif()
if (NOT SDP_FUNC_INCLUDE_INSTALL_DIR)
    set(SDP_FUNC_INCLUDE_INSTALL_DIR "include")
endif()
if (NOT SDP_FUNC_BIN_INSTALL_DIR)
    set(SDP_FUNC_BIN_INSTALL_DIR "bin")
endif()

# Set compiler include directories.
include_directories(
    ${PROJECT_BINARY_DIR}
    ${PROJECT_BINARY_DIR}/src
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/src
)

# Automatically set the build type if not specified.
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    # Use debug mode if building in dbg or debug directory.
    get_filename_component(dirname ${CMAKE_BINARY_DIR} NAME)
    if (${dirname} MATCHES "dbg" OR ${dirname} MATCHES "debug")
        set(CMAKE_BUILD_TYPE Debug)
    else()
        set(CMAKE_BUILD_TYPE Release)
    endif()
    message(STATUS "INFO: Setting CMAKE_BUILD_TYPE to ${CMAKE_BUILD_TYPE}")
endif()
include(set_version)

# Find dependencies.
if (FIND_CUDA OR NOT DEFINED FIND_CUDA)
    find_package(CUDA 7.0)
    find_library(CUDA_CUSOLVER_LIBRARY cusolver HINTS /usr/local/cuda/lib64)
endif()
if (CUDA_FOUND)
    add_definitions(-DSDP_HAVE_CUDA)
endif()
find_package(OpenMP QUIET)

# Set compiler options.
include(compiler_options)
include(sdp_func_macros)
enable_testing()

if (CMAKE_BUILD_TYPE MATCHES Debug AND COVERAGE_REPORT)
    message(STATUS "INFO: Adding code coverage build target")
    include(CodeCoverage)
    set(COVERAGE_EXCLUDES
        "${PROJECT_BINARY_DIR}/*"
        ".*.h"
        "*.h"
        "/usr/*"
    )
    append_coverage_compiler_flags()
    setup_target_for_coverage_lcov(NAME coverage EXECUTABLE ctest)
endif()

# Generate source list from subdirectories.
add_subdirectory(src)

# Build library using list of source files.
if (CUDA_FOUND)
    # Add CUDA kernel sources.
    list(APPEND ${CMAKE_PROJECT_NAME}_SRC ${${CMAKE_PROJECT_NAME}_CUDA_SRC})
    set(CUDA_GENERATED_OUTPUT_DIR
        ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/${CMAKE_PROJECT_NAME}.dir/src)
    cuda_add_library(${CMAKE_PROJECT_NAME} ${${CMAKE_PROJECT_NAME}_SRC})
    target_link_libraries(${CMAKE_PROJECT_NAME} ${CUDA_CUFFT_LIBRARIES} ${CUDA_CUSOLVER_LIBRARY})
else()
    add_library(${CMAKE_PROJECT_NAME} ${${CMAKE_PROJECT_NAME}_SRC})
endif()
if (NOT DEFINED PYTHON_EXECUTABLE)
    set_target_properties(${CMAKE_PROJECT_NAME} PROPERTIES
        SOVERSION ${SDP_FUNC_VERSION}
        VERSION ${SDP_FUNC_VERSION})
else()
    message(STATUS "Building for Python: ${PYTHON_EXECUTABLE}")
endif()

# Add OpenMP compiler flags if found.
if (OpenMP_CXX_FOUND)
    target_link_libraries(${CMAKE_PROJECT_NAME} OpenMP::OpenMP_CXX)
endif()

# Build tests.
add_subdirectory(tests)

install(TARGETS ${CMAKE_PROJECT_NAME}
    ARCHIVE DESTINATION ${SDP_FUNC_LIB_INSTALL_DIR} COMPONENT libraries
    LIBRARY DESTINATION ${SDP_FUNC_LIB_INSTALL_DIR} COMPONENT libraries
    RUNTIME DESTINATION ${SDP_FUNC_BIN_INSTALL_DIR} COMPONENT runtime
)
install(DIRECTORY "${PROJECT_SOURCE_DIR}/src/ska-sdp-func"
    DESTINATION ${SDP_FUNC_INCLUDE_INSTALL_DIR}
    COMPONENT headers
    FILES_MATCHING REGEX "(.*h)|(.*hpp)"
)
