cmake_minimum_required(VERSION 3.9 FATAL_ERROR)
if (APPLE)
    set(CMAKE_OSX_DEPLOYMENT_TARGET "10.13" CACHE STRING "Minimum macOS version")
endif()
project(SDP_FUNC CXX C)

# Library name.
set(libname ska_sdp_func)

# List of C and C++ source files.
set(${libname}_SRC
    src/ska-sdp-func/rfi_flagger/sdp_rfi_flagger.cpp
    src/ska-sdp-func/dft/sdp_dft.cpp
    src/ska-sdp-func/vector/sdp_vector_add.cpp
    src/ska-sdp-func/function_example_a/sdp_function_example_a.cpp
    src/ska-sdp-func/utility/sdp_device_wrapper.cpp
    src/ska-sdp-func/utility/sdp_logging.c
    src/ska-sdp-func/utility/sdp_mem.c
)

# List of CUDA kernel source files.
set(${libname}_CUDA_SRC
    src/ska-sdp-func/dft/sdp_dft.cu
    src/ska-sdp-func/vector/sdp_vector_add.cu
)

# List of tests to build, in the "tests" directory.
set(${libname}_TESTS
    test_rfi_flagger
    test_dft
    test_vector_add
    test_function_example_a
)

# Project version.
set(SDP_FUNC_VERSION_ID "0x000000")
set(SDP_FUNC_VERSION_MAJOR 0)
set(SDP_FUNC_VERSION_MINOR 0)
set(SDP_FUNC_VERSION_PATCH 0)
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
endif()
if (CUDA_FOUND)
    add_definitions(-DSDP_HAVE_CUDA)
endif()
find_package(OpenMP QUIET)

# Set compiler options.
include(compiler_options)
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

# Build library using list of source files.
if (CUDA_FOUND)
    # Add CUDA kernel sources.
    list(APPEND ${libname}_SRC ${${libname}_CUDA_SRC})
    set(CUDA_GENERATED_OUTPUT_DIR
        ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/${libname}.dir/src)
    cuda_add_library(${libname} ${${libname}_SRC})
    # Add the next line back when/if we actually need cuFFT.
    # target_link_libraries(${libname} ${CUDA_CUFFT_LIBRARIES})
else()
    add_library(${libname} ${${libname}_SRC})
endif()
if (NOT DEFINED PYTHON_EXECUTABLE)
    set_target_properties(${libname} PROPERTIES
        SOVERSION ${SDP_FUNC_VERSION}
        VERSION ${SDP_FUNC_VERSION})
else()
    message(STATUS "Building for Python: ${PYTHON_EXECUTABLE}")
endif()

# Add OpenMP compiler flags if found.
if (OpenMP_CXX_FOUND)
    target_link_libraries(${libname} PRIVATE OpenMP::OpenMP_CXX)
endif()

# Build tests.
foreach(test ${${libname}_TESTS})
    add_executable(${test} tests/${test}.cpp)
    target_link_libraries(${test} ${libname})
    add_test(${test} ${test})
endforeach()

install(TARGETS ${libname}
    ARCHIVE DESTINATION ${SDP_FUNC_LIB_INSTALL_DIR} COMPONENT libraries
    LIBRARY DESTINATION ${SDP_FUNC_LIB_INSTALL_DIR} COMPONENT libraries
    RUNTIME DESTINATION ${SDP_FUNC_BIN_INSTALL_DIR} COMPONENT runtime
)
install(DIRECTORY "${PROJECT_SOURCE_DIR}/src/ska-sdp-func"
    DESTINATION ${SDP_FUNC_INCLUDE_INSTALL_DIR}
    COMPONENT headers
    FILES_MATCHING REGEX "(.*h)|(.*hpp)"
)
