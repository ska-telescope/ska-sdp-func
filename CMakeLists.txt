cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
if (APPLE)
    set(CMAKE_OSX_DEPLOYMENT_TARGET "10.9" CACHE STRING "Minimum macOS version")
endif()
project(SDP_FUNC CXX C)

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
        "${PROJECT_SOURCE_DIR}/apps/*"
        "${PROJECT_SOURCE_DIR}/extern/*"
        "${PROJECT_SOURCE_DIR}/gui/*"
        "${PROJECT_SOURCE_DIR}/.*/test/.*"
        "${PROJECT_SOURCE_DIR}/*/test/*"
        ".*.h"
        "*.h"
        "/usr/*"
        "/Applications/*"
        "/Users/*/Qt/*"
    )
    append_coverage_compiler_flags()
    setup_target_for_coverage_lcov(NAME coverage EXECUTABLE ctest)
endif()

# List of source files.
set(libname ska_sdp_func)
set(${libname}_SRC
    src/dft/sdp_dft.cpp
    src/logging/sdp_logging.c
    src/mem/sdp_mem.c
    src/utility/sdp_device_wrapper.cpp
    src/vector_func/sdp_vector_add.cpp
)

if (CUDA_FOUND)
    # Add kernel sources.
    list(APPEND ${libname}_SRC
        src/dft/sdp_dft.cu
        src/vector_func/sdp_vector_add.cu
    )
    set(CUDA_GENERATED_OUTPUT_DIR
        ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/${libname}.dir/src)
    cuda_add_library(${libname} ${${libname}_SRC})
else()
    add_library(${libname} ${${libname}_SRC})
endif()

set_target_properties(${libname} PROPERTIES
    SOVERSION ${SDP_FUNC_VERSION}
    VERSION ${SDP_FUNC_VERSION})

add_executable(test_vector_add tests/test_vector_add.cpp)
target_link_libraries(test_vector_add ${libname})
add_test(test_vector_add test_vector_add)

add_executable(test_dft tests/test_dft.cpp)
target_link_libraries(test_dft ${libname})
add_test(test_dft test_dft)

install(TARGETS ${libname}
    ARCHIVE DESTINATION ${SDP_FUNC_LIB_INSTALL_DIR} COMPONENT libraries
    LIBRARY DESTINATION ${SDP_FUNC_LIB_INSTALL_DIR} COMPONENT libraries
    RUNTIME DESTINATION ${SDP_FUNC_BIN_INSTALL_DIR} COMPONENT runtime
)
