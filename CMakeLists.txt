cmake_minimum_required(VERSION 3.9 FATAL_ERROR)
if (APPLE)
    set(CMAKE_OSX_DEPLOYMENT_TARGET "10.13" CACHE STRING "Minimum macOS version")
endif()
project(ska_sdp_func CXX C)

# Project version.
set(SDP_FUNC_VERSION_ID "0x010202")
set(SDP_FUNC_VERSION_MAJOR 1)
set(SDP_FUNC_VERSION_MINOR 2)
set(SDP_FUNC_VERSION_PATCH 2)
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
    find_package(CUDAToolkit 7.0)
endif()
if (CUDAToolkit_FOUND)
    enable_language(CUDA)
    add_definitions(-DSDP_HAVE_CUDA)
endif()
find_package(CFITSIO QUIET)
find_package(OpenMP QUIET)
if (FIND_MKL OR NOT DEFINED FIND_MKL)
    find_package(MKL CONFIG QUIET PATHS $ENV{MKLROOT})
endif()
if (MKL_FOUND)
    add_definitions(-DSDP_HAVE_MKL)
    option(USE_POCKETFFT "Use PocketFFT for FFTs" OFF)
else()
    option(USE_POCKETFFT "Use PocketFFT for FFTs" ON)
endif()
if (USE_POCKETFFT)
    add_definitions(-DSDP_USE_POCKETFFT)
endif()
if (CFITSIO_FOUND)
    add_definitions(-DSDP_HAVE_CFITSIO)
endif()

# Set compiler options.
include(compiler_options)
include(sdp_func_macros)
enable_testing()

if ((CMAKE_BUILD_TYPE MATCHES Debug OR CMAKE_BUILD_TYPE MATCHES RelWithDebInfo)
        AND COVERAGE_REPORT)
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
if (CUDAToolkit_FOUND)
    # Add CUDA kernel sources.
    list(APPEND ${PROJECT_NAME}_SRC ${${PROJECT_NAME}_CUDA_SRC})
    add_library(${PROJECT_NAME} ${${PROJECT_NAME}_SRC})
    target_link_libraries(${PROJECT_NAME} CUDA::cufft CUDA::cudart_static)
else()
    add_library(${PROJECT_NAME} ${${PROJECT_NAME}_SRC})
endif()
if (NOT DEFINED PYTHON_EXECUTABLE)
    set_target_properties(${PROJECT_NAME} PROPERTIES
        SOVERSION ${SDP_FUNC_VERSION}
        VERSION ${SDP_FUNC_VERSION})
else()
    message(STATUS "Building for Python: ${PYTHON_EXECUTABLE}")
endif()

# Set target include directories at build time.
target_include_directories(${PROJECT_NAME} PUBLIC
    $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>
    $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/src>
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
)

# Link with MKL if found.
if (MKL_FOUND)
    target_link_libraries(${PROJECT_NAME} MKL::MKL)
endif()

# Add OpenMP compiler flags if found.
if (OpenMP_CXX_FOUND)
    target_link_libraries(${PROJECT_NAME} OpenMP::OpenMP_CXX)
endif()

# Build tests if this project is not a sub-project.
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    add_subdirectory(tests)
endif()

install(TARGETS "${PROJECT_NAME}"
    EXPORT ska_sdp_funcTargets
    ARCHIVE DESTINATION ${SDP_FUNC_LIB_INSTALL_DIR} COMPONENT libraries
    LIBRARY DESTINATION ${SDP_FUNC_LIB_INSTALL_DIR} COMPONENT libraries
    RUNTIME DESTINATION ${SDP_FUNC_BIN_INSTALL_DIR} COMPONENT runtime
    INCLUDES DESTINATION ${SDP_FUNC_INCLUDE_INSTALL_DIR}
)

install(DIRECTORY "${PROJECT_SOURCE_DIR}/src/ska-sdp-func"
    DESTINATION ${SDP_FUNC_INCLUDE_INSTALL_DIR}
    COMPONENT headers
    FILES_MATCHING REGEX "sdp_.*\.(h|hpp)$"
    PATTERN private* EXCLUDE
)
install(FILES "${PROJECT_BINARY_DIR}/src/sdp_func_version.h"
    DESTINATION "${SDP_FUNC_INCLUDE_INSTALL_DIR}/ska-sdp-func/"
    COMPONENT headers
)

install(EXPORT ska_sdp_funcTargets
    FILE "ska-sdp-funcTargets.cmake"
    NAMESPACE ska-sdp-func::
    DESTINATION "lib/cmake/ska-sdp-func")

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "ska-sdp-funcConfigVersion.cmake"
    VERSION "${SDP_FUNC_VERSION_MAJOR}.${SDP_FUNC_VERSION_MINOR}.${SDP_FUNC_VERSION_PATCH}"
    COMPATIBILITY AnyNewerVersion)

# The config file in the source tree has the suffix ".in" to keep
# find_package from mistaking the source tree for installed location.
# The suffix is removed upon installation
install(FILES "${PROJECT_SOURCE_DIR}/cmake/ska-sdp-funcConfig.cmake.in"
RENAME "ska-sdp-funcConfig.cmake"
DESTINATION "${SDP_FUNC_LIB_INSTALL_DIR}/cmake/ska-sdp-func")

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/ska-sdp-funcConfigVersion.cmake"
DESTINATION "${SDP_FUNC_LIB_INSTALL_DIR}/cmake/ska-sdp-func")
