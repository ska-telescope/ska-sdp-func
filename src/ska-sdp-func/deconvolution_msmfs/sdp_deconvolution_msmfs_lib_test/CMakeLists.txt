# CMakeLists.txt for the deconvolution_msmfs_lib_test executable
cmake_minimum_required(VERSION 3.23)

message(STATUS "CMake current directory is ${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "CMake project source directory is ${PROJECT_SOURCE_DIR}")

# Cuda and CMake checks
find_package(CUDA REQUIRED)
message(STATUS "CMake found CUDA ${CUDA_VERSION_STRING} at ${CUDA_TOOLKIT_ROOT_DIR}")

find_library(CUSOLVER_LIBRARY cusolver HINTS /usr/local/cuda/lib64)
message(STATUS "CUDA_INCLUDE_DIRS is ${CUDA_INCLUDE_DIRS}")
message(STATUS "CUSOLVER_LIBRARY is ${CUSOLVER_LIBRARY}")

# prepare the executable for sdp_deconvolution_msmfs_lib_test
add_executable(sdp_deconvolution_msmfs_lib_test)

target_include_directories(sdp_deconvolution_msmfs_lib_test PUBLIC
    ${CUDA_INCLUDE_DIRS}
    include
    ${PROJECT_SOURCE_DIR}/sdp_deconvolution_msmfs_lib/include
    )

target_sources(sdp_deconvolution_msmfs_lib_test
    PRIVATE src/Msmfstest.cpp include/Msmfstest.h
    PUBLIC FILE_SET HEADERS
    BASE_DIRS ${PROJECT_SOURCE_DIR}
    FILES include/Msmfstest.h
    )

target_link_libraries(sdp_deconvolution_msmfs_lib_test PUBLIC
    sdp_deconvolution_msmfs_lib
    ${CUSOLVER_LIBRARY}
    ${CUDA_LIBRARIES}
    m
    )
