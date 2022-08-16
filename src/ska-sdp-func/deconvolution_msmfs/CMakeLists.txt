# CMakeLists.txt base for the sdp_deconvolution_msmfs project
cmake_minimum_required(VERSION 3.23)
project(sdp_deconvolution_msmfs LANGUAGES C CXX CUDA)

message(STATUS "CMake current directory is ${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "CMake project source directory is ${PROJECT_SOURCE_DIR}")

# Cuda and CMake checks
find_package(CUDA REQUIRED)
message(STATUS "CMake found CUDA ${CUDA_VERSION_STRING} at ${CUDA_TOOLKIT_ROOT_DIR}")

find_library(CUSOLVER_LIBRARY cusolver HINTS /usr/local/cuda/lib64)
message(STATUS "CUDA_INCLUDE_DIRS is ${CUDA_INCLUDE_DIRS}")
message(STATUS "CUSOLVER_LIBRARY is ${CUSOLVER_LIBRARY}")

add_subdirectory(sdp_deconvolution_msmfs_cuda)
add_subdirectory(sdp_deconvolution_msmfs_test)

