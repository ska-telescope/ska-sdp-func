# CMakeLists.txt for the sdp_gain_calibration_svd_lib library
cmake_minimum_required(VERSION 3.23)

message(STATUS "CMake current directory is ${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "CMake project source directory is ${PROJECT_SOURCE_DIR}")

# Cuda and CMake checks
find_package(CUDA REQUIRED)
message(STATUS "CMake found CUDA ${CUDA_VERSION_STRING} at ${CUDA_TOOLKIT_ROOT_DIR}")

find_library(CUSOLVER_LIBRARY cusolver HINTS /usr/local/cuda/lib64)
message(STATUS "CUDA_INCLUDE_DIRS is ${CUDA_INCLUDE_DIRS}")
message(STATUS "CUSOLVER_LIBRARY is ${CUSOLVER_LIBRARY}")

set(gain_calibration_cuda_source_file_list 
    src/Gaincallogger.c
    src/Gaincalfunctionsdevice.cu
    src/Gaincalfunctionshost.cu
    src/Gaincalsimpletest.cu
    include/Gaincallogger.h
    include/Gaincalfunctionsdevice.h
    include/Gaincalfunctionshost.h
    include/Gaincalsimpletest.h
    )

# prepare the shared library for sdp_gain_calibration_svd_lib
add_library(sdp_gain_calibration_svd_lib SHARED)
target_include_directories(sdp_gain_calibration_svd_lib PRIVATE
    include)
target_sources(sdp_gain_calibration_svd_lib
    PRIVATE ${gain_calibration_cuda_source_file_list}
    PUBLIC FILE_SET HEADERS
    BASE_DIRS ${PROJECT_SOURCE_DIR}
    FILES include/Gaincallogger.h include/Gaincalfunctionshost.h include/Gaincalsimpletest.h
    )
set_target_properties(sdp_gain_calibration_svd_lib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(sdp_gain_calibration_svd_lib PROPERTIES CUDA_ARCHITECTURES native)
target_link_libraries(sdp_gain_calibration_svd_lib PUBLIC
    ${CUSOLVER_LIBRARY}
    m)
target_compile_options(sdp_gain_calibration_svd_lib PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
    -Xptxas;
    -O3
    >)
install(TARGETS sdp_gain_calibration_svd_lib FILE_SET HEADERS)
