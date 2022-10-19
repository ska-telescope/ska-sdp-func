# CMakeLists.txt for the sdp_gain_calibration_svd_int_test C library
cmake_minimum_required(VERSION 3.23)

message(STATUS "CMake current directory is ${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "CMake project source directory is ${PROJECT_SOURCE_DIR}")

# Cuda and CMake checks
find_package(CUDA REQUIRED)
message(STATUS "CMake found CUDA ${CUDA_VERSION_STRING} at ${CUDA_TOOLKIT_ROOT_DIR}")

find_library(CUSOLVER_LIBRARY cusolver HINTS /usr/local/cuda/lib64)
message(STATUS "CUDA_INCLUDE_DIRS is ${CUDA_INCLUDE_DIRS}")
message(STATUS "CUSOLVER_LIBRARY is ${CUSOLVER_LIBRARY}")

# NOTE sdp_mem.c sdp_mem.h sdp_logging.c have modified #include until cmake spfl building reorganised
set(sdp_processing_function_library_utility
    ${PROJECT_SOURCE_DIR}/../utility
    )

# prepare the executable for sdp_gain_calibration_svd_int
add_executable(sdp_gain_calibration_svd_int_test)

target_include_directories(sdp_gain_calibration_svd_int_test PUBLIC
    ${CUDA_INCLUDE_DIRS}
    include
    ${PROJECT_SOURCE_DIR}/sdp_gain_calibration_svd_int/include
    ${PROJECT_SOURCE_DIR}/sdp_gain_calibration_svd_lib/include
    ${sdp_processing_function_library_utility}
    )

target_sources(sdp_gain_calibration_svd_int_test
    PRIVATE src/Gaincalprocessingfunctiontest.cpp include/Gaincalprocessingfunctiontest.h
    PUBLIC FILE_SET HEADERS
    BASE_DIRS ${PROJECT_SOURCE_DIR} ${sdp_processing_function_library_utility}
    FILES include/Gaincalprocessingfunctiontest.h
    )

target_link_libraries(sdp_gain_calibration_svd_int_test PUBLIC
    sdp_gain_calibration_svd_int
    sdp_gain_calibration_svd_lib
    ${CUSOLVER_LIBRARY}
    ${CUDA_LIBRARIES}
    m
    )
