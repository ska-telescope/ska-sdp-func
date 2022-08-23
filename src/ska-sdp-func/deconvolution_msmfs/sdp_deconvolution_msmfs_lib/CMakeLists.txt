# CMakeLists.txt for the deconvolution_msmfs_cuda library
cmake_minimum_required(VERSION 3.23)

message(STATUS "CMake current directory is ${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "CMake project source directory is ${PROJECT_SOURCE_DIR}")

# Cuda and CMake checks
find_package(CUDA REQUIRED)
message(STATUS "CMake found CUDA ${CUDA_VERSION_STRING} at ${CUDA_TOOLKIT_ROOT_DIR}")

find_library(CUSOLVER_LIBRARY cusolver HINTS /usr/local/cuda/lib64)
message(STATUS "CUDA_INCLUDE_DIRS is ${CUDA_INCLUDE_DIRS}")
message(STATUS "CUSOLVER_LIBRARY is ${CUSOLVER_LIBRARY}")

set(msmfs_cuda_source_file_list 
    src/Msmfslogger.c
    src/Msmfsfunctionsdevice.cu
    src/Msmfsfunctionshost.cu
    src/Msmfssimpletest.cu
    include/Msmfscommon.h
    include/Msmfslogger.h
    include/Msmfsfunctionsdevice.h
    include/Msmfsfunctionshost.h
    include/Msmfssimpletest.h
    )

# prepare the shared library for sdp_deconvolution_msmfs_cuda
add_library(sdp_deconvolution_msmfs_lib SHARED)
target_include_directories(sdp_deconvolution_msmfs_lib PRIVATE
    include)
target_sources(sdp_deconvolution_msmfs_lib
    PRIVATE ${msmfs_cuda_source_file_list}
    PUBLIC FILE_SET HEADERS
    BASE_DIRS ${PROJECT_SOURCE_DIR}
    FILES include/Msmfscommon.h include/Msmfslogger.h include/Msmfsfunctionshost.h include/Msmfssimpletest.h
    )
set_target_properties(sdp_deconvolution_msmfs_lib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(sdp_deconvolution_msmfs_lib PROPERTIES CUDA_ARCHITECTURES native)
target_link_libraries(sdp_deconvolution_msmfs_lib PUBLIC
    ${CUSOLVER_LIBRARY}
    m)
target_compile_options(sdp_deconvolution_msmfs_lib PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
    -Xptxas;
    -O3
    >)
install(TARGETS sdp_deconvolution_msmfs_lib FILE_SET HEADERS)
