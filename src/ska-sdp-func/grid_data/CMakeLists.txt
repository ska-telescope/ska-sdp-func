SKA_SDP_FUNC_SRC(
    sdp_degrid_uvw_custom.cpp
    sdp_degrid_uvw_custom.cu
    sdp_grid_wstack_wtower.cpp
    sdp_gridder_clamp_channels.cpp
    sdp_gridder_clamp_channels.cu
    sdp_gridder_direct.cpp
    sdp_gridder_grid_correct.cpp
    sdp_gridder_grid_correct.cu
    sdp_gridder_utils.cpp
    sdp_gridder_utils.cu
    sdp_gridder_uvw_es_fft.cpp
    sdp_gridder_uvw_es_fft_kernels.cu
    sdp_gridder_uvw_es_fft_utils.cpp
    sdp_gridder_wtower_height.cpp
    sdp_gridder_wtower_uvw.cpp
    sdp_gridder_wtower_uvw.cu
)

set_source_files_properties(sdp_gridder_wtower_uvw.cpp PROPERTIES COMPILE_FLAGS 
    "-O3 -qopenmp -xCORE-AVX2 -fma -xHOST -fp-model=fast=2 -std=c++17 -unroll -qopt-report=5 -qopt-report-phase=vec -qopt-report-file=stdout -ftz -fno-exceptions")
