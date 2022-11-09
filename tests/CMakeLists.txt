# List of tests to build.
set(${CMAKE_PROJECT_NAME}_TESTS
    test_deconvolution_msmfs
    test_dft
    test_fft
    test_function_example_a
    test_gridder_uvw_es_fft
    test_phase_rotate
    test_rfi_flagger
    test_twosm_rfi_flagger
    test_vector_add
)

# Build tests.
foreach(test ${${CMAKE_PROJECT_NAME}_TESTS})
    add_executable(${test} ${test}.cpp)
    target_link_libraries(${test} ${CMAKE_PROJECT_NAME})
    add_test(${test} ${test})
endforeach()
