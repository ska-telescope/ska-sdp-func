# Define list of folders containing test files.
set(${PROJECT_NAME}_TEST_FOLDERS
    clean
    examples
    fourier_transforms
    grid_data
    numeric_functions
    rfi
    station_beam
    utility
    visibility
)

# Build tests.
foreach (module ${${PROJECT_NAME}_TEST_FOLDERS})
    add_subdirectory(${module})
    foreach (test ${${module}_TST})
        add_executable(${test} "${module}/${test}.cpp")
        target_link_libraries(${test} ${PROJECT_NAME})
        if (CFITSIO_FOUND)
            target_link_libraries(${test} ${CFITSIO_LIBRARIES})
        endif()
        add_test(${test} ${test})
    endforeach()
endforeach()
