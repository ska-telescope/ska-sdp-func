# Define list of folders containing test files.
set(${CMAKE_PROJECT_NAME}_TEST_FOLDERS
    examples
    fourier_transforms
    grid_data
    rfi
    utility
    visibility
    clean
)

# Build tests.
foreach (module ${${CMAKE_PROJECT_NAME}_TEST_FOLDERS})
    add_subdirectory(${module})
    foreach (test ${${module}_TST})
        add_executable(${test} "${module}/${test}.cpp")
        target_link_libraries(${test} ${CMAKE_PROJECT_NAME})
        add_test(${test} ${test})
    endforeach()
endforeach()
