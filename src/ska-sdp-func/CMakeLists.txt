# Define list of folders containing source files.
set(${CMAKE_PROJECT_NAME}_FOLDERS
    examples
    fourier_transforms
    gain_calibration_svd
    grid_data
    rfi
    utility
    visibility
)

# Auto-generate the source lists and pass up to the parent.
foreach (module ${${CMAKE_PROJECT_NAME}_FOLDERS})
    add_subdirectory(${module})
    foreach (file ${${module}_SRC})
        get_filename_component(fileext ${file} EXT)
        if (fileext STREQUAL ".cu")
            list(APPEND ${CMAKE_PROJECT_NAME}_CUDA_SRC "${file}")
        else()
            list(APPEND ${CMAKE_PROJECT_NAME}_SRC "${file}")
        endif()
    endforeach()
endforeach()
set(${CMAKE_PROJECT_NAME}_CUDA_SRC "${${CMAKE_PROJECT_NAME}_CUDA_SRC}" PARENT_SCOPE)
set(${CMAKE_PROJECT_NAME}_SRC "${${CMAKE_PROJECT_NAME}_SRC}" PARENT_SCOPE)
