# Define list of folders containing source files.
set(${PROJECT_NAME}_FOLDERS
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

# Auto-generate the source lists and pass up to the parent.
foreach (module ${${PROJECT_NAME}_FOLDERS})
    add_subdirectory(${module})
    foreach (file ${${module}_SRC})
        get_filename_component(fileext ${file} EXT)
        if (fileext STREQUAL ".cu")
            list(APPEND ${PROJECT_NAME}_CUDA_SRC "${file}")
        else()
            list(APPEND ${PROJECT_NAME}_SRC "${file}")
        endif()
    endforeach()
endforeach()
set(${PROJECT_NAME}_CUDA_SRC "${${PROJECT_NAME}_CUDA_SRC}" PARENT_SCOPE)
set(${PROJECT_NAME}_SRC "${${PROJECT_NAME}_SRC}" PARENT_SCOPE)
