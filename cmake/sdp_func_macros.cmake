macro(SKA_SDP_FUNC_SRC)
    # Get name of current directory.
    get_filename_component(module ${CMAKE_CURRENT_LIST_DIR} NAME_WE)

    # Iterate list of files supplied to the macro,
    # generate a list called <directory name>_SRC,
    # and add it to the parent scope.
    foreach (file ${ARGN})
        list(APPEND ${module}_SRC "src/ska-sdp-func/${module}/${file}")
    endforeach()
    set(${module}_SRC "${${module}_SRC}" PARENT_SCOPE)
endmacro()

macro(SKA_SDP_FUNC_TEST)
    # Get name of current directory.
    get_filename_component(module ${CMAKE_CURRENT_LIST_DIR} NAME_WE)

    # Iterate list of tests supplied to the macro,
    # generate a list called <directory name>_TST,
    # and add it to the parent scope.
    foreach (file ${ARGN})
        list(APPEND ${module}_TST "${file}")
    endforeach()
    set(${module}_TST "${${module}_TST}" PARENT_SCOPE)
endmacro()
