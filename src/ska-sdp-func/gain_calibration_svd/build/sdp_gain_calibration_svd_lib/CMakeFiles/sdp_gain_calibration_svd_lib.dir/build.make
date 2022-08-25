# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.24

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /opt/cmake-3.24.0-rc4-linux-x86_64/bin/cmake

# The command to remove a file.
RM = /opt/cmake-3.24.0-rc4-linux-x86_64/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/aensor/ska/spfl-gaincalibration/ska-sdp-func/src/ska-sdp-func/gain_calibration_svd

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/aensor/ska/spfl-gaincalibration/ska-sdp-func/src/ska-sdp-func/gain_calibration_svd/build

# Include any dependencies generated for this target.
include sdp_gain_calibration_svd_lib/CMakeFiles/sdp_gain_calibration_svd_lib.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include sdp_gain_calibration_svd_lib/CMakeFiles/sdp_gain_calibration_svd_lib.dir/compiler_depend.make

# Include the progress variables for this target.
include sdp_gain_calibration_svd_lib/CMakeFiles/sdp_gain_calibration_svd_lib.dir/progress.make

# Include the compile flags for this target's objects.
include sdp_gain_calibration_svd_lib/CMakeFiles/sdp_gain_calibration_svd_lib.dir/flags.make

sdp_gain_calibration_svd_lib/CMakeFiles/sdp_gain_calibration_svd_lib.dir/src/Gaincallogger.c.o: sdp_gain_calibration_svd_lib/CMakeFiles/sdp_gain_calibration_svd_lib.dir/flags.make
sdp_gain_calibration_svd_lib/CMakeFiles/sdp_gain_calibration_svd_lib.dir/src/Gaincallogger.c.o: /home/aensor/ska/spfl-gaincalibration/ska-sdp-func/src/ska-sdp-func/gain_calibration_svd/sdp_gain_calibration_svd_lib/src/Gaincallogger.c
sdp_gain_calibration_svd_lib/CMakeFiles/sdp_gain_calibration_svd_lib.dir/src/Gaincallogger.c.o: sdp_gain_calibration_svd_lib/CMakeFiles/sdp_gain_calibration_svd_lib.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/aensor/ska/spfl-gaincalibration/ska-sdp-func/src/ska-sdp-func/gain_calibration_svd/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object sdp_gain_calibration_svd_lib/CMakeFiles/sdp_gain_calibration_svd_lib.dir/src/Gaincallogger.c.o"
	cd /home/aensor/ska/spfl-gaincalibration/ska-sdp-func/src/ska-sdp-func/gain_calibration_svd/build/sdp_gain_calibration_svd_lib && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT sdp_gain_calibration_svd_lib/CMakeFiles/sdp_gain_calibration_svd_lib.dir/src/Gaincallogger.c.o -MF CMakeFiles/sdp_gain_calibration_svd_lib.dir/src/Gaincallogger.c.o.d -o CMakeFiles/sdp_gain_calibration_svd_lib.dir/src/Gaincallogger.c.o -c /home/aensor/ska/spfl-gaincalibration/ska-sdp-func/src/ska-sdp-func/gain_calibration_svd/sdp_gain_calibration_svd_lib/src/Gaincallogger.c

sdp_gain_calibration_svd_lib/CMakeFiles/sdp_gain_calibration_svd_lib.dir/src/Gaincallogger.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/sdp_gain_calibration_svd_lib.dir/src/Gaincallogger.c.i"
	cd /home/aensor/ska/spfl-gaincalibration/ska-sdp-func/src/ska-sdp-func/gain_calibration_svd/build/sdp_gain_calibration_svd_lib && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/aensor/ska/spfl-gaincalibration/ska-sdp-func/src/ska-sdp-func/gain_calibration_svd/sdp_gain_calibration_svd_lib/src/Gaincallogger.c > CMakeFiles/sdp_gain_calibration_svd_lib.dir/src/Gaincallogger.c.i

sdp_gain_calibration_svd_lib/CMakeFiles/sdp_gain_calibration_svd_lib.dir/src/Gaincallogger.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/sdp_gain_calibration_svd_lib.dir/src/Gaincallogger.c.s"
	cd /home/aensor/ska/spfl-gaincalibration/ska-sdp-func/src/ska-sdp-func/gain_calibration_svd/build/sdp_gain_calibration_svd_lib && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/aensor/ska/spfl-gaincalibration/ska-sdp-func/src/ska-sdp-func/gain_calibration_svd/sdp_gain_calibration_svd_lib/src/Gaincallogger.c -o CMakeFiles/sdp_gain_calibration_svd_lib.dir/src/Gaincallogger.c.s

sdp_gain_calibration_svd_lib/CMakeFiles/sdp_gain_calibration_svd_lib.dir/src/Gaincalfunctionsdevice.cu.o: sdp_gain_calibration_svd_lib/CMakeFiles/sdp_gain_calibration_svd_lib.dir/flags.make
sdp_gain_calibration_svd_lib/CMakeFiles/sdp_gain_calibration_svd_lib.dir/src/Gaincalfunctionsdevice.cu.o: /home/aensor/ska/spfl-gaincalibration/ska-sdp-func/src/ska-sdp-func/gain_calibration_svd/sdp_gain_calibration_svd_lib/src/Gaincalfunctionsdevice.cu
sdp_gain_calibration_svd_lib/CMakeFiles/sdp_gain_calibration_svd_lib.dir/src/Gaincalfunctionsdevice.cu.o: sdp_gain_calibration_svd_lib/CMakeFiles/sdp_gain_calibration_svd_lib.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/aensor/ska/spfl-gaincalibration/ska-sdp-func/src/ska-sdp-func/gain_calibration_svd/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CUDA object sdp_gain_calibration_svd_lib/CMakeFiles/sdp_gain_calibration_svd_lib.dir/src/Gaincalfunctionsdevice.cu.o"
	cd /home/aensor/ska/spfl-gaincalibration/ska-sdp-func/src/ska-sdp-func/gain_calibration_svd/build/sdp_gain_calibration_svd_lib && /usr/local/cuda-10.2/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT sdp_gain_calibration_svd_lib/CMakeFiles/sdp_gain_calibration_svd_lib.dir/src/Gaincalfunctionsdevice.cu.o -MF CMakeFiles/sdp_gain_calibration_svd_lib.dir/src/Gaincalfunctionsdevice.cu.o.d -x cu -rdc=true -c /home/aensor/ska/spfl-gaincalibration/ska-sdp-func/src/ska-sdp-func/gain_calibration_svd/sdp_gain_calibration_svd_lib/src/Gaincalfunctionsdevice.cu -o CMakeFiles/sdp_gain_calibration_svd_lib.dir/src/Gaincalfunctionsdevice.cu.o

sdp_gain_calibration_svd_lib/CMakeFiles/sdp_gain_calibration_svd_lib.dir/src/Gaincalfunctionsdevice.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/sdp_gain_calibration_svd_lib.dir/src/Gaincalfunctionsdevice.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

sdp_gain_calibration_svd_lib/CMakeFiles/sdp_gain_calibration_svd_lib.dir/src/Gaincalfunctionsdevice.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/sdp_gain_calibration_svd_lib.dir/src/Gaincalfunctionsdevice.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

# Object files for target sdp_gain_calibration_svd_lib
sdp_gain_calibration_svd_lib_OBJECTS = \
"CMakeFiles/sdp_gain_calibration_svd_lib.dir/src/Gaincallogger.c.o" \
"CMakeFiles/sdp_gain_calibration_svd_lib.dir/src/Gaincalfunctionsdevice.cu.o"

# External object files for target sdp_gain_calibration_svd_lib
sdp_gain_calibration_svd_lib_EXTERNAL_OBJECTS =

sdp_gain_calibration_svd_lib/CMakeFiles/sdp_gain_calibration_svd_lib.dir/cmake_device_link.o: sdp_gain_calibration_svd_lib/CMakeFiles/sdp_gain_calibration_svd_lib.dir/src/Gaincallogger.c.o
sdp_gain_calibration_svd_lib/CMakeFiles/sdp_gain_calibration_svd_lib.dir/cmake_device_link.o: sdp_gain_calibration_svd_lib/CMakeFiles/sdp_gain_calibration_svd_lib.dir/src/Gaincalfunctionsdevice.cu.o
sdp_gain_calibration_svd_lib/CMakeFiles/sdp_gain_calibration_svd_lib.dir/cmake_device_link.o: sdp_gain_calibration_svd_lib/CMakeFiles/sdp_gain_calibration_svd_lib.dir/build.make
sdp_gain_calibration_svd_lib/CMakeFiles/sdp_gain_calibration_svd_lib.dir/cmake_device_link.o: /usr/local/cuda/lib64/libcusolver.so
sdp_gain_calibration_svd_lib/CMakeFiles/sdp_gain_calibration_svd_lib.dir/cmake_device_link.o: sdp_gain_calibration_svd_lib/CMakeFiles/sdp_gain_calibration_svd_lib.dir/dlink.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/aensor/ska/spfl-gaincalibration/ska-sdp-func/src/ska-sdp-func/gain_calibration_svd/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CUDA device code CMakeFiles/sdp_gain_calibration_svd_lib.dir/cmake_device_link.o"
	cd /home/aensor/ska/spfl-gaincalibration/ska-sdp-func/src/ska-sdp-func/gain_calibration_svd/build/sdp_gain_calibration_svd_lib && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/sdp_gain_calibration_svd_lib.dir/dlink.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
sdp_gain_calibration_svd_lib/CMakeFiles/sdp_gain_calibration_svd_lib.dir/build: sdp_gain_calibration_svd_lib/CMakeFiles/sdp_gain_calibration_svd_lib.dir/cmake_device_link.o
.PHONY : sdp_gain_calibration_svd_lib/CMakeFiles/sdp_gain_calibration_svd_lib.dir/build

# Object files for target sdp_gain_calibration_svd_lib
sdp_gain_calibration_svd_lib_OBJECTS = \
"CMakeFiles/sdp_gain_calibration_svd_lib.dir/src/Gaincallogger.c.o" \
"CMakeFiles/sdp_gain_calibration_svd_lib.dir/src/Gaincalfunctionsdevice.cu.o"

# External object files for target sdp_gain_calibration_svd_lib
sdp_gain_calibration_svd_lib_EXTERNAL_OBJECTS =

sdp_gain_calibration_svd_lib/libsdp_gain_calibration_svd_lib.so: sdp_gain_calibration_svd_lib/CMakeFiles/sdp_gain_calibration_svd_lib.dir/src/Gaincallogger.c.o
sdp_gain_calibration_svd_lib/libsdp_gain_calibration_svd_lib.so: sdp_gain_calibration_svd_lib/CMakeFiles/sdp_gain_calibration_svd_lib.dir/src/Gaincalfunctionsdevice.cu.o
sdp_gain_calibration_svd_lib/libsdp_gain_calibration_svd_lib.so: sdp_gain_calibration_svd_lib/CMakeFiles/sdp_gain_calibration_svd_lib.dir/build.make
sdp_gain_calibration_svd_lib/libsdp_gain_calibration_svd_lib.so: /usr/local/cuda/lib64/libcusolver.so
sdp_gain_calibration_svd_lib/libsdp_gain_calibration_svd_lib.so: sdp_gain_calibration_svd_lib/CMakeFiles/sdp_gain_calibration_svd_lib.dir/cmake_device_link.o
sdp_gain_calibration_svd_lib/libsdp_gain_calibration_svd_lib.so: sdp_gain_calibration_svd_lib/CMakeFiles/sdp_gain_calibration_svd_lib.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/aensor/ska/spfl-gaincalibration/ska-sdp-func/src/ska-sdp-func/gain_calibration_svd/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking CUDA shared library libsdp_gain_calibration_svd_lib.so"
	cd /home/aensor/ska/spfl-gaincalibration/ska-sdp-func/src/ska-sdp-func/gain_calibration_svd/build/sdp_gain_calibration_svd_lib && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/sdp_gain_calibration_svd_lib.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
sdp_gain_calibration_svd_lib/CMakeFiles/sdp_gain_calibration_svd_lib.dir/build: sdp_gain_calibration_svd_lib/libsdp_gain_calibration_svd_lib.so
.PHONY : sdp_gain_calibration_svd_lib/CMakeFiles/sdp_gain_calibration_svd_lib.dir/build

sdp_gain_calibration_svd_lib/CMakeFiles/sdp_gain_calibration_svd_lib.dir/clean:
	cd /home/aensor/ska/spfl-gaincalibration/ska-sdp-func/src/ska-sdp-func/gain_calibration_svd/build/sdp_gain_calibration_svd_lib && $(CMAKE_COMMAND) -P CMakeFiles/sdp_gain_calibration_svd_lib.dir/cmake_clean.cmake
.PHONY : sdp_gain_calibration_svd_lib/CMakeFiles/sdp_gain_calibration_svd_lib.dir/clean

sdp_gain_calibration_svd_lib/CMakeFiles/sdp_gain_calibration_svd_lib.dir/depend:
	cd /home/aensor/ska/spfl-gaincalibration/ska-sdp-func/src/ska-sdp-func/gain_calibration_svd/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/aensor/ska/spfl-gaincalibration/ska-sdp-func/src/ska-sdp-func/gain_calibration_svd /home/aensor/ska/spfl-gaincalibration/ska-sdp-func/src/ska-sdp-func/gain_calibration_svd/sdp_gain_calibration_svd_lib /home/aensor/ska/spfl-gaincalibration/ska-sdp-func/src/ska-sdp-func/gain_calibration_svd/build /home/aensor/ska/spfl-gaincalibration/ska-sdp-func/src/ska-sdp-func/gain_calibration_svd/build/sdp_gain_calibration_svd_lib /home/aensor/ska/spfl-gaincalibration/ska-sdp-func/src/ska-sdp-func/gain_calibration_svd/build/sdp_gain_calibration_svd_lib/CMakeFiles/sdp_gain_calibration_svd_lib.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : sdp_gain_calibration_svd_lib/CMakeFiles/sdp_gain_calibration_svd_lib.dir/depend

