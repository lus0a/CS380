# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/lus0a/CS380/cs380-2021/2_assignment

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/lus0a/CS380/cs380-2021/2_assignment

# Include any dependencies generated for this target.
include CMakeFiles/assignment.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/assignment.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/assignment.dir/flags.make

CMakeFiles/assignment.dir/home/lus0a/CS380/cs380-2021/common/glad/glad.c.o: CMakeFiles/assignment.dir/flags.make
CMakeFiles/assignment.dir/home/lus0a/CS380/cs380-2021/common/glad/glad.c.o: /home/lus0a/CS380/cs380-2021/common/glad/glad.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/lus0a/CS380/cs380-2021/2_assignment/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object CMakeFiles/assignment.dir/home/lus0a/CS380/cs380-2021/common/glad/glad.c.o"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/assignment.dir/home/lus0a/CS380/cs380-2021/common/glad/glad.c.o   -c /home/lus0a/CS380/cs380-2021/common/glad/glad.c

CMakeFiles/assignment.dir/home/lus0a/CS380/cs380-2021/common/glad/glad.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/assignment.dir/home/lus0a/CS380/cs380-2021/common/glad/glad.c.i"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/lus0a/CS380/cs380-2021/common/glad/glad.c > CMakeFiles/assignment.dir/home/lus0a/CS380/cs380-2021/common/glad/glad.c.i

CMakeFiles/assignment.dir/home/lus0a/CS380/cs380-2021/common/glad/glad.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/assignment.dir/home/lus0a/CS380/cs380-2021/common/glad/glad.c.s"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/lus0a/CS380/cs380-2021/common/glad/glad.c -o CMakeFiles/assignment.dir/home/lus0a/CS380/cs380-2021/common/glad/glad.c.s

CMakeFiles/assignment.dir/src/CS380_prog.cpp.o: CMakeFiles/assignment.dir/flags.make
CMakeFiles/assignment.dir/src/CS380_prog.cpp.o: src/CS380_prog.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/lus0a/CS380/cs380-2021/2_assignment/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/assignment.dir/src/CS380_prog.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/assignment.dir/src/CS380_prog.cpp.o -c /home/lus0a/CS380/cs380-2021/2_assignment/src/CS380_prog.cpp

CMakeFiles/assignment.dir/src/CS380_prog.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/assignment.dir/src/CS380_prog.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/lus0a/CS380/cs380-2021/2_assignment/src/CS380_prog.cpp > CMakeFiles/assignment.dir/src/CS380_prog.cpp.i

CMakeFiles/assignment.dir/src/CS380_prog.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/assignment.dir/src/CS380_prog.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/lus0a/CS380/cs380-2021/2_assignment/src/CS380_prog.cpp -o CMakeFiles/assignment.dir/src/CS380_prog.cpp.s

CMakeFiles/assignment.dir/src/glslprogram.cpp.o: CMakeFiles/assignment.dir/flags.make
CMakeFiles/assignment.dir/src/glslprogram.cpp.o: src/glslprogram.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/lus0a/CS380/cs380-2021/2_assignment/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/assignment.dir/src/glslprogram.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/assignment.dir/src/glslprogram.cpp.o -c /home/lus0a/CS380/cs380-2021/2_assignment/src/glslprogram.cpp

CMakeFiles/assignment.dir/src/glslprogram.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/assignment.dir/src/glslprogram.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/lus0a/CS380/cs380-2021/2_assignment/src/glslprogram.cpp > CMakeFiles/assignment.dir/src/glslprogram.cpp.i

CMakeFiles/assignment.dir/src/glslprogram.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/assignment.dir/src/glslprogram.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/lus0a/CS380/cs380-2021/2_assignment/src/glslprogram.cpp -o CMakeFiles/assignment.dir/src/glslprogram.cpp.s

CMakeFiles/assignment.dir/src/vbocube.cpp.o: CMakeFiles/assignment.dir/flags.make
CMakeFiles/assignment.dir/src/vbocube.cpp.o: src/vbocube.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/lus0a/CS380/cs380-2021/2_assignment/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/assignment.dir/src/vbocube.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/assignment.dir/src/vbocube.cpp.o -c /home/lus0a/CS380/cs380-2021/2_assignment/src/vbocube.cpp

CMakeFiles/assignment.dir/src/vbocube.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/assignment.dir/src/vbocube.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/lus0a/CS380/cs380-2021/2_assignment/src/vbocube.cpp > CMakeFiles/assignment.dir/src/vbocube.cpp.i

CMakeFiles/assignment.dir/src/vbocube.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/assignment.dir/src/vbocube.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/lus0a/CS380/cs380-2021/2_assignment/src/vbocube.cpp -o CMakeFiles/assignment.dir/src/vbocube.cpp.s

CMakeFiles/assignment.dir/src/vbomesh.cpp.o: CMakeFiles/assignment.dir/flags.make
CMakeFiles/assignment.dir/src/vbomesh.cpp.o: src/vbomesh.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/lus0a/CS380/cs380-2021/2_assignment/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object CMakeFiles/assignment.dir/src/vbomesh.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/assignment.dir/src/vbomesh.cpp.o -c /home/lus0a/CS380/cs380-2021/2_assignment/src/vbomesh.cpp

CMakeFiles/assignment.dir/src/vbomesh.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/assignment.dir/src/vbomesh.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/lus0a/CS380/cs380-2021/2_assignment/src/vbomesh.cpp > CMakeFiles/assignment.dir/src/vbomesh.cpp.i

CMakeFiles/assignment.dir/src/vbomesh.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/assignment.dir/src/vbomesh.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/lus0a/CS380/cs380-2021/2_assignment/src/vbomesh.cpp -o CMakeFiles/assignment.dir/src/vbomesh.cpp.s

CMakeFiles/assignment.dir/src/vbodisc.cpp.o: CMakeFiles/assignment.dir/flags.make
CMakeFiles/assignment.dir/src/vbodisc.cpp.o: src/vbodisc.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/lus0a/CS380/cs380-2021/2_assignment/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object CMakeFiles/assignment.dir/src/vbodisc.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/assignment.dir/src/vbodisc.cpp.o -c /home/lus0a/CS380/cs380-2021/2_assignment/src/vbodisc.cpp

CMakeFiles/assignment.dir/src/vbodisc.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/assignment.dir/src/vbodisc.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/lus0a/CS380/cs380-2021/2_assignment/src/vbodisc.cpp > CMakeFiles/assignment.dir/src/vbodisc.cpp.i

CMakeFiles/assignment.dir/src/vbodisc.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/assignment.dir/src/vbodisc.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/lus0a/CS380/cs380-2021/2_assignment/src/vbodisc.cpp -o CMakeFiles/assignment.dir/src/vbodisc.cpp.s

CMakeFiles/assignment.dir/src/vbocylinder.cpp.o: CMakeFiles/assignment.dir/flags.make
CMakeFiles/assignment.dir/src/vbocylinder.cpp.o: src/vbocylinder.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/lus0a/CS380/cs380-2021/2_assignment/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Building CXX object CMakeFiles/assignment.dir/src/vbocylinder.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/assignment.dir/src/vbocylinder.cpp.o -c /home/lus0a/CS380/cs380-2021/2_assignment/src/vbocylinder.cpp

CMakeFiles/assignment.dir/src/vbocylinder.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/assignment.dir/src/vbocylinder.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/lus0a/CS380/cs380-2021/2_assignment/src/vbocylinder.cpp > CMakeFiles/assignment.dir/src/vbocylinder.cpp.i

CMakeFiles/assignment.dir/src/vbocylinder.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/assignment.dir/src/vbocylinder.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/lus0a/CS380/cs380-2021/2_assignment/src/vbocylinder.cpp -o CMakeFiles/assignment.dir/src/vbocylinder.cpp.s

CMakeFiles/assignment.dir/src/vbosphere.cpp.o: CMakeFiles/assignment.dir/flags.make
CMakeFiles/assignment.dir/src/vbosphere.cpp.o: src/vbosphere.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/lus0a/CS380/cs380-2021/2_assignment/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Building CXX object CMakeFiles/assignment.dir/src/vbosphere.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/assignment.dir/src/vbosphere.cpp.o -c /home/lus0a/CS380/cs380-2021/2_assignment/src/vbosphere.cpp

CMakeFiles/assignment.dir/src/vbosphere.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/assignment.dir/src/vbosphere.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/lus0a/CS380/cs380-2021/2_assignment/src/vbosphere.cpp > CMakeFiles/assignment.dir/src/vbosphere.cpp.i

CMakeFiles/assignment.dir/src/vbosphere.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/assignment.dir/src/vbosphere.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/lus0a/CS380/cs380-2021/2_assignment/src/vbosphere.cpp -o CMakeFiles/assignment.dir/src/vbosphere.cpp.s

# Object files for target assignment
assignment_OBJECTS = \
"CMakeFiles/assignment.dir/home/lus0a/CS380/cs380-2021/common/glad/glad.c.o" \
"CMakeFiles/assignment.dir/src/CS380_prog.cpp.o" \
"CMakeFiles/assignment.dir/src/glslprogram.cpp.o" \
"CMakeFiles/assignment.dir/src/vbocube.cpp.o" \
"CMakeFiles/assignment.dir/src/vbomesh.cpp.o" \
"CMakeFiles/assignment.dir/src/vbodisc.cpp.o" \
"CMakeFiles/assignment.dir/src/vbocylinder.cpp.o" \
"CMakeFiles/assignment.dir/src/vbosphere.cpp.o"

# External object files for target assignment
assignment_EXTERNAL_OBJECTS =

assignment: CMakeFiles/assignment.dir/home/lus0a/CS380/cs380-2021/common/glad/glad.c.o
assignment: CMakeFiles/assignment.dir/src/CS380_prog.cpp.o
assignment: CMakeFiles/assignment.dir/src/glslprogram.cpp.o
assignment: CMakeFiles/assignment.dir/src/vbocube.cpp.o
assignment: CMakeFiles/assignment.dir/src/vbomesh.cpp.o
assignment: CMakeFiles/assignment.dir/src/vbodisc.cpp.o
assignment: CMakeFiles/assignment.dir/src/vbocylinder.cpp.o
assignment: CMakeFiles/assignment.dir/src/vbosphere.cpp.o
assignment: CMakeFiles/assignment.dir/build.make
assignment: /usr/local/cuda-11.4/lib64/libcudart_static.a
assignment: /usr/lib/x86_64-linux-gnu/librt.so
assignment: /home/lus0a/CS380/cs380-2021/common/glfw-3.3/build/src/libglfw3.a
assignment: /usr/lib/x86_64-linux-gnu/librt.so
assignment: /usr/lib/x86_64-linux-gnu/libm.so
assignment: /usr/lib/x86_64-linux-gnu/libX11.so
assignment: CMakeFiles/assignment.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/lus0a/CS380/cs380-2021/2_assignment/CMakeFiles --progress-num=$(CMAKE_PROGRESS_9) "Linking CXX executable assignment"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/assignment.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/assignment.dir/build: assignment

.PHONY : CMakeFiles/assignment.dir/build

CMakeFiles/assignment.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/assignment.dir/cmake_clean.cmake
.PHONY : CMakeFiles/assignment.dir/clean

CMakeFiles/assignment.dir/depend:
	cd /home/lus0a/CS380/cs380-2021/2_assignment && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/lus0a/CS380/cs380-2021/2_assignment /home/lus0a/CS380/cs380-2021/2_assignment /home/lus0a/CS380/cs380-2021/2_assignment /home/lus0a/CS380/cs380-2021/2_assignment /home/lus0a/CS380/cs380-2021/2_assignment/CMakeFiles/assignment.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/assignment.dir/depend

