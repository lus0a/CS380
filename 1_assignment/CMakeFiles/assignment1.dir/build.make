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
CMAKE_SOURCE_DIR = /home/lus0a/CS380/cs380-2021/1_assignment

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/lus0a/CS380/cs380-2021/1_assignment

# Include any dependencies generated for this target.
include CMakeFiles/assignment1.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/assignment1.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/assignment1.dir/flags.make

CMakeFiles/assignment1.dir/src/CS380_prog.cpp.o: CMakeFiles/assignment1.dir/flags.make
CMakeFiles/assignment1.dir/src/CS380_prog.cpp.o: src/CS380_prog.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/lus0a/CS380/cs380-2021/1_assignment/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/assignment1.dir/src/CS380_prog.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/assignment1.dir/src/CS380_prog.cpp.o -c /home/lus0a/CS380/cs380-2021/1_assignment/src/CS380_prog.cpp

CMakeFiles/assignment1.dir/src/CS380_prog.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/assignment1.dir/src/CS380_prog.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/lus0a/CS380/cs380-2021/1_assignment/src/CS380_prog.cpp > CMakeFiles/assignment1.dir/src/CS380_prog.cpp.i

CMakeFiles/assignment1.dir/src/CS380_prog.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/assignment1.dir/src/CS380_prog.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/lus0a/CS380/cs380-2021/1_assignment/src/CS380_prog.cpp -o CMakeFiles/assignment1.dir/src/CS380_prog.cpp.s

CMakeFiles/assignment1.dir/home/lus0a/CS380/cs380-2021/common/glad/glad.c.o: CMakeFiles/assignment1.dir/flags.make
CMakeFiles/assignment1.dir/home/lus0a/CS380/cs380-2021/common/glad/glad.c.o: /home/lus0a/CS380/cs380-2021/common/glad/glad.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/lus0a/CS380/cs380-2021/1_assignment/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building C object CMakeFiles/assignment1.dir/home/lus0a/CS380/cs380-2021/common/glad/glad.c.o"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/assignment1.dir/home/lus0a/CS380/cs380-2021/common/glad/glad.c.o   -c /home/lus0a/CS380/cs380-2021/common/glad/glad.c

CMakeFiles/assignment1.dir/home/lus0a/CS380/cs380-2021/common/glad/glad.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/assignment1.dir/home/lus0a/CS380/cs380-2021/common/glad/glad.c.i"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/lus0a/CS380/cs380-2021/common/glad/glad.c > CMakeFiles/assignment1.dir/home/lus0a/CS380/cs380-2021/common/glad/glad.c.i

CMakeFiles/assignment1.dir/home/lus0a/CS380/cs380-2021/common/glad/glad.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/assignment1.dir/home/lus0a/CS380/cs380-2021/common/glad/glad.c.s"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/lus0a/CS380/cs380-2021/common/glad/glad.c -o CMakeFiles/assignment1.dir/home/lus0a/CS380/cs380-2021/common/glad/glad.c.s

# Object files for target assignment1
assignment1_OBJECTS = \
"CMakeFiles/assignment1.dir/src/CS380_prog.cpp.o" \
"CMakeFiles/assignment1.dir/home/lus0a/CS380/cs380-2021/common/glad/glad.c.o"

# External object files for target assignment1
assignment1_EXTERNAL_OBJECTS =

assignment1: CMakeFiles/assignment1.dir/src/CS380_prog.cpp.o
assignment1: CMakeFiles/assignment1.dir/home/lus0a/CS380/cs380-2021/common/glad/glad.c.o
assignment1: CMakeFiles/assignment1.dir/build.make
assignment1: /usr/local/cuda-11.4/lib64/libcudart_static.a
assignment1: /usr/lib/x86_64-linux-gnu/librt.so
assignment1: /home/lus0a/CS380/cs380-2021/common/glfw-3.3/build/src/libglfw3.a
assignment1: /usr/lib/x86_64-linux-gnu/librt.so
assignment1: /usr/lib/x86_64-linux-gnu/libm.so
assignment1: /usr/lib/x86_64-linux-gnu/libX11.so
assignment1: CMakeFiles/assignment1.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/lus0a/CS380/cs380-2021/1_assignment/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable assignment1"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/assignment1.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/assignment1.dir/build: assignment1

.PHONY : CMakeFiles/assignment1.dir/build

CMakeFiles/assignment1.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/assignment1.dir/cmake_clean.cmake
.PHONY : CMakeFiles/assignment1.dir/clean

CMakeFiles/assignment1.dir/depend:
	cd /home/lus0a/CS380/cs380-2021/1_assignment && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/lus0a/CS380/cs380-2021/1_assignment /home/lus0a/CS380/cs380-2021/1_assignment /home/lus0a/CS380/cs380-2021/1_assignment /home/lus0a/CS380/cs380-2021/1_assignment /home/lus0a/CS380/cs380-2021/1_assignment/CMakeFiles/assignment1.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/assignment1.dir/depend

