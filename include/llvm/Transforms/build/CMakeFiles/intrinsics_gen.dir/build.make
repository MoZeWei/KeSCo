# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.21

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
CMAKE_COMMAND = /data/public/wuk/env/v1/spack/opt/spack/linux-centos7-zen/gcc-8.5.0/cmake-3.21.4-nvydniyy3o7epsmkovhbb2r2gabeijca/bin/cmake

# The command to remove a file.
RM = /data/public/wuk/env/v1/spack/opt/spack/linux-centos7-zen/gcc-8.5.0/cmake-3.21.4-nvydniyy3o7epsmkovhbb2r2gabeijca/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /data/home/mzw/llvm-13.0.0/include/llvm/Transforms

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /data/home/mzw/llvm-13.0.0/include/llvm/Transforms/build

# Utility rule file for intrinsics_gen.

# Include any custom commands dependencies for this target.
include CMakeFiles/intrinsics_gen.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/intrinsics_gen.dir/progress.make

intrinsics_gen: CMakeFiles/intrinsics_gen.dir/build.make
.PHONY : intrinsics_gen

# Rule to build all files generated by this target.
CMakeFiles/intrinsics_gen.dir/build: intrinsics_gen
.PHONY : CMakeFiles/intrinsics_gen.dir/build

CMakeFiles/intrinsics_gen.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/intrinsics_gen.dir/cmake_clean.cmake
.PHONY : CMakeFiles/intrinsics_gen.dir/clean

CMakeFiles/intrinsics_gen.dir/depend:
	cd /data/home/mzw/llvm-13.0.0/include/llvm/Transforms/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /data/home/mzw/llvm-13.0.0/include/llvm/Transforms /data/home/mzw/llvm-13.0.0/include/llvm/Transforms /data/home/mzw/llvm-13.0.0/include/llvm/Transforms/build /data/home/mzw/llvm-13.0.0/include/llvm/Transforms/build /data/home/mzw/llvm-13.0.0/include/llvm/Transforms/build/CMakeFiles/intrinsics_gen.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/intrinsics_gen.dir/depend

