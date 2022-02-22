cmake_minimum_required(VERSION 3.1)
 project(Profiling)
 
 # support C++14 features used by LLVM 10.0.0
 set(CMAKE_CXX_STANDARD 14)
 SET(CMAKE_CXX_FLAGS "-Wall -frtti")
 find_package(LLVM REQUIRED CONFIG)
 add_definitions(${LLVM_DEFINITIONS})
 include_directories(/data/home/mzw/llvm-13.0.0/include)
 link_directories(/data/home/mzw/llvm-13.0.0)
 
 #add_subdirectory(skeleton)  # Use your pass name here.
 add_subdirectory(Profiling)
