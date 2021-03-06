cmake_minimum_required(VERSION 3.5.1)
project(tiny-dnn-benchmark)

# Enable c++11 support
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Locate own find_package CMake scripts
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

# Configure and add Caffe 
option(CPU_ONLY "" ON)
option(BUILD_python "" OFF)
option(BUILD_docs "" OFF)
option(USE_LEVELDB "" OFF)
option(USE_LMDB "" OFF)
add_subdirectory(caffe)
include_directories("${Caffe_SOURCE_DIR}/include"
                    "${Caffe_BINARY_DIR}/include/caffe/proto") # caffe.pb.h 

# Copy Caffe's preprocessor defines to configure Caffe's internal headers 
get_directory_property(CAFFE_DEFINITIONS DIRECTORY caffe COMPILE_DEFINITIONS)
foreach( d ${CAFFE_DEFINITIONS} )
  add_definitions(-D${d})
endforeach()

# Copy Caffe's include directories 
get_directory_property(CAFFE_INCLUDE_DIRS DIRECTORY caffe INCLUDE_DIRECTORIES)
foreach( d ${CAFFE_INCLUDE_DIRS} )
  include_directories(${d})
endforeach()

# Configure and google-benchmark
option(BENCHMARK_ENABLE_TESTING "" OFF)
option(BENCHMARK_USE_LIBCXX "" ON)
add_subdirectory(google-benchmark)

# Add tiny-dnn
add_subdirectory(tiny-dnn)

# Add tiny-dnn benchmarks
add_subdirectory(benchmarks)
