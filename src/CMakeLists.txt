cmake_minimum_required(VERSION 3.6.2)
project(tiny-dnn-benchmark)

# Enable c++11 support
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Caffe's gtest config
add_definitions(-DGTEST_USE_OWN_TR1_TUPLE=1 -DGTEST_HAS_TR1_TUPLE=1)
include_directories(caffe/src)

# Add subdir CmakeLists
add_subdirectory(caffe)
add_subdirectory(tiny-dnn)
add_subdirectory(tests)
