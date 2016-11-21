# Defines MODEL_MEAN_PATH, MODEL_PATH, PROTO_PATH and TEST_IMAGE_PATH   
include(${CMAKE_SOURCE_DIR}/cmake/bvlc_reference_caffenet_resources.cmake) 

# Add tiny-dnn forward pass benchmarks 
add_executable(bvlc_reference_caffenet bvlc_reference_caffenet.cpp)
target_link_libraries(bvlc_reference_caffenet benchmark caffe tiny_dnn)
