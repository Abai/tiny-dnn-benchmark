# Add tiny-dnn forward pass benchmarks 
add_executable(tiny-dnn-forward tiny_dnn_forward.cpp)
target_link_libraries(tiny-dnn-forward tiny_dnn gtest)
