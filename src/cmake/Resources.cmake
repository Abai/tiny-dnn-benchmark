# Define benchmark resources
set(MODEL_MEAN_PATH "${CMAKE_SOURCE_DIR}/caffe/data/ilsvrc12/imagenet_mean.binaryproto")
set(MODEL_PATH "${CMAKE_SOURCE_DIR}/caffe/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel")
set(PROTO_PATH "${CMAKE_SOURCE_DIR}/caffe/models/bvlc_reference_caffenet/deploy.prototxt")
set(TEST_IMAGE_PATH "${CMAKE_SOURCE_DIR}/caffe/examples/images/cat.jpg")

# Download imagenet_mean if required 
if(NOT EXISTS ${MODEL_MEAN_PATH})
  message(STATUS "Downloading imagenet_mean.binaryproto ...")
  execute_process(COMMAND ${CMAKE_SOURCE_DIR}/caffe/data/ilsvrc12/get_ilsvrc_aux.sh)
  if(NOT EXISTS ${MODEL_MEAN_PATH})
    message(FATAL_ERROR "Failed to download imagenet_mean.binaryproto to ${MODEL_MEAN_PATH}")
  else()
    add_definitions(-DMODEL_MEAN_PATH=${MODEL_MEAN_PATH})
  endif()
endif()

# Download bvlc_reference_caffenet model if required
if(NOT EXISTS ${MODEL_PATH})
  message(STATUS "Downloading bvlc_reference_caffenet.caffemodel ...")
  execute_process(COMMAND ${CMAKE_SOURCE_DIR}/caffe/scripts/download_model_binary.py ${CMAKE_SOURCE_DIR}/caffe/models/bvlc_reference_caffenet)
  if(NOT EXISTS ${MODEL_PATH})
    message(FATAL_ERROR "Failed to download bvlc_reference_caffenet.caffemodel to ${MODEL_PATH}")
  else()
    add_definitions(-DMODEL_PATH=${MODEL_PATH})
  endif()
endif()

# Verify test image and model proto can be found 
if(NOT EXISTS ${PROTO_PATH})
  message(FATAL_ERROR "Could not find prototxt at ${PROTO_PATH}")
else()
  add_definitions(-DPROTO_PATH=${PROTO_PATH})
endif()

if(NOT EXISTS ${TEST_IMAGE_PATH})
  message(FATAL_ERROR "Could not find test image at ${TEST_IMAGE_PATH}")
else()
  add_definitions(-DTEST_IMAGE_PATH=${TEST_IMAGE_PATH})
endif()
