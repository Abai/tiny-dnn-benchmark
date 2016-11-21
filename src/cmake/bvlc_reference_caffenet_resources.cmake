# Define benchmark resources
set(MODEL_MEAN_PATH "${CMAKE_SOURCE_DIR}/caffe/data/ilsvrc12/imagenet_mean.binaryproto")
set(MODEL_PATH "${CMAKE_SOURCE_DIR}/caffe/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel")
set(PROTO_PATH_IN "${CMAKE_SOURCE_DIR}/benchmarks/bvlc_reference_caffenet.prototxt.in")
set(PROTO_PATH "${CMAKE_CURRENT_BINARY_DIR}/bvlc_reference_caffenet.prototxt")
set(SOURCE_TXT_PATH_IN "${CMAKE_SOURCE_DIR}/benchmarks/bvlc_reference_caffenet_sources.txt.in")
set(SOURCE_TXT_PATH "${CMAKE_CURRENT_BINARY_DIR}/bvlc_reference_caffenet_sources.txt")
set(TEST_IMAGE_PATH "${CMAKE_SOURCE_DIR}/caffe/examples/images/cat.jpg")

# Download imagenet_mean if required 
if(NOT EXISTS ${MODEL_MEAN_PATH})
  message(STATUS "Downloading imagenet_mean.binaryproto ...")
  execute_process(COMMAND ${CMAKE_SOURCE_DIR}/caffe/data/ilsvrc12/get_ilsvrc_aux.sh)
  if(NOT EXISTS ${MODEL_MEAN_PATH})
    message(FATAL_ERROR "Failed to download imagenet_mean.binaryproto to ${MODEL_MEAN_PATH}")
  endif()
endif()

# Download bvlc_reference_caffenet model if required
if(NOT EXISTS ${MODEL_PATH})
  message(STATUS "Downloading bvlc_reference_caffenet.caffemodel ...")
  execute_process(COMMAND ${CMAKE_SOURCE_DIR}/caffe/scripts/download_model_binary.py ${CMAKE_SOURCE_DIR}/caffe/models/bvlc_reference_caffenet)
  if(NOT EXISTS ${MODEL_PATH})
    message(FATAL_ERROR "Failed to download bvlc_reference_caffenet.caffemodel to ${MODEL_PATH}")
  endif()
else()
  add_definitions(-DMODEL_PATH="${MODEL_PATH}")
endif()

if(NOT EXISTS ${TEST_IMAGE_PATH})
  message(FATAL_ERROR "Could not find test image at ${TEST_IMAGE_PATH}")
endif()

# Configure prototxt ImageData layer and test image source txt
configure_file(${PROTO_PATH_IN} ${PROTO_PATH})
configure_file(${SOURCE_TXT_PATH_IN} ${SOURCE_TXT_PATH})

# Verify model prototxt can be found 
if(NOT EXISTS ${PROTO_PATH})
  message(FATAL_ERROR "Could not find prototxt at ${PROTO_PATH}")
else()
  add_definitions(-DPROTO_PATH="${PROTO_PATH}")
endif()

# Verify sources txt can be found 
if(NOT EXISTS ${SOURCE_TXT_PATH})
  message(FATAL_ERROR "Could not find sources txt at ${SOURCE_TXT_PATH}")
endif()


