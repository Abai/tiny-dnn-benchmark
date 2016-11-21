#include <cassert>
#include <string>

#include "benchmark/benchmark.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "caffe/caffe.hpp"
//#include "caffe/util/io.hpp"
//using namespace std; // required for upgrade_proto.hpp
//#include "caffe/util/upgrade_proto.hpp"
//#include "caffe/proto/caffe.pb.h"

#include "tiny_dnn/tiny_dnn.h"

using namespace caffe;
using namespace tiny_dnn;
using namespace tiny_dnn::activation;
using namespace tiny_dnn::layers;

namespace {

// Verify CMake has found files required to run the benchmark
#if !defined(MODEL_MEAN_PATH) || !defined(MODEL_PATH) || !defined(PROTO_PATH) \
 || !defined(TEST_IMAGE_PATH)
#error  "Error: Could not find bvlc_reference_caffenet benchmark resources" 
#else
  const std::string mean_path  = MODEL_MEAN_PATH;  
  const std::string model_path = MODEL_PATH;  
  const std::string proto_path = PROTO_PATH;  
  const std::string image_path = TEST_IMAGE_PATH;  
#endif


class BVLCReferenceCaffenet : public ::benchmark::Fixture {
 public:
  // You can remove any or all of the following functions if its body
  // is empty.

  BVLCReferenceCaffenet() {
    // Initialization happens once per run
    if(!sIsInit) {
      SetUpTestCase();
    }
  }

  virtual ~BVLCReferenceCaffenet() {
    // You can do clean-up work that doesn't throw exceptions here.
  }

  // If the constructor and destructor are not enough for setting up
  // and cleaning up each test, you can define the following methods:

  static void SetUpTestCase() {
    // Read Caffe's model prototxt file and upgrade if needed
    sProto.reset(new NetParameter());
    ReadNetParamsFromTextFileOrDie(proto_path, sProto.get());

    // Read Caffe's binary model file and upgrade if needed
    sWeights.reset(new NetParameter());
    ReadNetParamsFromBinaryFileOrDie(model_path, sWeights.get());

    // Determine network input size
    cv::Size crop_size(sProto->layer(0).input_param().shape(0).dim(2),
                       sProto->layer(0).input_param().shape(0).dim(3));

    // Read Caffe's model mean
    sMean = load_mean(mean_path);

    // Load and scale test image to mean size 
    cv::Mat image = load_image(image_path);

    // Convert image to float and subtract mean
    cv::Mat normalized_image = zero_mean(image, sMean);

    // Apply center crop on image to match network input size
    cv::Mat cropped_image = center_crop(normalized_image, crop_size);

    // Deinterlace image into float features
    //cv::split(cropped_image, sTestFeatures);

    sIsInit = true;
  }

  static cv::Mat load_mean(const std::string & mean_file) {
    BlobProto blob_proto;
    ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);

    // Convert from BlobProto to Blob<float>
    Blob<float> mean_blob;
    mean_blob.FromProto(blob_proto);

    // Validate mean dimensions
    cv::Size mean_size(mean_blob.shape().at(2), mean_blob.shape().at(3));
    int num_channels = mean_blob.shape().at(1);
    assert(mean_size.width == 256 && mean_size.height == 256);
    assert(num_channels == 3);
  
    // The format of the mean file is planar 32-bit float BGR or grayscale
    std::vector<cv::Mat> channels;
    float* data = mean_blob.mutable_cpu_data();
    for (int i = 0; i < num_channels; ++i) {
      // Extract an individual channel
      cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
      channels.push_back(channel);
      data += mean_blob.height() * mean_blob.width();
    }
  
    // Merge the separate channels into a single image
    cv::Mat mean;
    cv::merge(channels, mean);
  
    // Compute the global mean pixel value and create a mean image
    // filled with this value
    cv::Scalar channel_mean = cv::mean(mean);
    return cv::Mat(mean_size, mean.type(), channel_mean);
  }

  static cv::Mat load_image(const std::string & image_path) {
    cv::Mat img = cv::imread(image_path, -1); 
    assert(img.channels() == sMean.channels());
 
    cv::Mat scaled_img; 
    if(img.size() != sMean.size()) {
      cv::resize(img, scaled_img, sMean.size()); 
    }
    else {
      scaled_img = img;
    }

    return scaled_img;
  }

  static cv::Mat zero_mean(const cv::Mat & scaled_img, const cv::Mat & mean) {
    cv::Mat float_img;
    if(sMean.channels() == 3) {
      scaled_img.convertTo(float_img, CV_32FC3);
    }
    else {
      scaled_img.convertTo(float_img, CV_32FC1);
    }

    cv::Mat normalized_img;
    cv::subtract(float_img, mean, normalized_img);

    return normalized_img;
  }

  static cv::Mat center_crop(const cv::Mat & normalized_img,
                             const cv::Size & crop_size) {
    cv::Mat cropped_img;
    if(normalized_img.size() == crop_size) {
      cropped_img = normalized_img;
    }
    else {
      int x = (normalized_img.cols - crop_size.width) / 2;
      int y = (normalized_img.rows - crop_size.height) / 2;
      cv::Rect crop = cv::Rect(x, y, crop_size.width, crop_size.height);
      cropped_img = normalized_img(crop);
    }

    return cropped_img;
  }

  virtual void SetUp() {
    // Code here will be called immediately after the constructor (right
    // before each test).
  }

  virtual void TearDown() {
    // Code here will be called immediately after each test (right
    // before the destructor).
  }

  // Objects declared here can be used by all tests in the test case for Foo.
  static bool sIsInit;
  static cv::Mat sMean;
  static std::unique_ptr<NetParameter> sProto;
  static std::unique_ptr<NetParameter> sWeights;
  static std::vector<float> sTestFeatures;
};

bool BVLCReferenceCaffenet::sIsInit = false;
cv::Mat BVLCReferenceCaffenet::sMean;
std::unique_ptr<NetParameter> BVLCReferenceCaffenet::sProto; 
std::unique_ptr<NetParameter> BVLCReferenceCaffenet::sWeights; 
std::vector<float> BVLCReferenceCaffenet::sTestFeatures;

BENCHMARK_F(BVLCReferenceCaffenet, FooTest)(benchmark::State& st) {
  while (st.KeepRunning()) {
    std::string mystring;
  }
}

BENCHMARK_DEFINE_F(BVLCReferenceCaffenet, BarTest)(benchmark::State& st) {
  while (st.KeepRunning()) {
    std::string mystring = "bla";
  }
}
/* BarTest is NOT registered */
BENCHMARK_REGISTER_F(BVLCReferenceCaffenet, BarTest)->Threads(2);
/* BarTest is now registered */

}  // namespace

BENCHMARK_MAIN()
