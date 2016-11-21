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
#if  !defined(PROTO_PATH) || !defined(MODEL_PATH)
#error  "Error: Could not find bvlc_reference_caffenet benchmark resources" 
#else
  const std::string proto_path = PROTO_PATH;
  const std::string model_path = MODEL_PATH;
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

    // Load caffe net
    Net<float> caffe_net(proto_path, TEST);
    caffe_net.CopyTrainedLayersFrom(model_path);

    // Perform one forward pass to fill blobs
    const std::vector<Blob<float>*>& result = caffe_net.Forward();

//    // Fill pointers to layers
//    sConv1 = 

    sIsInit = true;
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
  static std::unique_ptr<NetParameter> sProto;
  static std::unique_ptr<NetParameter> sWeights;
  static std::vector<float> sTestFeatures;
};

bool BVLCReferenceCaffenet::sIsInit = false;
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
