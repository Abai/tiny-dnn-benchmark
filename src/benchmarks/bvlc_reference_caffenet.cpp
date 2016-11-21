#include <cassert>
#include <string>

#include "benchmark/benchmark.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "caffe/caffe.hpp"

#include "tiny_dnn/tiny_dnn.h"

using namespace caffe;
using namespace tiny_dnn;
using namespace tiny_dnn::activation;
using namespace tiny_dnn::layers;

using BlobVec = std::vector<Blob<float>*>;

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

  BVLCReferenceCaffenet() {
    // Initialization happens once per run
    if(!sIsInit) {
      SetUpTestCase();
    }
  }

  virtual ~BVLCReferenceCaffenet() {}

  static void SetUpTestCase() {
    // Read Caffe's model prototxt file and upgrade if needed
    sProto.reset(new NetParameter());
    ReadNetParamsFromTextFileOrDie(proto_path, sProto.get());

    // Read Caffe's binary model file and upgrade if needed
    sWeights.reset(new NetParameter());
    ReadNetParamsFromBinaryFileOrDie(model_path, sWeights.get());

    // Load caffe net
    sCaffeNet.reset(new Net<float>(proto_path, TEST));
    sCaffeNet->CopyTrainedLayersFrom(model_path);

    //// Perform one forward pass to fill blobs
    sCaffeNet->Forward();

    sIsInit = true;
  }

  static bool sIsInit;
  static std::unique_ptr<NetParameter> sProto;
  static std::unique_ptr<NetParameter> sWeights;

  static std::unique_ptr<Net<float>> sCaffeNet;
};

bool BVLCReferenceCaffenet::sIsInit = false;
std::unique_ptr<NetParameter> BVLCReferenceCaffenet::sProto; 
std::unique_ptr<NetParameter> BVLCReferenceCaffenet::sWeights; 

std::unique_ptr<Net<float>> BVLCReferenceCaffenet::sCaffeNet;

static void CaffeLayers(benchmark::internal::Benchmark* b) {
  int num_layers = BVLCReferenceCaffenet::sCaffeNet->layers().size();
  for(int i = 1; i < num_layers; ++i) {
    b->Args({i});
  }
}

BENCHMARK_DEFINE_F(BVLCReferenceCaffenet, CaffeLayerTest)(
  benchmark::State& state) {
  while (state.KeepRunning()) {
    state.PauseTiming();
    unsigned int i = state.range(0);
    auto layer = sCaffeNet->layers().at(i);
    auto bottom_vec = sCaffeNet->bottom_vecs().at(i);
    auto top_vec = sCaffeNet->top_vecs().at(i);
    state.ResumeTiming();
    layer->Forward(bottom_vec, top_vec);
  }
}

BENCHMARK_REGISTER_F(BVLCReferenceCaffenet, CaffeLayerTest)->Apply(CaffeLayers);

}  // namespace

BENCHMARK_MAIN()
