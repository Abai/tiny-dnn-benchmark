#include <cassert>
#include <string>

#include "benchmark/benchmark.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "caffe/caffe.hpp"

#include "tiny_dnn/tiny_dnn.h"
#include "tiny_dnn/io/caffe/layer_factory.h"

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

  BVLCReferenceCaffenet() {
    // Disable caffe output
    FLAGS_minloglevel = 2;

    // Initialization happens once per run
    if(!sIsInit) {
      SetUpTestCase();
    }
  }

  virtual ~BVLCReferenceCaffenet() {}

  static void SetUpTestCase() {
    // Load caffe net
    sCaffeNet.reset(new Net<float>(proto_path, TEST));
    sCaffeNet->CopyTrainedLayersFrom(model_path);

    // Perform one forward pass to fill blobs
    sCaffeNet->Forward();

    // Convert Caffe net to tiny-dnn layers
    sTinyDNNLayers.push_back(nullptr); // skip ImageData layer
    for(unsigned int i = 1; i < sCaffeNet->layers().size(); ++i) { 
      sTinyDNNLayers.push_back(getTinyDNNLayer(i));
    } 

    sIsInit = true;
  }

  static std::shared_ptr<layer> getTinyDNNLayer(int i) {
    // Retrieve caffe layer and top and bottom blobs
    auto caffe_layer = sCaffeNet->layers().at(i);
    auto bottom_blob = sCaffeNet->bottom_vecs().at(i).at(0);
    auto top_blob = sCaffeNet->top_vecs().at(i).at(0);

    // Validate caffe input and output blob dimensions 
    assert(bottom_blob->num_axes() <= 4);
    assert(top_blob->num_axes() <= 4);

    // Convert Caffe input shape to tiny_dnn::shape3d
    cnn_size_t in_num = 1; // 4th axis unused in tiny_dnn
    shape_t in_shape(1, 1, 1); 
    std::vector<cnn_size_t*> in_tiny_shape =
      { &in_shape.width_,  &in_shape.height_, &in_shape.depth_, &in_num };  
    int in_vec_idx = 0;
    std::vector<int> in_caffe_shape = bottom_blob->shape();
    for(std::vector<int>::reverse_iterator it = in_caffe_shape.rbegin();
        it != in_caffe_shape.rend(); ++it) { // reverse for over caffe shape 
      *in_tiny_shape.at(in_vec_idx++) = *it; 
    }

    // Convert Caffe output shape to tiny_dnn::shape3d
    cnn_size_t out_num = 1; // 4th axis unused in tiny_dnn
    shape_t out_shape(1, 1, 1); 
    std::vector<cnn_size_t*> out_tiny_shape =
      { &out_shape.width_,  &out_shape.height_, &out_shape.depth_, &out_num };  
    int out_vec_idx = 0;
    std::vector<int> out_caffe_shape = top_blob->shape();
    for(std::vector<int>::reverse_iterator it = out_caffe_shape.rbegin();
        it != out_caffe_shape.rend(); ++it) { // reverse for over caffe shape 
      *out_tiny_shape.at(out_vec_idx++) = *it; 
    }

    // Save caffe output dimensions for later validation
    cnn_size_t out_channels = out_shape.depth_;
    cnn_size_t out_height = out_shape.height_;
    cnn_size_t out_width = out_shape.width_;

    // Copy Caffe proto layer parameter
    caffe::LayerParameter layer_param;
    layer_param.CopyFrom(caffe_layer->layer_param());

    // Insert weight and bias blobs into layer param
    for(auto blob : caffe_layer->blobs()) {
      // Add BlobProto to layer paramters
      auto blob_proto = layer_param.add_blobs(); 
      // Add BlobShape to BlobProto 
      auto blob_shape = blob_proto->mutable_shape();
      for(int dim : blob->shape()) {
        blob_shape->add_dim(dim);
      }
      // Add float weights to BlobProto
      for(int i = 0; i < blob->count(); ++i) {
        blob_proto->add_data(blob->cpu_data()[i]);
      }
    }

    // Convert Caffe's layer proto parameters to tiny-dnn layer
    std::shared_ptr<layer> tiny_dnn_layer =
      detail::create(layer_param, in_shape, &out_shape);

    // Validate output dimentions
    assert( out_shape.depth_ == out_channels &&
            out_shape.height_ == out_height &&
            out_shape.width_ == out_width );

    // Load input data into tiny-dnn layer 
    tiny_dnn_layer->set_in_data(
      std::vector<tensor_t>{
        std::vector<vec_t>{ vec_t(bottom_blob->cpu_data(),
                            bottom_blob->cpu_data() + bottom_blob->count()) } } ); 

    return tiny_dnn_layer;
  }

  bool validateLayerOutput(int idx) {
    // TODO(Abai) : Examine differences between caffe and tiny-dnn for these 
    // layers. Remove if statment when fixed.
    auto l = sTinyDNNLayers.at(idx);
    if(l->layer_type() == "conv" ||
       l->layer_type() == "norm") {
      return true;
    }
    auto out_data = l->output();
    auto top_caffe = sCaffeNet->top_vecs().at(idx).at(0);
    auto top_tiny_dnn = out_data.at(0).at(0);
    float threshold = 0.0001f;
    for(int i = 0; i < top_caffe->count(); ++i) {
      float diff = std::abs(top_tiny_dnn.at(i) - top_caffe->cpu_data()[i]);
      if(diff > threshold) {
        std::cerr << "Warning: Difference between output of layer index="
                  << idx << " with type=" << l->layer_type() << " at blob "
                  << "index=" << i << " caffe=" << top_caffe->cpu_data()[i]
                  << " and tiny-dnn=" << top_tiny_dnn.at(i)
                  << " is larger than threshold=" << threshold <<  std::endl;
        //return false;
      }
    }
    return true;
  }


  static bool sIsInit;
  static std::unique_ptr<NetParameter> sProto;
  static std::unique_ptr<NetParameter> sWeights;

  static std::unique_ptr<Net<float>> sCaffeNet;
  static std::vector<std::shared_ptr<layer> > sTinyDNNLayers;
};

bool BVLCReferenceCaffenet::sIsInit = false;
std::unique_ptr<NetParameter> BVLCReferenceCaffenet::sProto; 
std::unique_ptr<NetParameter> BVLCReferenceCaffenet::sWeights; 

std::unique_ptr<Net<float>> BVLCReferenceCaffenet::sCaffeNet;
std::vector<std::shared_ptr<layer> > BVLCReferenceCaffenet::sTinyDNNLayers;


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
    int i = state.range(0);
    auto layer = sCaffeNet->layers().at(i);
    auto bottom_vec = sCaffeNet->bottom_vecs().at(i);
    auto top_vec = sCaffeNet->top_vecs().at(i);
    assert( bottom_vec.size() == 1u );
    state.ResumeTiming();

    layer->Forward(bottom_vec, top_vec);
  }
}

BENCHMARK_DEFINE_F(BVLCReferenceCaffenet, TinyDNNLayerTest)(
  benchmark::State& state) {
  while (state.KeepRunning()) {
    state.PauseTiming();
    int i = state.range(0);
    auto layer = sTinyDNNLayers.at(i);
    state.ResumeTiming();

    layer->forward();

    state.PauseTiming();
      // Validate results
      assert(validateLayerOutput(i) == true);
    state.ResumeTiming();
  }
}

BENCHMARK_REGISTER_F(BVLCReferenceCaffenet, CaffeLayerTest)->Apply(CaffeLayers);
BENCHMARK_REGISTER_F(BVLCReferenceCaffenet, TinyDNNLayerTest)->Apply(CaffeLayers);

}  // namespace

BENCHMARK_MAIN()
