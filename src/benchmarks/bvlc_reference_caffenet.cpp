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

  std::shared_ptr<layer> getTinyDNNLayer(int i) {
    // Retrieve caffe layer and top and bottom blobs
    auto caffe_layer = sCaffeNet->layers().at(i);
    auto bottom_blob = sCaffeNet->bottom_vecs().at(i).at(0);
    auto top_blob = sCaffeNet->top_vecs().at(i).at(0);

    // Determine layer input and output dimensions
    cnn_size_t in_num = bottom_blob->shape(0);  
    cnn_size_t in_channels = bottom_blob->shape(1);  
    cnn_size_t in_height = bottom_blob->shape(2);  
    cnn_size_t in_width = bottom_blob->shape(3);  
    cnn_size_t out_num = top_blob->shape(0);  
    cnn_size_t out_channels = top_blob->shape(1);  
    cnn_size_t out_height = top_blob->shape(2);  
    cnn_size_t out_width = top_blob->shape(3);  

    //std::cerr << i << "\tInput: " << bottom_blob->shape_string() << std::endl;
    //std::cerr << "\tOutput: " << top_blob->shape_string() << std::endl;

    // Convert Caffe layer to tiny-dnn layer
    std::shared_ptr<layer> tiny_dnn_layer;
    std::string layer_type = caffe_layer->type();
    if(layer_type == "Convolution") {
      // Retrieve Caffe's layer parameters 
      auto c = caffe_layer->layer_param().convolution_param();

      // Determine kernel width and height
      cnn_size_t window_width = c.has_kernel_w() ? c.kernel_w() : c.kernel_size(0); 
      cnn_size_t window_height = c.has_kernel_h() ? c.kernel_h() : c.kernel_size(0); 

      // Determine stride
      cnn_size_t w_stride = c.has_stride_w() ? c.stride_w() : 1;
      cnn_size_t h_stride = c.has_stride_h() ? c.stride_h() : 1;
      assert(c.stride_size() < 3); // only 2D data
      if(c.stride_size() > 0) { 
        if(c.stride_size() == 1) { // c.stride_size() == 1
          w_stride = h_stride = c.stride(0);
        } else {                   // c.stride_size() == 2
          h_stride = c.stride(0);
          w_stride = c.stride(1);
        }
      }

      // Determine padding
      cnn_size_t w_pad = c.has_pad_w() ? c.pad_w() : 0;
      cnn_size_t h_pad = c.has_pad_h() ? c.pad_h() : 0;
      assert(c.pad_size() < 3); // only 2D data
      if(c.pad_size() > 0) { 
        if(c.pad_size() == 1) { // c.pad_size() == 1
          w_pad = h_pad = c.pad(0);
        } else { 		// c.pad_size() == 2
          h_pad = c.pad(0);
          w_pad = c.pad(1);
        }
      }
      
      // Check if padding supported by tiny-dnn
      assert(w_pad == h_pad);
      assert(w_pad == 0 || w_pad == (window_width - 1) / 2);

      // Convert padding to tiny-dnn::padding
      padding pad_type;
      if(w_pad == 0) {
        pad_type = padding::valid;
      } else { // w_pad = (windows_width - 1) / 2)
        pad_type = padding::same;
      }

      // Determine whether layer has bias
      bool has_bias = c.bias_term();

      // Validate number of outputs versus blob dimension 
      assert(out_channels == c.num_output());

      // Allocate tiny-dnn layer
      tiny_dnn_layer =
        std::make_shared<convolutional_layer<identity> >(in_width,
                                                         in_height,
                                                         window_width,
                                                         window_height,
                                                         in_channels,
                                                         out_channels,
                                                         pad_type,
                                                         has_bias,
                                                         w_stride,
                                                         h_stride);
     
      // Determine connection table
      connection_table table;
       if(c.has_group()) {
        table = connection_table(c.group(), in_channels, out_channels);
      }

      vec_t & b = *tiny_dnn_layer->weights()[1];

      // Fill weights
      int dst_idx = 0;
      int src_idx = 0;
      vec_t & w = *tiny_dnn_layer->weights()[0];
      auto weights = caffe_layer->blobs().at(0);
      for(int o = 0; o < out_channels; ++o) {
        for(int i = 0; i < in_channels; ++i) {
          if(!table.is_connected(o, i)) {
            dst_idx += window_width * window_height;
            continue;
          }
          for(int x = 0; x < window_width * window_height; ++x) {
            w[dst_idx++] =  weights->cpu_data()[src_idx++];
          }
        }
      }

      // Fill bias
      if(has_bias) {
        auto bias = caffe_layer->blobs().at(1);
        for(int o = 0; o < out_channels; o++) {
          b[o] = bias->cpu_data()[0];
        }
      }
    }

    // Load input into tiny-dnn layer 
    tiny_dnn_layer->set_in_data(
      std::vector<tensor_t>{
        std::vector<vec_t>{ vec_t(bottom_blob->cpu_data(),
                            bottom_blob->cpu_data() + bottom_blob->count()) } } ); 

    return tiny_dnn_layer;
  }

  bool validateLayerOutput(int i, std::vector<tensor_t> & out_data) {
    // TODO(Abai) : Examine differences between caffe and tiny-dnn output
    //auto top_caffe = sCaffeNet->top_vecs().at(i).at(0);
    //auto top_tiny_dnn = out_data.at(0).at(0);
    //float threshold = 0.01f;
    //for(int i = 0; i < top_caffe->count(); ++i) {
    //  float diff = std::abs(top_tiny_dnn.at(i) - top_caffe->cpu_data()[i]);
    //  if(diff > threshold) {
    //    std::cerr << "Warning: Difference between output blobs caffe="
    //              << top_caffe->cpu_data()[i] << " and tiny-dnn="
    //              << top_tiny_dnn.at(i) << " is larger than "
    //              << threshold <<  std::endl;
    //    //return false;
    //  }
    //}
    return true;
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
    int i = state.range(0);
    auto layer = sCaffeNet->layers().at(i);
    auto bottom_vec = sCaffeNet->bottom_vecs().at(i);
    auto top_vec = sCaffeNet->top_vecs().at(i);
    assert( bottom_vec.size() == 1u );
    state.ResumeTiming();

    std::string bla = "hi";
    //layer->Forward(bottom_vec, top_vec);
  }
}

BENCHMARK_DEFINE_F(BVLCReferenceCaffenet, TinyDNNLayerTest)(
  benchmark::State& state) {
  while (state.KeepRunning()) {
    state.PauseTiming();
    int i = state.range(0);
    // Convert caffe layer and blobs to tiny-dnn format
    auto layer = getTinyDNNLayer(i); 
    state.ResumeTiming();

    layer->forward();

    state.PauseTiming();
      // Validate results
      auto out_data = layer->output();
      assert(validateLayerOutput(i, out_data) == true);
    state.ResumeTiming();
  }
}

//BENCHMARK_REGISTER_F(BVLCReferenceCaffenet, CaffeLayerTest)->Apply(CaffeLayers);
BENCHMARK_REGISTER_F(BVLCReferenceCaffenet, TinyDNNLayerTest)->Apply(CaffeLayers);

}  // namespace

BENCHMARK_MAIN()
