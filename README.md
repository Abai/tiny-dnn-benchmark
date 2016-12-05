# tiny-dnn benchmark

A tiny-dnn forward-pass benchmark implemented using **google-benchmark** library. Generates CPU time and real time spent for forward-pass of each layer in Caffe and tiny-dnn. Currently uses *bvlc\_reference_caffenet.caffemodel* for these purposes.

Works on **OSX** and **Ubuntu** out of the box. Uses CMake so you should be able to get it to work on Windows too.

Here is an example output of tiny-dnn tiny-dnn/tiny-dnn@bc834c9 and Caffe BVLC/caffe@24d2f67 on [1st of December 2016](https://gist.github.com/Abai/273fe51faadb77807b79879507fe945a)

## Installing dependencies on Ubuntu

Tested on Ubuntu 16.04

It seems that google-benchmark currently has issues compiling on Ubuntu 16.04 with GCC 5.4.0. Therefore I would recommend to us clang-3.8:

```
sudo apt-get install build-essential clang
```

Now install the rest of dependencies

```
sudo apt-get install cmake libopencv-dev libhdf5-serial-dev libgflags-dev libgoogle-glog-dev libprotobuf-dev protobuf-compiler libatlas-base-dev libtbb-dev python-yaml
sudo apt-get install --no-install-recommends libboost-all-dev
```

## Compilation on Ubuntu

Tested on Ubuntu 16.04

Clone the repository including submodules:

```
git clone --recursive https://github.com/abai/tiny-dnn-benchmark.git
```

Firstly we need to set the compiler to clang:

```
export CC=/usr/bin/clang
export CXX=/usr/bin/clang++
```

CMake configuration triggers Caffe scripts to download model data and mean required to run the benchmark.

We configure the project in `RelWithDebInfo` so that C asserts function in case an errror occurs. Once you have executed the benchmark without errors, feel free to configure in `Release` mode.

Configure the project with CMake and compile. 

```
cd tiny-dnn-benchmark
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo ../src/
make
./benchmark/bvlc_reference_caffenet
```
## Installing dependencies on OSX

Tested on OSX 10.12.1

Install latest Xcode from the appstore and run it once to agree to the license. Then install [Homebrew](http://brew.sh/).

Now install the following homebrew packages:

```
brew tap homebrew/science
brew install cmake opencv hdf5 gflags glog protobuf tbb boost
```

Caffe scripts require python-yaml to download the models:

```
sudo easy-install pip
sudo pip install pyyaml
```

## Compilation on OSX

Tested on OSX 10.12.1

Clone the repository including submodules:

```
git clone --recursive https://github.com/abai/tiny-dnn-benchmark.git
```

CMake configuration triggers Caffe scripts to download model data and mean required to run the benchmark.

We configure the project in `RelWithDebInfo` so that C asserts function in case an errror occurs. Once you have executed the benchmark without errors, feel free to configure in `Release` mode.

Configure the project with CMake and compile. 

```
cd tiny-dnn-benchmark
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo ../src/
make
./benchmark/bvlc_reference_caffenet
```

## Updating the Caffe and tiny-dnn source

To update the submodules simply do this:

```
cd tiny-dnn-benchmark/src/caffe
git pull --recurse-submodules
cd ../tiny-dnn
git pull --recurse-submodules
```