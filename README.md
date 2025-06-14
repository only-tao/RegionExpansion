# Region Expansion: Optimization of Patch-fetching  Method for Point Cloud Denoising

### Environment

#### set pdflow env

refer to [pdflow](https://github.com/unknownue/pdflow)

#### set pointfilter env

refer to [pointfilter](https://github.com/dongbo-BUAA-VR/Pointfilter)

#### download libtorch

[libtorch ](https://pytorch.org/get-started/locally/)version: libtorch-cxx11-abi-shared-with-deps-2.1.0+cu118.zip

unzip to ./build  or where you want

> remember to change util-self/build_density/CMakeLists.txt  path variable

then set environment variable.

#### build libdensity.so

```
cd util-self/build
cmake ..
make -j4
```

>  After do this,  we can get libdensity.so file

#### build example...so

```
pip install pybind11
```

Before proceeding with the following steps, you should clarify the installation directories of your 

pybind11, Python, and PyTorch.

```
cd build_density && mkdir build
cd util-self/build_density/build
cmake  -Dpybind11_DIR=$(python -m pybind11 --cmakedir)
                -DPython3_EXECUTABLE=/path_to_you_pdflow_python../bin/python
                -DTorch_DIR=/path_to_you_pdflow_python/site-packages/torch/share/cmake/Torch
                ..
make -j4
```

> After do this,  we can get example.cpython-38-x86_64-linux-gnu.so

### Dataset

All training and evaluation data can be downloaded from repo of [pdflow ](https://github.com/unknownue/pdflow)and [pointfilter](https://github.com/dongbo-BUAA-VR/Pointfilter)

### Test algorithm

After prepared data, you should change test_time.py parameters, then run

```
python test_time.py
```
