---
title: Deploy by TensorRT
date: 2020-06-18 20:08:19
tags: [Deep Learning]
categories: TensorRT
---

## TensorRTX

Project Adress: https://github.com/wang-xinyu/tensorrtx


### Retina Face

Project Adress: https://github.com/wang-xinyu/Pytorch_Retinaface

### Jetson AGX Xavier

Deploy: https://developer.nvidia.com/embedded/jetson-agx-xavier-developer-kit

<!-- more -->

### Xavier Jetpack 4.4DP

#### JetPack 4.4 components:

* L4T R32.4.2
* **CUDA 10.2**
* **cuDNN 8.0.0 (Developer Preview)**
* **TensorRT 7.1.0 (Developer Preview)**
* VisionWorks 1.6
* **OpenCV 4.1**
* Vulkan 1.2
* VPI 0.2.0 (Developer Preview)
* Nsight Systems 2020.2
* Nsight Graphics 2020.1
* Nsight Compute 2019.3

#### 通过sdk-manager刷机

```
mkdir -p ~/sdkmanager 
cd ~/sdkmanager
下载 sdkmanager_1.1.0-6343_amd64.deb
cd ~/sdkmanager 
sudo apt install ./sdkmanager_1.1.0-6343_amd64.deb
sdkmanager
```

选择对应xavier型号安装jetpack 4.4环境，选择离线安装(多Retry几次)

手动方式就需要自己动手进入recovery模式：
0. 给xavier插上网线
1. 用原装usb先将host与Xavier连接，还要注意是连接**电源灯**旁边的插口(lsusb可以查看到NVidia Corp)；
2. 确保连接电源并保持Xavier为关闭状态；
3. 按住中间的按键（Force Recovery）不松手；
4. 按住左边的电源（Power）不松手；
5. 过一两秒，同时松手。
6. 安装完第一部分后配置xavier的ubantu系统
7. 完成剩余的安装
8. ifconfig 查看地址


#### 查看Xavier性能
```
tegrastats
```

or jetson monitor script
https://github.com/rbonghi/jetson_stats

PS: 测试TensorRT性能时需要跑多次测试


### Retina Face TensorRTX compile

1. generate retinaface.wts from pytorch

```
// download its weights 'Resnet50_Final.pth', put it in Pytorch_Retinaface/weights
cd Pytorch_Retinaface
python detect.py --save_model
python genwts.py
// a file 'retinaface.wts' will be generated.
```

2. put retinaface.wts into tensorrtx/retinaface
<br>

3. enviroment setting

```
mkdir cmake
cd cmake
touch FindTensorRT.cmake
```

Find TensorRT script

```cmake
# This module defines the following variables:
#
# ::
#
#   TensorRT_INCLUDE_DIRS
#   TensorRT_LIBRARIES
#   TensorRT_FOUND
#
# ::
#
#   TensorRT_VERSION_STRING - version (x.y.z)
#   TensorRT_VERSION_MAJOR  - major version (x)
#   TensorRT_VERSION_MINOR  - minor version (y)
#   TensorRT_VERSION_PATCH  - patch version (z)
#
# Hints
# ^^^^^
# A user may set ``TensorRT_ROOT`` to an installation root to tell this module where to look.
#
set(_TensorRT_SEARCHES)

if(TensorRT_ROOT)
  set(_TensorRT_SEARCH_ROOT PATHS ${TensorRT_ROOT} NO_DEFAULT_PATH)
  list(APPEND _TensorRT_SEARCHES _TensorRT_SEARCH_ROOT)
endif()

# appends some common paths
set(_TensorRT_SEARCH_NORMAL
    PATHS "/usr/src/tensorrt/" # or custom tensorrt path
)
list(APPEND _TensorRT_SEARCHES _TensorRT_SEARCH_NORMAL)

# Include dir
foreach(search ${_TensorRT_SEARCHES})
  find_path(TensorRT_INCLUDE_DIR NAMES NvInfer.h ${${search}} PATH_SUFFIXES include)
endforeach()

if(NOT TensorRT_LIBRARY)
  foreach(search ${_TensorRT_SEARCHES})
    find_library(TensorRT_LIBRARY NAMES nvinfer ${${search}} PATH_SUFFIXES lib)
  endforeach()
endif()

mark_as_advanced(TensorRT_INCLUDE_DIR)

if(TensorRT_INCLUDE_DIR AND EXISTS "${TensorRT_INCLUDE_DIR}/NvInfer.h")
    file(STRINGS "${TensorRT_INCLUDE_DIR}/NvInfer.h" TensorRT_MAJOR REGEX "^#define NV_TENSORRT_MAJOR [0-9]+.*$")
    file(STRINGS "${TensorRT_INCLUDE_DIR}/NvInfer.h" TensorRT_MINOR REGEX "^#define NV_TENSORRT_MINOR [0-9]+.*$")
    file(STRINGS "${TensorRT_INCLUDE_DIR}/NvInfer.h" TensorRT_PATCH REGEX "^#define NV_TENSORRT_PATCH [0-9]+.*$")

    string(REGEX REPLACE "^#define NV_TENSORRT_MAJOR ([0-9]+).*$" "\\1" TensorRT_VERSION_MAJOR "${TensorRT_MAJOR}")
    string(REGEX REPLACE "^#define NV_TENSORRT_MINOR ([0-9]+).*$" "\\1" TensorRT_VERSION_MINOR "${TensorRT_MINOR}")
    string(REGEX REPLACE "^#define NV_TENSORRT_PATCH ([0-9]+).*$" "\\1" TensorRT_VERSION_PATCH "${TensorRT_PATCH}")
    set(TensorRT_VERSION_STRING "${TensorRT_VERSION_MAJOR}.${TensorRT_VERSION_MINOR}.${TensorRT_VERSION_PATCH}")
endif()

include(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(TensorRT REQUIRED_VARS TensorRT_LIBRARY TensorRT_INCLUDE_DIR VERSION_VAR TensorRT_VERSION_STRING)

if(TensorRT_FOUND)
  set(TensorRT_INCLUDE_DIRS ${TensorRT_INCLUDE_DIR})

  if(NOT TensorRT_LIBRARIES)
    set(TensorRT_LIBRARIES ${TensorRT_LIBRARY})
  endif()

  if(NOT TARGET TensorRT::TensorRT)
    add_library(TensorRT::TensorRT UNKNOWN IMPORTED)
    set_target_properties(TensorRT::TensorRT PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${TensorRT_INCLUDE_DIRS}")
    set_property(TARGET TensorRT::TensorRT APPEND PROPERTY IMPORTED_LOCATION "${TensorRT_LIBRARY}")
  endif()
endif()
```

Modify CMakeLists.txt

```cmake
cmake_minimum_required(VERSION 2.6)

project(retinaface)

add_definitions(-std=c++11)

# CMake path
# For using Find TensorRT script
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
list(APPEND CMAKE_PREFIX_PATH "${PROJECT_SOURCE_DIR}/cmake")

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11) 
set(CMAKE_BUILD_TYPE Debug)

find_package(CUDA REQUIRED)
find_package(TensorRT REQUIRED)

set(CUDA_NVCC_PLAGS ${CUDA_NVCC_PLAGS};-std=c++11;-g;-G;-gencode;arch=compute_30;code=sm_30)

if (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    message("embed_platform on")
    include_directories(/usr/local/cuda/targets/aarch64-linux/include)
    link_directories(/usr/local/cuda/targets/aarch64-linux/lib)
else()
    message("embed_platform off")
    include_directories(/usr/local/cuda/include)
    link_directories(/usr/local/cuda/lib64)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")

cuda_add_library(decodeplugin SHARED ${PROJECT_SOURCE_DIR}/decode.cu)
target_include_directories(decodeplugin PUBLIC ${TensorRT_INCLUDE_DIRS})
target_link_libraries(decodeplugin ${TensorRT_LIBRARIES})
target_link_libraries(decodeplugin ${CUDA_LIBRARIES})

find_package(OpenCV)
include_directories(OpenCV_INCLUDE_DIRS)

add_executable(retina_50 ${PROJECT_SOURCE_DIR}/retina_r50.cpp)
target_link_libraries(retina_50 nvinfer)
target_link_libraries(retina_50 cudart)
target_link_libraries(retina_50 decodeplugin)
target_link_libraries(retina_50 ${OpenCV_LIBRARIES})
target_link_libraries(retina_50 ${TensorRT_LIBRARIES})
target_link_libraries(retina_50 ${CUDA_LIBRARIES})

add_definitions(-O2 -pthread)
```

4. Performance
Xavier:
Input IMG Size(768, 1344)
Speed: 102ms/per img