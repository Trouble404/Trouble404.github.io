---
title: NVIDIA边缘端设备刷机
date: 2021-03-03 17:00:00
tags: [Deep Learning]
categories: TensorRT
---

## NVIDIA Jetson Products

![image](https://cdn.jsdelivr.net/gh/Trouble404/Image/blog20210303161345.png)

[Jetson Products](https://developer.nvidia.com/buy-jetson)


<!-- more -->

### 刷机准备
* Host主机(Ubuntu 18.04.5)
* [NVIDIA SDK Manager](https://developer.nvidia.com/nvidia-sdk-manager)
* Jetson设备

### [通过sdk-manager刷机](https://docs.nvidia.com/sdk-manager/install-with-sdkm-jetson/index.html#repair-uninstall)
- 安装SDK Manager并且升级到最新版本
- 使用NVIDIA开发者账户登录
![image](https://cdn.jsdelivr.net/gh/Trouble404/Image/blog20210303162217.png)
- 选择Jetson设备以及安装配置(Host Machine可不选)
![image](https://cdn.jsdelivr.net/gh/Trouble404/Image/blog20210303162357.png)
- 选择离线安装(多Retry几次)
![image](https://cdn.jsdelivr.net/gh/Trouble404/Image/blog20210303162540.png)
- 用**原装usb**先将host与Jetson设备连接
  - Xavier连接
    - xavier注意是连接**电源灯**旁边的插口;
    - 确保连接电源并保持Xavier为关闭状态；
    - 按住中间的按键（Force Recovery）不松手；
    - 按住左边的电源（Power）不松手；
    - 过一两秒，同时松手。
  - TX2连接
    - USB 连接
    - 接通电源，按下 power 键，开机
    - 刚开机时，迅速 按住 Recovery 键不放开，然后按一下 Reset 键，过 2s 以后再松开 Recovery 键。此时开发板应该处于强制恢复模式。
  - lsusb可以查看到NVidia Corp;
![](https://raw.githubusercontent.com/Trouble404/Image/master/blog20210303164843.png)
  - 安装完第一部分后Jetson设备会开机，此时进行系统配置
  - 输入系统配置的用户名以及密码
![](https://raw.githubusercontent.com/Trouble404/Image/master/blog20210303164803.png)
  - 完成剩余的安装(出现问题的话按照终端的Log输出在设备系统里进行修复)
  - ifconfig 查看ssh地址

![image](https://cdn.jsdelivr.net/gh/Trouble404/Image/blog20210303164359.png)

### 查看Jetson设备性能
```sh
## system
tegrastats

## thrid party jetson monitor script
sudo -H pip install -U jetson-stats
jtop
```

### 工作模式
TX2:
![image](https://cdn.jsdelivr.net/gh/Trouble404/Image/blog20210303165137.png)
支持FP32, FP16推理
```sh
# 查看当前工作模式
sudo nvpmodel -q verbose
# 如果想切换到模式0
sudo nvpmodel -m 0
```

Xavier:
![](https://raw.githubusercontent.com/Trouble404/Image/master/blog20210303165241.png)
支持FP32, FP16， INT8推理
```sh
# 查看当前工作模式
sudo nvpmodel --query
# 如果想切换到模式0
sudo nvpmodel -m 0
```

### 环境配置(个人习惯)

```sh
# 生成ssh key
ssh-keygen -t rsa -C ”yourEmail@example.com”
cat ~/.ssh/id_rsa.pub

# 删除python2
sudo apt-get remove python2.7
sudo apt-get remove --auto-remove python2.7

# 安装pip
sudo apt-get install python3-pip

# 默认python为python3
sudo ln -s /usr/bin/python3.6 /usr/bin/python

# 默认pip为pip3
sudo ln -s /usr/bin/pip3 /usr/bin/pip

# 升级pip并切换源
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple pip -U
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# Pytorch aarch64
# https://forums.developer.nvidia.com/t/pytorch-for-jetson-version-1-7-0-now-available/72048
# 下载对应pytorch进行安装
sudo apt-get install python3-pip libopenblas-base libopenmpi-dev 
pip install Cython
pip install numpy torch-1.x.x-cp36-cp36m-linux_aarch64.whl

# 编译对应Pytorch版本的torchvision
PyTorch v1.0 - torchvision v0.2.2
PyTorch v1.1 - torchvision v0.3.0
PyTorch v1.2 - torchvision v0.4.0
PyTorch v1.3 - torchvision v0.4.2
PyTorch v1.4 - torchvision v0.5.0
PyTorch v1.5 - torchvision v0.6.0
PyTorch v1.6 - torchvision v0.7.0
PyTorch v1.7 - torchvision v0.8.1
sudo apt-get install libjpeg-dev zlib1g-dev libpython3-dev libavcodec-dev libavformat-dev libswscale-dev
# 编译其余的库
# 1. git clone 对应库
# 2. python setup.py bdist_wheel
# 3. 在dist中找到编译好的whl文件进行安装

# 升级cmake
sudo apt-get install software-properties-common
sudo add-apt-repository ppa:george-edison55/cmake-3.x
sudo apt-get update
apt-get install libprotobuf-dev protobuf-compiler

# ONNX-TENSORRT
git clone https://github.com.cnpmjs.org/onnx/onnx-tensorrt.git
cd onnx-tensorrt
git checkout 7.1
rm -r third_party/onnx
cd third_party
git clone https://github.com.cnpmjs.org/onnx/onnx.git
cd onnx
git checout 553df22
cd ../../
mkdir build
cd build
cmake .. -DTENSORRT_ROOT=/usr/src/tensorrt
export LD_LIBRARY_PATH=/home/nvidia/workspace/thrid_party/whl/pytorch/onnx-tensorrt/build:$LD_LIBRARY_PATH
sudo sh -c "echo '/usr/local/cuda/lib64' > /etc/ld.so.conf.d/cuda.conf"
sudo ldconfig
sudo make
sudo make install

# fix Illegal instruction (core dumped) -> numpy 1.19.5 issue
pip install numpy==1.19.4
```
