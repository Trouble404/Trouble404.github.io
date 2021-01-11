---
title: Ubuntu
date: 2018-11-12 09:04:14
tags: [Linux]
categories: System
---

## Ubuntu 配置

### NVIDIA驱动
1. 驱动删除
```
sudo apt --purge autoremove nvidia*
or
sudo /usr/bin/nvidia-uninstall
```

2. 驱动安装
```
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt update
sudo apt upgrade
ubuntu-drivers list
sudo apt install nvidia-driver-VERSION_NUMBER_HERE
```

Reboot your computer so that the new driver is loaded.

<!-- more -->
### CUDA+Cudnn
1. 下载对应驱动版本的cuda以及cudnn
```shell
chmod 755 cuda_%version%_linux.run
sudo sh cuda_%version%_linux.run
```

2. 安装cuda后配置环境变量
```shell
export CUDA_HOME=/usr/local/cuda 
export PATH=$PATH:$CUDA_HOME/bin 
export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```

3. 查看cuda版本
```shell
nvcc --version
```

4. 把cudnn对应文件移入 /usr/local/cuda/ 中
- **cudnn7.6.3**
    ```shell
    sudo cp cuda/include/cudnn.h /usr/local/cuda/include/
    sudo cp cuda/lib64/libcudnn.so.7.6.5 /usr/local/cuda/lib64/
    sudo cp cuda/lib64/libcudnn_static.a /usr/local/cuda/lib64/
    sudo chmod a+r /usr/local/cuda/include/cudnn.h
    sudo chmod a+r /usr/local/cuda/lib64/libcudnn*
    sudo ln -s /usr/local/cuda/lib64/libcudnn.so.7.6.5 /usr/local/cuda/lib64/libcudnn.so.7
    sudo ln -s /usr/local/cuda/lib64/libcudnn.so.7 /usr/local/cuda/lib64/libcudnn.so
    ```

- **cudnn8.0.5**
    ```shell
    sudo cp cuda/include/cudnn* /usr/local/cuda/include/
    sudo cp cuda/lib64/libcudnn.so.8.0.5 /usr/local/cuda/lib64/
    sudo cp cuda/lib64/libcudnn_adv_infer.so.8.0.5 /usr/local/cuda/lib64/
    sudo cp cuda/lib64/libcudnn_adv_train.so.8.0.5 /usr/local/cuda/lib64/
    sudo cp cuda/lib64/libcudnn_cnn_infer.so.8.0.5 /usr/local/cuda/lib64/
    sudo cp cuda/lib64/libcudnn_cnn_train.so.8.0.5 /usr/local/cuda/lib64/
    sudo cp cuda/lib64/libcudnn_ops_infer.so.8.0.5 /usr/local/cuda/lib64/
    sudo cp cuda/lib64/libcudnn_ops_train.so.8.0.5 /usr/local/cuda/lib64/
    sudo cp cuda/lib64/libcudnn_static.a /usr/local/cuda/lib64/
    sudo chmod a+r /usr/local/cuda/include/cudnn*
    sudo chmod a+r /usr/local/cuda/lib64/libcudnn*
    sudo ln -s /usr/local/cuda/lib64/libcudnn.so.8.0.5 /usr/local/cuda/lib64/libcudnn.so.8
    sudo ln -s /usr/local/cuda/lib64/libcudnn.so.8 /usr/local/cuda/lib64/libcudnn.so
    sudo ln -s /usr/local/cuda/lib64/libcudnn_adv_infer.so.8.0.5 /usr/local/cuda/lib64/libcudnn_adv_infer.so.8
    sudo ln -s /usr/local/cuda/lib64/libcudnn_adv_infer.so.8 /usr/local/cuda/lib64/libcudnn_adv_infer.so
    sudo ln -s /usr/local/cuda/lib64/libcudnn_adv_train.so.8.0.5 /usr/local/cuda/lib64/libcudnn_adv_train.so.8
    sudo ln -s /usr/local/cuda/lib64/libcudnn_adv_train.so.8 /usr/local/cuda/lib64/libcudnn_adv_train.so
    sudo ln -s /usr/local/cuda/lib64/libcudnn_cnn_infer.so.8.0.5 /usr/local/cuda/lib64/libcudnn_cnn_infer.so.8
    sudo ln -s /usr/local/cuda/lib64/libcudnn_cnn_infer.so.8 /usr/local/cuda/lib64/libcudnn_cnn_infer.so
    sudo ln -s /usr/local/cuda/lib64/libcudnn_cnn_train.so.8.0.5 /usr/local/cuda/lib64/libcudnn_cnn_train.so.8
    sudo ln -s /usr/local/cuda/lib64/libcudnn_cnn_train.so.8 /usr/local/cuda/lib64/libcudnn_cnn_train.so
    sudo ln -s /usr/local/cuda/lib64/libcudnn_ops_infer.so.8.0.5 /usr/local/cuda/lib64/libcudnn_ops_infer.so.8
    sudo ln -s /usr/local/cuda/lib64/libcudnn_ops_infer.so.8 /usr/local/cuda/lib64/libcudnn_ops_infer.so
    sudo ln -s /usr/local/cuda/lib64/libcudnn_ops_train.so.8.0.5 /usr/local/cuda/lib64/libcudnn_ops_train.so.8
    sudo ln -s /usr/local/cuda/lib64/libcudnn_ops_train.so.8 /usr/local/cuda/lib64/libcudnn_ops_train.so
    ```

5. 查看cudnn安装
```shell
cat /usr/local/cuda/include/cudnn_version.h | grep CUDNN_MAJOR -A 2
```

### Vim
1. 升级vim到8.0以上

2. 下载Vundl管理插件
```
git clone https://github.com/VundleVim/Vundle.vim.git ~/.vim/bundle/Vundle.vim
```

3. 配置 .vimrc
```bash
vim  ~/.vimrc

set nocompatible              " be iMproved, required
filetype off                  " required
set rtp+=~/.vim/bundle/Vundle.vim
call vundle#begin()
Plugin 'VundleVim/Vundle.vim'
Plugin 'tpope/vim-fugitive'
Plugin 'git://git.wincent.com/command-t.git'
Plugin 'rstacruz/sparkup', {'rtp': 'vim/'}
Plugin 'davidhalter/jedi-vim'
call vundle#end()            " required
filetype plugin indent on    " required
let g:jedi#completions_enabled = 1
set backspace=indent,eol,start
" 设置utf8编码
set fileencodings=utf-8,gbk,cp936
set fileencoding=utf-8
set encoding=utf-8
set termencoding=utf-8
" 设置标签栏
set showtabline=2

" 去除启动界面
set shortmess=atI

" 设置帮助信息为中文
set helplang=cn

" 设置自动对齐
set autoindent
set smartindent

" 设置c系缩进方式
set cindent
set tabstop=4

" 空格替换Tab
set expandtab
set shiftwidth=4
set softtabstop=4
set smarttab

" 增强命令补全
set wildmenu
" 设置语法高亮
syntax enable
"
" 显示行数
" set nu
```

3. 安装插件
```
vim ~/.vimrc
:PluginInstall ##插入模式下输入
```


### Tmux
1. 安装
```
sudo apt-get install tmux
```

2. tmux 内无法连接X
```
set-option -g update-environment "SSH_ASKPASS SSH_AUTH_SOCK SSH_AGENT_PID SSH_CONNECTION WINDOWID XAUTHORITY"
```
上面命令放到 tmux.conf 中
```shell
tmux source-file ~/.tmux.conf
```


### Anaconda3
[官网下载安装包](https://www.anaconda.com/download/#linux)  
**For Linux Installer**

打开命令行
1. /path/filename 替换为安装包路径
```
sha256sum /path/filename
```

2. 安装
```
bash ~/path/filename
```

3. 安装过程中出现说明以及选择的地方选择YES

4. 修改环境变量


```
vim ~/.bashrc
```
按"i"进入编辑模式，在最后一行添加
```
export PATH=~/anaconda3/bin:$PATH
```
然后重启环境变量
```
source ~/.bashrc
```

5. 配置完成，命令行输入
```
anaconda-navigator
```
6. 启动

### Anaconda环境管理
**断开VPN!!!**

1. 创建新环境(自定义python版本)
```
conda create -n pytorch python=3.7
```

2. 启动环境
```
source activate pytorch
```

3. 关联环境到Jupyter-Notebook 
```
conda install ipykernel
```

#### 切换国内源
1. 升级pip>10.0
```
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple pip -U
```
2. 设置清华源作为镜像
```
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

3. Anaconda 镜像
```
conda config --add channels 'https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/'
conda config --set show_channel_urls yes
```

### Pytorch
1. 各个pytorch以及torchvision版本地址 [here](https://download.pytorch.org/whl/torch_stable.html)


### TensorRT
1. 下载对应版本tensorrt(cuda, cudnn, linux)
2. 进入conda虚拟环境
3. 解压
```
tar xzvf TensorRT-${version}.${os}.${arch}-gnu.${cuda}.${cudnn}.tar.gz
```
4. 添加环境变量
```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:<TensorRT-${version}/lib>
export LIBRARY_PATH=$LIBRARY_PATH:<TensorRT-${version}/lib>
```
5. 安装
```
cd TensorRT-${version}/python
sudo pip install tensorrt-*-cp3x-none-linux_x86_64.whl
cd TensorRT-${version}/graphsurgeon
sudo pip install graphsurgeon-0.4.4-py2.py3-none-any.whl
cd TensorRT-${version}/onnx_graphsurgeon
sudo pip install onnx_graphsurgeon-0.2.6-py2.py3-none-any.whl
```
6. 重启虚拟环境
```
source ~/.bashrc
```

7. tensorrt bug记录
---
```
Assertion failed: !_importer_ctx.network()->hasImplicitBatchDimension() && "This version of the ONNX parser only supports TensorRT INetworkDefinitions with an explicit batch dimension. Please ensure t
he network was created using the EXPLICIT_BATCH NetworkDefinitionCreationFlag."
```
build trt enginn时候设定 EXPLICIT_BATCH
```python
EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
builder = trt.Builder()
network = builder.create_network(EXPLICIT_BATCH)
```
---

### Docker
使用脚本自动安装
```shell
curl -fsSL get.docker.com -o get-docker.sh
sudo sh get-docker.sh --mirror Aliyun
sudo systemctl enable docker
sudo systemctl start docker
sudo groupadd docker
sudo usermod -aG docker $USER
sudo gpasswd -a $USER docker
newgrp docker
```

退出当前终端并重新登录，进行如下测试
```shell
docker run hello-world
```

### Cmake
官网下载Cmake压缩包，解压后
```shell
sudo apt-get install libssl-dev
./bootstrap --prefix=/usr
make
sudo make install
cmake --version
```

### OpenCV
1. 安装
```
git clone https://github.com/opencv/opencv.git
cd opencv
mkdir release
cd release
cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local ..
sudo make
sudo make install
sudo /bin/bash -c 'echo "/usr/local/lib" > /etc/ld.so.conf.d/opencv.conf'  
sudo ldconfig
```
