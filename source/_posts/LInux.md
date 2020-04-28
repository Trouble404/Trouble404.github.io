---
title: Ubuntu
date: 2018-11-12 09:04:14
tags: [Linux]
categories: System
---

## Ubuntu 配置

### Anaconda3
[官网下载安装包](https://www.anaconda.com/download/#linux)  
**For Linux Installer**<!-- more -->

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

1. 创建新环境```
conda create -n pytorch python=3.5
```

2. 启动环境```
source activate pytorch
```

3. 安装Pytorch以及torchvision [具体版本命令网址](https://pytorch.org/)```
conda install pytorch-cpu torchvision-cpu -c pytorch
```

4. 关联环境到Jupyter-Notebook 
```
conda install nb_conda
```

### 切换国内源
1. 升级pip>10.0
```
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple pip -U
```
2. 设置
```
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

3. Anaconda 镜像
```
conda config --add channels 'https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/'
conda config --set show_channel_urls yes
```









