---
title: 【转】论文阅读 - Semantic Soft Segmentation
date: 2018-11-29 11:00:00
tags: [Deep Learning]
categories: 实习
---

# 论文阅读 - Semantic Soft Segmentation

<center>![image](https://raw.githubusercontent.com/Trouble404/Blog_Pics/master/Semantic-soft-Segmentation/1.png)</center>

---

题目：[Semantic Soft Segmentation - SIGGRAPH2018](http://people.inf.ethz.ch/aksoyy/papers/TOG18-sss.pdf)

作者：[Yagiz Aksoy](http://people.inf.ethz.ch/aksoyy/), [Tae-Hyun Oh](http://taehyunoh.com/), [Sylvain Paris](http://people.csail.mit.edu/sparis/), [Marc Pollefeys](https://www.inf.ethz.ch/personal/marc.pollefeys/) and [Wojciech Matusik](http://people.csail.mit.edu/wojciech/)

团队：MIT CSAIL, Adobe Research<!-- more -->

---

[[Paper - Semantic Soft Segmentation - SIGGRAPH2018]](http://people.inf.ethz.ch/aksoyy/papers/TOG18-sss.pdf)

[[Supplementary Material - Semantic Soft Segmentation - SIGGRAPH2018]](http://people.inf.ethz.ch/aksoyy/papers/TOG18-sss-supp.pdf)

[[HomePage]](http://people.inf.ethz.ch/aksoyy/sss/)

[[Github - SIGGRAPH18SSS - Semantic feature generator- 特征提取源码]](https://github.com/iyah4888/SIGGRAPH18SSS)

[[Github - Spectral segmentation implementation - 分割源码]](https://github.com/yaksoy/SemanticSoftSegmentation)

[[YouTube - Video]](https://youtu.be/QYIQbfnS9jA)

语义软分割(Semantic Soft Segments)，旨在精确表示图像不同区域间的软过渡. 类似与磁力套索(magnetic lasso) 和魔术棒(magic wand) 的功能.

从谱分割(spectral segmentation) 角度来解决 soft segmentation 问题，提出的图结构(Graph Structure)，既考虑了图片的纹理和颜色特征，也利用了由深度神经网络生成的更高层的语义信息. 根据仔细构建的 Laplacian 矩阵的特征分解(eigendecomposition) 自动的生成 soft segments.

出发点：
1. 能够分割图片中的不同物体，同时精确表示出分割物体间的过渡情况.
2. 自动完成分割，不用手工操作.

Semantic Soft Segmentation，自动将图像分解为不同的层，以覆盖场景的物体对象，并通过软过渡(soft transitions) 来分离不同的物体对象.

相关研究方向：

*   Soft segmentation - 将图像分解为两个或多个分割，每个像素可能属于不止一个分割部分.
*   Natural image matting - 估计用于定义的前景区域中每个像素的不透明度. 一般输入是 trimap，其分别定义了不透明的前景，透明的背景以及未知透明度的区域.
*   Targeted edit propagation
*   Semantic segmentation - 语义分割

## 技术路线

**问题描述**：
给定输入图片，自动生成其 soft 分割结果，即，分解为表示了场景内物体的不同层，包括物体的透明度和物体间的软过渡.
每一层的各个像素由一个透明度值alpha表示. alpha=0 表示完全不透明(fully opaque)，alpha=1 表示完全透明(fully transparent)，alpha 值在 0-1 之间，则表示部分不透明度.

$$(R,G,B)_{input} = \sum_{i} \alpha_{i}(R,G,B)_{i}$$
$$\sum_{i}\alpha_{i}=1$$

输入图片的 RGB 像素可以表示为每一层中的像素值与对应的 alpha 值的加权和.

<center>![image](https://raw.githubusercontent.com/Trouble404/Blog_Pics/master/Semantic-soft-Segmentation/2.png)</center>

### 1\. 低层特征构建 - Nonlocal Color Affinity

构建低层次的仿射关系项，以表示基于颜色的像素间较大范围的关联性特征. Nonloal Color Affinity可以提升分解恢复过程中isolated的区域的效果。
<center>![image](https://raw.githubusercontent.com/Trouble404/Blog_Pics/master/Semantic-soft-Segmentation/3.png)</center>

主要构建过程：
1. 采用 SLIC(超像素分割) 生成 2500 个超像素;
2. 估计每个超像素和对应于图像 20% 尺寸半径内所有超像素的仿射关系.

### 2\. 高层特征构建 - High-Level Semantic Affinity

虽然 nonlocal color affinity 添加了像素间大范围间的相互作用关系，但仍是低层特征.
这里构建高层语义仿射关系项，以使得属于同一场景物体的像素尽可能的接近，不同场景物体的像素间的关系远离.

<center>![image](https://raw.githubusercontent.com/Trouble404/Blog_Pics/master/Semantic-soft-Segmentation/4.png)</center>

### 3\. 图像层创建 - Creating the Layers

通过对 Laplacian 矩阵进行特征分解，提取特征向量，并对特征向量进行两步稀疏处理，来创建图像层.

1. 构建 Laplacian 矩阵
3. 受约束的稀疏化(Constrained sparsification)
3. 松弛的稀疏化(Relaxed sparsification)

<center>![image](https://raw.githubusercontent.com/Trouble404/Blog_Pics/master/Semantic-soft-Segmentation/5.png)</center>

### 4\. 语义特征向量 - Semantic Feature Vectors
在高层特征构建时，相同物体的像素的特征向量相似，不同物体的像素的特征向量不同.
特征向量是采用语义分割的深度网络模型训练和生成的.

这里采用了 DeepLab-ResNet-101 作为特征提取器，但网络训练是采用的是度量学习方法，最大化不同物体的特征间的 L2 距离(稍微修改了 N-Pair loss).

<center>![image](https://raw.githubusercontent.com/Trouble404/Blog_Pics/master/Semantic-soft-Segmentation/6.png)</center>

### 5\. 个人看法
使用了底层特征和高层特征(包括deep learning)产生的语义特征构建的拉普拉斯矩阵的特征分解创建了精细的图层来聚类区分最大可能的前景和背景。分割的效果特别不错，不过计算量特别的庞大，3~4分钟处理一张图片。并且比较依赖图像中的颜色信息，对颜色相近的物体的效果不是特别的好。 在影视方面有不错的前景，也可能可以考虑用来帮助标注人员产生不错的分割图并且进行进一步的标注。