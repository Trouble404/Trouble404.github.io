---
title: MalongTech 学习内容:目标检测
date: 2018-11-12 11:00:00
tags: [Deep Learning]
categories: 实习
---

# Object detection

![image](https://cdn.jsdelivr.net/gh/Trouble404/Blog_Pics/Object-detection-learning/14.png)

| one-stage系  | two-stage系|
| ---------- | -----------|
| YOLO V1,V2,V3   | FPN   |
| SSD   | RFCN   |
| RetinalNet | LIghthead |
<!-- more -->
## One Stage

### YOLO V1
** You only look once unified real-time object detection**

[PAPER ADDRESS](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Redmon_You_Only_Look_CVPR_2016_paper.pdf)

作者在YOLO算法中把物体检测（object detection）问题处理成回归问题，用一个卷积神经网络结构就可以从输入图像直接预测bounding box和类别概率。

**优点**: 
1. YOLO的速度非常快。在Titan X GPU上的速度是45 fps, 加速版155 fps。
2. YOLO是基于图像的全局信息进行预测的。这一点和基于sliding window以及region proposal等检测算法不一样。与Fast R-CNN相比，YOLO在误检测（将背景检测为物体）方面的错误率能降低一半多。
3. 泛化能力强。

![image](https://cdn.jsdelivr.net/gh/Trouble404/Blog_Pics/Object-detection-learning/1.png)

**缺点**:
1. accuracy 还落后于同期 state-of-the-art 目标检测方法。
2. 难于检测小目标。
3. 定位不够精准。
4. 虽然降低了背景检测为物体的概率但同事导致了召回率较低。

**流程**
1. 调整图像大小至$448\times448$.
2. 运行卷积网络同时预测多目标的边界框和所属类的概率
3. NMX(非极大值抑制)

**Unified Detection**
![image](https://cdn.jsdelivr.net/gh/Trouble404/Blog_Pics/Object-detection-learning/2.png)
1. 将图片分为$S\times S$格子。
2. 对每个格子都预测B个边界框并且每个边界框包含5个预测值：x,t,w,h以及confidence。x,y就是bounding box的中心坐标，与grid cell对齐（即相对于当前grid cell的偏移值），使得范围变成0到1；w和h进行归一化（分别除以图像的w和h，这样最后的w和h就在0到1范围。
3. 每个格子都预测C个假定类别的概率。
4. 在Pascal VOC中， S=7,B=2,C=20. 所以有 $S\times S\times (B \times 5 + C)$ 即 $7\times 7\times 30$ 维张量。

Confidence计算：$Pr(Object) * IOU_{pred}^{turth} $

每个bounding box都对应一个confidence score，如果grid cell里面没有object，confidence就是0，如果有，则confidence score等于预测的box和ground truth的IOU值，见上面公式。并且如果一个object的ground truth的中心点坐标在一个grid cell中，那么这个grid cell就是包含这个object，也就是说这个object的预测就由该grid cell负责。 
每个grid cell都预测C个类别概率，表示一个grid cell在包含object的条件下属于某个类别的概率：$Pr(Class_{i}|Object)$

每个bounding box的confidence和每个类别的score相乘，得到每个bounding box属于哪一类的confidence score。

即得到每个bounding box属于哪一类的confidence score。也就是说最后会得到20\*(7\*7\*2)的score矩阵，括号里面是bounding box的数量，20代表类别。接下来的操作都是20个类别轮流进行：在某个类别中（即矩阵的某一行），将得分少于阈值（0.2）的设置为0，然后再按得分从高到低排序。最后再用NMS算法去掉重复率较大的bounding box（NMS:针对某一类别，选择得分最大的bounding box，然后计算它和其它bounding box的IOU值，如果IOU大于0.5，说明重复率较大，该得分设为0，如果不大于0.5，则不改；这样一轮后，再选择剩下的score里面最大的那个bounding box，然后计算该bounding box和其它bounding box的IOU，重复以上过程直到最后）。最后每个bounding box的20个score取最大的score，如果这个score大于0，那么这个bounding box就是这个socre对应的类别（矩阵的行），如果小于0，说明这个bounding box里面没有物体，跳过即可。


**网络设计**
灵感来源于GoogLeNet,如下图：
![image](https://cdn.jsdelivr.net/gh/Trouble404/Blog_Pics/Object-detection-learning/3.png)
训练过程中：
1. 作者先在ImageNet数据集上预训练网络，而且网络只采用图中的前面20个卷积层，输入是224\*224大小的图像。然后在检测的时候再加上随机初始化的4个卷积层和2个全连接层，同时输入改为更高分辨率的448\*448。
2. Relu层改为leaky Relu，即当x<0时，激活值是0.1\*x，而不是传统的0。
3. 作者采用sum-squared error的方式把localization error（bounding box的坐标误差）和classificaton error整合在一起。但是如果二者的权值一致，容易导致模型不稳定，训练发散。因为很多grid cell是不包含物体的，这样的话很多grid cell的confidence score为0。所以采用设置不同权重方式来解决，一方面提高localization error的权重，另一方面降低没有object的box的confidence loss权值，loss权重分别是5和0.5。而对于包含object的box的confidence loss权值还是原来的1。
4. 用宽和高的开根号代替原来的宽和高，这样做主要是因为相同的宽和高误差对于小的目标精度影响比大的目标要大.
5. Loss Function如下：![image](https://cdn.jsdelivr.net/gh/Trouble404/Blog_Pics/Object-detection-learning/4.png)

训练的时候：输入N个图像，每个图像包含M个objec，每个object包含4个坐标（x，y，w，h）和1个label。然后通过网络得到7\*7\*30大小的三维矩阵。每个1\*30的向量前5个元素表示第一个bounding box的4个坐标和1个confidence，第6到10元素表示第二个bounding box的4个坐标和1个confidence。最后20个表示这个grid cell所属类别。注意这30个都是预测的结果。然后就可以计算损失函数的第一、二 、五行。至于第二三行，confidence可以根据ground truth和预测的bounding box计算出的IOU和是否有object的0,1值相乘得到。真实的confidence是0或1值，即有object则为1，没有object则为0。 这样就能计算出loss function的值了。

测试的时候：输入一张图像，跑到网络的末端得到7\*7\*30的三维矩阵，这里虽然没有计算IOU，但是由训练好的权重已经直接计算出了bounding box的confidence。然后再跟预测的类别概率相乘就得到每个bounding box属于哪一类的概率。

**YOLO效果**
![image](https://cdn.jsdelivr.net/gh/Trouble404/Blog_Pics/Object-detection-learning/5.png)
由于yolo更少的识别背景为物体对比Faster RCNN,因此结合YOLO作为背景检测器与Faster RCNN可以带来更大的提升，不过速度方面就没有优势了。

### YOLO V2 and YOLO 9000
[论文地址](https://arxiv.org/pdf/1612.08242.pdf)

与分类和标记等其他任务的数据集相比，目前目标检测数据集是有限的。最常见的检测数据集包含成千上万到数十万张具有成百上千个标签的图像。分类数据集有数以百万计的图像，数十或数十万个类别。为了扩大当前检测系统的范围。我们的方法使用目标分类的分层视图，允许我们将不同的数据集组合在一起。此外联合训练算法，使我们能够在检测和分类数据上训练目标检测器。我们的方法利用标记的检测图像来学习精确定位物体，同时使用分类图像来增加词表和鲁棒性。

**对比YOLO V1的改进**
1. YOLO有许多缺点。YOLO与Fast R-CNN相比的误差分析表明，YOLO造成了大量的定位误差。此外，与基于区域提出的方法相比，YOLO召回率相对较低。因此，我们主要侧重于提高召回率和改进定位，同时保持分类准确性。
2. 在YOLOv2中，一个更精确的检测器被设计出来，它仍然很快。但是不是通过扩大网络，而是简化网络，然后让其更容易学习。结合了以往的一些新方法，以提高YOLO的性能：
    * Batch Normalization: map提升2%
    * High Resolution Classifier: 先在ImageNet上以448×448的分辨率对分类网络进行10个迭代周期的微调。这给了网络时间来调整其滤波器以便更好地处理更高分辨率的输入。map提升4%,
    * Convolutional With Anchor Boxes: 从YOLO中移除全连接层，并使用锚盒来预测边界框。首先，我们消除了一个池化层，使网络卷积层输出具有更高的分辨率。我们还缩小了网络，操作416×416的输入图像而不是448×448。我们这样做是因为我们要在我们的特征映射中有奇数个位置，所以只有一个中心单元。目标，特别是大目标，往往占据图像的中心，所以在中心有一个单独的位置来预测这些目标，而不是四个都在附近的位置是很好的。YOLO的卷积层将图像下采样32倍，所以通过使用416的输入图像，我们得到了13×13的输出特征映射。map有所下降但是召回率达到了88%.
    * Dimension Clusters: Anchors 尺寸的选择用k-means聚类来挑选合适的锚盒尺寸。如果我们使用具有欧几里得距离的标准k-means，那么较大的边界框比较小的边界框产生更多的误差。然而，我们真正想要的是导致好的IOU分数的先验，这是独立于边界框大小的。因此，对于我们的距离度量，我们使用：$d(box,centroid)=1-IOU(box,centroid)$. 在voc和coco的测试中，更薄更高的边界框会带来更好的结果在k=5的时候，如图所示：![image](https://cdn.jsdelivr.net/gh/Trouble404/Blog_Pics/Object-detection-learning/6.png)
    * Direct location prediction: 传统的RPN中，特别是在早期的迭代过程中。大部分的不稳定来自预测边界框的(x,y)位置，他的位置修正方法是不受限制的，所以任何锚盒都可以在图像任一点结束，而不管在哪个位置预测该边界框。随机初始化模型需要很长时间才能稳定以预测合理的偏移量。所以这一步就优化为直接预测相对于网格单元位置的位置坐标。逻辑激活备用来限制网络的预测落在这个范围内。Sigmoid使输出在0~1之间这样映射到原图中时候不会位于其他的网格（在中心目标处）。![image](https://cdn.jsdelivr.net/gh/Trouble404/Blog_Pics/Object-detection-learning/7.png)网络预测输出特征映射中每个单元的5个边界框。网络预测每个边界框的5个坐标，$t_{x}$,$t_{y}$,$t_{w}$,$t_{h}$,$t_{o}$,如果单元从图像的左上角偏移了$(c_{x},c_{y})$,并且边界框先验的宽度和高度为$p_{w}$,$p_{h}$. 预测就可以对应如图公式计算。$Pr(object)\times IOU(b,object)=\sigma(t_{o}) $ 该方法结合维度聚类，map 提升了5%对比与其他的锚盒方法。
    * Fine-Grained Features(细粒度特征): 对于小目标物体，更细的力度特种可以带来更好的效果，因此直通层通过将相邻特征堆叠到不同的通道而不是空间位置来连接较高分辨率特征和较低分辨率特征，类似于ResNet中的恒等映射。这将26×26×512特征映射变成13×13×2048特征映射，其可以与原始特征连接。我们的检测器运行在这个扩展的特征映射的顶部，以便它可以访问细粒度的特征。这会使性能提高1%。
    * Multi-Scale Training:  由于模型只使用卷积层和池化层，因此它可以实时调整大小。所以每隔10个批次会随机选择一个新的图像尺寸大小（330到608，从32的倍数中选择因为模型缩减了32倍），强迫网络学习在不同维度上预测，并且小尺度的网络运行更快。![image](https://cdn.jsdelivr.net/gh/Trouble404/Blog_Pics/Object-detection-learning/8.png)

3. Darknet-19， 它有19个卷积层和5个最大池化层并且只需要55.8亿次运算处理图像获得了比复杂运算VGG和前一代YOLO更高的top-5精度在ImageNet上，结构如下：
![image](https://cdn.jsdelivr.net/gh/Trouble404/Blog_Pics/Object-detection-learning/9.png)
4. 分类训练：使用Darknet神经网络结构，使用随机梯度下降，初始学习率为0.1，学习率多项式衰减系数为4，权重衰减为0.0005，动量为0.9，在标准ImageNet 1000类分类数据集上训练网络160个迭代周期。在训练过程中，标准的数据增强技巧，包括随机裁剪，旋转，色调，饱和度和曝光偏移被使用来防止over-fitting. 在在对224×224的图像进行初始训练之后，对网络在更大的尺寸448上进行了微调。
5. 检测训练：删除了最后一个卷积层，加上了三个具有1024个滤波器的3×3卷积层，其后是最后的1×1卷积层与我们检测需要的输出数量。对于VOC，我们预测5个边界框，每个边界框有5个坐标和20个类别，所以有125个滤波器。还添加了从最后的3×3×512层到倒数第二层卷积层的直通层，以便模型可以使用细粒度特征。
6. 联合训练分类和检测数据：
    * 网络看到标记为检测的图像时，可以基于完整的YOLOv2损失函数进行反向传播。当它看到一个分类图像时，只能从该架构的分类特定部分反向传播损失。
    * Hierarchical classification(分层分类)：ImageNet标签是从WordNet中提取的，这是一个构建概念及其相互关系的语言数据库，在这里通过构建简单的分层树简化问题。最终的结果是WordTree，一个视觉概念的分层模型。为了使用WordTree进行分类，我们预测每个节点的条件概率，以得到同义词集合中每个同义词下义词的概率。如果想要计算一个特定节点的绝对概率，只需沿着通过树到达根节点的路径，再乘以条件概率。可以使用WordTree以合理的方式将多个数据集组合在一起。只需将数据集中的类别映射到树中的synsets即可。![image](https://cdn.jsdelivr.net/gh/Trouble404/Blog_Pics/Object-detection-learning/10.png)
    * YOLO 9000（anchors尺寸3个限制输出大小）: 使用COCO检测数据集和完整的ImageNet版本中的前9000个类来创建的组合数据集。该数据集的相应WordTree有9418个类别。ImageNet是一个更大的数据集，所以通过对COCO进行过采样来平衡数据集，使得ImageNet仅仅大于4:1的比例。当分析YOLO9000在ImageNet上的表现时，发现它很好地学习了新的动物种类，但是却在像服装和设备这样的学习类别中挣扎。新动物更容易学习，因为目标预测可以从COCO中的动物泛化的很好。相反，COCO没有任何类型的衣服的边界框标签，只针对人，因此效果不好3

### YOLO V3
[论文地址](https://pjreddie.com/media/files/papers/YOLOv3.pdf)
**模型**:
![image](https://cdn.jsdelivr.net/gh/Trouble404/Image/blog20201022194457.jpeg)

**改进**：
1. 将YOLO V3替换了V2中的Softmax loss变成Logistic loss(每个类一个logistic)，而且每个GT只匹配一个先验框.
2. Anchor bbox prior不同：V2用了5个anchor，V3用了9个anchor，提高了IOU.
3. Detection的策略不同：V2只有一个detection，V3设置有3个，分别是一个下采样的，Feature map为13\*13，还有2个上采样的eltwise sum(feature pyramid networks)，Feature map分别为26*\26和52\*52，也就是说，V3的416版本已经用到了52的Feature map，而V2把多尺度考虑到训练的data采样上，最后也只是用到了13的Feature map，这应该是对小目标影响最大的地方。
总结：![image](https://cdn.jsdelivr.net/gh/Trouble404/Blog_Pics/Object-detection-learning/11.png)
4. 网络改进 DarkNet-53: 融合了YOLOv2、Darknet-19以及其他新型残差网络，由连续的3×3和1×1卷积层组合而成，当然，其中也添加了一些shortcut connection，整体体量也更大。因为一共有53个卷积层。![image](https://cdn.jsdelivr.net/gh/Trouble404/Blog_Pics/Object-detection-learning/12.png)
5. LOSS: 除了w, h的损失函数依然采用总方误差之外，其他部分的损失函数用的是二值交叉熵。最后加到一起
```python
xy_loss = object_mask * box_loss_scale * K.binary_crossentropy(raw_true_xy, raw_pred[..., 0:2], from_logits=True)
wh_loss = object_mask * box_loss_scale * 0.5 * K.square(raw_true_wh - raw_pred[..., 2:4])
confidence_loss = object_mask * K.binary_crossentropy(object_mask, raw_pred[..., 4:5], from_logits=True) + (1 - object_mask) * K.binary_crossentropy(object_mask, raw_pred[..., 4:5], from_logits=True) * ignore_mask
class_loss = object_mask * K.binary_crossentropy(true_class_probs, raw_pred[..., 5:], from_logits=True)

xy_loss = K.sum(xy_loss) / mf
wh_loss = K.sum(wh_loss) / mf
confidence_loss = K.sum(confidence_loss) / mf
class_loss = K.sum(class_loss) / mf
loss += xy_loss + wh_loss + confidence_loss + class_loss
```

**结果**:
1. YOLO V3在Pascal Titan X上处理608x608图像速度达到20FPS，在 COCO test-dev 上 mAP@0.5 达到 57.9%，与RetinaNet的结果相近，并且速度快了4倍。  YOLO V3的模型比之前的模型复杂了不少，可以通过改变模型结构的大小来权衡速度与精度。  速度对比如下：![image](https://cdn.jsdelivr.net/gh/Trouble404/Blog_Pics/Object-detection-learning/13.png)
2. 失败的尝试：
    * Anchor box坐标的偏移预测
    * 用线性方法预测x,y，而不是使用逻辑方法
    * focal loss
    * 双IOU阈值和真值分配

### YOLO V4
[论文地址](https://arxiv.org/pdf/2004.10934.pdf)
**模型**:
![image](https://cdn.jsdelivr.net/gh/Trouble404/Image/blog20201022194538.png)

1. **CBM**: Yolov4网络结构中的最小组件，由Conv+Bn+Mish激活函数三者组成。
2. **CBL**: 由Conv+Bn+Leaky_relu激活函数三者组成。
3. **Res unit**: 借鉴Resnet网络中的残差结构，让网络可以构建的更深。
4. **CSPX**: 借鉴CSPNet网络结构，由卷积层和X个Res unint模块Concate组成。
5. **SPP**: 采用1×1，5×5，9×9，13×13的最大池化的方式，进行多尺度融合。

每个CSPX中包含3+2*X个卷积层，因此整个主干网络Backbone中一共包含$2+（3+2*1）+2+（3+2*2）+2+（3+2*8）+2+（3+2*8）+2+（3+2*4）+1=72$。

**评估过的技术**
![image](https://cdn.jsdelivr.net/gh/Trouble404/Image/blog20201112153043.png)

**改进**:
1. **输入端**: Mosaic数据增强、cmBN、SAT自对抗训练、类别标签平滑化
2. **Backbone**: CSPDarknet53、Mish激活函数、Dropblock
3. **Nect**: SPP模块、PAN结构
4. **Prediction**: CmBN、最优超参数、CIOU_Loss，DIOU_nms

**mosaic**:
![image](https://cdn.jsdelivr.net/gh/Trouble404/Image/blog20201111174653.png)
pro:
* 丰富数据集：随机使用4张图片，随机缩放，再随机分布进行拼接，大大丰富了检测数据集，特别是随机缩放增加了很多小目标，让网络的鲁棒性更好。

**CSPNet**:
![image](https://cdn.jsdelivr.net/gh/Trouble404/Image/blog20201110164131.png)
CSPNet的作者认为推理计算过高的问题是由于网络优化中的梯度信息重复导致的。

因此采用CSP模块先将基础层的特征映射划分为两部分，然后通过跨阶段层次结构将它们合并，在减少了计算量的同时可以保证准确率。

因此Yolov4在主干网络Backbone采用CSPDarknet53网络结构，主要有三个方面的优点：
* 增强CNN的学习能力，使得在轻量化的同时保持准确性。
* 降低计算瓶颈
* 降低内存成本

**Mish**:
![image](https://cdn.jsdelivr.net/gh/Trouble404/Image/blog20201111175218.png)
正值可以达到任何高度, 避免了由于封顶而导致的饱和。理论上对负值的轻微允许更好的梯度流，而不是像ReLU中那样的硬零边界。

**Dropblock**:
![image](https://cdn.jsdelivr.net/gh/Trouble404/Image/blog20201110170309.png)
卷积层对于dropout丢弃并不敏感，因为卷积层通常是三层连用：卷积+激活+池化层，池化层本身就是对相邻单元起作用。而且即使随机丢弃，卷积层仍然可以从相邻的激活单元学习到相同的信息。

**SPP**:
采用SPP模块的方式，比单纯的使用k*k最大池化的方式，更有效的增加主干特征的接收范围，显著的分离了最重要的上下文特征。
在SPP模块中，使用$k=[1*1, 5*5, 9*9, 13*13]$的最大池化的方式，再将不同尺度的特征图进行Concat操作。

**PAN**:
PAN模型也叫金字塔注意力模型，主要由FPA(特征金字塔注意力模块)和GAU两个模型组成
* FPA
该模块能够融合来自 U 型网络 (如特征金字塔网络 FPN) 所提取的三种不同尺度的金字塔特征
![image](https://cdn.jsdelivr.net/gh/Trouble404/Image/blog20201111173615.png)
* GAU
GAU是用在decode时候的单元，并且引入注意力机制
![image](https://cdn.jsdelivr.net/gh/Trouble404/Image/blog20201111173723.png)

yolov4包含两个PAN结构，并且只取最后一个特征图，FPN层自顶向下传达强语义特征，而特征金字塔则自底向上传达强定位特征，两两联手，从不同的主干层对不同的检测层进行参数聚合。

**IOU LOSS**:
Bounding Box Regeression的Loss近些年的发展过程是：Smooth L1 Loss-> IoU Loss（2016）-> GIoU Loss（2019）-> DIoU Loss（2020）->CIoU Loss（2020）

1. [IOU loss](https://arxiv.org/pdf/1608.01471.pdf)
![image](https://cdn.jsdelivr.net/gh/Trouble404/Image/blog20201109144545.png)

**pro**:
* IoU损失将位置信息作为一个整体进行训练, 对比L2损失对4个独立变量进行训练能得到更为准确的效果.

**con**:
* 当预测框和目标框不相交时，IoU(A,B)=0时，不能反映A,B距离的远近，此时损失函数不可导，IoU Loss 无法优化两个框不相交的情况。
* 假设预测框和目标框的大小都确定，只要两个框的相交值是确定的，其IoU值是相同时，IoU值不能反映两个框是如何相交的。
![image](https://cdn.jsdelivr.net/gh/Trouble404/Image/blog20201109151903.png)


2. [GIOU loss](https://arxiv.org/pdf/1902.09630.pdf)
![image](https://cdn.jsdelivr.net/gh/Trouble404/Image/blog20201109153716.png)
找到一个最小的外接矩形C，让C可以将A和B包围在里面，然后我们计算C中没有覆盖A和B的面积占C总面积的比例，然后用A和B的IOU值减去这个比值

**pro**:
* 预测框和目标不相交的情况下也可以进行优化，且对物体的尺度大小不敏感(因为比值的原因)

**con**:
* 当目标框完全包裹预测框的时候，IoU和GIoU的值都一样，此时GIoU退化为IoU, 无法区分其相对位置关系
![image](https://cdn.jsdelivr.net/gh/Trouble404/Image/blog20201109153852.png)


3. [DIOU loss](https://arxiv.org/pdf/1911.08287.pdf)
![image](https://cdn.jsdelivr.net/gh/Trouble404/Image/blog20201109164442.png)
通常基于IoU-based的loss可以定义为$L=1-IOU+R(B,B^{gt})$其中$R(B,B^{gt})$)定义为预测框$B$和目标框$B^{gt}$的惩罚项。
$DIoU=IoU - \frac{p^{2}(b,b^{gt})}{c^{2}}$
其中，$b$和$b^{gt}$分别代表了预测框和真实框的中心点，且代表的是计算两个中心点间的欧式距离。$c$代表的是能够同时包含预测框和真实框的最小闭包区域的对角线距离。

**pro**:
* 即使目标框完全包裹预测框的时候也可以进行优化
* 将目标与anchor之间的距离，重叠率以及尺度都考虑进去，使得目标框回归变得更加稳定，不会像IoU和GIoU一样出现训练过程中发散等问题
* DIoU还可以替换普通的IoU评价策略，应用于NMS中，使得NMS得到的结果更加合理和有效。

**con**:
* 没有考虑预测框的长宽比，当预测框与目标框中心点距离相同时无法进行优化
![image](https://cdn.jsdelivr.net/gh/Trouble404/Image/blog20201109170205.png)


4. [CIOU loss](https://arxiv.org/pdf/1911.08287.pdf)
$CIoU=IoU - \frac{p^{2}(b,b^{gt})}{c^{2}} - \alpha v$
$v=\frac{4}{\pi^{2}(arctan(w^{gt}/h^{gt})-arctan(w/h))^{2}}$
$\alpha = v / ((1-IoU)+v)$
其中，$\alpha$是用于做trade-off的权重函数，$v$是用来衡量长宽比一致性的参数

**pro**:
* 目标框回归函数应该考虑三个重要几何因素：重叠面积、中心点距离，长宽比全都考虑进去了


**结果**:
![image](https://cdn.jsdelivr.net/gh/Trouble404/Image/blog20201022195126.png)

### SSD: Single Shot MultiBox Detector
[论文地址](https://arxiv.org/pdf/1512.02325.pdf)

SSD将边界框的输出空间离散化为不同长宽比的一组默认框和并缩放每个特征映射的位置。在预测时，网络会在每个默认框中为每个目标类别的出现生成分数，并对框进行调整以更好地匹配目标形状。此外，网络还结合了不同分辨率的多个特征映射的预测，自然地处理各种尺寸的目标。

**改进**:
1. 针对多个类别的单次检测器
2. 预测固定的一系列默认边界框的类别分数和边界框偏移，使用更小的卷积滤波器应用到特征映射上
3. 根据不同尺度的特征映射生成不同尺度的预测，并通过纵横比明确分开预测
4. 在低分辨率输入图像上也能实现简单的端到端训练和高精度，从而进一步提高速度与精度之间的权衡。

**模型**:
![image](https://cdn.jsdelivr.net/gh/Trouble404/Blog_Pics/Object-detection-learning/15.png)
SSD方法基于前馈卷积网络，该网络产生固定大小的边界框集合，并对这些边界框中存在的目标类别实例进行评分，然后进行非极大值抑制步骤来产生最终的检测结果。早期的网络层基于用于高质量图像分类的标准架构将其称为基础网络。然后，将辅助结构添加到网络中以产生具有以下关键特征的检测：
* **用于检测的多尺度特征映射**。我们将卷积特征层添加到截取的基础网络的末端。这些层在尺寸上逐渐减小，并允许在多个尺度上对检测结果进行预测。用于预测检测的卷积模型对于每个特征层都是不同的
* **用于检测的卷积预测器**。每个添加的特征层（或者任选的来自基础网络的现有特征层）可以使用一组卷积滤波器产生固定的检测预测集合。
* **默认边界框和长宽比**。默认边界框与Faster R-CNN[2]中使用的锚边界框相似，但是我们将它们应用到不同分辨率的几个特征映射上。在几个特征映射中允许不同的默认边界框形状可以有效地离散可能的输出框形状的空间。![image](https://cdn.jsdelivr.net/gh/Trouble404/Blog_Pics/Object-detection-learning/16.png)
* 总结： 末尾添加的特征层预测不同尺度的长宽比的默认边界框的偏移量以及相关的置信度。PS: 空洞版本VGG更快。

**训练**：
1. **匹配策略**：默认边界框匹配到IOU重叠高于阈值（0.5）的任何实际边界框。这简化了学习问题，允许网络为多个重叠的默认边界框预测高分，而不是要求它只挑选具有最大重叠的一个边界框。
2. **训练目标函数**：定位损失加上置信度损失![image](https://cdn.jsdelivr.net/gh/Trouble404/Blog_Pics/Object-detection-learning/17.png)
3. **为默认边界框选择尺度和长宽比**: ![image](https://cdn.jsdelivr.net/gh/Trouble404/Blog_Pics/Object-detection-learning/18.png)
4. **难例挖掘**: 在匹配步骤之后，大多数默认边界框为负例，尤其是当可能的默认边界框数量较多时。这在正的训练实例和负的训练实例之间引入了显著的不平衡。不使用所有负例，而是使用每个默认边界框的最高置信度损失来排序它们，并挑选最高的置信度，以便负例和正例之间的比例至多为3:1。
5. **数据增强**![image](https://cdn.jsdelivr.net/gh/Trouble404/Blog_Pics/Object-detection-learning/20.png)
6. **小目标数据增强**： 将图像随机放置在填充了平均值的原始图像大小为16x的画布上，然后再进行任意的随机裁剪操作。因为通过引入这个新的“扩展”数据增强技巧，有更多的训练图像，所以必须将训练迭代次数加倍。

**优劣**:
SSD对类似的目标类别（特别是对于动物）有更多的混淆，部分原因是共享多个类别的位置。SSD对边界框大小非常敏感。换句话说，它在较小目标上比在较大目标上的性能要差得多。这并不奇怪，因为这些小目标甚至可能在顶层没有任何信息。增加输入尺寸（例如从300×300到512×512）可以帮助改进检测小目标，但仍然有很大的改进空间。积极的一面，SSD在大型目标上的表现非常好。而且对于不同长宽比的目标，它是非常鲁棒的，因为使用每个特征映射位置的各种长宽比的默认框。

### RetinaNet Focal Loss for Dense Object Detection
[论文地址](https://arxiv.org/abs/1708.02002)

![image](https://cdn.jsdelivr.net/gh/Trouble404/Blog_Pics/Object-detection-learning/32.png)

稠密分类的后者精度不够高，核心问题（central issus）是稠密proposal中前景和背景的极度不平衡。以我更熟悉的YOLO举例子，比如在PASCAL VOC数据集中，每张图片上标注的目标可能也就几个。但是YOLO V2最后一层的输出是13×13×5，也就是845个候选目标！大量（简单易区分）的负样本在loss中占据了很大比重，使得有用的loss不能回传回来。基于此，作者将经典的交叉熵损失做了变形（见下），给那些易于被分类的简单例子小的权重，给不易区分的难例更大的权重。同时，作者提出了一个新的one-stage的检测器RetinaNet，达到了速度和精度很好地trade-off。

$FL(p_{t}=-(1-p_{t})^{\gamma}log(p_{t})$

**Focal Loss**
Focal Loss从交叉熵损失而来。二分类的交叉熵损失如下：
![image](https://cdn.jsdelivr.net/gh/Trouble404/Blog_Pics/Object-detection-learning/33.png)

对应的，多分类的交叉熵损失是这样的：

$CE(p,y)=-log(p_{y})$

因此可以使用添加权重的交叉熵损失：
$CE(p)=-\alpha_{t}log(p_{t})$

而作者提出的是一个自适应调节的权重：(可加入权重$\alpha$平衡)

**$FL(p_{t}=-(1-p_{t})^{\gamma}log(p_{t})$**

Pytorch实现：

$L=-\sum_{i}^{C}onehot\odot(1-p_{t})^{\gamma}log(p_{t})$

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def one_hot(index, classes):
    size = index.size() + (classes,)
    view = index.size() + (1,)
    mask = torch.Tensor(*size).fill_(0)
    index = index.view(*view)
    ones = 1.
    if isinstance(index, Variable):
        ones = Variable(torch.Tensor(index.size()).fill_(1))
        mask = Variable(mask, volatile=index.volatile)
    return mask.scatter_(1, index, ones)
    
class FocalLoss(nn.Module):
    def __init__(self, gamma=0, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        
    def forward(self, input, target):
        y = one_hot(target, input.size(-1))
        logit = F.softmax(input)
        logit = logit.clamp(self.eps, 1. - self.eps)
        loss = -1 * y * torch.log(logit) # cross entropy
        loss = loss * (1 - logit) ** self.gamma # focal loss
        return loss.sum()
```

**模型**:

1. 模型初始化： 对于一般的分类网络，初始化之后，往往其输出的预测结果是均等的（随机猜测）。然而作者认为，这种初始化方式在类别极度不均衡的时候是有害的。作者提出，应该初始化模型参数，使得初始化之后，模型输出稀有类别的概率变小（如0.01），作者发现这种初始化方法对于交叉熵损失和Focal Loss的性能提升都有帮助。首先，从imagenet预训练得到的base net不做调整，新加入的卷积层权重均初始化为$\sigma$=0.01的高斯分布，偏置项为0.对于分类网络的最后一个卷积层，偏置项为$b=-log(\frac{(1-\pi)}{\pi})$, $\pi$是一个超参数，其意义是在训练的初始阶段，每个anchor被分类为前景的概率。
![image](https://cdn.jsdelivr.net/gh/Trouble404/Blog_Pics/Object-detection-learning/34.png)

作者利用前面介绍的发现和结论，基于ResNet和Feature Pyramid Net（FPN）设计了一种新的one-stage检测框架.RetinaNet 是由一个骨干网络和两个特定任务子网组成的单一网络。骨感网络负责在整个输入图像上计算卷积特征图，并且是一个现成的卷积网络。
* 第一个子网在骨干网络的输出上执行卷积对象分类(small FCN attached to each FPN level)子网的参数在所有金字塔级别共享
* 第二个子网执行卷积边界框回归(attach another samll FCN to each pyramid level)
* 对象分类子网和框回归子网，尽管共享一个共同的结构，使用单独的参数。

Anchors: 
在金字塔等级P3到P7上，锚点的面积分别为$32^{2}$到$512^{2}$, 使用的长宽比为[1:2,1:1,2:1]. 对于更密集的比例覆盖，每个级别添加锚点的尺寸$[2^{0},2^{\frac{1}{3}},2^{\frac{1}{2}}]$

IOU:
\[0,0.4)的为背景，\[0.4,0.5)的忽略，大于0.5的为前景

## Two Stage

### R-FCN Object Detection via Region-based Fully Convolutional Networks

[论文地址](https://link.jianshu.com/?t=https%3A%2F%2Farxiv.org%2Fpdf%2F1605.06409.pdf)

R-FCN 通过添加 Position-sensitive score map 解决了把 ROI pooling 放到网络最后一层降低平移可变性的问题，以此改进了 Faster R-CNN 中检测速度慢的问题。

PS: 
* 分类需要特征具有平移不变性，检测则要求对目标的平移做出准确响应。论文中作者给了测试的数据：ROI放在ResNet-101的conv5后，mAP是68.9%；ROI放到conv5前（就是标准的Faster R-CNN结构）的mAP是76.4%，差距是巨大的，这能证明平移可变性对目标检测的重要性。

* Faster R-CNN检测速度慢的问题，速度慢是因为ROI层后的结构对不同的proposal是不共享的，试想下如果有300个proposal，ROI后的全连接网络就要计算300次, 非常耗时。

**模型**：
![image](https://cdn.jsdelivr.net/gh/Trouble404/Blog_Pics/Object-detection-learning/19.png)

* Backbone architecture: ResNet-101有100个卷积层，后面是全局平均池化和1000类的全连接层。删除了平均池化层和全连接层，只使用卷积层来计算特征映射。最后一个卷积块是2048维，附加一个随机初始化的1024维的1×1卷积层来降维
* $k^{2}(C+1)$Conv: ResNet101的输出是W\*H\*1024，用$k^{2}(C+1)$个1024\*1\*1的卷积核去卷积即可得到$k^{2}(C+1)$个大小为W\*H的position sensitive的score map。这步的卷积操作就是在做prediction。k = 3，表示把一个ROI划分成3\*3，对应的9个位置
* ROI pooling: 一层的SPP结构。主要用来将不同大小的ROI对应的feature map映射成同样维度的特征![image](https://cdn.jsdelivr.net/gh/Trouble404/Blog_Pics/Object-detection-learning/21.png)
* Vote:k\*k个bin直接进行求和（每个类单独做）得到每一类的score，并进行softmax得到每类的最终得分，并用于计算损失![image](https://cdn.jsdelivr.net/gh/Trouble404/Blog_Pics/Object-detection-learning/22.png)

**训练**:
* 损失函数： $L(s,t_{x,y,w,h})=L_{cls}(s_{c^{\star}}+ \lambda [c^{\star}>0]L_{reg}(t,t^{\star})$ 
将正样本定义为与真实边界框IOU至少为0.5的ROI，否则为负样本
* 在线难例挖掘（**OHEM**): 其主要考虑训练样本集总是包含较多easy examples而相对较少hard examples，而自动选择困难样本能够使得训练更为有效，此外还有：S-OHEM: Stratified Online Hard Example Mining for Object Detection. **S-OHEM** 利用OHEM和stratified sampling技术。其主要考虑OHEM训练过程忽略了不同损失分布的影响，因此S-OHEM根据分布抽样训练样本。A-Fast-RCNN: Hard positive generation via adversary for object detection从更好的利用数据的角度出发，OHEM和S-OHEM都是发现困难样本，而A-Fast-RCNN的方法则是通过**GAN**的方式在特征空间产生具有部分遮挡和形变的困难样本。
* **空洞和步长**:我们的全卷积架构享有FCN广泛使用的语义分割的网络修改的好处。特别的是，将ResNet-101的有效步长从32像素降低到了16像素，增加了分数图的分辨率。第一个conv5块中的stride=2操作被修改为stride=1，并且conv5阶段的所有卷积滤波器都被“hole algorithm” 修改来弥补减少的步幅.

**位置敏感分数图**：
我们可以想想一下这种情况，M 是一个 5\*5 大小，有一个蓝色的正方形物体在其中的特征图，我们将方形物体平均分割成 3\*3 的区域。现在我们从 M 中创建一个新的特征图并只用其来检测方形区域的左上角。这个新的特征图如下右图，只有黄色网格单元被激活![image](https://cdn.jsdelivr.net/gh/Trouble404/Blog_Pics/Object-detection-learning/23.png)

因为我们将方形分为了 9 个部分，我们可以创建 9 张特征图分别来检测对应的物体区域。因为每张图检测的是目标物体的子区域，所以这些特征图被称为位置敏感分数图（position-sensitive score maps）。![image](https://cdn.jsdelivr.net/gh/Trouble404/Blog_Pics/Object-detection-learning/24.png)
比如，我们可以说，下图由虚线所画的红色矩形是被提议的 ROIs 。我们将其分为 3\*3 区域并得出每个区域可能包含其对应的物体部分的可能性。我们将此结果储存在 3\*3 的投票阵列（如下右图）中。比如，投票阵列 [0][0] 中数值的意义是在此找到方形目标左上区域的可能性。![image](https://cdn.jsdelivr.net/gh/Trouble404/Blog_Pics/Object-detection-learning/25.png)
将分数图和 ROIs 映射到投票阵列的过程叫做位置敏感 ROI 池化（position-sensitive ROI-pool）。
![image](https://cdn.jsdelivr.net/gh/Trouble404/Blog_Pics/Object-detection-learning/26.png)
在计算完位置敏感 ROI 池化所有的值之后，分类的得分就是所有它元素的平均值
![image](https://cdn.jsdelivr.net/gh/Trouble404/Blog_Pics/Object-detection-learning/27.png)
如果说我们有 C 类物体需要检测。我们将使用 C+1个类，因为其中多包括了一个背景（无目标物体）类。每类都分别有一个 3×3 分数图，因此一共有 (C+1)×3×3 张分数图。通过使用自己类别的那组分数图，我们可以预测出每一类的分数。然后我们使用 softmax 来操作这些分数从而计算出每一类的概率。


### FPN Feature Pyramid Networks for Object Detection
[论文地址](http://openaccess.thecvf.com/content_cvpr_2017/papers/Lin_Feature_Pyramid_Networks_CVPR_2017_paper.pdf)

![image](https://cdn.jsdelivr.net/gh/Trouble404/Blog_Pics/Object-detection-learning/28.png)

a. 通过缩放图片获取不同尺度的特征图

b. ConvNET

c. 通过不同特征分层

d. FPN

特征金字塔网络FPN，网络直接在原来的单网络上做修改，每个分辨率的 feature map 引入后一分辨率缩放两倍的 feature map 做 element-wise 相加的操作。通过这样的连接，每一层预测所用的 feature map 都融合了不同分辨率、不同语义强度的特征，融合的不同分辨率的 feature map 分别做对应分辨率大小的物体检测。这样保证了每一层都有合适的分辨率以及强语义特征。同时，由于此方法只是在原网络基础上加上了额外的跨层连接，在实际应用中几乎不增加额外的时间和计算量。

**FPN**:
![image](https://cdn.jsdelivr.net/gh/Trouble404/Blog_Pics/Object-detection-learning/29.png)

1. 图中feature map用蓝色轮廓表示，较粗的表示语义上较强特征
2. 采用的bottom-up 和 top-down 的方法， bottom-up这条含有较低级别的语义但其激活可以更精确的定位因为下采样的次数更少。 Top-down的这条路更粗糙但是语义更强。
3. top-down的特征随后通过bottom-up的特征经由横向连接进行增强如图，使用较粗糙分辨率的特征映射时候将空间分辨率上采样x2倍。bottom-up的特征要经过1x1卷积层来生成最粗糙分辨率映射。
4. 每个横向连接合并来自自下而上路径和自顶向下路径的具有相同空间大小的特征映射。
![image](https://cdn.jsdelivr.net/gh/Trouble404/Blog_Pics/Object-detection-learning/30.png)


**RPN结合FPN**
1. 通过用FPN替换单尺度特征映射来适应RPN。在特征金字塔的每个层级上附加一个相同设计的头部（3x3 conv（目标/非目标二分类和边界框回归）和 两个1x1convs（分类和回归））由于头部在所有金字塔等级上的所有位置密集滑动，所以不需要在特定层级上具有多尺度锚点。相反，为每个层级分配单尺度的锚点。定义锚点$[P_{2},P_{3},P_{4},P_{5},P_{6}]$分别具有$[32^{2},64^{2},128^{2},256^{2},512^{2}]$个像素 面积，以及多个长宽比{1:2,1:1,2:1}所以总共15个锚点在金字塔上。
2. 其余同RPN网络
2. 不同的尺度ROI用不同层的特征，每个box根据公式计算后提取其中某一层特征图对应的特征ROI，大尺度就用后面一些的金字塔层（P5），小尺度就用前面一点的层（P4）
可根据公式：$k=[k_{0}+log_{2}(\frac{\sqrt{wh}}{224}]$ 计算需要哪一层的特征
![image](https://cdn.jsdelivr.net/gh/Trouble404/Blog_Pics/Object-detection-learning/31.png)

### Light-Head R-CNN: In Defense of Two-Stage Object Detector
[论文地址](https://arxiv.org/abs/1711.07264)

two-stage 的方法在本身的一个基础网络上都会附加计算量很大的一个用于 classification+regression 的网络，导致速度变慢
* Faster R-CNN: two fully connected layers for RoI recognition
* R-FCN: produces a large score maps.

因此，作者为了解决 detection 的速度问题，提出了一种新的 two-stage detector，就是Light-Head R-CNN。速度和准确率都有提升。Light-Head R-CNN 重点是 head 的结构设计。包括两部分： R-CNN subnet（ROI pooling 之后的network） 和ROI warping。

![image](https://cdn.jsdelivr.net/gh/Trouble404/Blog_Pics/Object-detection-learning/35.png)

**方法**
* **Thin feature map**: ![image](https://cdn.jsdelivr.net/gh/Trouble404/Blog_Pics/Object-detection-learning/36.png)降低进入head部分feature map的channel数也就是将r-fcn的score map从$P\times P(C+1)$减小到$P\times P\times \alpha$，$\alpha$是一个与类别数无关且较小的值，比如10。这样，score map的channel数与类别数无关，使得后面的分类不能像r-fcn那样vote，于是在roi pooling之后添加了一个fc层进行预测类别和位置。
* **Large separable convolution**: 


![image](https://cdn.jsdelivr.net/gh/Trouble404/Blog_Pics/Object-detection-learning/38.png)![image](https://cdn.jsdelivr.net/gh/Trouble404/Blog_Pics/Object-detection-learning/39.png)  
* **Cheap R-CNN**: ![image](https://cdn.jsdelivr.net/gh/Trouble404/Blog_Pics/Object-detection-learning/37.png) 

**模型**：
* **Large**: (1) ResNEt-101; (2) atrous algorithm; (3) RPN chanels 512; (4) Large separable convolution with $C_{mid}=256$
* **Small**: (1) Xception like; (2) abbandon atrous algorithm; (3) RPN convolution to 256; (4) Large separable convolution with $C_{mid}=64$; (5) Apply PSPooling with alignment techniques as RoI warping(better results involve RoI-align).