# ThunderNet-Review

## 参考：https://blog.csdn.net/u011344545/article/details/88927579

# ThunderNet: Towards Real-time Generic Object Detection
论文下载：https://pan.baidu.com/s/1v2gIkhqR5nqBszo5BmNVkw  
提取码：f59z 
https://arxiv.org/pdf/1903.11752.pdf

# TensorMask: A Foundation for Dense Object Segmentation
https://blog.csdn.net/u011344545/article/details/88913042

# 摘要
移动平台上的通用对象实时检测是一项关键而又具有挑战性的计算机视觉任务。然而，以往的基于cnn的检测器存在着巨大的计算成本，这使得它们无法在计算受限的情况下进行实时推断。本文研究了两级检测器在实时通用检测中的有效性，并提出了一种轻量级的两级检测器ThunderNet。在主干网部分，分析了现有轻量化主干网的不足，提出了一种面向对象检测设计的轻量化主干网。在检测部分，我们采用了一种非常有效的RPN和检测头设计。为了生成更具鉴别性的特征表示，我们设计了两个高效的架构模块:上下文增强模块和空间注意模块。最后，我们研究了输入分辨率、主干和检测头之间的平衡。与轻量级单级检测器相比，迅雷网仅用单级检测器就能取得优异的性能占PASCALVOC和COCO基准计算成本的40%。没有铃铛和哨子，我们的模型运行在基于ARM设备上帧率为24,1fps。据我们所知，这是在ARM平台上报道的第一个实时探测器。代码将被发布用于纸张复制。

# 引言
移动设备上通用对象的实时检测是计算机视觉领域的一项重要而又具有挑战性的任务。与服务器级gpu相比，移动设备计算受限，对检测器的计算成本提出了更严格的限制。然而，现代基于cnn的检测器资源匮乏，需要大量的计算才能达到理想的检测精度，这阻碍了它们在移动场景中的实时推理。

# ThunderNet的总体架构
ThunderNet的输入分辨率为320×320像素。SNet骨干网是基于ShuffleNetV2是专门为对象检测而设计的。检测部分对RPN进行压缩，R-CNN子网采用1024-d fc层，效率更高。上下文增强模块利用来自多个尺度的语义和上下文信息。空间注意模块引入RPN中的信息来细化特征分布。

# 相关工作

## 1、CNN-based object detectors（基于CNN的目标检测器）——一般分为一步法和二步法

神经网络 | TensorMask: A Foundation for Dense Object Segmentation（何凯明团队新作）
https://blog.csdn.net/u011344545/article/details/88913042

## 2、Real-time generic object detection（实时通用对象检测）

## YOLO：
You Only Look Once: Unified, Real-Time Object Detection 
https://arxiv.org/pdf/1506.02640.pdf

## Yolo9000: better, faster, stronger.
https://arxiv.org/pdf/1612.08242.pdf

## Yolov3: Anincrementalimprovement
https://arxiv.org/pdf/1804.02767.pdf
https://pjreddie.com/    YOLO主页

## SSD：
Ssd: Single shot multibox detector.
https://arxiv.org/pdf/1512.02325.pdf         论文
https://zhuanlan.zhihu.com/p/31427288  知乎
https://github.com/MichaelBeechan/SSD-Tensorflow    代码

## MobileNet-SSD：
https://arxiv.org/pdf/1704.04861.pdf    论文
https://github.com/MichaelBeechan/MobileNet-SSD   代码

## MobileNetV2-SSDLite： 
Mobilenetv2: Inverted residuals and linear bottlenecks.
https://arxiv.org/pdf/1801.04381v3.pdf    论文
https://github.com/MichaelBeechan/MobileNetv2-SSDLite  代码
https://github.com/MichaelBeechan/MobileNetv2

## Tiny-DSOD：
Tiny-dsod: Lightweight object detection for resource-restricted usages  
https://arxiv.org/pdf/1807.11013.pdf   论文
https://github.com/MichaelBeechan/Tiny-DSOD     代码

## Light-Head R-CNN：
Light-head r-cnn: In defense of two-stage object detector
https://arxiv.org/pdf/1807.11013.pdf
https://github.com/MichaelBeechan/light_head_rcnn

## RetinaNet：
Focal Loss for Dense Object Detection 
https://arxiv.org/pdf/1708.02002.pdf
https://github.com/MichaelBeechan/keras-retinanet

## CornerNet：
CornerNet: Detecting Objects as Paired Keypoints 
https://arxiv.org/pdf/1808.01244.pdf
https://github.com/MichaelBeechan/CornerNet

## 3、Backbone networks for detection

## Feature pyramid networks for object detection
https://arxiv.org/pdf/1612.03144.pdf
https://github.com/MichaelBeechan/FPN
https://github.com/MichaelBeechan/FPN_Tensorflow

## MobileNet：
ShuffleNet：
Shufflenet: An extremelyefficientconvolutionalneuralnetworkformobiledevices
https://arxiv.org/pdf/1707.01083.pdf
https://github.com/MichaelBeechan/ShuffleNet
https://github.com/MichaelBeechan/ShuffleNet-1

## DetNet：
DetNet: A Backbone network for Object Detection 
https://arxiv.org/pdf/1804.06215.pdf
https://github.com/MichaelBeechan/DetNet_pytorch


# 实验

## 实验所用数据集 ：PASCAL VOC  and  COCO
ThePascalVisual Object Classes Challenge: A Retrospective 
https://link.springer.com/content/pdf/10.1007%2Fs11263-014-0733-5.pdf
[benchmark.tgz](http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz)
https://blog.csdn.net/u011344545/article/details/84977173
https://github.com/MichaelBeechan/CV-Datasets

## Microsoft coco: Common objects in context
https://arxiv.org/pdf/1405.0312.pdf
https://github.com/MichaelBeechan/cocoapi
http://cocodataset.org/#download
http://cocodataset.org/

Context Enhancement Module与Spatial Attention Module
可参考：https://www.jianshu.com/p/aa7502e3547d
