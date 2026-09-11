---
created: 2024-10-17
title: "Rethinking Features-Fused-Pyramid-Neck for Object Detection"
aliases:
  - "Rethinking Features-Fused-Pyramid-Neck for Object Detection"
tags:
  - object-detection
  - efficient-models
---

> [Rethinking Features-Fused-Pyramid-Neck for Object Detection](https://eccv.ecva.net/virtual/2024/poster/1537)

## Abstract

Multi-head detectors typically employ a features-fused-pyramid-neck for multi-scale detection and are widely adopted in the industry. However, this approach faces feature misalignment when representations from different hierarchical levels of the feature pyramid are forcibly fused point-to-point. To address this issue, we designed an independent hierarchy pyramid (IHP) architecture to evaluate the effectiveness of the features-unfused-pyramid-neck for multi-head detectors. Subsequently, we introduced soft nearest neighbor interpolation (SNI) with a weightdownscaling factor to mitigate the impact of feature fusion at different hierarchies while preserving key textures. Furthermore, we present a feature adaptive selection method for downsampling in extended spatial windows (ESD) to retain spatial features and enhance lightweight convolutional techniques (GSConvE). These advancements culminate in our secondary features alignment solution (SA) for real-time detection, achieving state-of-the-art results on Pascal VOC and MS COCO.

## Introduction

- 早期基于深度学习的目标检测模型的探索为该领域提供了重要的见解：多尺度特征的有效利用对于提高检测器的整体性能（准确度与速度的权衡）至关重要。
- 后续基于 FPN 的各种变体越来越复杂，表征学习表明模型最终需要精细的表示，而不是大量的复杂表示。
- 这些带有表征偏置的特征图直接进行元素级融合可能会导致部分的表征被破坏，并加剧融合过程中的特征错位问题。
- 此外，FPN 的最初目的不仅是为了展示特征融合在解决多尺度检测挑战方面的优势，也是为了适当降低模型复杂度。在硬件资源有限的行业中，复杂的模型通常难以应用。

## Related Work

### 多头检测和特征融合

深度学习的第一代通用检测模型由两个主要部分组成：主干和检测头。这些模型通常使用最终特征图进行预测。SSD 引入了使用多层次特征图进行物体检测，标志着多头检测的开始，并影响了后续的实时检测器设计。在 SSD 模型中，每个检测头直接使用来自主干网不同层级的原始特征图，无需进行任何融合。多头检测方法的主要优点是能够使用具有不同感受野的特征图预测不同尺度的物体。高层特征图具有宽广的全局感受野，有利于识别大型物体，而低层特征图具有精细的局部感受野，有助于识别小型物体。FPN 引入了多级特征图的融合方案，以提高检测精度，开创了特征融合的潮流，并成为一种普遍做法。这导致了 "主干-颈部-头部"（Backbone-Neck-Head）架构的发展。此后，又提出了基于 FPN 的更复杂的变体，如 BiFPN 和 PANet，从而可以使用更多特征进行预测。然而，随着预测准确率的提高，由于增加了融合层，网络变得越来越复杂。一些方法试图绕过特征融合方案，探索多尺度检测挑战的新视角，如优化多尺度训练策略或使用扩张卷积捕捉多尺度感受野。但随着分类 Backbone 特征提取能力的不断优化和特征融合技术的持续发展，这些方法已逐渐失去竞争力。

### 轻量化

一个模型是选择云计算还是边缘计算，取决于其参数和浮点运算（FLOP）的数量。因此，减少参数或浮点运算次数是轻量级模型研究的首要重点。直接的轻量级方法包括减少模型的深度（层数）或宽度（神经元/Filter数量），如 YOLO-fast/tiny/nano 等模型。此外，还可以采用稀疏计算技术，如 MobileNets 和 ShuffleNet 中使用的深度可分离卷积。然而，由于非线性表示能力不足，低深度网络往往会出现拟合不足的问题。因此，减少网络层数并不总是一种经济有效的简化模型的方法。

## 次要特征对齐解决方案（Secondary Features Alignment Solution - SA）

- Independent hierarchy pyramid（IHP）：通过放弃融合解决了特征错位问题。
- Soft nearest neighbor interpolation（SNI）：缓解点对点特征融合过程中的错位问题。
- Feature adaptive selection in extended spatial windows（ESD）：增强下采样阶段的空间特征捕获。
- Lightweight GSConv enhancement（GSConvE）：提高轻量级模型的准确性和速度之间的性能。

### Independent hierarchy pyramid architecture

FPN 的提出将所有实时检测器带入了特征融合的领域。但作者发现了不同层次特征的点对点融合所产生的一个重要问题就是局部局部特征变得错位。这种融合过程类似于在错位的特征组合时添加噪音，因为与目标空间无关的特征被强行整合在一起，导致空间混乱。

通常情况下，近邻插值法用于快速提高高层特征图的分辨率，使其与低层特征图的分辨率相匹配，然后通过元素相加、通道串联或加权和等方法进行融合。然而，下采样往往是非线性和不可逆的，会导致上采样产生的附加特征之间的空间特征不一致。

![IHP 和其他七种典型的颈部结构。三张热图分别来自 FPN 颈部模型的第三层（P3）、第四层（P4）和第五层（P5）。每个层次的特征所关注的区域用红色标出；颜色越深，特征权重越高。很明显，较低层次的层次结构优先考虑局部特征，而较高层次的层次结构则更注重全局特征。换句话说，在不同的层次结构中存在表征偏差。如果不先解决错位问题，就通过点对点融合的方式将这些特征结合起来，可能会引入噪音，而不是丰富语义。](Paper%20Reading/attachments/GSConv_V2/001.webp)

IHP 采用了一种激进的方法，放弃了颈部的所有融合，从根本上规避了特征错位问题，并精简了结构。通过利用多头检测模式的固有优势，IHP 利用不同层次的特征图直接预测不同大小的物体。

但还是有一些问题存在。首先， IHP 可能会减少语义信息。其次，与定位分类耦合头检测器不同，IHP 没有对定位分类解耦头检测器实现同样的正向反应。最后，FPN 在某种程度上仍然适用于解耦头检测器，但特征未对准问题仍然需要解决。

### Soft nearest neighbor interpolation

在检测或分割模型中，不同级别的特征融合通常发生在特征图上采样之后，近邻插值和转置卷积是流行的上采样方法。扩展的特征图代表抽象的高层语义特征，而不是原始图像细节信息。转置卷积会增加计算量和延迟。

![融合和 SNI 中的特征错位的插图。这些数字，22-88，只是不同局部特征的标记，而不是真正的特征值。](Paper%20Reading/attachments/GSConv_V2/002.webp)

近邻插值类似于平均去池化，但被认为是一种 “硬 ”操作。通过软化这一操作（类似于将 Max 转换为 SoftMax 函数），减轻 “硬 ”特征错位问题。为此，在上采样过程中为近邻插值引入了一个软因子（用$\alpha$表示）：

$$
Y=\alpha \cdot f(X), \alpha=\frac{ResolutionX}{ResolutionY}
$$

- $X$：高层特征图的分辨率。
- $Y$：低层特征图的分辨率。
- $f$：近邻插值算法。

SNI 会根据特征图的缩放因子来调整高层语义特征对低层特征的影响。具体来说，随着缩放因子的增加，高层语义特征对低层特征的影响会减弱。与 SoftMax 不同，SNI 不会将输出转换为概率，而是在通过点对点融合将高层特征与低层特征整合时，在不增加成本的情况下减轻错位。SNI 在保留关键纹理的同时软化特征权重，从而减少融合过程中不同层次特征之间的相互影响。

![SNI 与传统方法的比较。这个结果可以通过使用 SNI 源代码（模型：Yolov5n-panet）来重现。上采样前后相同尺寸缩放是为了直观比较。](Paper%20Reading/attachments/GSConv_V2/003.webp)

### Feature adaptive selection in extended spatial windows

在视觉表征学习中，模型通常依赖于低分辨率、高级特征图进行预测，而不是直接处理高分辨率原始图像。然而，过度的下采样可能会导致空间细节的严重损失。

![图 4](Paper%20Reading/attachments/GSConv_V2/004.webp)

作者提出了下采样扩展空间窗口（Extended Spatial Windows for Downsampling - ESD）中的特征自适应选择方法，以增强空间信息保留能力。ESD 包括两个非线性分支和一个线性分支。在非线性分支中，标准卷积层和扩展窗口最大池化层增强了局部特征捕捉能力，而扩展窗口平均池化层增强了全局特征捕捉能力。随后，线性特征与非线性特征进行融合。

ESD-I 采用逐元素加法合并特征，计算成本低，适用于轻量级模型；ESD-II 采用可学习的线性自适应融合，计算成本略有增加，适用于标准的模型。重要的是，扩展窗口的局部和全局特征采用简单的池化技术进行采样，在很大程度上保留了输入信息。这减少了下采样过程中的信息损失，并使后续层能够从上一层学习部分表征，类似于间接实现隐藏捷径连接，让人想起 ResNet。

### Lightweight GSConv enhancement

![GSConv](Paper%20Reading/attachments/GSConv_V2/005.webp)

![GSConvE](Paper%20Reading/attachments/GSConv_V2/006.webp)

GSConvE-I 是专为标准模型设计的，其中辅助分支上的中间特征映射来自密集线性映射，而输出特征映射来自稀疏线性映射。这种方法既能保持特征的丰富性，又不会显著增加计算成本。

另一方面，GSConvE-II 面向轻量级模型。它包含三个深度卷积辅助分支，内核大小分别为 9×9、13×13 和 17×17。通过利用更大的内核尺寸，该变体可以直接捕捉更大的感受野和全局特征，而只需较少的层累加，为优先考虑减少 FLOP 和层数的轻量级模型提供了较高的成本效益。

在 GSConvE-I 中，作者探索去除深度卷积分支上的批量归一化层，同时保留主分支上的批量归一化层，类似于 ConvNeXt。这种调整避免了梯度消失问题并简化了计算，从而提高了预测精度。

## Experiments

### Ablation studies

![IHP、SNI 和 ESD-I/II 在 VOC 07+12 上的消融实验。](Paper%20Reading/attachments/GSConv_V2/007.webp)

![在 VOC 07+12 上进行的 GSConv 和 GSConvE-I/II 消融实验。](Paper%20Reading/attachments/GSConv_V2/008.webp)

### Comparasion experiments

![SYOLO 的结构。E-ELAN/C2f 来自 YOLOv7/8。](Paper%20Reading/attachments/GSConv_V2/009.webp)

![VOC 07+12 SOTA 实时检测器的比较。](Paper%20Reading/attachments/GSConv_V2/010.webp)

![COCO SOTA 实时检测器的比较。](Paper%20Reading/attachments/GSConv_V2/011.webp)

![COCO SOTA 实时检测器的比较。](Paper%20Reading/attachments/GSConv_V2/012.webp)

## Conclusion

作者探索了表征学习的本质，发现不同层次特征图的融合面临着特征错位的问题。为了解决这个问题，作者引入了 SNI。此外，作者还提出了 ESD，以减轻下采样过程中的空间信息损失，并进一步优化轻量级卷积方法，以提高轻量级模型的计算效率。在这些进步的基础上，作者提出了用于实时检测的 SA 解决方案，所有测试的检测器都超越了其原有性能，达到了 SOTA 的结果。在过去几年中，目标检测从 FPN 的范式中获益匪浅。然而，最近的发展表明，包括创新架构和训练策略在内的更新技术可能会为现代检测器提供更优越的性能，从而使类 FPN 范式具有潜在的局限性。在许多情况下，通过应用 SNI 来解决特征错位问题，可以提高类 FPN 范式的有效性。要在现有视觉模型中完全解决特征融合过程中的错位问题，还需要进一步的研究努力。
