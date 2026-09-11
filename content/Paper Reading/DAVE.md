---
created: 2024-06-25
title: "DAVE – A Detect-and-Verify Paradigm for Low-Shot Counting"
aliases:
  - "DAVE – A Detect-and-Verify Paradigm for Low-Shot Counting"
tags:
  - few-shot
  - counting
---

> [DAVE – A Detect-and-Verify Paradigm for Low-Shot Counting](https://doi.org/10.1109/CVPR52733.2024.02198)

## Abstract

Low-shot counters estimate the number of objects corresponding to a selected category, based on only few or no exemplars annotated in the image. The current state-of-the-art estimates the total counts as the sum over the object location density map, but does not provide individual object locations and sizes, which are crucial for many applications. This is addressed by detection-based counters, which, however fall behind in the total count accuracy. Furthermore, both approaches tend to overestimate the counts in the presence of other object classes due to many false positives. We propose DAVE, a low-shot counter based on a detect-and-verify paradigm, that avoids the aforementioned issues by first generating a high-recall detection set and then verifying the detections to identify and remove the outliers. This jointly increases the recall and precision, leading to accurate counts. DAVE outperforms the top densitybased counters by ∼20% in the total count MAE, it outperforms the most recent detection-based counter by ∼20% in detection quality and sets a new state-of-the-art in zero-shot as well as text-prompt-based counting.

## Introduction

在 Low-shot Counting(Few-shot & Zero-shot) 常见的方法有两类：

- Density-based
- Detection-based

Density-based 计数性能优于 Detection-based，但是不提供详细的输出，例如对象位置和大小。两种方法对于一些场景会出现计数错误情况，原因在于明确与泛化的权衡。通过提出 DAVE(detect-and-verify paradigm) 这一二阶段 pipeline 解决上述问题，在检测阶段生成一个高召回率（查的全）的检测结果，在验证阶段，剔除异常检测结果，最后得到一个正确的密度图用于计数。

![图 1](Paper%20Reading/attachments/DAVE/001.webp)

## Counting by detection and verification

![图 2](Paper%20Reading/attachments/DAVE/002.webp)

Given an input image$\mathit{I} \in \mathbb{R^{\mathit{H_0} \times \mathit{W_0} \times \mathrm{3}}}$and a set of$k$exemplar bounding boxes$\mathrm{B^E = \{\mathit{b_i}\}_{\mathit{i}=1:k}}$denoting object exemplars, a low-shot detection counter is required to report bounding boxes$\mathrm{B^P = \{\mathit{b_i}\}_{\mathit{i}=1:N_P}}$of all detected objects of the same category and their estimated count.

### Detection stage

The aim of this stage is to predict candidate bounding boxes$\mathrm{B^C = \{\mathit{b_i}\}_{\mathit{i}=1:N_C}}$with a high recall. Detection is thus split into first estimating the object centers$\mathrm{C} = \{(x^i_c, y^i_c)\}_{i=1:N_c}$, and then predicting the corresponding bounding box parameters. We re-purpose the architecture of the recent low-shot counter LOCA [6] for estimating the object location density map$\tilde{G}$, from which we obtain the center locations$C$by non-maxima suppression.

![图 3](Paper%20Reading/attachments/DAVE/003.webp)

Next, features are constructed for regressing the bounding box parameters for each detected center. the selected object category shape information is injected by fusing$\mathrm{f^1}$and the upscaled similarity tensor$\tilde{R}$using the feature fusion module (FFM), i.e., $\mathrm{\tilde{f} = FFM(f^1,\tilde{R})}$. The constructed features$\mathrm{\tilde{f}}$are then fed into a bounding box regression head$\Omega{(\cdot)}$, which predicts for each location a distance to the left, right, top and bottom bounding box edge of the underlying object.

### Verification stage

In practice, the candidate detections$B^C$retain a high recall, but are also contaminated by false positives. The goal of the verification stage is thus to increase the precision by analysing the appearance of the detections and rejecting the outliers. First, a verification feature vector$\mathrm{f^v_\mathit{i}}$is extracted for each detected bounding box$\mathit{b_i}$as follows. The backbone features$\mathrm{f^0}$are pooled into a feature tensor$\mathrm{f}_i \in \mathbb{R^{\mathit{s} \times \mathit{s} \times \mathit{d}}}$and transformed by a shallow network$\phi(\cdot)$. The verification features are also extracted for the annotated exemplars, leading to$N_C + k$features in total, i.e.,$\mathit{F}^\mathrm{V}  = {\{\mathrm{f^V_i}\}_{\mathit{i}=1:N_C + k}}$. The verification features are then clustered by unsupervised clustering. Specifically, spectral clustering [21] is applied to an affinity matrix computed from cosine similarities between pairs of features in$\mathit{F}^\mathrm{V}$, yielding several clusters. The final set of$\mathit{N_P}$object detections$\mathit{B^P} = {\{\mathit{b_i}\}_{\mathit{i}=1:N_P}}$. Finally, the density map$\tilde{G}$from the detection stage is updated by setting all values outside of the detected bounding boxes to zero, yielding$G$, from which the improved density-based count is estimated.

### Zero-shot and prompt-based adaptation

#### Zero shot counting

First, the location density prediction part is replaced by its zero-shot variant to account for the absence of exemplars. The only change in the verification stage is the cluster selection method: all clusters whose size is at least 45% of the largest cluster are kept as positive detections and the rest are identified as outliers.

#### Prompt-based counting

The only modification is the cluster selection protocol in the verification stage. The text prompt embedding is extracted by CLIP and compared to the CLIP embedding of each identified cluster. The latter is obtained by masking the image regions outside the bounding boxes corresponding to the cluster and computing the CLIP embedding. Cosine distances between the text embedding and individual cluster embeddings are computed, and clusters with less than 85% of the highest prompt-to-cluster similarity are identified as outliers.

### Train

Since DAVE employs LOCA for the initial density prediction, we use the publicly available pretrained version of LOCA, and train only the free parameters of the detection and verification stages in two phases.

In the first phase, the detection stage (i.e., the$\mathrm{FFM}$and$\Omega(\cdot)$) is trained by a bounding box loss evaluated on the available ground truth exemplar bounding boxes, i.e.,$\mathcal{L}_{box} = {\textstyle \sum_{i=1}^{k=3} 1 - \mathrm{GIoU(v(\mathit{x^c, y^c}), \mathit{b_i^{GT}})}}$, where$(\mathit{x_c^{(i)}, y_c^{(i)}})$are locations in the central regions of the ground truth bounding boxes$\mathit{b_i^{GT}}$and$\mathrm{GIoU(\cdot)}$is the generalized intersection over union.

In the second phase, the verification feature extraction network$\phi(\cdot)$is trained. Training examples are generated by stitching together a pair of images with annotated exemplar objects of different classes. The stitched image thus contains 2 × 3 = 6 bounding boxes, yielding two sets of features extracted by$\phi(\cdot)$, corresponding to the two sets of exemplars:$\{\mathrm{z}_j^1\}_{j=1:3}$and$\{\mathrm{z}_j^2\}_{j=1:3}$. The verification network$\phi(\cdot)$is then trained by a contrastive loss:

$$
\begin{align}
\mathcal{L}_{cos} = \begin{cases} 1 - c(z_{j_1}^{i^1}, z_{j_2}^{i^2}), & i_1 = i_2
\\
\text{max}(0, c(z_{j_1}^{i^1}, z_{j_2}^{i^2}) - \lambda), & else,\end{cases}
\end{align}
$$

where$c(z_{j_1}^{i^1}, z_{j_2}^{i^2})$is the cosine similarity between a pair of features, and$\lambda$is the margin.

## Experiments

### Density-based counting performance

#### Few-shot counting

![图 4](Paper%20Reading/attachments/DAVE/004.webp)

#### One-shot counting

![图 5](Paper%20Reading/attachments/DAVE/005.webp)

#### Prompt-based counting

![图 6](Paper%20Reading/attachments/DAVE/006.webp)

#### Zero-shot counting

![图 7](Paper%20Reading/attachments/DAVE/007.webp)

### Detection performance

#### Few-shot detection

![图 8](Paper%20Reading/attachments/DAVE/008.webp)

![图 9](Paper%20Reading/attachments/DAVE/009.webp)

#### Zero-shot detection

![图 10](Paper%20Reading/attachments/DAVE/010.webp)

#### Few-shot detection counting

![图 11](Paper%20Reading/attachments/DAVE/011.webp)

### Ablation study

#### Impact of mixed-class training

首先要验证的是，是否只需在具有多个物体类别的图像上进行训练，就能降低最先进方法的误报率。因此在多类别图像上重新训练了当前 SOTA: LOCA，将 FSCD147 训练图像与另一张包含不同类别物体的图像连接起来，作为困难负样本训练实例。用$LOCA_{mul}$表示这一版本，并将 CounTR 也纳入比较范围，因为它已经应用了这种训练设置。我们还构建了一个 FSCD147 子集，该子集由包含不同类别（图像取自 FSCD147 的测试和评估分割）对象的图像组成（记为$FSCD147_{mul}$），以揭示计数方法对其他类别对象的敏感性。

![图 12](Paper%20Reading/attachments/DAVE/012.webp)

#### Architecture design

![图 13](Paper%20Reading/attachments/DAVE/013.webp)

#### Limitations

DAVE outputs detections (i.e., bounding boxes), as well as total counts estimated from the density. To expose limitations, we inspect the discrepancy between the total count estimates and the number of detections with respect to the number of objects in the image (Figure 5). The discrepancy is most apparent for images with very large object counts, which typically contain many small objects packed together (i.e., extremely dense regions). Further error reductions are thus expected by improving DAVE detection stage in the presence of extreme density. The limitation is common to all low-shot counters, and we defer this to future research.

![图 14](Paper%20Reading/attachments/DAVE/014.webp)
