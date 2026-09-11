---
created: 2026-08-24
title: "Understanding Diffusion Models: A Unified Perspective"
aliases:
  - "Understanding Diffusion Models: A Unified Perspective"
tags:
  - generative-models
  - diffusion
---

> [Understanding Diffusion Models: A Unified Perspective](https://arxiv.org/abs/2208.11970)

## Part 1：生成模型基础与 VAE

本部分对应原论文的 “Introduction: Generative Models” 以及 “Background: ELBO, VAE, and Hierarchical VAE” 中进入 Hierarchical VAE 之前的内容。我们的目标不是先记住 VAE 的损失函数，而是沿着论文的问题链逐步回答：生成模型究竟要学习什么，隐藏变量为什么会带来难算的积分，VAE 为什么需要一个输出分布的 encoder，以及 ELBO 如何把无法直接优化的似然变成可以训练的目标。

---

### 1.1 为什么需要生成模型

#### 背景

在监督学习中，我们经常学习从输入到标签的条件分布，例如给定图片 $x$，预测类别 $y$ 的 $p_\theta(y\mid x)$。生成模型的方向不同：它试图学习数据本身是怎样出现的，也就是学习 $x$ 的分布 $p_\theta(x)$。如果模型真的掌握了这个分布，我们就不仅能判断一张图片“像不像训练数据”，还可以从分布中采样，生成以前没有出现过的新图片。

在理解“学习分布”之前，需要先区分概率论中的几个基本概念。

**随机变量**不是一个“会随机变化的普通变量”，而是一个把随机实验结果映射为数值的规则。例如掷骰子的结果可以用随机变量 $X$ 表示，$X$ 可能取 $1$ 到 $6$；一张图片也可以看成一个高维随机变量 $X$，它的每个分量对应一个像素通道的数值。大写 $X$ 常表示随机变量，小写 $x$ 常表示它的一次具体取值。

**概率**描述某个事件发生的可能性。例如 $P(X=6)$ 表示骰子结果为 $6$ 的概率。离散随机变量可以直接给每个取值分配概率；连续随机变量通常使用概率密度 $p(x)$。严格来说，连续变量在单个点上的概率通常为零，真正有意义的是区间概率，例如 $P(a\le X\le b)=\int_a^b p(x)\,dx$。机器学习文献常把“概率质量”和“概率密度”都简写成 $p(x)$，后文也沿用这种写法，但要记住二者在连续情形下并不完全相同。

**概率分布**描述随机变量所有可能取值及其相对可能性。合法分布必须非负，并把全部可能性的总量归一化为 $1$：离散情形满足 $\sum_x p(x)=1$，连续情形满足 $\int p(x)\,dx=1$。求和符号 $\sum$ 表示把所有离散取值的概率相加；积分符号 $\int$ 表示把连续空间中的概率密度累积起来；$dx$ 表示积分变量 $x$ 上一个无穷小的区间或体积元素。

真实世界的数据分布记为 $p_{\mathrm{data}}(x)$。我们只能看到从中抽取的有限训练样本，通常不能写出它的解析表达式。因此引入带参数的模型分布 $p_\theta(x)$，其中 $\theta$ 表示神经网络的权重等所有可训练参数。训练的目标是让 $p_\theta(x)$ 尽可能接近未知的 $p_{\mathrm{data}}(x)$。

#### 数学推导

假设训练集为 $\mathcal D=\{x^{(1)},x^{(2)},\ldots,x^{(N)}\}$，其中 $N$ 是样本数量，$x^{(n)}$ 是第 $n$ 个观测。最常见的假设是这些样本独立且来自同一个分布，简称 i.i.d.。在独立假设下，整套数据同时出现的概率或密度等于每个样本概率或密度的乘积：

$$
p_\theta(\mathcal D)=\prod_{n=1}^{N}p_\theta\!\left(x^{(n)}\right).
$$

这个公式描述训练集在模型 $p_\theta$ 下有多“合理”。符号 $\prod$ 表示连乘；$p_\theta(\mathcal D)$ 是整套训练数据的联合似然；$p_\theta(x^{(n)})$ 是模型分配给第 $n$ 个样本的概率质量或概率密度。之所以能够直接相乘，来源正是样本独立的假设。

当数据已经观察到以后，$x^{(n)}$ 是固定的，变化的是模型参数 $\theta$。这时，同一个表达式 $p_\theta(\mathcal D)$ 被称为参数 $\theta$ 的**似然**。概率和似然使用相同的数值表达式，但观察角度不同：概率固定参数、讨论数据会出现什么；似然固定数据、比较哪些参数更能解释数据。

最大似然估计选择让训练集似然最大的参数：

$$
\theta^\ast=\arg\max_\theta p_\theta(\mathcal D)
=\arg\max_\theta\prod_{n=1}^{N}p_\theta\!\left(x^{(n)}\right).
$$

这里 $\arg\max_\theta$ 返回的不是最大值本身，而是使目标达到最大值的参数；$\theta^\ast$ 表示训练后希望得到的最佳参数。这个目标之所以必要，是因为如果模型给真实样本很低的可能性，那么从模型采样时也很难产生与真实数据相似的结果。

实际训练通常最大化**对数似然**。因为对数函数 $\log u$ 在 $u>0$ 时严格单调递增，若 $a>b>0$，就一定有 $\log a>\log b$，所以对目标取对数不会改变使它最大的参数。利用对数把乘法变成加法的性质 $\log(ab)=\log a+\log b$，可得

$$
\theta^\ast
=\arg\max_\theta\sum_{n=1}^{N}\log p_\theta\!\left(x^{(n)}\right).
$$

这个公式仍然是最大似然估计，只是把许多很小概率的连乘变成了对数概率的求和。这样做有三个作用：第一，避免大量小数相乘造成计算机下溢；第二，加法比长乘积更容易求导和实现小批量训练；第三，后续 ELBO 的推导天然围绕 $\log p_\theta(x)$ 展开。

优化器通常执行最小化，因此代码里经常最小化平均负对数似然

$$
\mathcal L_{\mathrm{NLL}}(\theta)
=-\frac{1}{N}\sum_{n=1}^{N}\log p_\theta\!\left(x^{(n)}\right).
$$

$\mathcal L_{\mathrm{NLL}}$ 是 negative log-likelihood，简称 NLL；前面的负号把“最大化对数似然”改写成“最小化损失”；除以 $N$ 只是取样本平均，不改变最优参数。需要注意，负对数似然不是另外发明的经验损失，它就是最大似然目标的等价写法。

#### 公式解释

本章最核心的单样本目标是 $\log p_\theta(x)$。其中 $x$ 是一个已经观察到的数据样本，$p_\theta$ 是模型分布，$\theta$ 是要学习的参数，$\log$ 是自然对数。这个量越大，说明模型越愿意相信样本 $x$ 来自自己所描述的分布。

论文从“给观测样本高似然”的生成模型路线出发。后面引入 VAE 和 Diffusion，并不是放弃最大似然，而是因为复杂数据的 $\log p_\theta(x)$ 很难直接计算，需要寻找一个可以优化的替代目标。

#### 直觉理解

可以把分布想象成一张覆盖整个数据空间的“地形图”，地势越高的位置代表概率密度越大。真实图片集中在某些复杂区域，随机像素噪声则位于另一些区域。训练生成模型就是调整 $\theta$，让训练图片所在的位置逐渐变高，同时由于整个分布必须归一化，总概率质量会相应从不合理区域移走。

最大化 $\log p_\theta(x)$ 不是要求模型背下每一张训练图片，而是要求它学习一套能够共同解释训练样本的概率规律。只有当模型学到可泛化的结构，从这个分布重新采样时才可能得到新的合理样本。

#### 本章总结

- **本章解决了什么问题：** 明确了生成模型学习的是数据分布 $p_\theta(x)$，训练目标来自最大似然估计，而不是凭经验随意指定的损失。
- **核心公式：** $\theta^\ast=\arg\max_\theta\sum_{n=1}^{N}\log p_\theta(x^{(n)})$。
- **需要记住的直觉：** 训练就是让模型分布在真实数据附近分配更高的概率密度；取对数只改变计算方式，不改变最优解。
- **与下一章的关系：** 直接给复杂图片建立 $p_\theta(x)$ 很困难。下一章会引入隐藏变量 $z$，把“直接描述图片”改写成“先产生潜在因素，再由潜在因素生成图片”。

---

### 1.2 Latent Variable Model：为什么引入隐藏变量

#### 背景

图片的像素维度很高，但决定图片内容的因素往往更抽象。例如一张人脸可以由身份、姿态、光照、表情和背景共同决定。我们观察到的是最终像素 $x$，这些背后的生成因素却没有直接出现在数据集中。概率模型把这种不可直接观测、但参与生成数据的变量称为**隐藏变量**或**潜变量**，记为 $z$。

图式 $z\rightarrow x$ 表示一个生成方向：模型先得到 $z$，再根据 $z$ 生成 $x$。箭头表达依赖关系，而不一定表示一个完全确定的函数。如果固定 $z$ 后仍然允许多种可能的 $x$，这种不确定性就由条件分布 $p_\theta(x\mid z)$ 描述。

理解隐藏变量模型需要三个概率概念。

**联合概率** $p(x,z)$ 描述 $x$ 与 $z$ 同时取某些值的可能性。**条件概率** $p(x\mid z)$ 描述已知 $z$ 后 $x$ 的分布。**先验分布** $p(z)$ 描述在看到具体数据 $x$ 之前，我们认为潜变量 $z$ 应该如何分布。这里“先验”不是说它永远正确，而是说它位于生成过程的起点。

#### 数学推导

条件概率的定义为 $p(x\mid z)=p(x,z)/p(z)$，前提是 $p(z)>0$。把等式两边乘以 $p(z)$，得到概率的乘法法则

$$
p_\theta(x,z)=p_\theta(x\mid z)\,p(z).
$$

这个公式描述完整的生成过程。右侧的 $p(z)$ 负责产生潜变量，$p_\theta(x\mid z)$ 负责在给定潜变量后产生观测；二者相乘得到 $x$ 与 $z$ 同时出现的联合分布。下标 $\theta$ 通常放在需要学习的 decoder 分布上；先验 $p(z)$ 常被预先选定，所以这里没有写 $\theta$。

训练集只有 $x$，没有对应的 $z$。为了得到单独的 $x$ 的分布，必须把所有可能的 $z$ 都考虑进去。离散潜变量用求和，连续潜变量用积分。VAE 通常使用连续潜变量，因此

$$
p_\theta(x)
=\int p_\theta(x,z)\,dz
=\int p_\theta(x\mid z)p(z)\,dz.
$$

第一步称为对 $z$ **边缘化**：联合分布中包含了 $x$ 和 $z$，对全部 $z$ 积分后，$z$ 被消去，只留下 $x$ 的边缘分布。第二步代入了刚才的联合分布分解。$dz$ 不是一个需要单独相乘的普通变量，它表示在 $z$ 空间上累积概率密度的体积元素。如果 $z$ 有上百个维度，这实际上是一个上百维积分。

这个积分可以理解为加权混合：对每一种潜在状态 $z$，先计算它出现的密度 $p(z)$，再计算它生成当前 $x$ 的密度 $p_\theta(x\mid z)$，最后把所有可能路径的贡献相加。它之所以难算，不是因为积分符号本身神秘，而是因为潜空间通常是高维连续空间，可能的 $z$ 有无限多个，并且 $p_\theta(x\mid z)$ 由非线性神经网络参数化，通常没有可直接求出的解析积分。

看到某个 $x$ 以后，我们还会反过来问：“它最可能由哪些 $z$ 生成？”这对应后验分布。由 Bayes 公式，

$$
p_\theta(z\mid x)
=\frac{p_\theta(x,z)}{p_\theta(x)}
=\frac{p_\theta(x\mid z)p(z)}
{\int p_\theta(x\mid z')p(z')\,dz'}.
$$

分子衡量特定 $z$ 与观测 $x$ 的共同可能性；分母 $p_\theta(x)$ 负责归一化，使后验对所有 $z$ 的积分为 $1$。分母中的 $z'$ 只是积分用的临时变量，特意写成 $z'$ 是为了避免和分子中正在评估的 $z$ 混淆。问题在于这个分母正是刚才难算的高维积分，因此真实后验 $p_\theta(z\mid x)$ 通常也难以直接计算。

#### 公式解释

生成方向 $z\rightarrow x$、联合分解 $p_\theta(x,z)=p_\theta(x\mid z)p(z)$ 和边缘化 $p_\theta(x)=\int p_\theta(x,z)\,dz$ 描述的是同一个模型的三个角度：

- $z\rightarrow x$ 是结构图，说明谁先生成、谁依赖谁。
- $p_\theta(x,z)=p_\theta(x\mid z)p(z)$ 是概率因子分解，说明联合概率如何计算。
- $p_\theta(x)=\int p_\theta(x,z)\,dz$ 把不可观测的 $z$ 消去，得到训练最大似然真正需要的观测分布。

隐藏变量缓解了“如何表达复杂数据”的问题，却引入了“如何对所有隐藏状态积分”的计算问题。VAE 的核心正是处理这个新问题。

#### 直觉理解

想象一个画图程序先随机选择“圆形还是方形”“颜色”“大小”和“位置”，再把这些属性渲染成像素。属性集合就是 $z$，渲染结果就是 $x$。同一张近似的图片可能由许多略有不同的属性组合生成，所以要求 $p_\theta(x)$ 时不能只选择一个 $z$，而要把所有可能生成路径的贡献加起来，这就是边缘化积分。

隐藏变量的价值在于把高维像素规律转化为较低维、较有结构的生成因素；代价是这些因素没有标签，我们既不知道每个样本对应什么 $z$，也难以精确完成对所有 $z$ 的积分。

#### 本章总结

- **本章解决了什么问题：** 用潜变量 $z$ 表达观测数据背后的生成因素，并说明了生成模型的联合分布、边缘分布和后验分布。
- **核心公式：** $p_\theta(x)=\int p_\theta(x\mid z)p(z)\,dz$。
- **需要记住的直觉：** 一个观测 $x$ 的总概率等于所有潜在生成路径的概率贡献之和；潜变量让模型更容易表达，却让似然和真实后验更难计算。
- **与下一章的关系：** 下一章引入 VAE 的 encoder $q_\phi(z\mid x)$，用一个可训练、可采样的分布去近似难以计算的真实后验 $p_\theta(z\mid x)$。

---

### 1.3 VAE：用可学习分布近似隐藏变量

#### 背景

隐藏变量模型有两个方向。生成方向从 $z$ 到 $x$，由 $p_\theta(x\mid z)$ 描述；推断方向从已经观察到的 $x$ 反推 $z$，理论上应由真实后验 $p_\theta(z\mid x)$ 描述。上一章已经看到，真实后验的分母包含难算的 $p_\theta(x)$，所以我们无法直接把它当作 encoder。

VAE 增加一个参数为 $\phi$ 的近似分布 $q_\phi(z\mid x)$。它接收 $x$，输出一个关于 $z$ 的分布，并尝试逼近 $p_\theta(z\mid x)$。字母 $q$ 用来提醒我们：它是为推断而引入的近似分布，不是原始生成模型的联合分布 $p_\theta(x,z)$。

VAE 中的两个网络可以写为：

- Encoder：$q_\phi(z\mid x)$，从观测 $x$ 推断潜变量 $z$ 的近似后验。
- Decoder：$p_\theta(x\mid z)$，从潜变量 $z$ 给出观测 $x$ 的条件生成分布。

需要特别注意，decoder 神经网络的计算本身可以是确定的，但符号 $p_\theta(x\mid z)$ 表示一个概率分布。网络通常输出这个分布的参数，例如 Gaussian 分布的均值，或 Bernoulli 分布的像素概率，而不是在数学上只输出一个没有不确定性的固定图片。

#### 数学推导

为了理解 VAE 常用的分布形式，先看一维 Gaussian（高斯或正态）分布。若 $u\sim\mathcal N(\mu,\sigma^2)$，其概率密度为

$$
\mathcal N(u;\mu,\sigma^2)
=\frac{1}{\sqrt{2\pi\sigma^2}}
\exp\!\left(-\frac{(u-\mu)^2}{2\sigma^2}\right).
$$

$u$ 是随机变量的一次取值；$\mu$ 是均值，控制钟形曲线的中心；$\sigma^2$ 是方差，衡量分布的分散程度；$\sigma=\sqrt{\sigma^2}$ 是标准差，与 $u$ 使用相同单位；$\exp(v)=e^v$ 是指数函数。距离均值越远，负的平方项越大，密度越低。前面的归一化系数保证整条曲线下的面积为 $1$。

方差的定义是 $\operatorname{Var}(u)=\mathbb E[(u-\mu)^2]$。期望符号 $\mathbb E$ 表示按概率加权的平均；平方使正负偏差不会互相抵消。标准差是方差的平方根，所以当采样公式中要把标准正态缩放到目标分布时，乘的是 $\sigma$，而不是 $\sigma^2$。

对于多维潜变量 $z=(z_1,\ldots,z_d)$，VAE 常让 encoder 输出每个维度的均值与方差：

$$
q_\phi(z\mid x)
=\mathcal N\!\left(
z;\mu_\phi(x),
\operatorname{diag}\!\big(\sigma_\phi^2(x)\big)
\right).
$$

$\mu_\phi(x)$ 是 $d$ 维均值向量；$\sigma_\phi^2(x)$ 是 $d$ 维方差向量；$\operatorname{diag}(\sigma_\phi^2(x))$ 表示把这些方差放在协方差矩阵的对角线上，非对角元素为零。这个“对角协方差”假设意味着给定 $x$ 后，各潜变量维度在近似后验中条件独立。它降低了表达能力，却让采样和 KL 计算非常简单。实践中网络常输出 $\log\sigma_\phi^2(x)$，因为对数方差可以取任意实数，再通过指数恢复正方差，从而避免网络直接产生非法的负方差。

Encoder 为什么不能只输出一个固定点 $z=f_\phi(x)$？原因有三层。第一，真实后验 $p_\theta(z\mid x)$ 本来就是一个分布：同一个观测可能由多个潜在解释产生，固定点无法表达这种不确定性。第二，后面的 ELBO 需要对 $q_\phi(z\mid x)$ 求期望，分布形式使我们能够从中采样并近似这个期望。第三，如果每个样本只对应一个互不相关的孤立点，训练集之间的潜空间可能出现大量空洞，从任意位置采样时 decoder 未必知道如何生成合理数据；用分布表示并与共同先验匹配，会鼓励潜空间更加连续。

VAE 通常选择标准多元 Gaussian 作为先验：

$$
p(z)=\mathcal N(z;0,I).
$$

$0$ 表示每个潜变量维度的均值都是零；$I$ 是单位矩阵，表示每个维度方差为 $1$，不同维度协方差为 $0$。因此“标准”指零均值、单位方差，而不是“所有 VAE 必须如此”。选择它主要有四个实际原因：容易采样；每个坐标使用统一尺度；与对角 Gaussian 近似后验之间的 KL 有解析解；训练完成后有一个明确且简单的生成入口。先验也可以换成更复杂的分布，但相应的推导、计算和采样会更复杂。

VAE 的完整概率角色现在可以区分为

$$
\underbrace{p(z)p_\theta(x\mid z)}_{\text{生成模型}}
\qquad\text{与}\qquad
\underbrace{q_\phi(z\mid x)}_{\text{近似推断模型}}.
$$

左侧先采样 $z$ 再生成 $x$，用于定义模型和生成新样本；右侧从已有 $x$ 推断 $z$，主要用于训练。两者暂时只是结构，还缺少一个原则来决定如何同时学习 $\theta$ 与 $\phi$。这个原则就是下一章推导的 ELBO。

#### 公式解释

符号 $p_\theta(x\mid z)$ 中的 $\theta$ 是 decoder 参数；$q_\phi(z\mid x)$ 中的 $\phi$ 是 encoder 参数。真实后验 $p_\theta(z\mid x)$ 与近似后验 $q_\phi(z\mid x)$ 不能混为一谈：前者由生成模型和 Bayes 公式决定，但通常难算；后者由 encoder 直接给出，容易计算和采样，但需要通过训练逼近前者。

$p(z)=\mathcal N(z;0,I)$ 是生成前的先验，$q_\phi(z\mid x)$ 是观察数据后的近似后验。训练时可以从近似后验取得与当前样本相关的潜变量；生成时没有输入 $x$，所以直接从先验取得潜变量，再送入 decoder。

#### 直觉理解

普通 autoencoder 像是让 encoder 在地图上为每张图片插一根针：输入图片对应一个确定坐标。VAE 的 encoder 更像是在地图上画一个带中心和范围的云团：$\mu_\phi(x)$ 决定云团中心，$\sigma_\phi(x)$ 决定各方向的宽度，从云团中不同位置采样仍应能够重建同一类输入。

标准 Gaussian 先验则像一片预先规定的主要活动区域。后续的 KL 项会让每个样本的云团不要无约束地跑到极远处。这样，生成阶段从这片区域随机取点时，更可能落在 decoder 训练过的范围内。

#### 本章总结

- **本章解决了什么问题：** 用可训练的 $q_\phi(z\mid x)$ 近似难算的真实后验，并明确了 encoder、decoder、先验与后验各自的角色。
- **核心公式：** $q_\phi(z\mid x)=\mathcal N(z;\mu_\phi(x),\operatorname{diag}(\sigma_\phi^2(x)))$，以及 $p(z)=\mathcal N(z;0,I)$。
- **需要记住的直觉：** VAE 的 encoder 为每个输入输出潜空间中的“概率云团”，而不是孤立坐标；decoder 接收其中的样本并描述数据分布。
- **与下一章的关系：** 仅仅定义 $q_\phi(z\mid x)$ 还不知道怎样训练它。下一章将从 $\log p_\theta(x)$ 出发，完整推导 ELBO，并说明 VAE 的两项训练目标为什么必然出现。

---

### 1.4 ELBO：从难算的似然到可训练目标

#### 背景

我们希望最大化 $\log p_\theta(x)$，但隐藏变量模型要求先计算 $p_\theta(x)=\int p_\theta(x,z)\,dz$。这个高维积分通常不可解。VAE 的关键不是直接猜一个“重建损失加 KL 损失”，而是引入容易采样的 $q_\phi(z\mid x)$，从最大似然目标一步步推导出一个可计算的下界。

在推导前需要补充期望、Jensen 不等式和 KL 散度。

如果 $z$ 服从密度 $q_\phi(z\mid x)$，函数 $f(z)$ 的期望为 $\mathbb E_{q_\phi(z\mid x)}[f(z)]=\int q_\phi(z\mid x)f(z)\,dz$。它描述反复从 $q_\phi(z\mid x)$ 采样并计算 $f(z)$ 时的长期平均。后文把较长的下标简写为 $\mathbb E_q$，但所指分布仍然是 $q_\phi(z\mid x)$。

对数函数是**凹函数**。凹函数的图像位于任意两点连线的上方，因此对任意正数 $a,b$ 和 $0\le\lambda\le1$，有 $\log(\lambda a+(1-\lambda)b)\ge\lambda\log a+(1-\lambda)\log b$。左侧先对 $a,b$ 做加权平均再取对数，右侧先取对数再做加权平均。把两个点推广为随机变量 $Y>0$ 的所有可能取值，就得到 Jensen 不等式

$$
\log\mathbb E[Y]\ge\mathbb E[\log Y].
$$

例如 $Y$ 以相同概率取 $1$ 和 $9$，左侧是 $\log 5$，右侧是 $(\log1+\log9)/2=\log3$，确有 $\log5\ge\log3$。在 ELBO 推导中，这个不等式把“期望外面的对数”移动到“期望里面”，代价是等号通常变成下界。

两个分布 $q(z)$ 与 $p(z)$ 的 KL 散度定义为

$$
D_{\mathrm{KL}}(q\|p)
=\mathbb E_{q(z)}
\left[\log\frac{q(z)}{p(z)}\right].
$$

它按 $q$ 的取值频率，平均比较 $q(z)$ 与 $p(z)$ 的对数密度比。顺序很重要，通常 $D_{\mathrm{KL}}(q\|p)\ne D_{\mathrm{KL}}(p\|q)$，所以它不是普通几何距离。KL 散度非负可以由 Jensen 不等式推出：

$$
D_{\mathrm{KL}}(q\|p)
=-\mathbb E_q\!\left[\log\frac{p(z)}{q(z)}\right]
\ge-\log\mathbb E_q\!\left[\frac{p(z)}{q(z)}\right]
=-\log\int p(z)\,dz=0.
$$

第一步只是把对数比值取倒数并加负号；第二步对凹函数 $\log$ 使用 Jensen 后再乘 $-1$，不等号方向随之反转；第三步按照期望定义，$q(z)$ 与分母中的 $q(z)$ 抵消；最后合法分布的积分为 $1$，而 $\log1=0$。当 $q=p$（几乎处处相等）时 KL 为零。这个结论会直接解释 ELBO 为什么是下界。

#### 数学推导

##### 第一步：把边缘似然改写成关于近似后验的期望

从边缘化公式开始：

$$
\log p_\theta(x)
=\log\int p_\theta(x,z)\,dz.
$$

在积分内部乘以 $q_\phi(z\mid x)/q_\phi(z\mid x)=1$，得到

$$
\begin{aligned}
\log p_\theta(x)
&=\log\int q_\phi(z\mid x)
\frac{p_\theta(x,z)}{q_\phi(z\mid x)}\,dz\\
&=\log\mathbb E_{q_\phi(z\mid x)}
\left[
\frac{p_\theta(x,z)}{q_\phi(z\mid x)}
\right].
\end{aligned}
$$

为什么要特意引入 $q/q$？它没有改变原积分的值，却把被积函数整理成“分布 $q_\phi(z\mid x)$ 乘某个函数”的形式，而这正是期望的定义。于是我们可以通过从 encoder 分布采样来估计它。技术上还要求 $q_\phi(z\mid x)$ 在 $p_\theta(x,z)$ 有贡献的区域不能为零；常用 Gaussian 具有全空间支持，通常满足这一需要。

令正随机变量 $Y=p_\theta(x,z)/q_\phi(z\mid x)$，对上式应用 Jensen 不等式：

$$
\log p_\theta(x)
\ge
\mathbb E_{q_\phi(z\mid x)}
\left[
\log\frac{p_\theta(x,z)}{q_\phi(z\mid x)}
\right]
\equiv\mathcal L_{\mathrm{ELBO}}(x;\theta,\phi).
$$

右侧就是 Evidence Lower Bound，证据下界。Evidence 指观测 $x$ 的边缘似然；Lower Bound 表示它不超过 $\log p_\theta(x)$。$\mathcal L_{\mathrm{ELBO}}$ 是这个下界的名称，不是额外假设出来的新量。

##### 第二步：说明下界与真实似然究竟差在哪里

仅凭 Jensen 可以知道“不超过”，但还看不出差距是什么。利用 Bayes 关系 $p_\theta(x,z)=p_\theta(z\mid x)p_\theta(x)$，从 ELBO 开始：

$$
\begin{aligned}
\mathcal L_{\mathrm{ELBO}}
&=\mathbb E_q\left[
\log\frac{p_\theta(z\mid x)p_\theta(x)}{q_\phi(z\mid x)}
\right]\\
&=\log p_\theta(x)
+\mathbb E_q\left[
\log\frac{p_\theta(z\mid x)}{q_\phi(z\mid x)}
\right]\\
&=\log p_\theta(x)
-D_{\mathrm{KL}}\!\left(
q_\phi(z\mid x)\|p_\theta(z\mid x)
\right).
\end{aligned}
$$

第一行代入联合分布分解；第二行使用 $\log(ab)=\log a+\log b$，并注意 $\log p_\theta(x)$ 与积分变量 $z$ 无关，所以它的期望仍是自身；第三行根据 KL 定义，把对数比值的方向反过来并出现负号。移项得到论文强调的关系：

$$
\log p_\theta(x)=
\mathcal L_{\mathrm{ELBO}}(x;\theta,\phi)
+D_{\mathrm{KL}}\!\left(
q_\phi(z\mid x)\|p_\theta(z\mid x)
\right).
$$

这条公式精确描述了 ELBO 与真实对数似然的差距。因为 KL 总是非负，ELBO 必然不高于 $\log p_\theta(x)$；当 $q_\phi(z\mid x)$ 与真实后验完全一致时，KL 为零，下界变成等号。

固定生成模型参数 $\theta$ 时，$\log p_\theta(x)$ 不依赖 encoder 参数 $\phi$。因此，增大 ELBO 等价于减小近似后验与真实后验之间的 KL。这里必须强调“固定 $\theta$”：联合训练 VAE 时 $\log p_\theta(x)$ 会随 decoder 参数 $\theta$ 改变。最大化 ELBO 一方面让 $q_\phi$ 更好地近似后验，另一方面也提高生成模型对数据的下界似然。

##### 第三步：把 ELBO 拆成 VAE 可以理解的两项

代入联合分布 $p_\theta(x,z)=p_\theta(x\mid z)p(z)$：

$$
\begin{aligned}
\mathcal L_{\mathrm{ELBO}}
&=\mathbb E_q\left[
\log\frac{p_\theta(x\mid z)p(z)}{q_\phi(z\mid x)}
\right]\\
&=\mathbb E_q[\log p_\theta(x\mid z)]
+\mathbb E_q\left[
\log\frac{p(z)}{q_\phi(z\mid x)}
\right]\\
&=\mathbb E_q[\log p_\theta(x\mid z)]
-D_{\mathrm{KL}}\!\left(q_\phi(z\mid x)\|p(z)\right).
\end{aligned}
$$

第一项 $\mathbb E_q[\log p_\theta(x\mid z)]$ 是 reconstruction term，即期望重建对数似然。从 $q_\phi(z\mid x)$ 取得的 $z$ 应让 decoder 给原始 $x$ 较高的条件似然，因此它迫使 $z$ 保存重建 $x$ 所需的信息。

第二项前面有负号，而 KL 本身非负。最大化 ELBO 时就会最小化 $D_{\mathrm{KL}}(q_\phi(z\mid x)\|p(z))$，让样本相关的近似后验不要偏离统一先验太远。这一项称为 prior matching term。它使生成阶段从 $p(z)$ 采到的点更可能位于 decoder 在训练中见过的区域。

两项不是随意拼接：它们是把联合概率的乘法法则代入 ELBO 后，由对数把乘法拆成加法、再由 KL 定义自然得到的。

##### 第四步：把期望变成可以计算的样本平均

重建项仍然包含对 $q_\phi(z\mid x)$ 的期望，但现在我们能够从这个分布采样。若独立采样 $L$ 个潜变量 $z^{(1)},\ldots,z^{(L)}\sim q_\phi(z\mid x)$，Monte Carlo 估计为

$$
\mathbb E_q[\log p_\theta(x\mid z)]
\approx\frac{1}{L}\sum_{l=1}^{L}
\log p_\theta\!\left(x\mid z^{(l)}\right).
$$

$L$ 是采样次数；$l$ 是样本编号；$z^{(l)}$ 表示第 $l$ 次潜变量采样。大数定律告诉我们，$L$ 增大时样本平均会接近期望。训练 VAE 时常对每个 $x$ 只采一个 $z$，也就是 $L=1$；单次估计有噪声，但在许多训练批次上平均后仍可提供有效梯度，而且计算便宜。

##### 第五步：重参数化让梯度穿过采样过程

若直接写 $z\sim q_\phi(z\mid x)$，采样结果会随 $\phi$ 改变，但“从分布中随机抽一个值”不是普通的确定性网络节点，无法直接沿这条写法对 $\phi$ 做反向传播。对 Gaussian 近似后验，可以先采样与参数无关的标准噪声，再做确定性变换：

$$
\epsilon\sim\mathcal N(0,I),
\qquad
z=\mu_\phi(x)+\sigma_\phi(x)\odot\epsilon.
$$

$\epsilon$ 是辅助噪声；$\odot$ 表示逐元素相乘；$\mu_\phi(x)$ 与 $\sigma_\phi(x)$ 都是 encoder 的确定性输出。这个变换确实产生目标 Gaussian，因为 $\mathbb E[\epsilon]=0$、$\operatorname{Var}(\epsilon)=I$，所以 $\mathbb E[z]=\mu_\phi(x)$，而每个维度都有 $\operatorname{Var}(z_j)=\sigma_{\phi,j}^2(x)\operatorname{Var}(\epsilon_j)=\sigma_{\phi,j}^2(x)$。因此 $z$ 的均值和方差恰好与 $q_\phi(z\mid x)$ 的定义一致。

重参数化没有删除随机性，而是把随机性集中到不依赖模型参数的 $\epsilon$ 中。给定一次采到的 $\epsilon$ 后，$z$ 对 $\mu_\phi(x)$ 和 $\sigma_\phi(x)$ 都是可微的确定性函数，decoder 的重建梯度便可以继续传回 encoder。

##### 第六步：说明 Gaussian KL 项如何直接计算

当一维近似后验为 $q(z)=\mathcal N(\mu,\sigma^2)$，先验为 $p(z)=\mathcal N(0,1)$ 时，把两个 Gaussian 密度代入对数比可得

$$
\log\frac{q(z)}{p(z)}
=-\log\sigma
-\frac{(z-\mu)^2}{2\sigma^2}
+\frac{z^2}{2}.
$$

标准先验和近似后验中共同的 $-\tfrac12\log(2\pi)$ 已经抵消。对 $q$ 求期望时，按方差定义有 $\mathbb E_q[(z-\mu)^2]=\sigma^2$；再将 $z=(z-\mu)+\mu$ 展开平方，可得 $\mathbb E_q[z^2]=\mu^2+\sigma^2$。代回上式：

$$
D_{\mathrm{KL}}\!\left(
\mathcal N(\mu,\sigma^2)\|\mathcal N(0,1)
\right)
=\frac12\left(\mu^2+\sigma^2-1-\log\sigma^2\right).
$$

对角 Gaussian 的各维度相互独立，联合密度的对数比会变成各维度对数比的和，所以多维结果是

$$
D_{\mathrm{KL}}\!\left(q_\phi(z\mid x)\|p(z)\right)
=\frac12\sum_{j=1}^{d}
\left(
\mu_j^2+\sigma_j^2-1-\log\sigma_j^2
\right).
$$

$j$ 是潜变量维度编号，$d$ 是潜变量维数。这个解析式只依赖 encoder 输出的 $\mu_j$ 和 $\sigma_j^2$，所以 KL 项不需要额外 Monte Carlo 采样。若 $\mu_j=0$ 且 $\sigma_j^2=1$，该维度的近似后验等于标准先验，对应 KL 正好为零。

##### 第七步：得到训练代码中的 VAE loss

理论上联合优化 encoder 参数 $\phi$ 和 decoder 参数 $\theta$ 时要最大化 ELBO：

$$
\max_{\theta,\phi}
\left\{
\mathbb E_{q_\phi(z\mid x)}[\log p_\theta(x\mid z)]
-D_{\mathrm{KL}}\!\left(q_\phi(z\mid x)\|p(z)\right)
\right\}.
$$

深度学习优化器通常最小化损失，因此取负号得到完全等价的形式：

$$
\mathcal L_{\mathrm{VAE}}
=-\mathbb E_{q_\phi(z\mid x)}[\log p_\theta(x\mid z)]
+D_{\mathrm{KL}}\!\left(q_\phi(z\mid x)\|p(z)\right).
$$

第一项是负的重建对数似然，代码里常称 reconstruction loss；第二项是 KL loss。最大化“重建对数似然减 KL”与最小化“负重建对数似然加 KL”只是符号约定不同。

重建损失的具体形式来自对 decoder 分布的选择。例如若假设 $p_\theta(x\mid z)=\mathcal N(x;\hat x_\theta(z),\sigma_x^2I)$，其中 decoder 输出均值 $\hat x_\theta(z)$，并把方差 $\sigma_x^2$ 固定，那么对 Gaussian 密度取负对数可得

$$
-\log p_\theta(x\mid z)
=\frac{1}{2\sigma_x^2}
\|x-\hat x_\theta(z)\|_2^2+C.
$$

$\|x-\hat x_\theta(z)\|_2^2$ 是所有维度误差平方之和；$C$ 收集与可训练均值无关的归一化常数。固定 $\sigma_x^2$ 时，最小化这个负对数似然就等价于最小化均方误差，差别只在固定比例和常数。因此 MSE 不是凭空选择的，它对应一个固定方差 Gaussian 观测模型。若选择 Bernoulli 观测模型，则会得到交叉熵形式。

一次完整训练迭代由此自然产生：encoder 根据 $x$ 输出 $\mu_\phi(x)$ 和 $\log\sigma_phi^2(x)$；采样 $\epsilon\sim\mathcal N(0,I)$ 并通过重参数化得到 $z$；decoder 用 $z$ 参数化 $p_\theta(x\mid z)$；计算重建负对数似然与解析 KL；将二者相加并反向传播，同时更新 $\theta$ 与 $\phi$。

训练完成后，重建已有数据时使用 $q_\phi(z\mid x)$ 取得 $z$；生成新数据时没有输入 $x$，直接采样 $z\sim p(z)=\mathcal N(0,I)$，再从 $p_\theta(x\mid z)$ 采样或取其均值。KL 先验匹配项正是训练路径与生成路径能够衔接的原因。

#### 公式解释

ELBO 的三种等价视角分别回答不同问题：

- $\mathcal L_{\mathrm{ELBO}}=\mathbb E_q[\log(p_\theta(x,z)/q_\phi(z\mid x))]$ 来自 Jensen 推导，说明下界如何构造。
- $\log p_\theta(x)=\mathcal L_{\mathrm{ELBO}}+D_{\mathrm{KL}}(q_\phi(z\mid x)\|p_\theta(z\mid x))$ 说明下界与真实似然之间的缺口是什么。
- $\mathcal L_{\mathrm{ELBO}}=\mathbb E_q[\log p_\theta(x\mid z)]-D_{\mathrm{KL}}(q_\phi(z\mid x)\|p(z))$ 说明 VAE 实际可以优化的两个组成部分。

Reconstruction term 保证 $z$ 对当前 $x$ 有解释力，prior matching term 保证这些 $z$ 的总体位置与生成时使用的先验相容。只要忘记其中任意一项，VAE 的训练逻辑都会断裂：没有重建项，$z$ 不需要包含数据内容；没有 KL 项，encoder 可以把样本编码到任意互不相连的区域，从简单先验采样就没有可靠依据。

#### 直觉理解

可以把 $\log p_\theta(x)$ 想成一份无法直接核算的“真实总成绩”。ELBO 是一份可以计算的保守成绩，二者相差近似后验与真实后验之间的 KL。提升 ELBO 时，我们既在提高可证明的似然下界，也在固定 decoder 时迫使 encoder 的推断接近真正的 Bayes 后验。

重建项与先验匹配项分别像两股力量：重建项要求每个潜变量云团携带足够信息，能够找回原数据；KL 项要求云团不要随意散落，而要与统一的标准 Gaussian 坐标系保持联系。VAE 的可生成性来自两者的共同作用，而不是只来自“加了随机噪声”。

#### 本章总结

- **本章解决了什么问题：** 从难以直接计算的 $\log p_\theta(x)$ 完整推导出 ELBO，并进一步得到可用采样、解析 KL 和反向传播训练的 VAE loss。
- **核心公式：** $\mathcal L_{\mathrm{ELBO}}=\mathbb E_{q_\phi(z\mid x)}[\log p_\theta(x\mid z)]-D_{\mathrm{KL}}(q_\phi(z\mid x)\|p(z))$。
- **需要记住的直觉：** ELBO 不是经验拼装的损失，而是对数似然的可计算下界；重建项保存信息，KL 项让训练时的潜变量分布与生成时的先验接轨。
- **与下一章的关系：** 普通 VAE 只有一层潜变量 $z\rightarrow x$。Part 2 将沿论文顺序把它扩展为多层潜变量的 Hierarchical VAE，再通过三个特殊假设把这种层次模型变成 Diffusion Model。

## Part 2：从 VAE 到 Diffusion

Part 1 只有一个潜变量层：模型从先验取得 $z$，再由 $z$ 生成观测 $x$。原论文接下来先把单层 VAE 扩展为 Hierarchical VAE，再对这种层次模型加入三个限制，得到 Variational Diffusion Model。本部分的重点是完成模型结构的转换；forward process 中 Gaussian 加噪公式的逐步计算将在 Part 3 展开。

---

### 2.1 Hierarchical VAE：从一个潜变量到一条潜变量链

#### 背景

普通 VAE 的生成过程是 $z\rightarrow x$：先从先验 $p(z)$ 采样一个潜变量，再由 decoder 分布 $p_\theta(x\mid z)$ 生成数据。这个结构把所有抽象层次都压在单个 $z$ 中。对于复杂数据，一层潜变量有时需要同时表达非常不同的因素，例如整体类别、物体布局、局部纹理和像素细节。

Hierarchical Variational Autoencoder（HVAE，层次变分自编码器）把一个潜变量扩展为多个层级：

$$
z_T\rightarrow z_{T-1}\rightarrow\cdots\rightarrow z_2\rightarrow z_1\rightarrow x.
$$

$T$ 是潜变量层数；$z_T$ 是生成过程最顶层的潜变量；$z_1$ 最靠近数据；箭头表示右侧变量的分布依赖左侧变量。模型先生成较高层潜变量，再逐层生成更靠近观测的潜变量，最终产生 $x$。

为什么要使用多个 latent？核心原因不是“层数越多一定越好”，而是把一个复杂生成映射拆成若干较简单的条件转移。一般 HVAE 可以让高层变量表达较全局、抽象的信息，让低层变量补充局部细节。对 Diffusion 而言，这些中间 latent 不一定具有可解释的语义；它们更重要的作用是把“Gaussian 噪声到数据”的困难转换拆成许多小的去噪步骤。

论文进一步关注 HVAE 的一个特殊情况：Markovian Hierarchical VAE（MHVAE）。要理解它，需要先理解 Markov chain。

**Markov chain（马尔可夫链）**是一串按顺序排列的随机变量。它的关键假设是：已知当前状态以后，下一状态不再需要依赖更久以前的全部历史。这个性质常被概括为“未来在给定现在后与过去条件独立”。它并不是说过去没有影响，而是说过去的影响已经由当前状态概括。

#### 数学推导

先回忆单层 VAE 的生成联合分布：$p_\theta(x,z)=p(z)p_\theta(x\mid z)$。层次模型只是在 $x$ 与最高层先验之间插入更多随机变量。记 $z_{1:T}=(z_1,z_2,\ldots,z_T)$，论文中的 Markovian HVAE 生成联合分布为

$$
p_\theta(x,z_{1:T})
=p(z_T)\,p_\theta(x\mid z_1)
\prod_{t=2}^{T}p_\theta(z_{t-1}\mid z_t).
$$

这个公式描述从顶层到数据的完整生成概率。$p(z_T)$ 是最高层先验；$p_\theta(z_{t-1}\mid z_t)$ 是从第 $t$ 层向第 $t-1$ 层的生成转移；$p_\theta(x\mid z_1)$ 是最后一步 decoder；$\prod_{t=2}^{T}$ 表示把 $t=2,3,\ldots,T$ 的全部相邻层转移相乘。

以 $T=3$ 为例，乘积展开为 $p_\theta(x,z_1,z_2,z_3)=p(z_3)p_\theta(z_2\mid z_3)p_\theta(z_1\mid z_2)p_\theta(x\mid z_1)$。这个展开正好对应 $z_3\rightarrow z_2\rightarrow z_1\rightarrow x$ 的每一条箭头。

Markov 限制体现在每个条件分布只依赖相邻的上层变量。例如生成 $z_{t-1}$ 时，

$$
p_\theta(z_{t-1}\mid z_t,z_{t+1},\ldots,z_T)
=p_\theta(z_{t-1}\mid z_t).
$$

左侧看似允许 $z_{t-1}$ 依赖所有更高层变量；Markov 性质说明，一旦已经知道直接相邻的 $z_t$，更高的 $z_{t+1},\ldots,z_T$ 不再提供额外条件信息。正因为如此，联合分布才能分解成相邻转移的乘积。

训练时还需要从数据向上推断全部 latent。普通 VAE 的 encoder 是 $q_\phi(z\mid x)$；Markovian HVAE 将它扩展为反方向的推断链 $x\rightarrow z_1\rightarrow z_2\rightarrow\cdots\rightarrow z_T$：

$$
q_\phi(z_{1:T}\mid x)
=q_\phi(z_1\mid x)
\prod_{t=2}^{T}q_\phi(z_t\mid z_{t-1}).
$$

$q_\phi(z_1\mid x)$ 从观测得到第一层 latent；$q_\phi(z_t\mid z_{t-1})$ 继续从较低层推断较高层；$\phi$ 表示整条推断链的参数。生成链与推断链方向相反：$p_\theta$ 从抽象 latent 走向数据，$q_\phi$ 从数据走向 latent。

一般 HVAE 可以让某一层同时依赖多个先前层，而论文选择相邻依赖的 Markovian 版本。这样做大幅简化了联合分布、采样过程和后续 ELBO 的拆解，也是 Diffusion 能被解释为层次 VAE 的结构基础。

Part 1 的 ELBO 推导可以原封不动地应用到整条 latent 链，只需把单个 $z$ 看成组合随机变量 $z_{1:T}$：

$$
\log p_\theta(x)
\ge
\mathbb E_{q_\phi(z_{1:T}\mid x)}
\left[
\log\frac{p_\theta(x,z_{1:T})}
{q_\phi(z_{1:T}\mid x)}
\right].
$$

它仍然来自三个相同步骤：对 $z_{1:T}$ 全部边缘化，在积分中乘 $q_\phi/q_\phi$ 改写成期望，再对对数使用 Jensen 不等式。区别只是积分从一个 latent 扩展为整条 latent chain。将上面的 $p_\theta$ 与 $q_\phi$ 乘积分解代入，就得到 Markovian HVAE 的 ELBO；论文暂时保留这个比值形式，稍后再针对 Diffusion 将它分解成多个 KL 项。

#### 公式解释

生成分布 $p_\theta(x,z_{1:T})$ 与推断分布 $q_\phi(z_{1:T}\mid x)$ 描述两条方向相反的链。前者必须先给最高层 $z_T$ 一个先验，因为生成新数据时没有输入 $x$；后者以已有数据 $x$ 为起点，因为它的任务是在训练阶段推断各层 latent。

符号 $z_{1:T}$ 是一组变量的缩写，不是从 $z_1$ 到 $z_T$ 的减法或比值。积分 $dz_{1:T}$ 若出现，表示 $dz_1\,dz_2\cdots dz_T$，也就是对每一层 latent 都积分。乘积符号则把链上每个局部条件概率组合成完整路径的联合概率。

Markov 性质是可计算性的关键。若每一层都依赖所有其他层，联合分布虽然仍可定义，但条件关系和训练目标会迅速变复杂；相邻依赖让一条长链可以由重复的局部规则描述。

#### 直觉理解

普通 VAE 像一次从“抽象概念”跳到“完整图片”；Hierarchical VAE 像分阶段绘图，先确定大体结构，再逐层加入布局、形状和细节。每一步只处理相邻抽象层级之间的变化，因此单步任务可能比一次完成全部转换更简单。

Markov 性质可以用接力赛理解：第 $t$ 层把继续生成所需的信息全部交给 $z_{t-1}$。当 $z_{t-1}$ 已经拿到接力棒后，下一步不必再次询问更早的所有队员。Diffusion 将把这种“多次局部转换”的思想推到极端：使用很多与数据同维的中间状态，每一步只增加或去除少量噪声。

#### 本章总结

- **本章解决了什么问题：** 把普通 VAE 的单层 latent 扩展为多层 latent chain，并用 Markov 性质将生成与推断过程分解为相邻条件分布。
- **核心公式：** $p_\theta(x,z_{1:T})=p(z_T)p_\theta(x\mid z_1)\prod_{t=2}^{T}p_\theta(z_{t-1}\mid z_t)$。
- **需要记住的直觉：** 多层 latent 把一次困难的大转换拆成许多局部转换；Markov 性质让每一步只需查看相邻状态。
- **与下一章的关系：** 下一章会让每个 latent 与数据同维，把 $x$ 和 $z_t$ 统一重命名为 $x_t$，再固定 forward 推断链为逐步加 Gaussian 噪声的过程。

---

### 2.2 Diffusion 作为 Hierarchical VAE

#### 背景

原论文给出的核心观点是：Variational Diffusion Model（VDM）并不是与 VAE 完全无关的新模型，而是 Markovian Hierarchical VAE 的一个特殊情况。它在上一章结构上加入三项限制。

第一，所有 latent 的维度与原始数据完全相同。如果 $x$ 是形状为 $H\times W\times C$ 的图片，那么每个中间 latent 也具有相同形状。第二，推断链的每一步不再由带参数 $\phi$ 的 encoder 网络学习，而是预先规定为线性 Gaussian 转移。第三，噪声强度随层级逐渐变化，使最末端状态的分布接近标准 Gaussian。

这三项限制带来一个重要解释变化：中间 latent 不再主要被看作低维语义编码，而被看作原始数据在不同噪声等级下的版本。层级编号 $t$ 也常被称为 timestep，但它是离散的算法步骤，不一定对应现实世界中的物理时间。

#### 数学推导

##### 第一步：统一变量名称

普通 Markovian HVAE 的观测是 $x$，latent 是 $z_1,\ldots,z_T$。由于 VDM 要求每个 latent 与数据同维，论文进行如下变量替换：

$$
x\equiv x_0,
\qquad
z_t\equiv x_t\quad (t=1,2,\ldots,T).
$$

$x_0$ 表示真实数据；$x_t$ 表示第 $t$ 个噪声层级；$x_T$ 表示最末端、接近纯 Gaussian 噪声的状态。符号 $\equiv$ 表示在后续记号中把它们视为同一个角色。这个替换没有改变概率模型，只是突出所有状态具有相同数据形状，并把层级索引解释为扩散时间。

于是生成链从 $z_T\rightarrow\cdots\rightarrow z_1\rightarrow x$ 变为

$$
x_T\rightarrow x_{T-1}\rightarrow\cdots\rightarrow x_1\rightarrow x_0.
$$

这条从噪声到数据的方向是 reverse process，也就是模型真正用于生成的方向。它从简单先验中的 $x_T$ 出发，逐步恢复更干净的状态，最终得到数据样本 $x_0$。

##### 第二步：把推断链固定为 forward process

训练时从真实样本 $x_0$ 出发，沿相反方向得到越来越嘈杂的状态：$x_0\rightarrow x_1\rightarrow\cdots\rightarrow x_T$。由于保持 Markov 性质，整条 forward path 的条件分布分解为

$$
q(x_{1:T}\mid x_0)
=\prod_{t=1}^{T}q(x_t\mid x_{t-1}).
$$

$x_{1:T}$ 是 $(x_1,\ldots,x_T)$ 的缩写；$q(x_{1:T}\mid x_0)$ 描述给定干净样本后整条加噪轨迹的联合分布；每个 $q(x_t\mid x_{t-1})$ 只负责一次局部加噪。这里不再写 $q_\phi$，因为标准 VDM 的 forward 转移由预先设定的噪声 schedule 决定，不需要训练 encoder 参数 $\phi$。

论文使用线性 Gaussian 转移。用 $\alpha_t$ 表示第 $t$ 步保留信号的比例，可写为

$$
q(x_t\mid x_{t-1})
=\mathcal N\!\left(
x_t;\sqrt{\alpha_t}\,x_{t-1},
(1-\alpha_t)I
\right).
$$

这个分布以 $\sqrt{\alpha_t}x_{t-1}$ 为均值，以 $(1-\alpha_t)I$ 为协方差。$0<\alpha_t<1$ 时，均值中的原信号略微缩小，同时加入非零 Gaussian 方差。通常定义 $\beta_t=1-\alpha_t$，就得到 Part 3 将详细研究的 $\mathcal N(\sqrt{1-\beta_t}x_{t-1},\beta_t I)$。平方根、方差与采样形式为何这样配合，将在下一 Part 从 Gaussian 基础开始推导。

##### 第三步：写出 reverse generative model

生成时先从简单末端先验采样

$$
p(x_T)=\mathcal N(x_T;0,I),
$$

再使用带参数的反向条件分布逐步生成 $x_{T-1},\ldots,x_0$。完整生成联合分布为

$$
p_\theta(x_{0:T})
=p(x_T)
\prod_{t=1}^{T}p_\theta(x_{t-1}\mid x_t).
$$

$x_{0:T}$ 表示从 $x_0$ 到 $x_T$ 的全部状态；$p(x_T)$ 是已知标准 Gaussian 先验；$p_\theta(x_{t-1}\mid x_t)$ 是从较噪状态恢复较干净状态的 learned reverse transition；$\theta$ 是去噪网络参数。乘积从 $t=T$ 还是从 $t=1$ 依次执行并不改变联合概率的数值，但实际采样必须从已经得到的 $x_T$ 开始，按 $T,T-1,\ldots,1$ 的顺序向 $x_0$ 运行。

为什么 forward 使用 $q$，reverse 使用 $p_\theta$？$q$ 是训练时人为规定的推断或加噪分布，我们能够从真实 $x_0$ 直接运行它；$p_\theta$ 是生成模型，需要学习如何逆转信息逐渐丢失的加噪过程。由于许多不同的干净样本都可能在加噪后得到相似的 $x_t$，真实的反向分布不是简单地把某个确定函数倒过来，因此必须建模为条件概率分布。

三项限制可以用下面的对应关系概括：

| Markovian HVAE              | Variational Diffusion Model | 含义                                     |
| --------------------------- | --------------------------- | ---------------------------------------- |
| 观测 $x$                    | 数据 $x_0$                  | 把真实样本标记为时间零状态               |
| latent $z_t$                | 噪声状态 $x_t$              | 每个 latent 与数据同维                   |
| $q_\phi(z_t\mid z_{t-1})$   | 固定 $q(x_t\mid x_{t-1})$   | learned encoder 变成预定义 Gaussian 加噪 |
| $p(z_T)$                    | $p(x_T)=\mathcal N(0,I)$    | 最高层 latent 变成纯噪声先验             |
| $p_\theta(z_{t-1}\mid z_t)$ | $p_\theta(x_{t-1}\mid x_t)$ | 层间 decoder 变成 learned denoising step |

#### 公式解释

$q(x_{1:T}\mid x_0)$ 与 $p_\theta(x_{0:T})$ 都描述一整条状态链，但方向和用途不同。Forward distribution $q$ 以训练数据 $x_0$ 为条件，逐步破坏信息，它不是最终要学习的生成模型；reverse joint distribution $p_\theta$ 以 $p(x_T)$ 为起点，逐步恢复信息，才是生成新数据时使用的模型。

末端先验 $p(x_T)=\mathcal N(0,I)$ 之所以重要，是因为生成开始时没有任何真实图片可供加噪。模型必须从一个容易独立采样的已知分布开始。Forward schedule 被设计为在足够大的 $T$ 后，让 $q(x_T\mid x_0)$ 几乎不再携带 $x_0$ 的信息并接近这个先验，从而把训练过程的终点与生成过程的起点接起来。

Diffusion 中真正学习的是 $p_\theta(x_{t-1}\mid x_t)$，而不是 forward transition。实际实现通常用同一个神经网络处理所有 $t$，并把 timestep 作为额外输入；数学上仍可把它理解为一族随 $t$ 变化的条件分布。

#### 直觉理解

可以把 forward process 想成把清晰图片依次复印到越来越差的纸上。每一步只损失少量信息，最终只剩近似随机噪声。Reverse process 则要学习在每一张受损复印件上恢复一点信息。单步恢复仍然有不确定性，但许多小步串联后可以从纯噪声逐渐形成完整图片。

从 VAE 角度看，$x_1,\ldots,x_T$ 全部是 latent variables；从 Diffusion 角度看，它们是同一数据在不同噪声等级下的状态。这两个说法并不冲突。所谓“Diffusion 是 Hierarchical VAE”，正是把同一条概率链分别用 latent hierarchy 和 iterative noising/denoising 两种语言解释。

#### 本章总结

- **本章解决了什么问题：** 通过变量同维、固定 Gaussian forward chain 和标准 Gaussian 末端先验三项限制，把 Markovian HVAE 转换成 Variational Diffusion Model。
- **核心公式：** Forward 为 $q(x_{1:T}\mid x_0)=\prod_{t=1}^{T}q(x_t\mid x_{t-1})$；reverse generative model 为 $p_\theta(x_{0:T})=p(x_T)\prod_{t=1}^{T}p_\theta(x_{t-1}\mid x_t)$。
- **需要记住的直觉：** 训练时沿 forward chain 把数据逐步变成噪声，生成时沿 reverse chain 把噪声逐步变回数据；中间噪声状态就是层次 VAE 的 latent variables。
- **与下一章的关系：** Part 3 将进入 forward diffusion process，解释 $\beta_t$、方差、标准差与 Gaussian 采样，并完整推导如何不经过所有中间步骤而直接从 $x_0$ 得到任意 $x_t$。

## Part 3：Forward Diffusion Process

Part 2 把 Diffusion 看成一条固定的 Gaussian encoder chain：从真实数据 $x_0$ 出发，经过 $T$ 次局部随机转移得到越来越嘈杂的 $x_1,\ldots,x_T$。本部分沿论文的线性 Gaussian 假设，先解释一次加噪究竟如何采样，再推导无需依次经过中间状态、可以直接从 $x_0$ 得到任意 $x_t$ 的闭式公式。

---

### 3.1 单步加噪：均值、方差与重参数化采样

#### 背景

Forward diffusion 的任务是有控制地破坏数据。每一步不能一下子抹掉全部信息，否则 reverse model 必须一次完成“纯噪声到图片”的困难映射；它也不能让数值尺度不断膨胀，否则经过很多步以后变量方差会失控。因此论文选择一种 variance-preserving linear Gaussian transition：每一步略微缩小已有信号，同时加入少量 Gaussian 噪声。

第 $t$ 步的 forward transition 定义为

$$
q(x_t\mid x_{t-1})
=\mathcal N\!\left(
x_t;\sqrt{1-\beta_t}\,x_{t-1},
\beta_t I
\right).
$$

这是一个条件分布。给定 $x_{t-1}$ 后，$x_t$ 仍然是随机变量；其条件均值为 $\sqrt{1-\beta_t}x_{t-1}$，条件协方差为 $\beta_t I$。$I$ 是与数据维度匹配的单位矩阵，表示每个分量加入方差为 $\beta_t$ 的独立噪声。

$\beta_t$ 是第 $t$ 步的 noise variance，也叫 noise schedule 中的一个值。通常要求 $0<\beta_t<1$，使 $1-\beta_t$ 为正，并让单步只破坏一小部分信号。整组 $\{\beta_1,\ldots,\beta_T\}$ 决定噪声随时间如何累积；它们可以预先设定，也可以在某些变体中学习。

理解这个公式必须区分方差和标准差。方差衡量平方尺度上的分散程度；标准差是方差的平方根。若一维随机变量 $u$ 的方差为 $\beta_t$，其标准差就是 $\sqrt{\beta_t}$。在采样公式中，标准 Gaussian 噪声前面乘的是标准差，而不是方差本身。

#### 数学推导

##### 从一般线性加噪形式开始

先假设第 $t$ 步采用一般形式

$$
x_t=a_t x_{t-1}+b_t\epsilon_t,
\qquad
\epsilon_t\sim\mathcal N(0,I).
$$

$a_t$ 控制保留多少原信号，$b_t$ 控制噪声振幅，$\epsilon_t$ 是与 $x_{t-1}$ 同形状的标准 Gaussian 向量。$\epsilon_t\sim\mathcal N(0,I)$ 表示它的每个分量均值为 $0$、方差为 $1$，不同分量之间独立。不同 timestep 通常使用彼此独立的新噪声。

在条件 $x_{t-1}$ 已知时，$a_t x_{t-1}$ 是常量，随机性只来自 $b_t\epsilon_t$。因此条件均值为

$$
\mathbb E[x_t\mid x_{t-1}]
=a_t x_{t-1}+b_t\mathbb E[\epsilon_t]
=a_t x_{t-1},
$$

因为标准 Gaussian 满足 $\mathbb E[\epsilon_t]=0$。条件协方差为

$$
\operatorname{Cov}(x_t\mid x_{t-1})
=b_t^2\operatorname{Cov}(\epsilon_t)
=b_t^2 I.
$$

乘以常数 $b_t$ 会把偏差放大 $b_t$ 倍，因此偏差平方的平均，也就是方差，会放大 $b_t^2$ 倍。若希望加入的噪声方差恰好为 $\beta_t I$，就必须令 $b_t^2=\beta_t$，即取正标准差 $b_t=\sqrt{\beta_t}$。如果错误地写成 $\beta_t\epsilon_t$，所得噪声方差将是 $\beta_t^2I$，并不是定义要求的 $\beta_t I$。

##### 为什么原信号乘以 $\sqrt{1-\beta_t}$

接着确定 $a_t$。为了说明论文的 variance-preserving 选择，假设数据已经大致标准化，使 $x_{t-1}$ 的每个分量方差约为 $1$，并且新噪声 $\epsilon_t$ 与 $x_{t-1}$ 独立。独立变量线性组合的方差等于各项方差之和，因此

$$
\operatorname{Var}(x_t)
=a_t^2\operatorname{Var}(x_{t-1})
+b_t^2\operatorname{Var}(\epsilon_t)
\approx a_t^2+b_t^2.
$$

我们已经选定 $b_t^2=\beta_t$。若希望 $x_t$ 的总体方差仍大致为 $1$，就令 $a_t^2+\beta_t=1$，从而 $a_t=\sqrt{1-\beta_t}$。这解释了均值中平方根系数的来源：信号方差贡献为 $1-\beta_t$，噪声方差贡献为 $\beta_t$，两者相加仍为 $1$。

代回一般形式，得到关键采样公式：

$$
x_t
=\sqrt{1-\beta_t}\,x_{t-1}
+\sqrt{\beta_t}\,\epsilon_t,
\qquad
\epsilon_t\sim\mathcal N(0,I).
$$

这个公式来自 Gaussian 的重参数化。它与 $q(x_t\mid x_{t-1})=\mathcal N(\sqrt{1-\beta_t}x_{t-1},\beta_t I)$ 完全等价：前一项给出条件均值，后一项的标准差是 $\sqrt{\beta_t}$，所以条件协方差是 $\beta_t I$。

需要注意，“variance-preserving”并不表示某个具体样本的像素幅度在每一步完全不变，也不表示任意未经标准化的数据都严格保持单位方差。它表示在数据近似单位方差且新噪声独立的建模条件下，转移不会系统性地让总体方差随步数爆炸或消失。

#### 公式解释

单步分布中的每个符号都有明确角色：$t$ 是离散 timestep；$x_{t-1}$ 是较干净状态；$x_t$ 是加噪后的随机状态；$\beta_t$ 是本步噪声方差；$\sqrt{1-\beta_t}$ 是信号的标准差尺度系数；$\sqrt{\beta_t}$ 是新噪声的标准差；$I$ 表示各维度使用相同方差且相互独立。

分布写法 $q(x_t\mid x_{t-1})$ 回答“给定上一状态，下一状态服从什么分布”；采样写法回答“程序里怎样实际产生一个 $x_t$”。两者不是两个不同假设，而是同一个 Gaussian transition 的概率形式与计算形式。

$q$ 没有参数下标 $\phi$，因为标准 DDPM/VDM 中的 forward process 是预先规定的。训练并不是让网络学习怎样加噪，而是利用这个已知加噪过程制造训练样本，随后学习未知的 reverse transition。

#### 直觉理解

可以把每一步想成调节一张图片的两个音量旋钮：第一个旋钮把原图音量从 $1$ 略微降到 $\sqrt{1-\beta_t}$，第二个旋钮把新噪声音量调到 $\sqrt{\beta_t}$。由于功率对应振幅平方，两路“功率”分别是 $1-\beta_t$ 和 $\beta_t$，总功率仍约为 $1$。

当 $\beta_t$ 很小时，一次加噪前后的状态非常接近。Forward process 把许多这样的微小随机变化串起来，逐渐删除关于 $x_0$ 的信息。Reverse model 因而只需要学习许多局部小修正，而不是一次把纯噪声直接变成完整图片。

#### 本章总结

- **本章解决了什么问题：** 从一般线性 Gaussian 采样出发，解释了单步 forward transition 的均值、方差、标准差以及两个平方根系数的来源。
- **核心公式：** $x_t=\sqrt{1-\beta_t}x_{t-1}+\sqrt{\beta_t}\epsilon_t$，其中 $\epsilon_t\sim\mathcal N(0,I)$。
- **需要记住的直觉：** $\beta_t$ 是方差，所以噪声振幅是 $\sqrt{\beta_t}$；信号与噪声的方差贡献相加为 $1$，使数值尺度保持稳定。
- **与下一章的关系：** 若每次获得 $x_t$ 都必须从 $x_0$ 连续运行 $t$ 步，训练会很低效。下一章将把全部中间噪声合并为一个 Gaussian 噪声，直接得到 $q(x_t\mid x_0)$。

---

### 3.2 从 $x_0$ 直接得到 $x_t$

#### 背景

Forward chain 的定义是局部的：$q(x_t\mid x_{t-1})$ 只告诉我们如何走一步。但训练 diffusion network 时，经常随机选择一个 timestep $t$，然后需要立刻构造对应的 noisy sample $x_t$。若每次都从 $x_0$ 开始依次计算 $x_1,x_2,\ldots,x_t$，计算量会随 $t$ 增长。

线性 Gaussian 转移有一个重要性质：若干独立 Gaussian 随机变量的线性组合仍然是 Gaussian。其均值等于各项均值的线性组合；当各项独立时，协方差等于各项协方差之和。正是这个性质允许我们把多步加入的许多噪声合并成一次标准 Gaussian 噪声。

#### 数学推导

##### 第一步：简化符号

定义

$$
\alpha_t=1-\beta_t,
\qquad
\bar\alpha_t=\prod_{i=1}^{t}\alpha_i.
$$

$\alpha_t$ 是第 $t$ 步的 signal retention factor，对应本步保留的信号方差比例；$\bar\alpha_t$ 读作 “alpha bar t”，是从第 $1$ 步到第 $t$ 步所有 $\alpha_i$ 的乘积，不是平均值。由定义可得递推关系 $\bar\alpha_t=\alpha_t\bar\alpha_{t-1}$。若约定 $\bar\alpha_0=1$，这个关系对 $t=1$ 也成立。

单步采样公式变为

$$
x_t=\sqrt{\alpha_t}\,x_{t-1}
+\sqrt{1-\alpha_t}\,\epsilon_t,
\qquad
\epsilon_t\sim\mathcal N(0,I).
$$

这里只是把 $1-\beta_t$ 重命名为 $\alpha_t$，没有改变模型。新记号使多步相乘更容易阅读。

##### 第二步：先展开两步

第一步和第二步分别为 $x_1=\sqrt{\alpha_1}x_0+\sqrt{1-\alpha_1}\epsilon_1$ 与 $x_2=\sqrt{\alpha_2}x_1+\sqrt{1-\alpha_2}\epsilon_2$。将 $x_1$ 代入 $x_2$：

$$
\begin{aligned}
x_2
&=\sqrt{\alpha_2}
\left(
\sqrt{\alpha_1}x_0
+\sqrt{1-\alpha_1}\epsilon_1
\right)
+\sqrt{1-\alpha_2}\epsilon_2\\
&=\sqrt{\alpha_1\alpha_2}\,x_0
+\sqrt{\alpha_2(1-\alpha_1)}\,\epsilon_1
+\sqrt{1-\alpha_2}\,\epsilon_2.
\end{aligned}
$$

信号部分的系数已经变成 $\sqrt{\alpha_1\alpha_2}$。现在处理两个噪声项。令 $\eta=\sqrt{\alpha_2(1-\alpha_1)}\epsilon_1+\sqrt{1-\alpha_2}\epsilon_2$。因为 $\epsilon_1$ 与 $\epsilon_2$ 独立、均值都为零，所以 $\eta$ 的均值为零，协方差为

$$
\begin{aligned}
\operatorname{Cov}(\eta)
&=\alpha_2(1-\alpha_1)I+(1-\alpha_2)I\\
&=(1-\alpha_1\alpha_2)I.
\end{aligned}
$$

为什么没有两个噪声的交叉项？展开方差时会出现 $\mathbb E[\epsilon_1\epsilon_2^\top]$；独立且零均值意味着这个交叉协方差为零。又因为独立 Gaussian 的线性组合仍是 Gaussian，所以可以用一个新的标准 Gaussian $\bar\epsilon_2\sim\mathcal N(0,I)$ 表示相同分布：$\eta\overset{d}{=}\sqrt{1-\alpha_1\alpha_2}\bar\epsilon_2$。符号 $\overset{d}{=}$ 表示“分布相同”，并不表示对每次具体采样，右侧与原来两个噪声的数值都逐点相等。

因此

$$
x_2
=\sqrt{\alpha_1\alpha_2}\,x_0
+\sqrt{1-\alpha_1\alpha_2}\,\bar\epsilon_2.
$$

两步之后，信号方差比例是 $\alpha_1\alpha_2$，累计噪声方差比例是 $1-\alpha_1\alpha_2$。这已经显示出最终公式的结构。

##### 第三步：用递推推广到任意 $t$

假设在第 $t-1$ 步已经有

$$
x_{t-1}
=\sqrt{\bar\alpha_{t-1}}\,x_0
+\sqrt{1-\bar\alpha_{t-1}}\,\bar\epsilon_{t-1},
$$

其中 $\bar\epsilon_{t-1}\sim\mathcal N(0,I)$。把它代入第 $t$ 步：

$$
\begin{aligned}
x_t
&=\sqrt{\alpha_t}\,x_{t-1}
+\sqrt{1-\alpha_t}\,\epsilon_t\\
&=\sqrt{\alpha_t\bar\alpha_{t-1}}\,x_0
+\sqrt{\alpha_t(1-\bar\alpha_{t-1})}\,\bar\epsilon_{t-1}
+\sqrt{1-\alpha_t}\,\epsilon_t.
\end{aligned}
$$

根据 $\bar\alpha_t=\alpha_t\bar\alpha_{t-1}$，信号系数变为 $\sqrt{\bar\alpha_t}$。两个独立噪声项的总协方差系数为

$$
\alpha_t(1-\bar\alpha_{t-1})+(1-\alpha_t)
=1-\alpha_t\bar\alpha_{t-1}
=1-\bar\alpha_t.
$$

因此它们同样可以合并成一个新的标准 Gaussian 噪声 $\epsilon\sim\mathcal N(0,I)$。得到从 $x_0$ 直接采样任意 $x_t$ 的关键公式：

$$
x_t
=\sqrt{\bar\alpha_t}\,x_0
+\sqrt{1-\bar\alpha_t}\,\epsilon,
\qquad
\epsilon\sim\mathcal N(0,I).
$$

这里的 $\epsilon$ 是前 $t$ 步全部独立噪声合并后的等价标准 Gaussian，不必等于某个具体单步使用的 $\epsilon_i$。由于我们只关心给定 $x_0$ 后 $x_t$ 的分布，可以直接重新采一个标准 Gaussian 来代表累计噪声。

由采样形式读取条件均值和协方差，最终得到

$$
q(x_t\mid x_0)
=\mathcal N\!\left(
x_t;\sqrt{\bar\alpha_t}\,x_0,
(1-\bar\alpha_t)I
\right).
$$

这个分布描述：给定干净数据 $x_0$，第 $t$ 个 noisy state 的平均位置仍沿着 $x_0$，但信号幅度缩小为 $\sqrt{\bar\alpha_t}$；围绕这个平均位置的不确定性由累计噪声协方差 $(1-\bar\alpha_t)I$ 决定。

#### 公式解释

$\alpha_t$ 只描述第 $t$ 个局部步骤；$\bar\alpha_t$ 描述从 $0$ 到 $t$ 的累计信号保留。由于每一步都乘一次 $\sqrt{\alpha_i}$，经过 $t$ 步后的信号振幅是 $\prod_{i=1}^{t}\sqrt{\alpha_i}=\sqrt{\prod_{i=1}^{t}\alpha_i}=\sqrt{\bar\alpha_t}$。

累计噪声方差不是简单的 $\sum_i\beta_i$，因为较早加入的噪声还会在后续步骤中被 $\sqrt{\alpha}$ 再次缩放。递推合并后，所有噪声的总方差恰好为 $1-\bar\alpha_t$。这就是为什么闭式公式中信号方差比例与噪声方差比例仍相加为 $1$。

这个公式对训练至关重要。训练时可以先随机选择一个 $t$，再采样一个 $\epsilon\sim\mathcal N(0,I)$，用一次向量运算直接构造 $x_t$，计算复杂度不再随 $t$ 增长。更重要的是，构造 $x_t$ 时使用的噪声 $\epsilon$ 是已知的，它将在 Part 6 成为神经网络的监督目标。

当噪声 schedule 使 $\bar\alpha_T$ 非常接近零时，$\sqrt{\bar\alpha_T}x_0$ 几乎消失，而 $\sqrt{1-\bar\alpha_T}\epsilon$ 几乎就是标准 Gaussian 噪声。这正好解释了 Part 2 为什么能够把 $p(x_T)=\mathcal N(0,I)$ 选作 reverse process 的起点。

#### 直觉理解

$\bar\alpha_t$ 可以看成原图信号经过前 $t$ 个衰减器以后剩下的“累计功率比例”。它从接近 $1$ 逐渐降向 $0$。相应地，$1-\bar\alpha_t$ 是累计噪声功率比例，从接近 $0$ 逐渐升向 $1$。

虽然真实 forward chain 每一步都加入了不同噪声，但 Gaussian 的封闭性允许我们只看起点和终点：许多独立小噪声的加权和，在分布上等价于一个具有正确总方差的大 Gaussian 噪声。因此训练时不需要真的重放整条破坏过程。

#### 本章总结

- **本章解决了什么问题：** 完整推导了多步 Gaussian 加噪的闭式分布，使任意 $x_t$ 都能由 $x_0$ 一步采样得到。
- **核心公式：** $x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon$，等价地，$q(x_t\mid x_0)=\mathcal N(\sqrt{\bar\alpha_t}x_0,(1-\bar\alpha_t)I)$。
- **需要记住的直觉：** 信号系数逐步相乘，独立 Gaussian 噪声的方差逐步相加；最终只需用累计信号比例 $\bar\alpha_t$ 描述任意噪声等级。
- **与下一章的关系：** Forward transition 与 $q(x_t\mid x_0)$ 现在都已知。Part 4 将解释为什么反向条件 $q(x_{t-1}\mid x_t)$ 仍然未知，以及为什么需要学习 $p_\theta(x_{t-1}\mid x_t)$。

## Part 4：Reverse Diffusion

Forward process 的每个转移 $q(x_t\mid x_{t-1})$ 都由 noise schedule 明确规定，因此给定 $x_{t-1}$ 后很容易采样 $x_t$。生成模型却需要沿相反方向从 $x_t$ 得到 $x_{t-1}$。本部分将说明随机加噪为什么不能像普通可逆函数那样直接倒过来，并利用训练时已知的干净样本 $x_0$ 推导一个解析 Gaussian posterior，作为 learned reverse process 的训练依据。

---

### 4.1 为什么已知 Forward Process 仍不知道 Reverse Process

#### 背景

已知函数 $y=f(x)$ 时，如果 $f$ 是一一对应的可逆函数，或许可以写出 $x=f^{-1}(y)$。Diffusion 的 forward step 不是这种确定性变换，而是 $x_t=\sqrt{\alpha_t}x_{t-1}+\sqrt{1-\alpha_t}\epsilon_t$，其中每次都会采样未知噪声 $\epsilon_t$。许多不同的 $x_{t-1}$ 配合不同噪声，都可能产生同一个或非常接近的 $x_t$。加噪会丢失信息，所以 reverse 不是代数意义上的唯一逆函数，而是“给定 $x_t$ 后，哪些 $x_{t-1}$ 比较可能”的条件分布。

Forward kernel

$$
q(x_t\mid x_{t-1})
=\mathcal N\!\left(
x_t;\sqrt{\alpha_t}x_{t-1},
(1-\alpha_t)I
\right)
$$

只回答从某个已知 $x_{t-1}$ 出发会得到怎样的 $x_t$。我们真正希望用于反向采样的是 $q(x_{t-1}\mid x_t)$。交换条件方向必须使用 Bayes 公式，而且 Bayes 公式不仅需要 forward likelihood，还需要 $x_{t-1}$ 与 $x_t$ 的边缘分布。

#### 数学推导

对 forward joint distribution 使用 Bayes 公式：

$$
q(x_{t-1}\mid x_t)
=\frac{q(x_t\mid x_{t-1})\,q_{t-1}(x_{t-1})}
{q_t(x_t)}.
$$

$q(x_t\mid x_{t-1})$ 是已知的单步 Gaussian；$q_{t-1}(x_{t-1})$ 是把真实数据分布经过 $t-1$ 步加噪后得到的边缘分布；$q_t(x_t)$ 是经过 $t$ 步后的边缘分布。这里用下标 $q_t$ 强调它是 timestep $t$ 的总体分布，而不是给定某个固定 $x_0$ 的条件分布。

这些边缘分布需要对真实数据来源进行平均。例如

$$
q_t(x_t)
=\int q(x_t\mid x_0)p_{\mathrm{data}}(x_0)\,dx_0.
$$

虽然 Part 3 已知 $q(x_t\mid x_0)=\mathcal N(\sqrt{\bar\alpha_t}x_0,(1-\bar\alpha_t)I)$，但 $p_{\mathrm{data}}(x_0)$ 是我们只有样本、没有解析密度的真实数据分布。积分把所有可能的干净图片经过加噪后形成的 Gaussian 成分混合起来，得到一个极其复杂的分布。因此 $q_{t-1}(x_{t-1})$ 与 $q_t(x_t)$ 通常无法显式计算，Bayes 公式中的反向条件也就未知。

还可以从“所有可能的干净来源”角度写成

$$
q(x_{t-1}\mid x_t)
=\int q(x_{t-1}\mid x_t,x_0)
q(x_0\mid x_t)\,dx_0.
$$

这个公式来自全概率法则：给定当前噪声状态 $x_t$ 后，先枚举它可能源自的每个 $x_0$，再对相应的 $q(x_{t-1}\mid x_t,x_0)$ 加权平均。难点转移到了 $q(x_0\mid x_t)$，它要求从 noisy sample 推断未知的真实数据来源，正是生成模型需要学习的能力。

因此，“forward 已知”只表示我们知道怎样破坏一个给定样本，并不表示知道所有真实样本被破坏后形成的总体分布，也不表示知道如何恢复丢失的信息。

#### 公式解释

$q(x_t\mid x_{t-1})$ 是人为设定的局部 corruption rule；$q_t(x_t)$ 是该规则作用于整个真实数据分布后的边缘结果；$q(x_{t-1}\mid x_t)$ 是依赖这个复杂边缘分布的真实反向条件。三者都写字母 $q$，但知道第一个不等于知道后两个。

Bayes 公式中的先验项 $q_{t-1}(x_{t-1})$ 很重要。只看 Gaussian likelihood $q(x_t\mid x_{t-1})$，任意能够配合某个噪声解释 $x_t$ 的前一状态似乎都可能；数据先验会让“像真实图片逐步加噪而来”的状态获得更高权重。Diffusion network 要学习的正是这种无法手写的数据结构偏好。

#### 直觉理解

想象看到一张被严重打码的照片。你知道打码算法怎样把清晰照片变模糊，却仍然不能唯一恢复原图，因为许多清晰照片都可能产生相似的马赛克。若知道自然人脸通常长什么样，就能排除大量不合理恢复结果；这个关于自然图像的知识来自数据分布，而不是来自打码公式本身。

Forward process 像已知的破坏机器，reverse process 像需要经验的修复师。知道机器如何随机破坏物品，并不会自动赋予修复师理解物品结构的能力。

#### 本章总结

- **本章解决了什么问题：** 说明了随机加噪不是可逆确定函数，并从 Bayes 公式揭示真实反向条件依赖未知的数据边缘分布。
- **核心公式：** $q(x_{t-1}\mid x_t)=q(x_t\mid x_{t-1})q_{t-1}(x_{t-1})/q_t(x_t)$。
- **需要记住的直觉：** Forward kernel 只教我们怎样破坏一个已知样本；reverse distribution 还必须知道什么样的干净结构在真实数据中更可能出现。
- **与下一章的关系：** 训练数据提供了真实 $x_0$。下一章将在额外给定 $x_0$ 的条件下使用 Bayes，推导可计算的 $q(x_{t-1}\mid x_t,x_0)$，并用它监督 $p_\theta(x_{t-1}\mid x_t)$。

---

### 4.2 用 Bayes 推导可计算后验，并学习 Reverse Process

#### 背景

生成时我们只有 $x_t$，不知道 $x_0$；训练时却从数据集取得了干净样本 $x_0$，并且能够亲自从 $q(x_t\mid x_0)$ 构造 noisy sample $x_t$。额外给定 $x_0$ 后，单步反向 posterior $q(x_{t-1}\mid x_t,x_0)$ 可以完全由 forward process 的已知 Gaussian 分布算出。

这个 posterior 不等于生成时真正可直接使用的 $q(x_{t-1}\mid x_t)$。它更像训练阶段的 oracle：它知道当前噪声状态 $x_t$ 最终应该回到哪个具体 $x_0$，因而能告诉我们合理的前一状态 $x_{t-1}$ 应服从什么分布。模型 $p_\theta(x_{t-1}\mid x_t)$ 不接收 $x_0$，但可以在大量训练样本上学习逼近这种 denoising behavior。

#### 数学推导

##### 第一步：写出带 $x_0$ 条件的 Bayes 公式

由 Bayes 公式，

$$
q(x_{t-1}\mid x_t,x_0)
=\frac{
q(x_t\mid x_{t-1},x_0)q(x_{t-1}\mid x_0)
}{q(x_t\mid x_0)}.
$$

Forward chain 具有 Markov 性质：给定 $x_{t-1}$ 后，$x_t$ 不再额外依赖更早的 $x_0$，所以 $q(x_t\mid x_{t-1},x_0)=q(x_t\mid x_{t-1})$。三个因子都已由 Part 3 得到：

$$
\begin{aligned}
q(x_t\mid x_{t-1})
&=\mathcal N\!\left(
x_t;\sqrt{\alpha_t}x_{t-1},(1-\alpha_t)I
\right),\\
q(x_{t-1}\mid x_0)
&=\mathcal N\!\left(
x_{t-1};\sqrt{\bar\alpha_{t-1}}x_0,
(1-\bar\alpha_{t-1})I
\right),\\
q(x_t\mid x_0)
&=\mathcal N\!\left(
x_t;\sqrt{\bar\alpha_t}x_0,
(1-\bar\alpha_t)I
\right).
\end{aligned}
$$

分母 $q(x_t\mid x_0)$ 保证 posterior 归一化，但在把 posterior 看作关于 $x_{t-1}$ 的函数时，$x_t$ 与 $x_0$ 都已固定，所以分母是常数。为了识别 posterior 的 Gaussian 均值与方差，可以先只保留分子中依赖 $x_{t-1}$ 的部分。

##### 第二步：展开 Gaussian 指数并配平方

为减少下标干扰，暂时令 $y=x_{t-1}$。忽略与 $y$ 无关的归一化常数后，两个 Gaussian 分子的乘积满足

$$
q(y\mid x_t,x_0)
\propto
\exp\!\left\{
-\frac12\left[
\frac{\|x_t-\sqrt{\alpha_t}y\|_2^2}{1-\alpha_t}
+\frac{\|y-\sqrt{\bar\alpha_{t-1}}x_0\|_2^2}
{1-\bar\alpha_{t-1}}
\right]
\right\}.
$$

符号 $\propto$ 表示两边只差一个与 $y$ 无关的正常数；$\|v\|_2^2=v^\top v$ 是向量各分量平方之和。展开两个平方，把所有含 $y$ 的项收集起来，可写成

$$
q(y\mid x_t,x_0)
\propto
\exp\!\left\{
-\frac12
\left[
A_t\|y\|_2^2-2B_t^\top y+C(x_t,x_0)
\right]
\right\},
$$

其中 $C(x_t,x_0)$ 不含 $y$，而二次项系数与线性项系数分别为

$$
\begin{aligned}
A_t
&=\frac{\alpha_t}{1-\alpha_t}
+\frac{1}{1-\bar\alpha_{t-1}},\\
B_t
&=\frac{\sqrt{\alpha_t}}{1-\alpha_t}x_t
+\frac{\sqrt{\bar\alpha_{t-1}}}
{1-\bar\alpha_{t-1}}x_0.
\end{aligned}
$$

$A_t$ 是标量，因为所有协方差都是单位矩阵的倍数；$B_t$ 是与数据同维的向量；$B_t^\top y$ 是内积。利用 $\bar\alpha_t=\alpha_t\bar\alpha_{t-1}$，$A_t$ 可化简为

$$
A_t
=\frac{1-\bar\alpha_t}
{(1-\alpha_t)(1-\bar\alpha_{t-1})}.
$$

配平方使用恒等式 $A\|y\|_2^2-2B^\top y=A\|y-B/A\|_2^2-\|B\|_2^2/A$。最后一项与 $y$ 无关，可以并入归一化常数。因此 posterior 是 Gaussian，方差系数是 $A_t^{-1}$，均值是 $B_t/A_t$。

##### 第三步：得到 posterior 均值和方差

定义

$$
\tilde\beta_t
=A_t^{-1}
=\frac{(1-\alpha_t)(1-\bar\alpha_{t-1})}
{1-\bar\alpha_t}
=\frac{\beta_t(1-\bar\alpha_{t-1})}
{1-\bar\alpha_t}.
$$

$\tilde\beta_t$ 是给定 $x_t$ 与 $x_0$ 后，$x_{t-1}$ 的 posterior variance；最后一步使用 $\beta_t=1-\alpha_t$。posterior mean 为

$$
\tilde\mu_t(x_t,x_0)
=\frac{
\sqrt{\alpha_t}(1-\bar\alpha_{t-1})x_t
+\sqrt{\bar\alpha_{t-1}}(1-\alpha_t)x_0
}{1-\bar\alpha_t}.
$$

因此完整解析 posterior 是

$$
q(x_{t-1}\mid x_t,x_0)
=\mathcal N\!\left(
x_{t-1};\tilde\mu_t(x_t,x_0),
\tilde\beta_t I
\right).
$$

均值是 $x_t$ 与 $x_0$ 的已知加权组合，权重完全由 noise schedule 决定。$x_t$ 提供当前噪声状态，$x_0$ 提供最终应恢复到的干净目标。$\tilde\beta_t$ 仍然大于或等于零，表示即使已知两端状态，中间一步通常仍有随机不确定性。当 $t=1$ 且约定 $\bar\alpha_0=1$ 时，$\tilde\beta_1=0$，因为 $x_{t-1}$ 就是已经给定的 $x_0$；这个边界情形将在 Diffusion ELBO 中形成单独的 reconstruction term。

##### 第四步：用神经网络近似生成时需要的反向条件

生成时 $x_0$ 尚不存在，不能把上面的 oracle mean $\tilde\mu_t(x_t,x_0)$ 直接用于采样。我们定义带参数的 reverse transition

$$
p_\theta(x_{t-1}\mid x_t)
=\mathcal N\!\left(
x_{t-1};\mu_\theta(x_t,t),
\Sigma_\theta(t)
\right).
$$

$\mu_\theta(x_t,t)$ 是神经网络给出的反向均值；输入包括 noisy state $x_t$ 与 timestep $t$；$\Sigma_\theta(t)$ 是反向方差，可以固定为与 $\tilde\beta_t I$ 相关的已知形式，也可以采用其他参数化。网络没有 $x_0$ 输入，必须从训练数据中学会根据 $x_t$ 统计性地推断被噪声遮蔽的结构。

训练时，先从数据集采样 $x_0$，再由已知 $q(x_t\mid x_0)$ 构造 $x_t$。此时 oracle posterior $q(x_{t-1}\mid x_t,x_0)$ 可解析计算，于是可以通过减小

$$
D_{\mathrm{KL}}\!\left(
q(x_{t-1}\mid x_t,x_0)
\|p_\theta(x_{t-1}\mid x_t)
\right)
$$

来训练反向分布。左侧 $q$ 可以看见 $x_0$，右侧 $p_\theta$ 不能看见 $x_0$；在大量 $(x_0,x_t)$ 样本上取期望后，$p_\theta$ 学习的是适用于未知数据来源的 denoising rule。这个 KL 项如何从 Hierarchical VAE ELBO 中系统出现，将在 Part 5 完整推导。

训练完成后，从 $x_T\sim\mathcal N(0,I)$ 开始，依次采样 $x_{t-1}\sim p_\theta(x_{t-1}\mid x_t)$，便能沿 $T,T-1,\ldots,1$ 逐步生成 $x_0$。

#### 公式解释

$q(x_{t-1}\mid x_t,x_0)$ 是由 forward process 和训练样本共同决定的真实条件 posterior，它没有可训练参数；$p_\theta(x_{t-1}\mid x_t)$ 是生成时实际可用的近似 reverse transition，它的参数 $\theta$ 需要学习。称前者为“ground-truth denoising distribution”时，指的是给定训练样本 $x_0$ 后它可解析计算，并不是说生成阶段能访问真实答案。

$\tilde\mu_t$ 与 $\tilde\beta_t$ 上方的波浪号用于区分 posterior 参数和 forward 参数。$\beta_t$ 是 forward 单步噪声方差，$\tilde\beta_t$ 是在同时知道 $x_t$ 与 $x_0$ 后，对前一状态 $x_{t-1}$ 剩余不确定性的方差；两者相关但并不相同。

Reverse network 最终可以等价地参数化为预测 $x_0$、预测噪声 $\epsilon$ 或预测 score。当前先把它写成学习均值 $\mu_\theta$，三种等价解释将在 Part 7 统一推导。

#### 直觉理解

训练时像是在做带答案的图像修复练习：我们先拿一张清晰图 $x_0$ 主动加噪得到 $x_t$，所以知道它原来是什么。Bayes posterior 能利用“受损图”和“标准答案”计算上一步合理修复结果的分布。神经网络只看受损图与噪声等级，却在大量带答案练习中逐渐学会自然图像的结构规律。

生成时不再有标准答案，但网络已经学习了统计意义上的修复经验。它从纯噪声开始，每一步都给出一个更可能属于真实数据演化路径的前一状态，许多小步最终累积成完整样本。

#### 本章总结

- **本章解决了什么问题：** 使用 Bayes、Markov 性质和 Gaussian 配平方，完整求出了训练时可计算的 $q(x_{t-1}\mid x_t,x_0)$，并说明为什么生成时必须用 $p_\theta(x_{t-1}\mid x_t)$ 近似反向过程。
- **核心公式：** $q(x_{t-1}\mid x_t,x_0)=\mathcal N(x_{t-1};\tilde\mu_t(x_t,x_0),\tilde\beta_t I)$。
- **需要记住的直觉：** 带 $x_0$ 的 posterior 是训练 oracle，不带 $x_0$ 的 learned reverse transition 才是生成时可用的模型；网络通过大量已知干净答案的加噪样本学习去噪先验。
- **与下一章的关系：** Part 5 将从 Hierarchical VAE ELBO 出发，把 Diffusion 的训练目标拆成 $L_T$、$L_{t-1}$ 与 $L_0$，其中中间项正是这里出现的 posterior 与 learned reverse transition 之间的 KL。

## Part 5：Diffusion ELBO

Diffusion 作为 Markovian Hierarchical VAE，原则上仍然通过最大化 ELBO 训练。真正需要解决的问题是：把整条 forward path 与 reverse path 的概率比值，整理成可以解释、可以计算并可以随机估计的损失。本部分严格沿论文顺序先给出直接分解，再说明它为何具有较高 Monte Carlo 方差，最后利用 Bayes 得到标准的 $L_T$、$L_{t-1}$ 与 $L_0$。

---

### 5.1 从 Hierarchical VAE ELBO 到第一种 Diffusion 分解

#### 背景

在 Diffusion 的层次 VAE 解释中，$x_0$ 是观测数据，$x_{1:T}$ 是全部 latent variables。Forward inference distribution 与 reverse generative model 分别为 $q(x_{1:T}\mid x_0)=\prod_{t=1}^{T}q(x_t\mid x_{t-1})$ 和 $p_\theta(x_{0:T})=p(x_T)\prod_{t=1}^{T}p_\theta(x_{t-1}\mid x_t)$。

Part 1 的 ELBO 只有一个 latent $z$；现在只是把 $z$ 换成整条 latent path $x_{1:T}$。推导的逻辑仍然是边缘化、乘以 $q/q$、改写成期望并应用 Jensen 不等式。

#### 数学推导

##### 第一步：为整条路径构造 ELBO

从观测似然开始：

$$
\begin{aligned}
\log p_\theta(x_0)
&=\log\int p_\theta(x_{0:T})\,dx_{1:T}\\
&=\log\int q(x_{1:T}\mid x_0)
\frac{p_\theta(x_{0:T})}{q(x_{1:T}\mid x_0)}\,dx_{1:T}\\
&=\log\mathbb E_{q(x_{1:T}\mid x_0)}
\left[
\frac{p_\theta(x_{0:T})}{q(x_{1:T}\mid x_0)}
\right]\\
&\ge
\mathbb E_{q(x_{1:T}\mid x_0)}
\left[
\log\frac{p_\theta(x_{0:T})}{q(x_{1:T}\mid x_0)}
\right]
\equiv\mathcal L_{\mathrm{ELBO}}(x_0).
\end{aligned}
$$

$dx_{1:T}$ 表示对 $x_1,\ldots,x_T$ 全部积分；第二行乘入的 $q/q=1$ 没有改变似然；第三行使用期望定义；第四行利用对数是凹函数的 Jensen 不等式。于是右侧成为 $\log p_\theta(x_0)$ 的下界。

将 forward 与 reverse 的 Markov 乘积分解代入：

$$
\mathcal L_{\mathrm{ELBO}}
=\mathbb E_q\left[
\log
\frac{
p(x_T)\prod_{t=1}^{T}p_\theta(x_{t-1}\mid x_t)
}{
\prod_{t=1}^{T}q(x_t\mid x_{t-1})
}
\right].
$$

这里 $\mathbb E_q$ 是 $\mathbb E_{q(x_{1:T}\mid x_0)}$ 的简写。分子是从 $x_T$ 向 $x_0$ 的生成路径概率，分母是从 $x_0$ 向 $x_T$ 的加噪路径概率。

##### 第二步：直接按链上的对应转移分组

先把 reverse product 的第一步 $p_\theta(x_0\mid x_1)$ 与 forward product 的最后一步 $q(x_T\mid x_{T-1})$ 单独取出，再把 reverse product 的索引从 $p_\theta(x_{t-1}\mid x_t)$ 改写成 $p_\theta(x_t\mid x_{t+1})$：

$$
\frac{p_\theta(x_{0:T})}{q(x_{1:T}\mid x_0)}=
\frac{p(x_T)p_\theta(x_0\mid x_1)}
{q(x_T\mid x_{T-1})}
\prod_{t=1}^{T-1}
\frac{p_\theta(x_t\mid x_{t+1})}
{q(x_t\mid x_{t-1})}.
$$

这个式子没有改变任何因子，只是重新排列：$p_\theta(x_0\mid x_1)$ 对应链的最终重建，$p(x_T)$ 对应最高层先验，中间每个 $x_t$ 同时连接来自较干净状态的 forward transition 与来自较噪状态的 reverse transition。

对乘积取对数后变成求和，并只保留每项实际依赖的随机变量，得到论文的第一种分解：

$$
\begin{aligned}
\mathcal L_{\mathrm{ELBO}}
={}&\mathbb E_{q(x_1\mid x_0)}
[\log p_\theta(x_0\mid x_1)]\\
&-\mathbb E_{q(x_{T-1}\mid x_0)}
\left[
D_{\mathrm{KL}}\!\left(
q(x_T\mid x_{T-1})\|p(x_T)
\right)
\right]\\
&-\sum_{t=1}^{T-1}
\mathbb E_{q(x_{t-1},x_{t+1}\mid x_0)}
\left[
D_{\mathrm{KL}}\!\left(
q(x_t\mid x_{t-1})
\|p_\theta(x_t\mid x_{t+1})
\right)
\right].
\end{aligned}
$$

第一行是 reconstruction term：从第一层 noisy latent $x_1$ 恢复观测 $x_0$。第二行是 prior matching term：给定 $x_{T-1}$，最后一步 forward Gaussian 应与标准 Gaussian prior $p(x_T)$ 匹配。第三行是 consistency term：从较干净状态 $x_{t-1}$ 加噪到 $x_t$ 的分布，应与从较噪状态 $x_{t+1}$ 去噪到 $x_t$ 的分布一致。

每个 KL 前都有负号，因为 ELBO 要最大化，而 KL 越小越好。对于 prior term，外层先从 $q(x_{T-1}\mid x_0)$ 取得条件环境，再计算两个关于 $x_T$ 的分布之间的 KL。对于 consistency term，外层需要取得 $x_{t-1}$ 与 $x_{t+1}$，KL 内部再比较两个关于同一中间变量 $x_t$ 的分布。

##### 第三步：为什么还要重新推导

上述形式理论上正确，也可以用 Monte Carlo 采样估计。但每个 consistency term 同时依赖随机的 $x_{t-1}$ 与 $x_{t+1}$；当 $T$ 很大时还要累加 $T-1$ 项。随机估计中涉及的随机变量越多、累加项越多，估计方差通常越大，训练信号会更不稳定。

论文因此没有停在第一种分解，而是寻找一种每个 timestep 最多只需对一个 noisy state $x_t$ 做外层采样的等价 ELBO。这不是更换训练理论，而是利用 Bayes 对同一个概率比重新整理。

#### 公式解释

第一种分解中的三个组成部分与普通 VAE 一一对应并扩展：reconstruction term 对应 decoder 重建；prior matching term 对应最高层 posterior 与先验匹配；新增的一串 consistency KL 负责让相邻 forward 与 reverse transition 在每个中间层协调一致。

Consistency KL 的两个分布都以 $x_t$ 为随机变量，但条件不同：$q(x_t\mid x_{t-1})$ 知道更干净的前一状态，$p_\theta(x_t\mid x_{t+1})$ 只知道更嘈杂的后一状态。训练希望 reverse model 在缺少干净信息时仍产生与 forward chain 相容的分布。

#### 直觉理解

可以把整条链想成一排相邻房间。Forward process 规定从左边房间走到当前房间的方式，reverse model 学习从右边房间返回当前房间的方式。第一种 ELBO 要求两条路线在每个当前房间汇合，并要求最右端匹配标准 Gaussian、最左端重建真实数据。

这种比较需要同时观察当前房间两侧的随机状态，因此像用两个不断晃动的参照物校准位置。下一种推导会利用已知起点 $x_0$，把校准改写成只围绕当前 $x_t$ 的 denoising posterior，降低 Monte Carlo 估计的复杂度。

#### 本章总结

- **本章解决了什么问题：** 从整条 Hierarchical VAE 路径的 ELBO 出发，得到 reconstruction、prior matching 和 consistency 三类项，并说明第一种分解的估计方差问题。
- **核心公式：** $\mathcal L_{\mathrm{ELBO}}=\mathbb E_q[\log p_\theta(x_{0:T})/q(x_{1:T}\mid x_0)]$，代入两条 Markov 链后可拆成一个重建项、一个终点 KL 和 $T-1$ 个一致性 KL。
- **需要记住的直觉：** ELBO 要求 reverse chain 在每个层级都与已知 forward chain 相容；第一种写法正确，但中间项需要同时采样两侧状态，训练估计方差较高。
- **与下一章的关系：** 下一章利用 $q(x_{t-1}\mid x_t,x_0)$ 的 Bayes 恒等式重新组织 forward product，得到标准的 $L_T+\sum L_{t-1}+L_0$。

---

### 5.2 低方差 ELBO：$L_T$、$L_{t-1}$ 与 $L_0$

#### 背景

Part 4 已推导出可计算的 denoising posterior $q(x_{t-1}\mid x_t,x_0)$。论文的关键改写是用它替换 forward factor $q(x_t\mid x_{t-1})$，使每个中间损失只需先采样一个 $x_t$，再在 KL 内部解析处理 $x_{t-1}$。

这里还要注意优化方向。ELBO 是对数似然的下界，需要最大化；深度学习代码通常最小化损失，因此定义负 ELBO。由 $\log p_\theta(x_0)\ge\mathcal L_{\mathrm{ELBO}}(x_0)$ 可得 $-\log p_\theta(x_0)\le-\mathcal L_{\mathrm{ELBO}}(x_0)$。所以负 ELBO 是 negative log-likelihood 的可优化上界，也常称 variational lower bound loss，记为 $\mathcal L_{\mathrm{VLB}}$。

#### 数学推导

##### 第一步：用 Bayes 改写每个 forward transition

由 Part 4 的 Bayes 公式和 Markov 性质，

$$
q(x_{t-1}\mid x_t,x_0)
=\frac{q(x_t\mid x_{t-1})q(x_{t-1}\mid x_0)}
{q(x_t\mid x_0)}.
$$

把等式移项，得到

$$
q(x_t\mid x_{t-1})
=q(x_{t-1}\mid x_t,x_0)
\frac{q(x_t\mid x_0)}{q(x_{t-1}\mid x_0)}.
$$

对 $t=2,\ldots,T$ 全部相乘，并保留最前面的 $q(x_1\mid x_0)$：

$$
\begin{aligned}
q(x_{1:T}\mid x_0)
&=q(x_1\mid x_0)
\prod_{t=2}^{T}q(x_t\mid x_{t-1})\\
&=q(x_1\mid x_0)
\prod_{t=2}^{T}
q(x_{t-1}\mid x_t,x_0)
\frac{q(x_t\mid x_0)}{q(x_{t-1}\mid x_0)}\\
&=q(x_T\mid x_0)
\prod_{t=2}^{T}q(x_{t-1}\mid x_t,x_0).
\end{aligned}
$$

最后一步发生了 telescoping cancellation（望远镜式消去）：边缘比值依次为 $q(x_2\mid x_0)/q(x_1\mid x_0)$、$q(x_3\mid x_0)/q(x_2\mid x_0)$，一直到 $q(x_T\mid x_0)/q(x_{T-1}\mid x_0)$。中间分子分母全部抵消，再与开头的 $q(x_1\mid x_0)$ 相乘，只留下 $q(x_T\mid x_0)$。

这条公式说明，同一个 forward path posterior 既可以从 $x_0$ 向右分解，也可以在给定 $x_0$ 后从 $x_T$ 向左分解。后一种分解正好包含训练时可计算的 denoising posterior。

##### 第二步：代回 ELBO 并逐项识别 KL

将新的 $q$ 分解与 reverse model 代入概率比：

$$
\frac{p_\theta(x_{0:T})}{q(x_{1:T}\mid x_0)}=
\frac{p(x_T)}{q(x_T\mid x_0)}\,
p_\theta(x_0\mid x_1)
\prod_{t=2}^{T}
\frac{p_\theta(x_{t-1}\mid x_t)}
{q(x_{t-1}\mid x_t,x_0)}.
$$

右侧包含三个相乘的部分：终点比值、$p_\theta(x_0\mid x_1)$ 和中间比值乘积。取对数后，乘法变为加法。于是

$$
\begin{aligned}
\mathcal L_{\mathrm{ELBO}}(x_0)
={}&\mathbb E_{q(x_1\mid x_0)}
[\log p_\theta(x_0\mid x_1)]\\
&-D_{\mathrm{KL}}\!\left(
q(x_T\mid x_0)\|p(x_T)
\right)\\
&-\sum_{t=2}^{T}
\mathbb E_{q(x_t\mid x_0)}
\left[
D_{\mathrm{KL}}\!\left(
q(x_{t-1}\mid x_t,x_0)
\|p_\theta(x_{t-1}\mid x_t)
\right)
\right].
\end{aligned}
$$

上式与第一种分解是同一个 ELBO，但每个中间项的外层只需从 $q(x_t\mid x_0)$ 采样一个 noisy state。KL 内部的 $q(x_{t-1}\mid x_t,x_0)$ 是 Part 4 的解析 Gaussian，不需要再随机模拟整条两侧路径。

##### 第三步：切换为最小化损失并定义三个部分

对 ELBO 取负号，并对训练数据 $x_0\sim p_{\mathrm{data}}$ 求平均：

$$
\mathcal L_{\mathrm{VLB}}(\theta)
=\mathbb E_{p_{\mathrm{data}}(x_0)}
\left[
L_T+\sum_{t=2}^{T}L_{t-1}+L_0
\right].
$$

三个部分定义为

$$
\begin{aligned}
L_T
&=D_{\mathrm{KL}}\!\left(
q(x_T\mid x_0)\|p(x_T)
\right),\\
L_{t-1}
&=\mathbb E_{q(x_t\mid x_0)}
\left[
D_{\mathrm{KL}}\!\left(
q(x_{t-1}\mid x_t,x_0)
\|p_\theta(x_{t-1}\mid x_t)
\right)
\right],\quad 2\le t\le T,\\
L_0
&=-\mathbb E_{q(x_1\mid x_0)}
[\log p_\theta(x_0\mid x_1)].
\end{aligned}
$$

命名中的 $L_{t-1}$ 表示该项训练的是从 $x_t$ 恢复 $x_{t-1}$ 的转移；当 $t$ 从 $2$ 到 $T$ 时，它覆盖 $L_1,\ldots,L_{T-1}$。$L_T$ 位于 latent chain 顶端，$L_0$ 位于数据端。

##### 第四步：逐项解释 $L_T$

$L_T$ 衡量 forward process 的终点是否匹配生成模型的起点。由 Part 3，$q(x_T\mid x_0)=\mathcal N(\sqrt{\bar\alpha_T}x_0,(1-\bar\alpha_T)I)$，而 $p(x_T)=\mathcal N(0,I)$。使用 Part 1 推导的 diagonal Gaussian KL，若数据维数为 $d$，可得

$$
L_T
=\frac12\left[
\bar\alpha_T\|x_0\|_2^2
+d\left(
(1-\bar\alpha_T)-1-\log(1-\bar\alpha_T)
\right)
\right].
$$

均值差贡献 $\bar\alpha_T\|x_0\|_2^2$，方差差贡献后面的 $d$ 倍项。当 schedule 使 $\bar\alpha_T\approx0$ 时，$q(x_T\mid x_0)$ 几乎等于标准 Gaussian，$L_T$ 接近零。若 noise schedule 固定，这一项不含 $\theta$，所以不会训练 denoising network；它仍然必须出现在完整似然界中，因为它验证 forward 终点与 reverse prior 是否衔接。

##### 第五步：逐项解释 $L_{t-1}$

$L_{t-1}$ 是 denoising matching term，也是主要训练成本。Oracle posterior 与 learned reverse transition 分别为 $q(x_{t-1}\mid x_t,x_0)=\mathcal N(\tilde\mu_t,\tilde\beta_t I)$ 和 $p_\theta(x_{t-1}\mid x_t)=\mathcal N(\mu_\theta,\Sigma_\theta)$。它要求只看 $x_t$ 的模型分布逼近同时知道 $x_t$ 与 $x_0$ 的解析 posterior。

若选择 $\Sigma_\theta=\tilde\beta_t I$，两个 Gaussian 的协方差相同。Gaussian KL 中的 log-determinant、trace 与维数常数互相抵消，只剩均值差：

$$
L_{t-1}
=\mathbb E_{q(x_t\mid x_0)}
\left[
\frac{1}{2\tilde\beta_t}
\left\|
\tilde\mu_t(x_t,x_0)-\mu_\theta(x_t,t)
\right\|_2^2
\right].
$$

$1/(2\tilde\beta_t)$ 是由共享协方差的逆矩阵产生的 timestep 权重；posterior 方差越小，均值预测误差受到的惩罚越大。若 $\Sigma_\theta$ 使用其他固定值或由模型学习，KL 还会包含方差之间的 log-determinant 与 trace 项，但由于两边都是 Gaussian，仍可解析计算。

##### 第六步：逐项解释 $L_0$

$L_0$ 是最后一步 reconstruction term，直接衡量从 $x_1$ 生成真实观测 $x_0$ 的 likelihood。这里不再写成普通的中间 KL，因为 $x_0$ 已经是观察到的数据，而在额外给定 $x_0$ 时，$q(x_0\mid x_1,x_0)$ 是集中在真实 $x_0$ 上的退化分布。

若示意性地设 $p_\theta(x_0\mid x_1)=\mathcal N(x_0;\mu_\theta(x_1,1),\sigma_0^2I)$，则

$$
L_0
=\mathbb E_{q(x_1\mid x_0)}
\left[
\frac{1}{2\sigma_0^2}
\|x_0-\mu_\theta(x_1,1)\|_2^2
\right]+C.
$$

$C$ 是与可训练均值无关的常数。对于离散像素，也可以选择更合适的离散 likelihood；无论具体 decoder 形式如何，$L_0$ 的职责都是保证链的最后一步真正给观测数据高概率。

##### 第七步：随机选择 timestep 估计中间求和

每个训练样本都计算 $T-1$ 个中间 KL 代价很高。因为

$$
\sum_{t=2}^{T}L_{t-1}
=(T-1)\,
\mathbb E_{t\sim\mathcal U\{2,\ldots,T\}}
[L_{t-1}],
$$

可以均匀随机选择一个 $t$，只构造一次 $x_t$ 并估计对应损失。$\mathcal U\{2,\ldots,T\}$ 表示离散均匀分布；乘回 $T-1$ 后是原求和的无偏估计。实际目标还可能采用非均匀 timestep sampling 或重新加权，但理论 VLB 的来源仍是上述完整分解。

#### 公式解释

三个损失覆盖 chain 的三个位置：$L_T$ 检查最噪端是否接上简单 prior；$L_{t-1}$ 训练每个中间 denoising transition；$L_0$ 检查最干净端是否真正重建观测。省略任何一类都不再是完整的 variational likelihood bound。

“最大化 ELBO”和“最小化 $L_T+\sum L_{t-1}+L_0$”完全等价，因为后者就是负 ELBO。KL 在 ELBO 中带负号，在 loss 中带正号；reconstruction log-likelihood 在 ELBO 中带正号，在 loss 中变成 negative log-likelihood。

在固定 schedule 的常见设置中，$L_T$ 对网络参数是常数；实际梯度主要来自 $L_{t-1}$ 与 $L_0$。但是“对梯度没有贡献”不等于“理论上可以假装它不存在”，它仍解释了为何 forward 终点必须接近 $\mathcal N(0,I)$。

#### 直觉理解

可以把完整 Diffusion loss 想成一条从数据到噪声、再从噪声返回数据的铁路验收。$L_T$ 检查最远端车站是否接上统一的 Gaussian 起点；每个 $L_{t-1}$ 检查一段返程轨道是否与已知 forward 轨道相容；$L_0$ 检查列车最终是否真正回到数据站。

Bayes 重写没有改变铁路本身，只是把“同时观察某一站左右两边”改成“给定起点答案后，检查从当前站返回上一站”。后者每次只需采一个 $x_t$，因此更适合随机梯度训练。

#### 本章总结

- **本章解决了什么问题：** 利用 Bayes 与望远镜式消去，把高方差 consistency ELBO 改写成标准的 variational loss，并逐项解释 $L_T$、$L_{t-1}$ 与 $L_0$。
- **核心公式：** $\mathcal L_{\mathrm{VLB}}=\mathbb E_{p_{\mathrm{data}}}[L_T+\sum_{t=2}^{T}L_{t-1}+L_0]$。
- **需要记住的直觉：** $L_T$ 连接 Gaussian prior，$L_{t-1}$ 学习逐步去噪，$L_0$ 保证最终重建数据；三者来自同一个 Hierarchical VAE ELBO，而不是人为拼接。
- **与下一章的关系：** Part 6 将继续化简占主导地位的 $L_{t-1}$，把 Gaussian 均值匹配转换为神经网络预测累计噪声 $\epsilon$ 的均方误差。

## Part 6：DDPM Training Objective

Part 5 已把主要的中间损失 $L_{t-1}$ 化为两个 Gaussian reverse transition 之间的 KL；当协方差固定相同时，它等价于 posterior mean 与 neural mean 的加权均方误差。本部分继续沿论文推导，把“预测反向均值”改写为“预测构造 $x_t$ 时加入的累计 Gaussian 噪声”，最终得到 DDPM 最常见的训练目标。

---

### 6.1 从 Reverse Mean Matching 到 Noise Prediction MSE

#### 背景

Part 4 的真实 posterior mean 可以写成 $x_t$ 与 $x_0$ 的线性组合：

$$
\tilde\mu_t(x_t,x_0)
=A_t x_t+B_t x_0,
$$

其中

$$
A_t=\frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}
{1-\bar\alpha_t},
\qquad
B_t=\frac{\sqrt{\bar\alpha_{t-1}}\beta_t}
{1-\bar\alpha_t}.
$$

这里使用了 $\beta_t=1-\alpha_t$。$A_t$ 与 $B_t$ 都由固定 noise schedule 决定；$A_t x_t$ 是已知部分，困难只在真实 $x_0$。因此可以让神经网络从 $(x_t,t)$ 预测干净样本 $\hat x_\theta(x_t,t)$，再定义 neural reverse mean

$$
\mu_\theta(x_t,t)
=A_t x_t+B_t\hat x_\theta(x_t,t).
$$

若网络准确预测 $x_0$，neural mean 就等于真实 posterior mean。论文先由此得到 $x_0$ prediction objective，再利用 forward 重参数化把它变成 noise prediction objective。

#### 数学推导

##### 第一步：KL 先化为 $x_0$ 预测误差

在 $q(x_{t-1}\mid x_t,x_0)$ 与 $p_\theta(x_{t-1}\mid x_t)$ 使用同一协方差 $\tilde\beta_t I$ 时，Part 5 已得到

$$
L_{t-1}
=\mathbb E_{q(x_t\mid x_0)}
\left[
\frac{1}{2\tilde\beta_t}
\|\tilde\mu_t-\mu_\theta\|_2^2
\right].
$$

将两个均值的线性形式代入，公共项 $A_t x_t$ 抵消：

$$
\tilde\mu_t(x_t,x_0)-\mu_\theta(x_t,t)
=B_t\left(x_0-\hat x_\theta(x_t,t)\right).
$$

因此

$$
L_{t-1}
=\mathbb E_{q(x_t\mid x_0)}
\left[
\frac{B_t^2}{2\tilde\beta_t}
\|x_0-\hat x_\theta(x_t,t)\|_2^2
\right].
$$

这说明 reverse mean matching 已经等价于带 timestep 权重的 clean-image regression。权重 $B_t^2/(2\tilde\beta_t)$ 与网络参数无关，只由 schedule 和 timestep 决定。

##### 第二步：用 forward 公式表示真实 $x_0$

Part 3 的直接采样公式为

$$
x_t=\sqrt{\bar\alpha_t}x_0
+\sqrt{1-\bar\alpha_t}\epsilon,
\qquad
\epsilon\sim\mathcal N(0,I).
$$

将噪声项移到左侧，再除以 $\sqrt{\bar\alpha_t}$：

$$
x_0
=\frac{x_t-\sqrt{1-\bar\alpha_t}\epsilon}
{\sqrt{\bar\alpha_t}}.
$$

这个公式描述什么？训练时我们先采样真实 $x_0$ 与噪声 $\epsilon$，再合成 $x_t$，所以等式中的所有量都已知。它也说明，若能从 $x_t$ 预测累计噪声 $\epsilon$，就能通过简单代数恢复 $x_0$。

让神经网络输出 $\epsilon_\theta(x_t,t)$，并用同一代数关系定义隐含的 clean prediction：

$$
\hat x_\theta(x_t,t)
=\frac{x_t-\sqrt{1-\bar\alpha_t}\epsilon_\theta(x_t,t)}
{\sqrt{\bar\alpha_t}}.
$$

真实 $x_0$ 与预测 $\hat x_\theta$ 的差为

$$
\hat x_\theta(x_t,t)-x_0
=\sqrt{\frac{1-\bar\alpha_t}{\bar\alpha_t}}
\left(
\epsilon-\epsilon_\theta(x_t,t)
\right).
$$

因此，预测 $x_0$ 与预测 $\epsilon$ 不是两项互不相关的任务；给定 $x_t$ 和 $t$ 后，它们通过可逆线性关系一一转换。

##### 第三步：得到严格 ELBO 对应的 weighted noise loss

把上面的差代入 $x_0$ prediction loss，并使用 $B_t^2=\bar\alpha_{t-1}\beta_t^2/(1-\bar\alpha_t)^2$ 与 $\bar\alpha_t=\alpha_t\bar\alpha_{t-1}$：

$$
\begin{aligned}
\frac{B_t^2}{2\tilde\beta_t}
\|x_0-\hat x_\theta\|_2^2
&=\frac{B_t^2}{2\tilde\beta_t}
\frac{1-\bar\alpha_t}{\bar\alpha_t}
\|\epsilon-\epsilon_\theta(x_t,t)\|_2^2\\
&=\frac{\beta_t^2}
{2\tilde\beta_t\alpha_t(1-\bar\alpha_t)}
\|\epsilon-\epsilon_\theta(x_t,t)\|_2^2.
\end{aligned}
$$

定义正权重

$$
w_t=\frac{\beta_t^2}
{2\tilde\beta_t\alpha_t(1-\bar\alpha_t)},
$$

则严格由中间 ELBO 项导出的噪声目标是

$$
\mathcal L_{\epsilon,\mathrm{weighted}}
=\mathbb E_{x_0,t,\epsilon}
\left[
w_t
\left\|
\epsilon-epsilon_\theta(x_t,t)
\right\|_2^2
\right],
$$

其中 $x_0\sim p_{\mathrm{data}}$，$t$ 从选定的 timestep 分布采样，$\epsilon\sim\mathcal N(0,I)$，而 $x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon$。若 reverse variance 选择的不是 $\tilde\beta_t I$，权重分母中的 $\tilde\beta_t$ 会相应替换为该 variance coefficient。

##### 第四步：从 weighted objective 到 DDPM simplified objective

实践中常用的 DDPM simplified objective 去掉了显式的 $w_t$：

$$
\mathcal L_{\mathrm{simple}}
=\mathbb E_{x_0,t,\epsilon}
\left[
\left\|
\epsilon-epsilon_\theta(x_t,t)
\right\|_2^2
\right].
$$

压缩记号后，这正是常见的 $\mathbb E[\|\epsilon-\epsilon_\theta(x_t,t)\|_2^2]$。

必须准确理解“变成”的含义。对固定 timestep $t$，$w_t>0$ 且不依赖 $\theta$，所以 weighted 与 unweighted MSE 的最优预测器相同：都要求 $\epsilon_\theta(x_t,t)$ 逼近真实条件噪声。可是同一个网络共享所有 timestep 时，删除 $w_t$ 会改变不同噪声等级对总梯度的相对贡献。因此 $\mathcal L_{\mathrm{simple}}$ 不是与完整 VLB 数值完全相等的恒等变换，而是保留相同回归目标、重新平衡 timestep 的实用训练目标。

如果按与 $w_t$ 成比例的概率采样 timestep，并做相应归一化，也可以把 weighted sum 写成对无显式权重误差的期望。常见实现则直接均匀采样 $t$ 并使用 simplified loss，这是一项经验上有效的 objective reweighting。

#### 公式解释

目标中的真实 $\epsilon$ 是构造当前 $x_t$ 时使用的**累计源噪声**。它来自 $q(x_t\mid x_0)$ 的一步闭式采样，不是 Markov chain 中最后一次局部转移使用的单步噪声 $\epsilon_t$。网络输出 $\epsilon_\theta(x_t,t)$ 与 $x_t$ 形状相同，逐维估计这份累计噪声。

均方误差 $\|\epsilon-\epsilon_\theta\|_2^2$ 是所有像素或特征维度预测误差的平方和。外层期望表示训练会平均覆盖真实数据、不同 noise level 和不同 Gaussian corruption，而不是只针对一张图片或一个 timestep。

#### 直觉理解

与其要求网络直接画出完全干净的图片，可以让它回答：“当前 noisy sample 中，哪一部分是我人工加入的 Gaussian 噪声？”训练时这份答案由我们亲自采样，因此监督信号免费且精确。去掉预测噪声后，剩余部分就对应干净信号估计。

噪声目标还有一个实用好处：无论 $t$ 是多少，监督目标始终来自标准 Gaussian，数值尺度相对统一。预测 $x_0$ 时，输入在高噪声等级几乎不含可见信号，回归难度随 $t$ 变化很大；noise parameterization 通常提供更规整的学习问题。

#### 本章总结

- **本章解决了什么问题：** 从 Gaussian reverse mean matching 逐步推导出 $x_0$ MSE、weighted noise MSE 和常用 DDPM simplified noise objective，并说明它们的等价范围与权重差异。
- **核心公式：** $\mathcal L_{\mathrm{simple}}=\mathbb E_{x_0,t,\epsilon}[\|\epsilon-\epsilon_\theta(x_t,t)\|_2^2]$。
- **需要记住的直觉：** 训练时噪声是我们亲自加入的已知答案；预测累计噪声就等价于预测干净数据，并能进一步确定反向 Gaussian 的均值。
- **与下一章的关系：** 下一章将把这个目标落实成网络的输入、输出和一次完整训练迭代，并说明预测噪声怎样真正产生 $x_{t-1}$。

---

### 6.2 网络输入、输出与“预测噪声为何能够去噪”

#### 背景

数学目标确定以后，神经网络只需要实现一个时间条件函数 $\epsilon_\theta(x_t,t)$。它不接收 $x_0$，因为生成阶段不存在真实干净答案；它接收 noisy state $x_t$ 和 timestep $t$，输出对累计源噪声 $\epsilon$ 的估计。

$x_t$ 与输出通常具有相同空间形状。例如输入是 $H\times W\times C$ 的 noisy image，输出也是 $H\times W\times C$ 的噪声张量。$t$ 通常先编码为 time embedding，再注入网络的多个层。具体网络可以是 U-Net 或其他架构，但概率推导只要求它能表示随 $x_t$ 与 $t$ 变化的函数。

为什么必须输入 $t$？相同数值范围的 $x_t$ 在不同 noise level 下含有不同的信噪比，去噪尺度也不同。若网络不知道 $t$，就无法判断应把多少结构解释为信号、多少解释为噪声，也无法使用对应的 $\alpha_t$、$\bar\alpha_t$ 关系。

#### 数学推导

##### 一次完整训练迭代

一次 DDPM noise-prediction 训练迭代可以从目标公式逐项读出：

1. 从数据集采样干净样本 $x_0\sim p_{\mathrm{data}}$。
2. 从 $\{1,\ldots,T\}$ 采样 timestep $t$；simplified objective 常使用均匀分布。
3. 采样累计源噪声 $\epsilon\sim\mathcal N(0,I)$。
4. 一步构造 $x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon$。
5. 计算网络输出 $\epsilon_\theta(x_t,t)$。
6. 计算 $\|\epsilon-\epsilon_\theta(x_t,t)\|_2^2$，对 batch 求平均并反向传播更新 $\theta$。

把这些随机步骤写成一个完整期望，就是

$$
\mathcal L_{\mathrm{simple}}(\theta)
=\mathbb E_{\substack{
x_0\sim p_{\mathrm{data}},\;
t\sim\mathcal U\{1,\ldots,T\},\\
\epsilon\sim\mathcal N(0,I)
}}
\left[
\left\|
\epsilon-epsilon_\theta\!\left(
\sqrt{\bar\alpha_t}x_0
+\sqrt{1-\bar\alpha_t}\epsilon,
t
\right)
\right\|_2^2
\right].
$$

公式把网络看到的 $x_t$ 直接展开为 $x_0$ 与 $\epsilon$ 的组合。训练无需顺序计算 $x_1,\ldots,x_{t-1}$，也无需知道真实 reverse transition。

##### 从预测噪声恢复 $x_0$

给定网络预测，可以立刻计算 clean estimate：

$$
\hat x_0(x_t,t)
=\frac{x_t-\sqrt{1-\bar\alpha_t}\epsilon_\theta(x_t,t)}
{\sqrt{\bar\alpha_t}}.
$$

若 $\epsilon_\theta=\epsilon$，将 forward 公式代入后，分子变为 $\sqrt{\bar\alpha_t}x_0$，再除以 $\sqrt{\bar\alpha_t}$ 就精确恢复 $x_0$。网络并非只学到一个抽象噪声标签；它输出的信息足以重建 clean prediction。

##### 从预测噪声得到反向均值

将 $\hat x_0$ 代入 Part 4 的 posterior mean 参数化并化简，可得到常用的 epsilon-parameterized reverse mean：

$$
\mu_\theta(x_t,t)
=\frac{1}{\sqrt{\alpha_t}}
\left(
x_t-rac{\beta_t}{\sqrt{1-\bar\alpha_t}}
\epsilon_\theta(x_t,t)
\right).
$$

这个公式的来源不是经验猜测。Part 4 的 posterior mean 为 $A_t x_t+B_t x_0$；用 $\hat x_0=(x_t-\sqrt{1-\bar\alpha_t}\epsilon_\theta)/\sqrt{\bar\alpha_t}$ 替换 $x_0$，再使用 $\bar\alpha_t=\alpha_t\bar\alpha_{t-1}$ 与 $\beta_t=1-\alpha_t$，$x_t$ 的系数合并为 $1/\sqrt{\alpha_t}$，噪声系数合并为 $-\beta_t/(\sqrt{\alpha_t}\sqrt{1-\bar\alpha_t})$，正好得到上式。

若 reverse transition 定义为 $p_\theta(x_{t-1}\mid x_t)=\mathcal N(\mu_\theta(x_t,t),\sigma_t^2I)$，一次反向采样为

$$
x_{t-1}
=\mu_\theta(x_t,t)+\sigma_t\zeta,
\qquad
\zeta\sim\mathcal N(0,I).
$$

$\epsilon_\theta$ 估计已经存在于 $x_t$ 中的 forward corruption；$\zeta$ 是反向 Gaussian sampling 新加入的随机性，两者不能混为一谈。通常在 $t>1$ 时采样 $\zeta$，而最后到 $x_0$ 的步骤按所选 decoder 处理，不再加入普通中间噪声。

从 $x_T\sim\mathcal N(0,I)$ 开始，对 $t=T,T-1,\ldots,1$ 重复“预测 $\epsilon_\theta$、计算 $\mu_\theta$、采样 $x_{t-1}$”，就得到完整生成过程。由此可见，噪声预测已经充分参数化了每一步 reverse mean，所以能够实际完成去噪采样。

#### 公式解释

网络输入 $x_t$ 提供当前 noisy observation，输入 $t$ 提供 noise level，输出 $\epsilon_\theta(x_t,t)$ 提供累计 corruption estimate。利用已知 schedule，三者共同确定 $\hat x_0$ 与 $\mu_\theta(x_t,t)$；再结合预设或学习的 reverse variance，便确定完整 Gaussian transition。

训练只运行一次随机 timestep，是因为 forward closed form 能直接构造 $x_t$；生成仍需按顺序运行多个 reverse step，是因为每个 $x_{t-1}$ 都依赖当前模型采样得到的 $x_t$，不存在训练时那种已知 $x_0$ 的直接捷径。

#### 直觉理解

可以把 $x_t$ 想成“信号与噪声的混音”。网络根据混音本身和已知的混音比例 $t$，估计其中的噪声音轨。减去估计噪声并按 schedule 调整音量，就得到原始信号估计；把这份估计代入 Gaussian posterior，就能生成稍微更干净的上一状态。

一次噪声预测不会直接从高噪声 $x_T$ 跳到完美图片。它只为当前 timestep 提供正确方向和尺度的局部反向均值，随后新的 $x_{t-1}$ 又成为下一次预测的输入。许多局部去噪步骤串联，才把标准 Gaussian 转换为数据样本。

#### 本章总结

- **本章解决了什么问题：** 明确了 DDPM 网络的输入、输出和监督信号，并从预测噪声推导出 clean estimate、reverse mean 与实际反向采样步骤。
- **核心公式：** $\mu_\theta(x_t,t)=\frac{1}{\sqrt{\alpha_t}}(x_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\epsilon_\theta(x_t,t))$。
- **需要记住的直觉：** 网络预测的不是随意噪声，而是构造当前 $x_t$ 的累计 Gaussian corruption；知道它就能恢复 $x_0$ 估计并确定下一步去噪分布。
- **与下一章的关系：** Part 7 将把这里的 $\epsilon$ parameterization 与直接预测 $x_0$、预测 score 的形式并列，完整证明三种网络输出如何相互转换。

## Part 7：Three Equivalent Interpretations

前面已经得到一个可以训练和采样的 DDPM：训练时让网络预测加入 $x_0$ 的噪声，采样时再把这个预测代入反向高斯分布的均值。但“预测噪声”并不是唯一的表达方式。同一个反向去噪方向也可以用原始样本 $x_0$ 或概率分布的 score 表示。本 Part 将按照论文的顺序证明三种表示如何互相转换，并解释“等价”究竟指什么。

### 7.1 预测原始样本 $x_0$ 与预测噪声 $\epsilon$

#### 背景

在 Part 3 中，我们把 $t$ 次前向加噪压缩成了一个公式：$x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon$，其中
$\epsilon\sim\mathcal N(0,I)$。这个公式同时包含四个量：干净样本 $x_0$、带噪样本 $x_t$、时间步 $t$ 和本次采样的累计噪声 $\epsilon$。只要知道其中三个量，就可以通过代数变形求出第四个量。因此，网络输出 $x_0$ 还是输出 $\epsilon$，首先是同一条关系的两种参数化。

这里的“参数化”是指：我们仍然要构造同一个反向均值 $\mu_\theta(x_t,t)$，只是选择让神经网络直接输出哪个中间量。它不意味着不同输出的未经加权损失在数值上完全相同；这一点会在本节后面单独说明。

#### 数学推导

先从前向重参数化公式出发：

$$
x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon,\qquad \epsilon\sim\mathcal N(0,I).
$$

为了求 $x_0$，先把噪声项移到等号左边，得到 $x_t-\sqrt{1-\bar\alpha_t}\epsilon=\sqrt{\bar\alpha_t}x_0$，然后两边同时除以 $\sqrt{\bar\alpha_t}$：

$$
x_0=\frac{x_t-\sqrt{1-\bar\alpha_t}\epsilon}{\sqrt{\bar\alpha_t}}.
$$

这不是新的概率结论，只是前向公式的代数变形。若网络预测噪声 $\epsilon_\theta(x_t,t)$，便可以定义相应的原始样本预测
$\hat x_{0,\theta}(x_t,t)=\frac{x_t-\sqrt{1-\bar\alpha_t}\epsilon_\theta(x_t,t)}{\sqrt{\bar\alpha_t}}$。反过来，若网络直接输出 $\hat x_{0,\theta}(x_t,t)$，则相应的噪声预测为 $\epsilon_\theta(x_t,t)=\frac{x_t-\sqrt{\bar\alpha_t}\hat x_{0,\theta}(x_t,t)}{\sqrt{1-\bar\alpha_t}}$。

接下来需要证明：这不只是能够互换两个输出，而且会给出同一个反向转移均值。Part 4 已经推导出真实后验 $q(x_{t-1}\mid x_t,x_0)$ 的均值：

$$
\tilde\mu_t(x_t,x_0)=\frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}x_t+\frac{\sqrt{\bar\alpha_{t-1}}\beta_t}{1-\bar\alpha_t}x_0.
$$

把刚才的 $x_0=\frac{x_t-\sqrt{1-\bar\alpha_t}\epsilon}{\sqrt{\bar\alpha_t}}$ 代入第二项，可得

$\tilde\mu_t=\frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}x_t+\frac{\sqrt{\bar\alpha_{t-1}}\beta_t}{(1-\bar\alpha_t)\sqrt{\bar\alpha_t}}x_t-\frac{\sqrt{\bar\alpha_{t-1}}\beta_t}{\sqrt{\bar\alpha_t}\sqrt{1-\bar\alpha_t}}\epsilon$。

现在逐项化简。由于 $\bar\alpha_t=\alpha_t\bar\alpha_{t-1}$，所以 $\frac{\sqrt{\bar\alpha_{t-1}}}{\sqrt{\bar\alpha_t}}=\frac{1}{\sqrt{\alpha_t}}$。因此噪声项的系数变成
$-\frac{\beta_t}{\sqrt{\alpha_t}\sqrt{1-\bar\alpha_t}}$。两个 $x_t$ 项的系数相加为

$\frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})+\beta_t/\sqrt{\alpha_t}}{1-\bar\alpha_t}=\frac{\alpha_t(1-\bar\alpha_{t-1})+\beta_t}{\sqrt{\alpha_t}(1-\bar\alpha_t)}$。

再利用 $\beta_t=1-\alpha_t$，分子为 $\alpha_t-\alpha_t\bar\alpha_{t-1}+1-\alpha_t=1-\bar\alpha_t$。它与分母中的 $1-\bar\alpha_t$ 约掉后，$x_t$ 的系数就是 $1/\sqrt{\alpha_t}$。于是完整结果为

$$
\tilde\mu_t(x_t,x_0)=\frac{1}{\sqrt{\alpha_t}}\left(x_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\epsilon\right).
$$

训练时我们不知道反向采样中的真实 $x_0$ 和真实 $\epsilon$，所以用神经网络的输出替代它们。若网络预测 $x_0$，反向均值写成

$\mu_\theta^{(x_0)}(x_t,t)=\frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})x_t+\sqrt{\bar\alpha_{t-1}}\beta_t\hat x_{0,\theta}(x_t,t)}{1-\bar\alpha_t}$。

若网络预测噪声，反向均值写成

$$
\mu_\theta^{(\epsilon)}(x_t,t)=\frac{1}{\sqrt{\alpha_t}}\left(x_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\epsilon_\theta(x_t,t)\right).
$$

只要 $\hat x_{0,\theta}$ 与 $\epsilon_\theta$ 按前面的转换公式对应，这两个均值就完全相同。也就是说，采样器最终使用的反向高斯分布没有改变，改变的是神经网络描述它的坐标。

最后讨论损失。由真实 $x_0$ 与预测 $\hat x_{0,\theta}$ 的两个重构公式相减，可得

$\hat x_{0,\theta}-x_0=\sqrt{\frac{1-\bar\alpha_t}{\bar\alpha_t}}(\epsilon-\epsilon_\theta)$，因而

$$
\|\hat x_{0,\theta}-x_0\|_2^2=\frac{1-\bar\alpha_t}{\bar\alpha_t}\|\epsilon-\epsilon_\theta\|_2^2.
$$

所以，两种误差只差一个由时间步 $t$ 决定的系数。若为损失配上相应权重，它们表示同一个优化问题；若都直接使用“不加权 MSE”，它们会以不同强度训练不同时间步，优化轨迹和实际效果可能不同。

#### 公式解释

前向公式 $x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon$ 描述从数据 $x_0$ 一步采样任意噪声级别 $x_t$ 的方法。$x_0$ 是原始数据，$x_t$ 是第 $t$ 步的带噪数据，$\epsilon$ 是与数据同形状的标准高斯噪声，$\bar\alpha_t=\prod_{i=1}^t\alpha_i$，而 $\alpha_i=1-\beta_i$。$\sqrt{\bar\alpha_t}$ 控制保留多少原始信号，$\sqrt{1-\bar\alpha_t}$ 控制加入多少累计噪声。这个公式之所以需要，是因为所有三种参数化都从它出发。

$x_0=\frac{x_t-\sqrt{1-\bar\alpha_t}\epsilon}{\sqrt{\bar\alpha_t}}$ 描述在知道 $x_t$ 和本次真实噪声 $\epsilon$ 时如何精确还原 $x_0$。分子先从 $x_t$ 中减去噪声分量，分母再撤销信号缩放。Diffusion 网络并不知道真实 $\epsilon$，但可以用 $\epsilon_\theta$ 代替它，从而得到 $\hat x_{0,\theta}$。

反向均值公式中的 $\tilde\mu_t$ 是已知 $x_t$ 和真实 $x_0$ 时，$x_{t-1}$ 的后验均值；$\mu_\theta$ 是模型用网络预测近似出来的均值；$\beta_t=1-\alpha_t$ 是第 $t$ 步新增噪声的方差。反向均值公式把网络输出真正连接到了生成过程：网络本身并不是直接输出下一张图，而是提供构造 $p_\theta(x_{t-1}\mid x_t)$ 所需的均值。

误差换算公式描述了两种参数化的尺度差异。当天然信号已经很弱，即 $\bar\alpha_t$ 很小时，系数 $(1-\bar\alpha_t)/\bar\alpha_t$ 会很大。因此，在各时间步上直接使用相同权重的 $x_0$-MSE，与直接使用相同权重的噪声 MSE 并不是同一个训练偏好。它在 Diffusion 中提醒我们区分“表达能力等价”和“具体损失加权等价”。

#### 直觉理解

可以把 $x_t$ 想成一杯由“原图信号”和“随机噪声”按已知比例混合的液体。预测 $x_0$ 是直接说出其中原图成分的内容；预测 $\epsilon$ 是先说出噪声成分，再用总混合物减去它。只要混合比例已知，两种答案可以互相换算。

但是，数值尺度会随 $t$ 改变。早期的 $x_t$ 几乎就是 $x_0$，从中预测原图较容易；晚期的 $x_t$ 几乎都是噪声，一个很小的噪声预测误差在除以 $\sqrt{\bar\alpha_t}$ 后可能变成很大的 $x_0$ 误差。因此，选择输出参数化也隐含了如何平衡不同噪声级别的训练问题。

#### 本章总结

**本章解决了什么问题：** 本章证明了预测原始样本 $x_0$ 与预测累计噪声 $\epsilon$ 都能构造同一个反向去噪均值，并说明了参数化等价不等于未经加权的训练损失完全相同。

**核心公式：** 两个输出通过 $\hat x_{0,\theta}=\frac{x_t-\sqrt{1-\bar\alpha_t}\epsilon_\theta}{\sqrt{\bar\alpha_t}}$ 与 $\epsilon_\theta=\frac{x_t-\sqrt{\bar\alpha_t}\hat x_{0,\theta}}{\sqrt{1-\bar\alpha_t}}$ 互相转换；噪声参数化的反向均值为 $\mu_\theta=\frac{1}{\sqrt{\alpha_t}}(x_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\epsilon_\theta)$。

**需要记住的直觉：** 预测原图是在估计混合物中的信号，预测噪声是在估计需要从混合物中减掉的部分。二者包含相同信息，但不同的损失尺度会改变各时间步受到的训练权重。

**与下一章的关系：** 下一章将引入第三种表示——score。它不直接回答“原图是什么”或“加入了什么噪声”，而是回答“在当前带噪数据空间中，朝哪个方向移动会进入概率更高的区域”。

### 7.2 预测 Score 并统一三种参数化

#### 背景

论文的第三种解释是让网络预测 score function。对一个密度 $q_t(x_t)$，score 定义为 $s_t(x_t)=\nabla_{x_t}\log q_t(x_t)$。符号 $\nabla_{x_t}$ 表示对向量 $x_t$ 的每个分量分别求偏导并排成一个向量；这个向量指向
$\log q_t(x_t)$ 增长最快的方向。由于对数函数严格递增，增加 $\log q_t$ 也就是增加 $q_t$，所以 score 可以直观理解为“走向更高概率密度区域的方向”。梯度与 score 的基础会在 Part 8 进一步展开，本节先专注于它与 $x_0$、$\epsilon$ 的代数和概率关系。

这里必须区分两个分布。给定某个固定 $x_0$ 时，前向条件分布是 $q(x_t\mid x_0)=\mathcal N(\sqrt{\bar\alpha_t}x_0,(1-\bar\alpha_t)I)$。如果先从真实数据分布 $p_{\text{data}}(x_0)$ 抽取 $x_0$，再对它加噪，所有可能的 $x_0$ 混合后形成第 $t$ 步的边缘分布
$q_t(x_t)=\int q(x_t\mid x_0)p_{\text{data}}(x_0)\,dx_0$。论文有时把这个边缘密度简写成 $p(x_t)$；本教程写成 $q_t(x_t)$，以免与反向生成模型 $p_\theta$ 混淆。

#### 数学推导

先计算条件分布的 score。令 $\sigma_t^2=1-\bar\alpha_t$，则高斯密度的对数可以写成

$\log q(x_t\mid x_0)=C-\frac{1}{2\sigma_t^2}\|x_t-\sqrt{\bar\alpha_t}x_0\|_2^2$，其中 $C$ 收集所有与 $x_t$ 无关的常数。向量平方距离对 $x_t$ 的梯度是 $2(x_t-\sqrt{\bar\alpha_t}x_0)$，所以

$$
\nabla_{x_t}\log q(x_t\mid x_0)=-\frac{x_t-\sqrt{\bar\alpha_t}x_0}{1-\bar\alpha_t}=-\frac{\epsilon}{\sqrt{1-\bar\alpha_t}}.
$$

最后一个等号使用了 $x_t-\sqrt{\bar\alpha_t}x_0=\sqrt{1-\bar\alpha_t}\epsilon$。这条等式对一次已知 $x_0$ 和 $\epsilon$ 的前向采样逐样本成立，但左边是条件 score $\nabla_{x_t}\log q(x_t\mid x_0)$，还不是网络在只看到 $x_t$ 时需要学习的边缘 score。

下面从条件 score 推导边缘 score。先对 $q_t(x_t)=\int q(x_t\mid x_0)p_{\text{data}}(x_0)\,dx_0$ 求梯度：

$\nabla_{x_t}q_t(x_t)=\int \nabla_{x_t}q(x_t\mid x_0)p_{\text{data}}(x_0)\,dx_0$。

这一步把梯度移入积分；在这里高斯核足够光滑且衰减良好，因此可以这样交换求导与积分。再用 $\nabla\log q_t=(\nabla q_t)/q_t$，并把 $\nabla q(x_t\mid x_0)$ 写成 $q(x_t\mid x_0)\nabla\log q(x_t\mid x_0)$，得到

$\nabla_{x_t}\log q_t(x_t)=\int \frac{q(x_t\mid x_0)p_{\text{data}}(x_0)}{q_t(x_t)}\nabla_{x_t}\log q(x_t\mid x_0)\,dx_0$。

根据 Bayes 公式，分数
$\frac{q(x_t\mid x_0)p_{\text{data}}(x_0)}{q_t(x_t)}$ 正是给定 $x_t$ 后 $x_0$ 的后验密度 $q(x_0\mid x_t)$。因此这个积分就是条件期望：

$$
s_t(x_t):=\nabla_{x_t}\log q_t(x_t)=\mathbb E\!\left[\nabla_{x_t}\log q(x_t\mid x_0)\mid x_t\right]= -\frac{\mathbb E[\epsilon\mid x_t]}{\sqrt{1-\bar\alpha_t}}.
$$

为什么 Part 6 的噪声网络恰好能够提供这里的 $\mathbb E[\epsilon\mid x_t]$？设网络能看到的输入为 $X=(x_t,t)$，训练目标为随机变量 $Y=\epsilon$，并记 $m(X)=\mathbb E[Y\mid X]$。对任何预测函数 $f(X)$，有平方误差分解

$\mathbb E\|Y-f(X)\|_2^2=\mathbb E\|Y-m(X)\|_2^2+\mathbb E\|m(X)-f(X)\|_2^2$。

推导方法是先写 $Y-f=(Y-m)+(m-f)$ 并展开平方。交叉项的期望为零，因为给定 $X$ 时，$m-f$ 是固定的，而 $\mathbb E[Y-m\mid X]=0$。第一项与 $f$ 无关，第二项非负，只有 $f(X)=m(X)$ 时达到最小。因此无限数据、足够模型容量和充分优化下，噪声 MSE 的最优预测为

$\epsilon_\theta^*(x_t,t)=\mathbb E[\epsilon\mid x_t,t]=\mathbb E[\epsilon\mid x_t]$。时间步 $t$ 已经决定了当前分布，最后一个等号只是省略了已知的 $t$。代入边缘 score 公式便得到模型输出之间的关系

$$
s_\theta(x_t,t)=-\frac{\epsilon_\theta(x_t,t)}{\sqrt{1-\bar\alpha_t}}.
$$

接着推导 $x_0$ 与边缘 score 的关系。把条件 score 的第一种形式放入条件期望，有

$s_t(x_t)=-\frac{x_t-\sqrt{\bar\alpha_t}\mathbb E[x_0\mid x_t]}{1-\bar\alpha_t}$。这里 $x_t$ 已经给定，所以它可以直接移出条件期望。移项后得到高斯加噪情形下的 Tweedie 公式：

$$
\mathbb E[x_0\mid x_t]=\frac{x_t+(1-\bar\alpha_t)\nabla_{x_t}\log q_t(x_t)}{\sqrt{\bar\alpha_t}}.
$$

这个公式描述什么？$x_t$ 是当前带噪样本，$\nabla_{x_t}\log q_t(x_t)$ 是把它推向更高数据密度区域的修正方向，$1-\bar\alpha_t$ 按噪声方差调节修正幅度，最后除以 $\sqrt{\bar\alpha_t}$ 撤销前向过程对信号的缩放。它从边缘 score 的定义和高斯条件 score 推导而来，在 Diffusion 中把 score 网络的输出转换为最佳平方误差意义下的干净样本估计。

需要注意，若知道一次采样所用的真实 $\epsilon$，Part 7.1 的公式可以精确恢复该次的 $x_0$；但网络只看到 $x_t$，同一个 $x_t$ 可能由多个 $x_0$ 与噪声组合产生，所以一般只能得到后验平均 $\mathbb E[x_0\mid x_t]$，而不能确定那一次唯一的真实 $x_0$。

现在可以把三种网络输出放在同一张换算表中：

| 网络直接输出        | 转换为 $\hat x_{0,\theta}$                                             | 转换为 $\epsilon_\theta$                                                 | 转换为 $s_\theta$                                                 |
| ------------------- | ---------------------------------------------------------------------- | ------------------------------------------------------------------------ | ----------------------------------------------------------------- |
| $\hat x_{0,\theta}$ | 本身                                                                   | $\frac{x_t-\sqrt{\bar\alpha_t}\hat x_{0,\theta}}{\sqrt{1-\bar\alpha_t}}$ | $\frac{\sqrt{\bar\alpha_t}\hat x_{0,\theta}-x_t}{1-\bar\alpha_t}$ |
| $\epsilon_\theta$   | $\frac{x_t-\sqrt{1-\bar\alpha_t}\epsilon_\theta}{\sqrt{\bar\alpha_t}}$ | 本身                                                                     | $-\frac{\epsilon_\theta}{\sqrt{1-\bar\alpha_t}}$                  |
| $s_\theta$          | $\frac{x_t+(1-\bar\alpha_t)s_\theta}{\sqrt{\bar\alpha_t}}$             | $-\sqrt{1-\bar\alpha_t}s_\theta$                                         | 本身                                                              |

最后验证 score 如何进入反向采样。把 $\epsilon_\theta=-\sqrt{1-\bar\alpha_t}s_\theta$ 代入噪声形式的反向均值：

$$
\mu_\theta^{(s)}(x_t,t)=\frac{1}{\sqrt{\alpha_t}}\left(x_t+\beta_t s_\theta(x_t,t)\right).
$$

因此，score 乘以 $\beta_t$ 后就是反向均值相对于缩放后 $x_t$ 的修正方向。由于 score 指向更高的带噪数据密度，这个修正会把随机噪声逐步推回数据分布。

三种表示的损失尺度同样可以换算。对一次训练样本，条件 score 目标为
$-\epsilon/\sqrt{1-\bar\alpha_t}$。因此

$\|\epsilon-\epsilon_\theta\|_2^2=(1-\bar\alpha_t)\|{-\epsilon}/{\sqrt{1-\bar\alpha_t}}-s_\theta\|_2^2$。

右边监督的是可计算的条件 score；在期望意义下，它的最优预测正是边缘 score。和上一节一样，若忽略这个随 $t$ 变化的系数，不加权的噪声 MSE 与不加权的 score MSE 会强调不同的噪声级别。

#### 公式解释

$q_t(x_t)=\int q(x_t\mid x_0)p_{\text{data}}(x_0)\,dx_0$ 描述第 $t$ 个噪声级别上所有带噪样本的总体分布。积分变量 $x_0$ 遍历所有可能的干净数据，$p_{\text{data}}(x_0)$ 表示数据出现的密度，$q(x_t\mid x_0)$ 表示该数据被加噪成 $x_t$ 的密度。需要这个公式，是因为采样时网络只有 $x_t$，不知道它具体来自哪个 $x_0$，所以网络面对的是混合后的边缘分布。

条件 score 公式
$\nabla_{x_t}\log q(x_t\mid x_0)=-\epsilon/\sqrt{1-\bar\alpha_t}$ 描述在已知干净样本时，高斯密度的上升方向恰好与所加噪声方向相反。负号表示去噪应逆着噪声移动，分母表示相同大小的噪声在不同噪声方差下应使用不同尺度。

边缘 score 公式
$\nabla_{x_t}\log q_t(x_t)=-\mathbb E[\epsilon\mid x_t]/\sqrt{1-\bar\alpha_t}$ 描述网络实际能够学习的目标。条件期望 $\mathbb E[\epsilon\mid x_t]$ 是在所有可能产生当前 $x_t$ 的噪声中，按照后验概率加权的平均值。需要条件期望，是因为网络输入没有包含那次随机采样的 $\epsilon$；MSE 训练会自动把无法从输入确定的目标平均起来。

Tweedie 公式描述如何由 noisy marginal 的 score 得到干净样本的后验均值。它不是凭空加入的新结论，而是把高斯条件 score 对未知 $x_0$ 取后验平均后再移项得到的。它在 Diffusion 中建立了 score 参数化与 $x_0$ 参数化的直接桥梁。

反向均值 $\mu_\theta^{(s)}=\frac{1}{\sqrt{\alpha_t}}(x_t+\beta_t s_\theta)$ 描述 score 如何真正影响一步生成：先沿高密度方向修正 $x_t$，修正强度由本步噪声方差 $\beta_t$ 控制，再用 $1/\sqrt{\alpha_t}$ 撤销本步的信号缩放。它与 $x_0$ 形式和 $\epsilon$ 形式是同一个反向均值。

#### 直觉理解

三种预测可以看作回答同一个去噪问题的三种问法。预测 $x_0$ 回答“干净终点在哪里”；预测 $\epsilon$ 回答“这次污染大致把我推向了哪里”；预测 score 回答“从当前位置出发，哪个方向更像数据”。已知噪声日程后，终点、污染方向和高密度方向之间可以确定地换算。

不过，面对一张严重受损的图，网络不可能知道前向过程当时抽到的那一个随机噪声。MSE 让网络给出所有合理解释的平均噪声，这个平均噪声的反方向恰好等于边缘分布的 score。于是“预测随机噪声”并不是要求网络读心，而是在用容易合成的监督信号学习概率密度的局部方向。

#### 本章总结

**本章解决了什么问题：** 本章从高斯条件密度出发，补全了条件 score、边缘 score、噪声预测与干净样本预测之间的推导，并说明了为什么噪声 MSE 的最优网络输出会成为边缘 score。

**核心公式：** 三个核心关系为 $s_t(x_t)=-\mathbb E[\epsilon\mid x_t]/\sqrt{1-\bar\alpha_t}$、
$\mathbb E[x_0\mid x_t]=\frac{x_t+(1-\bar\alpha_t)s_t(x_t)}{\sqrt{\bar\alpha_t}}$，以及
$\mu_\theta^{(s)}=\frac{1}{\sqrt{\alpha_t}}(x_t+\beta_t s_\theta)$。

**需要记住的直觉：** 单次噪声的反方向是已知 $x_0$ 时的条件 score；把所有可能来源按后验概率平均，才得到只依赖 $x_t$ 的边缘 score。网络通过噪声 MSE 学到的正是这个平均方向。

**与下一章的关系：** 本 Part 已经从 VAE/DDPM 的推导得到 score，但还没有系统解释梯度场、score matching 和 Tweedie 公式为什么能支撑一种独立的生成模型观点。Part 8 将进入 Score-based Model，从这些基础概念出发重新理解扩散生成。

## Part 8：Score Based Model

Part 7 是从 VAE/DDPM 的反向均值出发，通过 Tweedie’s Formula 得到 score 参数化。本 Part 按照论文接下来的逻辑换一个起点：先把概率分布写成 energy-based model，再解释为什么 score 能绕过难算的归一化常数、如何用 score matching 学到它，以及如何通过 Langevin dynamics 生成样本。最后，我们会看到多噪声级别的 score-based model 与离散 Diffusion 在训练目标和采样过程上描述的是同一件事。

### 8.1 从梯度到 Score：概率密度空间中的方向

#### 背景

理解 score 之前，需要先补回导数、偏导数和梯度。导数不是一个需要死记的符号，而是在回答“输入发生极小变化时，输出以多快的速度变化”。对于一元函数 $g(x)$，导数定义为 $g'(x)=\lim_{h\to 0}\frac{g(x+h)-g(x)}{h}$。其中 $h$ 是输入的微小改变量，分子是相应的输出改变量，两者之比是平均变化率；当 $h$ 趋近于零时，得到当前位置的瞬时变化率。

图像不是一个标量输入，而是一个高维向量。若一张图有 $d$ 个数值，便可写成 $x=(x_1,\ldots,x_d)\in\mathbb R^d$。函数 $g(x)$ 对第 $i$ 个分量的偏导数 $\frac{\partial g}{\partial x_i}$，表示只改变 $x_i$、暂时固定其他分量时，$g$ 的变化率。把所有偏导数排成向量，就得到梯度 $\nabla_x g(x)$。

#### 数学推导

##### 第一步：证明梯度是增长最快的方向

梯度的定义为

$$
\nabla_x g(x)=
\begin{bmatrix}
\frac{\partial g}{\partial x_1}\\
\vdots\\
\frac{\partial g}{\partial x_d}
\end{bmatrix}.
$$

这个公式描述 $g$ 对每个输入维度的局部敏感程度。为了知道整个向量向某个方向移动时会发生什么，令 $\delta\in\mathbb R^d$ 表示一个很小的位移。一阶 Taylor 近似给出 $g(x+\delta)\approx g(x)+\nabla_x g(x)^\top\delta$。这里的上标 $\top$ 表示转置，因此 $\nabla_x g(x)^\top\delta$ 是两个向量的内积，也就是各分量乘积之和。

为什么这个近似成立？在一维中，导数的局部线性定义给出 $g(x+h)\approx g(x)+g'(x)h$。多维时，每个分量分别贡献 $\frac{\partial g}{\partial x_i}\delta_i$，把它们相加便得到 $\sum_i\frac{\partial g}{\partial x_i}\delta_i=\nabla g^\top\delta$。

现在固定移动距离 $\|\delta\|_2=r$。根据 Cauchy-Schwarz 不等式，$\nabla g^\top\delta\leq\|\nabla g\|_2\|\delta\|_2=r\|\nabla g\|_2$。等号只在 $\delta$ 与 $\nabla g$ 同方向时取得。因此，在允许移动相同距离的所有方向中，沿梯度方向移动会带来最大的一阶增长；梯度的方向告诉我们往哪里走，梯度的长度告诉我们局部增长有多陡。

##### 第二步：从概率密度得到 score

设连续数据的概率密度为 $p(x)$。密度 $p(x)$ 不是“恰好取到单点 $x$ 的概率”，而是单位体积附近集中了多少概率；小区域 $A$ 的概率由 $\Pr(x\in A)=\int_A p(x)\,dx$ 给出。score function 定义为

$$
s(x):=\nabla_x\log p(x).
$$

这个公式描述概率密度的对数在数据空间中的梯度。$s(x)$ 是 score 向量，$\log$ 是自然对数，$\nabla_x$ 表示对数据坐标而不是模型参数求梯度。由链式法则，$\nabla_x\log p(x)=\frac{1}{p(x)}\nabla_x p(x)$。只要 $p(x)>0$，正数 $1/p(x)$ 只改变向量长度，不改变方向，所以 score 与密度本身的梯度指向相同的上升方向。使用对数的好处是把密度的巨大尺度差异压缩，并把乘法结构变成加法结构。

若 $x$ 位于一个光滑的局部众数，也就是附近概率密度的峰顶，那么一阶方向已经没有继续上升的空间，通常有 $s(x)=0$。因此，score 在整个空间中形成一个向量场：空间中每一点都有一支箭头，箭头大体指向附近的高密度区域。

##### 第三步：score 如何绕过 energy model 的归一化常数

论文从 energy-based model 出发，把一个灵活的概率模型写成

$$
p_\theta(x)=\frac{e^{-f_\theta(x)}}{Z_\theta},
\qquad
Z_\theta=\int e^{-f_\theta(u)}\,du.
$$

$f_\theta(x)$ 是由参数 $\theta$ 控制的能量函数，能量越低，未归一化权重 $e^{-f_\theta(x)}$ 越大；$u$ 只是积分中的占位变量；$Z_\theta$ 把所有位置的未归一化权重加总，使 $\int p_\theta(x)\,dx=1$。复杂神经网络定义的 $f_\theta$ 可以表达复杂分布，但高维积分 $Z_\theta$ 往往无法精确计算，这使最大似然中的 $\log p_\theta(x)$ 难以求值。

对概率模型取对数，可得 $\log p_\theta(x)=-f_\theta(x)-\log Z_\theta$。接着对数据 $x$ 求梯度：

$$
\nabla_x\log p_\theta(x)
=-\nabla_x f_\theta(x)-\nabla_x\log Z_\theta
=-\nabla_x f_\theta(x).
$$

最后一个等号成立，是因为 $Z_\theta$ 虽然依赖参数 $\theta$，却不依赖当前数据坐标 $x$，所以对 $x$ 的梯度为零。这条推导说明：只学习概率密度的局部方向时，不必知道全局归一化常数。我们可以用神经网络 $s_\theta(x)$ 直接近似真实 score $\nabla_x\log p_{\text{data}}(x)$。

#### 公式解释

导数定义 $g'(x)=\lim_{h\to0}\frac{g(x+h)-g(x)}h$ 描述一维函数的局部变化率；梯度公式则把这一概念推广到多维输入。Taylor 近似说明梯度如何预测一次微小移动带来的函数变化，Cauchy-Schwarz 不等式进一步证明了梯度方向是固定移动距离下的一阶最快上升方向。这些公式之所以需要，是因为“score 指向高概率区域”不是比喻，而是梯度定义的直接结果。

score 公式 $s(x)=\nabla_x\log p(x)$ 中，$x$ 是数据向量，$p(x)$ 是其概率密度，$s(x)$ 与 $x$ 形状相同。对图像而言，每个 score 分量表示稍微改变对应像素或特征时，对数密度如何变化。它在生成模型中的作用不是直接给出概率值，而是提供让样本朝高密度区域移动的局部导航信息。

energy model 公式中的负号表示低能量对应高概率，$Z_\theta$ 则保证概率总量为一。对 $x$ 求 score 后，$\log Z_\theta$ 消失，因此 score-based model 可以绕过高维归一化积分。但它只绕过了“算归一化常数”的困难，并没有自动解决“真实 score 从哪里来”的问题；这将由下一章的 score matching 处理。

#### 直觉理解

可以把 $\log p(x)$ 想成高维地形的海拔，score 就是每个位置的最陡上坡箭头。我们不知道整张地形图的绝对海拔，也不知道统一的海拔零点，但只要知道脚下最陡的上坡方向，仍然能够向山峰移动。归一化常数 $Z_\theta$ 只给整个密度添加同一个对数偏移量，就像把整张地形图整体抬高；它不会改变任何位置的坡度，所以求梯度后自然消失。

#### 本章总结

**本章解决了什么问题：** 本章从导数和偏导数出发，证明了梯度为何表示最快上升方向，并从 energy-based model 推导出不含归一化常数的 score。

**核心公式：** 核心关系是 $s(x)=\nabla_x\log p(x)=\nabla_xp(x)/p(x)$，以及 energy model 对应的 $\nabla_x\log p_\theta(x)=-\nabla_xf_\theta(x)$。

**需要记住的直觉：** 概率密度给出“哪里更常见”，score 给出“从当前位置往哪里会更常见”。它是一张局部导航图，而不是概率密度本身。

**与下一章的关系：** score 避开了难算的归一化常数，但真实数据分布 $p_{\text{data}}$ 仍然未知，因此也不能直接计算它的 score。下一章将推导如何只用数据样本训练 score，并解释如何沿 score 生成样本。

### 8.2 Fisher Divergence、Langevin Dynamics 与 Score Matching

#### 背景

若真实 score 已知，最自然的训练方法是让网络 $s_\theta(x)$ 拟合它。问题在于数据集只提供若干样本 $x$，并不会告诉我们 $p_{\text{data}}(x)$ 的解析表达式，更不会直接提供 $\nabla_x\log p_{\text{data}}(x)$。论文先写出理想目标，再说明 score 可以用于 Langevin dynamics 采样，随后指出必须借助 score matching 才能在未知真实 score 的情况下训练。

还需要补充两个概念。Fisher divergence 衡量两个分布的 score 之间有多大平方差；Markov chain 则是一串随机状态，其中下一状态只依赖当前状态和新抽取的随机噪声，而不需要保存更早的完整历史。Diffusion 的前向链和反向链都是 Markov chain，Langevin dynamics 也会构造一条这样的链。

#### 数学推导

##### 第一步：写出理想的显式 score 目标

令 $p(x)=p_{\text{data}}(x)$，理想训练目标为

$$
J_{\text{explicit}}(\theta)
=\frac12\mathbb E_{x\sim p}
\left[\left\|s_\theta(x)-\nabla_x\log p(x)\right\|_2^2\right].
$$

这个公式描述在真实数据较常出现的位置上，网络 score 与真实 score 的平均平方距离。期望 $x\sim p$ 表示训练位置按数据分布抽取，$\|\cdot\|_2^2$ 把各维误差平方后相加，前面的 $1/2$ 只是为了求导时抵消平方产生的系数 $2$。当目标达到零时，两套向量场在数据分布覆盖的位置完全一致。论文把这一类差异称为 Fisher divergence。

但是，损失中的 $\nabla_x\log p(x)$ 无法由有限数据集直接计算，所以这个式子目前只是“希望做到什么”，还不是可执行的训练算法。

##### 第二步：由 score 构造 Langevin dynamics

假设暂时已经拥有准确的 score。从初始点 $x^{(0)}$ 出发，Langevin dynamics 使用下面的迭代：

$$
x^{(k+1)}
=x^{(k)}+c\,s_\theta\!\left(x^{(k)}\right)
+\sqrt{2c}\,\xi^{(k)},
\qquad
\xi^{(k)}\sim\mathcal N(0,I).
$$

这里 $k=0,\ldots,K-1$ 是采样迭代编号，避免与 Diffusion 的噪声时间步 $t$ 混淆；$c>0$ 是很小的步长；第一项保留当前位置，第二项沿 score 上坡，第三项加入新的标准高斯噪声。

这个离散公式来自连续 Langevin 随机过程的一小段时间近似。在长度为 $c$ 的时间间隔内，确定性的速度 $s_\theta(x)$ 累积出位移 $c\,s_\theta(x)$；标准 Brownian motion 的增量在每个维度上方差为 $c$，写成 $\sqrt c\,\xi$，再乘扩散强度 $\sqrt2$ 就得到 $\sqrt{2c}\xi$。把当前状态、确定性位移和随机位移相加，便得到上面的 Euler-Maruyama 离散步。

为什么不能只做 $x^{(k+1)}=x^{(k)}+c\,s_\theta(x^{(k)})$？没有随机项时，这是对 $\log p(x)$ 的梯度上升，轨迹会确定地停在某个众数附近；相同起点每次都会得到相同结果，而且样本容易坍缩在峰顶。随机项使轨迹能够在高密度区域附近波动，也可能在多个可达区域之间探索。在准确 score、足够小步长和充分迭代等条件下，这条随机链的平稳分布接近目标 $p(x)$，所以保留下来的状态可以看作来自目标分布的样本。

##### 第三步：用分部积分消去未知的真实 score

现在回到不可计算的显式目标。展开平方并把与 $\theta$ 无关的项记为常数 $C$：

$J_{\text{explicit}}(\theta)=\frac12\int p(x)\|s_\theta(x)\|_2^2\,dx-\int p(x)s_\theta(x)^\top\nabla_x\log p(x)\,dx+C$。

由于 $p(x)\nabla_x\log p(x)=\nabla_xp(x)$，交叉项可写成 $-\int s_\theta(x)^\top\nabla_xp(x)\,dx$。按坐标展开，它等于 $-\sum_{i=1}^d\int s_{\theta,i}(x)\frac{\partial p(x)}{\partial x_i}\,dx$。

一维分部积分公式是 $\int u\,dv=uv-\int v\,du$。对每个坐标 $x_i$ 使用它，并假设当 $\|x\|\to\infty$ 时边界项 $p(x)s_{\theta,i}(x)\to0$，便有

$-\int s_{\theta,i}(x)\frac{\partial p(x)}{\partial x_i}\,dx=\int p(x)\frac{\partial s_{\theta,i}(x)}{\partial x_i}\,dx$。

对所有维度求和，定义向量场的散度 $\nabla_x\cdot s_\theta(x):=\sum_{i=1}^d\frac{\partial s_{\theta,i}(x)}{\partial x_i}$，就得到

$$
J_{\text{implicit}}(\theta)
=\mathbb E_{x\sim p}
\left[
\frac12\|s_\theta(x)\|_2^2
+\nabla_x\cdot s_\theta(x)
\right].
$$

$J_{\text{implicit}}$ 与 $J_{\text{explicit}}$ 只相差不依赖 $\theta$ 的常数 $C$，所以具有相同的最优解。新目标只需要从数据集中采样 $x$、计算网络输出以及网络输出对输入的导数，不再需要真实 $p(x)$ 或真实 score。这就是经典 score matching 的核心。

#### 公式解释

显式 Fisher divergence 描述“想让模型向量场匹配真实向量场”的最终目标。它的每个符号都有明确角色：$p$ 决定在哪些位置评估，$s_\theta$ 是待训练网络，$\nabla\log p$ 是未知目标，平方范数衡量方向和长度的误差。它在理论上定义了正确答案，却因真实 score 未知而不能直接训练。

Langevin 公式描述如何把局部 score 变成全局采样过程。步长 $c$ 同时控制确定性移动和噪声方差；$\xi^{(k)}$ 每一步重新采样，所以整个过程具有随机性。score 项将样本拉向高密度区域，噪声项防止所有轨迹只做确定性爬山。这个公式解释了为什么“只学习导数而不学习密度值”仍然能够生成数据。

隐式 score matching 公式中的 $\nabla\cdot s_\theta$ 衡量向量场在一点附近总体向外发散还是向内汇聚。它来自显式损失交叉项的分部积分，而不是额外猜出的正则项。这个目标可以从数据样本估计，但高维散度的精确计算可能昂贵；Diffusion 更常使用下一章介绍的 denoising score matching，它为每个加噪样本构造一个直接可算的监督目标。

#### 直觉理解

显式目标像是要求学生模仿一张看不见的标准答案图。分部积分所做的事情，是把“标准答案的导数”转移到我们能够求导的神经网络上，从而只凭题目样本判断向量场是否合理。学会向量场后，Langevin dynamics 像一名带有随机探索的登山者：大体沿上坡走，但每一步都会抖动，因此最终不是永远停在同一个山尖，而是在符合目标概率的区域中游走。

#### 本章总结

**本章解决了什么问题：** 本章说明了理想 score 学习目标、score 如何驱动 Langevin 采样，以及如何通过分部积分把含未知真实 score 的 Fisher divergence 改写成可训练的 score matching 目标。

**核心公式：** 训练端的核心是 $J_{\text{implicit}}=\mathbb E_p[\frac12\|s_\theta(x)\|^2+\nabla\cdot s_\theta(x)]$；采样端的核心是 $x^{(k+1)}=x^{(k)}+c\,s_\theta(x^{(k)})+\sqrt{2c}\xi^{(k)}$。

**需要记住的直觉：** score matching 负责学习“往哪里走”，Langevin dynamics 负责真的沿这些方向移动并保留随机性。二者合起来才构成 score-based generative modeling。

**与下一章的关系：** 经典 score matching 仍会在图像流形、低密度区域和互不连通的模式上遇到困难。下一章将按照论文的分析说明这三个问题，并证明 Gaussian 加噪为何能同时缓解它们。

### 8.3 Gaussian Perturbation、Denoising Score Matching 与 Tweedie

#### 背景

论文接着指出 vanilla score matching 的三个困难。第一，自然图像通常被认为集中在高维像素空间中的低维流形附近；流形外的密度可能为零，于是 $\log p(x)$ 和它的梯度没有良好定义。第二，score 目标的期望按 $p(x)$ 加权，低密度区域几乎不给训练信号，但生成时的初始随机噪声恰恰可能位于这些区域。第三，对于彼此不相交的多个模式，局部 score 可能丢失各模式的全局混合权重，使 Langevin 链难以正确地在模式之间混合。

Gaussian perturbation 的核心做法是先给真实数据加入高斯噪声，再学习扰动后分布的 score。高斯分布在整个 $\mathbb R^d$ 上都有正密度，因此扰动后的分布不再只落在原来的低维流形上；较大的噪声还会扩大各模式覆盖的区域，让远离数据的地方也得到学习信号。更重要的是，高斯条件分布的 score 有解析解，这会导出 denoising score matching。

#### 数学推导

##### 第一步：看清三个困难的数学来源

若 $x$ 不在数据流形上且 $p(x)=0$，则 $\log p(x)=-\infty$，无法像普通光滑函数那样求梯度。低密度问题则直接来自 $\mathbb E_{p(x)}[\cdot]=\int p(x)(\cdot)\,dx$：某个区域的 $p(x)$ 越小，它对损失和梯度的贡献就越小。

对混合问题，设两个分量的支撑集互不相交，$p(x)=c_1p_1(x)+c_2p_2(x)$，其中 $c_1,c_2>0$ 且 $c_1+c_2=1$。当 $x$ 位于只有第 $j$ 个分量非零的区域时，$p(x)=c_jp_j(x)$，所以 $\nabla_x\log p(x)=\nabla_x(\log c_j+\log p_j(x))=\nabla_x\log p_j(x)$。常数权重 $c_j$ 被梯度消掉了，因此局部 score 不再告诉采样器该模式应占多大比例。这个结论依赖“分量不相交”；若分量明显重叠，后验分量权重仍可能影响局部 score。

##### 第二步：定义 Gaussian 扰动分布

从 $x_0\sim p_{\text{data}}(x_0)$ 抽取真实数据，再采样 $\epsilon\sim\mathcal N(0,I)$，并令 $x_\sigma=x_0+\sigma\epsilon$。给定 $x_0$ 时，$x_\sigma$ 服从 $\mathcal N(x_0,\sigma^2I)$。把未知的 $x_0$ 积分掉，得到扰动后的边缘密度

$$
p_\sigma(x_\sigma)
=\int p_{\text{data}}(x_0)
\mathcal N(x_\sigma;x_0,\sigma^2I)\,dx_0.
$$

这个公式是数据分布与 Gaussian kernel 的卷积。$p_{\text{data}}(x_0)$ 决定干净样本从哪里来，$\mathcal N(x_\sigma;x_0,\sigma^2I)$ 决定它被扰动到哪里，积分把所有可能来源相加。只要 $\sigma>0$，高斯核在整个空间为正，因此通常有 $p_\sigma(x_\sigma)>0$。

##### 第三步：求出可计算的条件 score

Gaussian 条件密度的对数为 $\log p(x_\sigma\mid x_0)=C-\frac{1}{2\sigma^2}\|x_\sigma-x_0\|_2^2$，其中 $C$ 与 $x_\sigma$ 无关。对 $x_\sigma$ 求梯度，平方距离贡献 $2(x_\sigma-x_0)$，所以

$$
\nabla_{x_\sigma}\log p(x_\sigma\mid x_0)
=-\frac{x_\sigma-x_0}{\sigma^2}
=-\frac{\epsilon}{\sigma}.
$$

这个公式描述已知干净来源 $x_0$ 时，带噪点的 Gaussian score。负号表示它指回 Gaussian 中心 $x_0$；$\sigma^2$ 是噪声方差；最后一个等号使用 $x_\sigma-x_0=\sigma\epsilon$。训练时 $x_0$、$\epsilon$ 和 $\sigma$ 都由我们自己采样，因此这个条件 score 可以直接算出。

##### 第四步：证明 denoising score matching 学到边缘 score

用神经网络拟合上面的条件 score，得到单噪声级别的 denoising score matching 目标：

$$
J_{\text{DSM}}(\theta;\sigma)
=\mathbb E_{\substack{x_0\sim p_{\text{data}}\\
\epsilon\sim\mathcal N(0,I)}}
\left[
\left\|
s_\theta(x_0+\sigma\epsilon,\sigma)
+\frac{\epsilon}{\sigma}
\right\|_2^2
\right].
$$

式子中的加号来自目标是 $-\epsilon/\sigma$：预测减目标等于 $s_\theta-(-\epsilon/\sigma)=s_\theta+\epsilon/\sigma$。这个损失只依赖数据样本和人工噪声，不需要知道 $p_{\text{data}}$ 的密度。

但网络输入只有 $x_\sigma$ 和 $\sigma$，同一个 $x_\sigma$ 可能由不同 $x_0$ 与 $\epsilon$ 产生。根据 Part 7 证明过的平方误差条件期望性质，最优预测是监督目标在给定输入后的条件平均：

$s_\theta^*(x_\sigma,\sigma)=\mathbb E[-\epsilon/\sigma\mid x_\sigma]$。

再对边缘密度求梯度，并像 Part 7 那样把求导移入积分，可得

$\nabla_{x_\sigma}\log p_\sigma(x_\sigma)=\mathbb E[\nabla_{x_\sigma}\log p(x_\sigma\mid x_0)\mid x_\sigma]=\mathbb E[-\epsilon/\sigma\mid x_\sigma]$。

因此，虽然每条训练样本提供的是条件 score，MSE 回归的总体最优解恰好是扰动后边缘分布的 score。这里必须保留条件期望：某次抽到的 $-\epsilon/\sigma$ 并不逐样本等于边缘 score。

##### 第五步：从边缘 score 推导 Tweedie’s Formula

把条件 score 的第一种形式取条件期望：

$\nabla_{x_\sigma}\log p_\sigma(x_\sigma)=\mathbb E[-(x_\sigma-x_0)/\sigma^2\mid x_\sigma]=(\mathbb E[x_0\mid x_\sigma]-x_\sigma)/\sigma^2$。

两边乘 $\sigma^2$ 并移项，得到

$$
\mathbb E[x_0\mid x_\sigma]
=x_\sigma+\sigma^2\nabla_{x_\sigma}\log p_\sigma(x_\sigma).
$$

这就是 Gaussian 加性噪声下的 Tweedie’s Formula。它描述给定带噪观测后，平方误差意义下最优的干净样本估计等于“当前观测加上噪声方差乘 score”。它不是额外假设，而是 Gaussian 条件 score 与 Bayes 条件期望的代数结果。

DDPM 的前向均值不是 $x_0$，而是 $\sqrt{\bar\alpha_t}x_0$。把 $x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon$ 看作均值为 $\sqrt{\bar\alpha_t}x_0$、方差为 $1-\bar\alpha_t$ 的 Gaussian 观测，同一推导便给出 $\mathbb E[x_0\mid x_t]=\frac{x_t+(1-\bar\alpha_t)\nabla_{x_t}\log q_t(x_t)}{\sqrt{\bar\alpha_t}}$，正是 Part 7 使用的 Diffusion 版本。

#### 公式解释

扰动分布 $p_\sigma$ 描述“干净数据先按 $p_{\text{data}}$ 出现，再被方差 $\sigma^2$ 的 Gaussian 模糊后”的总体密度。它的 support 通常覆盖整个空间，从而修复流形外 score 无定义的问题；$\sigma$ 越大，分布越平滑，远离原始数据的位置也越可能获得训练信号。

条件 score $-(x_\sigma-x_0)/\sigma^2=-\epsilon/\sigma$ 是人为加噪后能够精确计算的监督标签。DSM 公式把这个标签与网络输出做 MSE；它之所以能学习边缘 score，是因为平方误差会对输入无法确定的随机目标取条件平均，而条件 score 的后验平均正好等于边缘 score。

Tweedie 公式中的 $x_\sigma$ 是带噪观测，$\sigma^2$ 决定应该沿 score 修正多远，$\nabla\log p_\sigma$ 给出朝高密度区域的方向。它在 score-based model 中把“概率地形的坡度”转化成实际的去噪估计，也把 DSM 与 Part 7 的 $x_0$/noise/score 三种参数化连接起来。

#### 直觉理解

原始图像分布像一张极薄、断裂而陡峭的山脉地图：离开山脊后几乎没有可用坡度，随机初始化的旅行者不知道该往哪里走。Gaussian 加噪像用不同粗细的刷子把山脉晕开。粗刷子让远处也能看到大致方向，细刷子保留精细结构。训练时，我们知道每个带噪点是从哪个干净点被推出来的，所以“往回指”的向量很容易构造；大量样本的平均回指方向，就是平滑后分布的 score。

#### 本章总结

**本章解决了什么问题：** 本章解释了 vanilla score model 的三个困难，推导了 Gaussian 条件 score 和 DSM 目标，并证明 DSM 的最优解是扰动后边缘 score，随后从同一等式推出 Tweedie’s Formula。

**核心公式：** DSM 使用可计算目标 $s_\theta(x_0+\sigma\epsilon,\sigma)\approx-\epsilon/\sigma$；其总体最优解满足 $\nabla\log p_\sigma(x_\sigma)=\mathbb E[-\epsilon/\sigma\mid x_\sigma]$；Tweedie 公式为 $\mathbb E[x_0\mid x_\sigma]=x_\sigma+\sigma^2\nabla\log p_\sigma(x_\sigma)$。

**需要记住的直觉：** 单个条件 score 是“从这次带噪点指回其来源”的箭头；边缘 score 是所有可能来源箭头的后验平均；Tweedie 把这支平均箭头乘以噪声方差，得到最佳去噪修正。

**与下一章的关系：** 单一 $\sigma$ 面临平滑程度的取舍：大噪声便于全局探索但缺少细节，小噪声保留细节却难以从随机区域开始。下一章将把多个噪声级别串起来，得到 Noise Conditional Score Network 和 annealed Langevin dynamics。

### 8.4 多噪声级别、Annealed Langevin 与 Diffusion 的统一

#### 背景

论文用一组逐渐变化的 Gaussian 噪声同时解决全局探索与精细生成。设 $0<\sigma_1<\sigma_2<\cdots<\sigma_T$：$\sigma_T$ 最大，对数据分布进行最强平滑；$\sigma_1$ 最小，最接近真实数据。一个共享网络 $s_\theta(x,t)$ 接收样本和噪声级别索引 $t$，学习每个扰动分布的 score。这类模型常被称为 Noise Conditional Score Network。

训练之后，采样从最大噪声级别开始，在每个级别运行若干步 Langevin dynamics，再把结果交给下一个更小的噪声级别。这个过程叫 annealed Langevin dynamics。“Annealed”可以理解为逐步降温：早期允许大范围随机探索，后期逐渐减小噪声和步长，收敛到细致的数据结构。

#### 数学推导

##### 第一步：定义一族扰动分布

对每个 $t$，定义

$p_{\sigma_t}(x_t)=\int p_{\text{data}}(x_0)\mathcal N(x_t;x_0,\sigma_t^2I)\,dx_0$。

这与上一章的单噪声公式相同，只是现在有 $T$ 个不同平滑程度的分布。网络的理想多尺度目标为

$$
\min_\theta
\sum_{t=1}^T
\lambda(t)\,
\mathbb E_{x_t\sim p_{\sigma_t}}
\left[
\left\|
s_\theta(x_t,t)-\nabla_{x_t}\log p_{\sigma_t}(x_t)
\right\|_2^2
\right].
$$

$\lambda(t)>0$ 是第 $t$ 个噪声级别的损失权重。需要它是因为条件 score 的典型尺度约随 $1/\sigma_t$ 变化：小噪声下目标可能更大，不加权时各级别对总梯度的贡献可能严重不平衡。具体权重属于训练设计；它不会改变每个单独级别的理想目标，却会影响有限模型如何分配容量。

真实边缘 score 仍然未知，所以实际使用 DSM 形式。采样 $x_0\sim p_{\text{data}}$、$t$、$\epsilon\sim\mathcal N(0,I)$，令 $x_t=x_0+\sigma_t\epsilon$，便可最小化

$$
\mathcal L_{\text{NCSN}}(\theta)=
\mathbb E_{t,x_0,\epsilon}
\left[
\lambda(t)
\left\|
s_\theta(x_0+\sigma_t\epsilon,t)
+\frac{\epsilon}{\sigma_t}
\right\|_2^2
\right].
$$

这个公式中的输入是带噪样本 $x_t$ 和噪声级别 $t$，输出是与 $x_t$ 同形状的 score 向量，监督目标是可计算的条件 score $-\epsilon/\sigma_t$。对每个 $t$ 应用上一章的条件期望证明，网络的总体最优输出就是 $\nabla_{x_t}\log p_{\sigma_t}(x_t)$。

##### 第二步：推导 annealed Langevin 采样

从易于采样的高噪声初始分布得到 $x_T^{(0)}$。随后按 $t=T,T-1,\ldots,1$ 依次处理，每个级别运行 $K_t$ 步：

$$
x_t^{(k+1)}
=x_t^{(k)}
+c_t\,s_\theta(x_t^{(k)},t)
+\sqrt{2c_t}\,\xi_t^{(k)},
\qquad
\xi_t^{(k)}\sim\mathcal N(0,I).
$$

在同一级别内，$k$ 是 Langevin 迭代编号；完成后把 $x_t^{(K_t)}$ 作为下一个较低噪声级别的初始状态。较大的 $\sigma_t$ 对应平滑分布，score 容易提供全局方向；随着 $t$ 下降，通常也减小步长 $c_t$，让采样器逐渐关注细节。最终状态接近最低噪声分布；若 $\sigma_1$ 足够小，它便近似真实数据分布。

##### 第三步：与离散 Diffusion 的训练目标对应

DDPM 的前向条件分布是

$q(x_t\mid x_0)=\mathcal N(\sqrt{\bar\alpha_t}x_0,(1-\bar\alpha_t)I)$，对应采样式 $x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon$。其条件 score 为

$\nabla_{x_t}\log q(x_t\mid x_0)=-\frac{x_t-\sqrt{\bar\alpha_t}x_0}{1-\bar\alpha_t}=-\frac{\epsilon}{\sqrt{1-\bar\alpha_t}}$。

因此，DDPM 的 noise-prediction MSE 与 DSM 只差一个已知尺度和时间权重：若定义 $s_\theta(x_t,t)=-\epsilon_\theta(x_t,t)/\sqrt{1-\bar\alpha_t}$，那么拟合噪声就等价于拟合 Gaussian 条件 score，期望最优解是 noisy marginal $q_t(x_t)$ 的 score。

NCSN 常写成 $x_t=x_0+\sigma_t\epsilon$，而 DDPM 还把信号乘以 $\sqrt{\bar\alpha_t}$。把 DDPM 变量除以该系数可得

$$
\frac{x_t}{\sqrt{\bar\alpha_t}}
=x_0+
\sqrt{\frac{1-\bar\alpha_t}{\bar\alpha_t}}\,\epsilon.
$$

所以在重新缩放的坐标中，DDPM 也是逐渐增大的加性 Gaussian 扰动，有效噪声标准差为 $\sqrt{(1-\bar\alpha_t)/\bar\alpha_t}$。两类模型的具体噪声日程和损失权重可以不同，但它们都在多个 Gaussian 噪声级别上学习 score。

##### 第四步：与采样过程及连续时间的对应

Annealed Langevin 从大噪声分布开始，依次使用越来越精细的 score；DDPM 从 $x_T$ 开始，依次采样 $p_\theta(x_{t-1}\mid x_t)$。二者都把一个简单的高噪声初始分布沿递减噪声级别逐步变成数据分布。前者从 MCMC/向量场角度描述，后者从反向 Markov transition/HVAE 角度描述。

当噪声级别数 $T\to\infty$、相邻级别差异趋近于零时，离散加噪链可以视为连续时间随机过程，并用 stochastic differential equation（SDE）描述。生成过程对应反转这个随机过程，而反向漂移需要每个连续时刻的 score。论文在这里指出：score-based 视角使“无限时间步 Diffusion”自然过渡到连续时间模型；不同 SDE 参数化对应不同的连续加噪方案。

#### 公式解释

多尺度训练目标描述一个网络同时拟合 $T$ 个扰动分布的 score。$t$ 告诉网络当前噪声级别，$\lambda(t)$ 平衡不同级别，期望保证训练覆盖相应的带噪样本。DSM 版本把未知边缘 score 替换成可采样的 Gaussian 条件 score，因此真正可用于随机梯度训练。

Annealed Langevin 公式与单尺度 Langevin 形式相同，区别在于 score、步长和当前分布都随 $t$ 改变。大噪声阶段负责跨越低密度间隔、选择全局模式；小噪声阶段负责沿细致 score 恢复局部结构。每一级别的末状态成为下一级别的起点，这让不同尺度形成连续的生成路径。

DDPM 重缩放公式说明 NCSN 与 VDM 并非表面相似：二者的 forward corruption 都能写成“干净信号加 Gaussian 噪声”，而 noise prediction 与 score prediction 通过确定比例互换。它们的训练目标都属于多噪声级别 DSM，采样过程都沿噪声由大到小的顺序细化样本。

#### 直觉理解

如果只拿一张极精细的地图，从完全陌生的远方出发时，很可能连目标城市在哪个方向都不知道；如果只拿一张世界地图，又无法找到最后一条街。多噪声 score network 同时学习从世界地图到街道地图的一整套导航。Annealed Langevin 先按粗地图完成大范围移动，再不断切换到更精细的地图。Diffusion reverse process 采用不同的概率语言安排同一段旅程。

#### 本章总结

**本章解决了什么问题：** 本章把单噪声 DSM 扩展为多噪声 score network，解释了 annealed Langevin 采样，并从训练目标、Gaussian 扰动和递减噪声采样三个层面建立了 Score-based Model 与 Variational Diffusion Model 的对应。

**核心公式：** 多尺度 DSM 为 $\mathbb E[\lambda(t)\|s_\theta(x_0+\sigma_t\epsilon,t)+\epsilon/\sigma_t\|^2]$；annealed Langevin 使用 $x^{(k+1)}=x^{(k)}+c_t s_\theta(x^{(k)},t)+\sqrt{2c_t}\xi^{(k)}$；DDPM 与 score 的转换为 $s_\theta=-\epsilon_\theta/\sqrt{1-\bar\alpha_t}$。

**需要记住的直觉：** 大噪声 score 提供可靠的全局方向，小噪声 score 恢复精细结构。把多个尺度从粗到细串联，就是 score-based sampling；用反向概率转移描述同一过程，就是 Diffusion sampling。

**与下一章的关系：** 到这里我们一直学习无条件分布 $p(x)$，只能控制“像数据”，不能指定“生成哪一类数据”。Part 9 将在 score 和 noise parameterization 上加入条件 $c$，依次推导 Conditional Diffusion、Classifier Guidance 与 Classifier-Free Guidance。

## Part 9：Guidance

前面的所有推导都在学习无条件数据分布 $p(x)$：模型知道“真实数据大致长什么样”，却不能接受“生成猫”“生成某个类别”或“把低分辨率图放大”这样的控制信息。Guidance 研究的就是如何学习和强化条件分布 $p(x\mid c)$。论文用 $y$ 表示条件，本教程统一写成 $c$；它可以是类别标签、文本编码、低分辨率图像或其他控制信号。

本 Part 按论文顺序分三步。首先，把条件直接加入每一步 reverse process，得到 vanilla conditional diffusion。其次，用 Bayes 公式把 conditional score 拆成 unconditional score 与噪声分类器的梯度，得到 Classifier Guidance。最后，用 conditional score 和 unconditional score 的差替代分类器梯度，得到 Classifier-Free Guidance，并从概率重加权角度解释 CFG 为什么有效。

### 9.1 Conditional Diffusion：从 $p(x)$ 到 $p(x\mid c)$

#### 背景

条件概率 $p(x\mid c)$ 表示“已知条件 $c$ 后，数据 $x$ 的分布”。例如，无条件分布 $p(x)$ 混合了所有类别的图片，而 $p(x\mid c=\text{猫})$ 只描述满足“猫”这一条件的图片。条件不是给最终样本贴一个事后标签，而是在整条生成路径中改变每一步应该朝哪里去噪。

条件概率来自联合概率。联合分布可以按两种顺序分解为 $p(x,c)=p(x\mid c)p(c)=p(c\mid x)p(x)$。把两边的 $p(c)$ 除掉，便得到 Bayes 公式 $p(x\mid c)=\frac{p(x)p(c\mid x)}{p(c)}$。本章先用第一种分解构造条件生成模型，下一章再用 Bayes 公式推导 guidance。

#### 数学推导

##### 第一步：给整条反向链加入条件

无条件 Diffusion 的生成联合分布为

$p_\theta(x_{0:T})=p(x_T)\prod_{t=1}^{T}p_\theta(x_{t-1}\mid x_t)$。

这个公式描述先从简单先验 $p(x_T)$ 抽取纯噪声，再从 $t=T$ 到 $1$ 依次采样反向条件分布。若希望整条路径受条件 $c$ 控制，只需让每一步反向转移都看到 $c$：

$$
p_\theta(x_{0:T}\mid c)
=p(x_T)\prod_{t=1}^{T}
p_\theta(x_{t-1}\mid x_t,c).
$$

左边是给定 $c$ 后整条路径的联合分布；$x_{0:T}$ 是 $x_0,\ldots,x_T$ 的简写；$p_\theta(x_{t-1}\mid x_t,c)$ 表示网络根据当前带噪状态、时间步和条件构造的一步反向分布。右边仍写 $p(x_T)$ 而不是 $p(x_T\mid c)$，因为前向过程在足够大的 $T$ 时几乎抹掉所有数据与条件信息，使 $x_T$ 近似统一的 $\mathcal N(0,I)$。

##### 第二步：把条件加入三种网络参数化

Part 7 的三种输出都可以条件化：

$\hat x_{0,\theta}(x_t,t,c)\approx x_0$、$\epsilon_\theta(x_t,t,c)\approx\epsilon$，或 $s_\theta(x_t,t,c)\approx\nabla_{x_t}\log p_t(x_t\mid c)$。

这里 $p_t(x_t\mid c)$ 是条件数据经过 $t$ 步加噪后的边缘分布。条件 $c$ 不是预测目标的一部分，而是帮助网络决定在当前噪声级别上应该输出哪一种去噪方向。工程上可以先把类别、文本或低分辨率图编码成向量，再通过拼接、条件归一化或 cross-attention 注入网络；数学上只需把网络函数的输入从 $(x_t,t)$ 扩展为 $(x_t,t,c)$。

##### 第三步：写出条件噪声预测训练目标

训练数据现在是成对样本 $(x_0,c)\sim p_{\text{data}}(x_0,c)$。与无条件 DDPM 相同，抽取时间步 $t$ 和噪声 $\epsilon\sim\mathcal N(0,I)$，构造 $x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon$，然后最小化

$$
\mathcal L_{\text{cond}}(\theta)=
\mathbb E_{(x_0,c),t,\epsilon}
\left[
\left\|
\epsilon-\epsilon_\theta(x_t,t,c)
\right\|_2^2
\right].
$$

这个公式描述网络在知道条件 $c$ 时预测人工加入的噪声。所有符号与 Part 6 相同，新增的 $c$ 来自与 $x_0$ 配对的标注或输入。MSE 的最优解为 $\mathbb E[\epsilon\mid x_t,t,c]$，因此根据 Part 7 的 score/noise 关系，$s_\theta(x_t,t,c)=-\epsilon_\theta(x_t,t,c)/\sqrt{1-\bar\alpha_t}$ 会逼近 conditional score $\nabla_{x_t}\log p_t(x_t\mid c)$。

##### 第四步：条件网络如何完成一步反向采样

把条件噪声预测代入 DDPM 反向均值：

$$
\mu_\theta(x_t,t,c)
=\frac{1}{\sqrt{\alpha_t}}
\left(
x_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}
\epsilon_\theta(x_t,t,c)
\right).
$$

然后从 $p_\theta(x_{t-1}\mid x_t,c)=\mathcal N(\mu_\theta(x_t,t,c),\Sigma_\theta(x_t,t,c))$ 采样。生成时固定同一个条件 $c$，从 $x_T\sim\mathcal N(0,I)$ 开始重复这一步，最终得到服从模型条件分布的 $x_0$。条件通过每一次均值预测持续影响轨迹，而不是只在最后一步起作用。

#### 公式解释

条件反向链公式描述如何把一个无条件生成模型改造成条件生成模型。乘积中的每个因子负责一步去噪，条件 $c$ 出现在每个因子中，意味着整条轨迹都可以根据条件调整。先验 $p(x_T)$ 仍然简单且与条件近似无关，所以我们不需要为不同条件准备不同的初始噪声分布。

条件训练目标中的网络输入是 $x_t,t,c$，输出是与 $x_t$ 同形状的噪声向量。$x_t$ 提供当前状态，$t$ 告诉网络噪声强度，$c$ 指定希望恢复的数据子分布。这个公式在 Diffusion 中的作用，是用与无条件训练几乎相同的监督信号学习 conditional score。

条件反向均值公式说明网络预测如何真正影响生成：同一个 $x_t$ 在不同 $c$ 下会产生不同的 $\epsilon_\theta$，从而得到不同的 $\mu_\theta$ 和下一状态。若条件被网络忽略，$\epsilon_\theta(x_t,t,c)$ 对不同 $c$ 几乎相同，生成结果也就不受控制；这正是 Guidance 要解决的问题。

#### 直觉理解

无条件去噪像是听到“请画一张合理图片”，模型只需朝任意真实图像方向移动。条件去噪则像每一步都听到“请画一只猫”：早期步骤决定大体布局时要考虑猫，后期步骤恢复纹理时也要考虑猫。只把条件放在最终一步，无法纠正前面已经走向错误模式的轨迹，因此条件必须伴随整条反向链。

#### 本章总结

**本章解决了什么问题：** 本章把无条件 reverse process 扩展为 $p_\theta(x_{0:T}\mid c)$，并给出条件 noise prediction 的训练和采样公式。

**核心公式：** 条件反向链为 $p_\theta(x_{0:T}\mid c)=p(x_T)\prod_t p_\theta(x_{t-1}\mid x_t,c)$，训练目标为 $\mathbb E\|\epsilon-\epsilon_\theta(x_t,t,c)\|^2$。

**需要记住的直觉：** 条件不是生成完成后的筛选器，而是每一步去噪的导航信息；同一带噪状态因为条件不同，会被推向不同的数据子分布。

**与下一章的关系：** Vanilla conditional diffusion 能够使用条件，却没有一个独立旋钮控制条件影响有多强。下一章将用 Bayes 公式把 conditional score 分解，并通过分类器梯度显式放大条件方向。

### 9.2 Classifier Guidance：用分类器梯度放大条件

#### 背景

Classifier Guidance 从 score 参数化出发。目标是得到条件分布在第 $t$ 个噪声级别的 score $s_{\text{cond}}(x_t,t,c)=\nabla_{x_t}\log p_t(x_t\mid c)$。它将这个向量拆成两部分：无条件 Diffusion 提供“怎样更像数据”的方向，额外分类器提供“怎样更像条件 $c$”的方向。

这里的分类器不是普通的干净图片分类器。它必须估计 $p_\phi(c\mid x_t,t)$，也就是在任意噪声级别上从带噪输入判断条件。参数 $\phi$ 属于分类器，参数 $\theta$ 属于 Diffusion model。采样时会固定类别 $c$，对分类器的对数概率关于输入 $x_t$ 求梯度。

#### 数学推导

##### 第一步：用 Bayes 公式分解 conditional score

在固定时间步 $t$ 下，Bayes 公式为

$p_t(x_t\mid c)=\frac{p_t(x_t)p_t(c\mid x_t)}{p(c)}$。

取对数把乘除变成加减：

$\log p_t(x_t\mid c)=\log p_t(x_t)+\log p_t(c\mid x_t)-\log p(c)$。

现在对 $x_t$ 求梯度。条件 $c$ 已经固定，$p(c)$ 不随 $x_t$ 改变，因此 $\nabla_{x_t}\log p(c)=0$。于是得到 Classifier Guidance 的基础恒等式：

$$
\nabla_{x_t}\log p_t(x_t\mid c)=
\underbrace{\nabla_{x_t}\log p_t(x_t)}_{\text{unconditional score}}
+
\underbrace{\nabla_{x_t}\log p_t(c\mid x_t)}_{\text{classifier gradient}}.
$$

第一项由无条件 Diffusion 的 score network 近似，负责维持样本处于真实数据高密度区域；第二项由 noise-aware classifier 提供，表示怎样微调当前 $x_t$ 会让分类器更相信条件 $c$。论文称它为 adversarial gradient，因为它与基于输入梯度修改分类器输出的方法形式相同；这里它被用于增加目标条件概率。

##### 第二步：训练能够处理任意噪声级别的分类器

从配对数据 $(x_0,c)$ 采样，抽取 $t$ 和 $\epsilon$，构造相应的 $x_t$。对于离散类别，分类器可以最小化交叉熵

$$
\mathcal L_{\text{classifier}}(\phi)=
\mathbb E_{(x_0,c),t,\epsilon}
\left[-\log p_\phi(c\mid x_t,t)\right].
$$

这个公式描述让分类器给真实条件 $c$ 分配更高概率。负号把最大化对数概率改成最小化损失；将 $t$ 作为输入，是因为同一 $x_t$ 在不同噪声级别上的判别难度和统计特征不同。训练后，在采样时计算 $\nabla_{x_t}\log p_\phi(c\mid x_t,t)$，梯度穿过分类器回到输入，但不更新分类器参数。

##### 第三步：加入 guidance scale

若所有模型都精确，前面的 Bayes 恒等式对应普通 conditional score。为了显式控制条件强度，引入 $\gamma\geq0$：

$$
s_{\text{guided}}(x_t,t,c)
=s_{\text{uncond}}(x_t,t)
+\gamma\,\nabla_{x_t}\log p_\phi(c\mid x_t,t).
$$

$\gamma=0$ 时完全忽略条件，只使用无条件 score；$\gamma=1$ 时，在模型理想的情况下恢复普通 $p_t(x_t\mid c)$ 的 score；$\gamma>1$ 时，分类器方向被额外放大。

为什么放大后会提高条件一致性并降低多样性？定义一个重新加权但尚未归一化的密度

$\tilde p_{\gamma,t}(x_t\mid c)\propto p_t(x_t)\,p_\phi(c\mid x_t,t)^\gamma$。

对其取对数梯度，未知归一化常数因不依赖 $x_t$ 而消失，得到

$\nabla_{x_t}\log\tilde p_{\gamma,t}(x_t\mid c)=\nabla_{x_t}\log p_t(x_t)+\gamma\nabla_{x_t}\log p_\phi(c\mid x_t,t)$，正好就是 guided score。把分类器概率提升到大于一的幂，会强烈压低分类器不确定的样本、集中到容易被识别为 $c$ 的区域，所以条件更明显，但原本合理而不典型的样本会被削弱。

##### 第四步：把 classifier guidance 转换为噪声预测

DDPM 常用 noise parameterization。令 $\sigma_t=\sqrt{1-\bar\alpha_t}$，并使用 $s_{\text{uncond}}=-\epsilon_{\text{uncond}}/\sigma_t$ 与 $s_{\text{guided}}=-\epsilon_{\text{guided}}/\sigma_t$。把它们代入 guided score：

$-\epsilon_{\text{guided}}/\sigma_t=-\epsilon_{\text{uncond}}/\sigma_t+\gamma\nabla_{x_t}\log p_\phi(c\mid x_t,t)$。

两边乘以 $-\sigma_t$，得到

$$
\epsilon_{\text{guided}}
=\epsilon_{\text{uncond}}
-\gamma\sigma_t\nabla_{x_t}\log p_\phi(c\mid x_t,t).
$$

这个噪声预测不是另一个网络的直接输出，而是无条件噪声预测经过分类器梯度修正后的结果。把它代入 $\mu_\theta=\frac1{\sqrt{\alpha_t}}(x_t-\frac{\beta_t}{\sigma_t}\epsilon_{\text{guided}})$，便能沿受条件强化的反向均值采样。

#### 公式解释

Bayes score 分解式描述条件信息如何改变数据分布的局部方向。Unconditional score 维护自然性，classifier gradient 增加条件概率；二者相加才得到条件 score。它之所以需要，是因为它把一个难以直接控制的 conditional model 拆成两个作用清晰的模块。

分类器损失中的 $p_\phi(c\mid x_t,t)$ 是对 noisy input 的条件概率，不能简单用只见过 $x_0$ 的现成分类器替代。若分类器在大噪声时梯度错误，生成轨迹从一开始就可能被推向异常区域。这也是 Classifier Guidance 需要额外训练专用分类器的主要成本。

Guidance scale 公式中的 $\gamma$ 不是“条件是否存在”的开关，而是对条件 log-likelihood 梯度的倍数。概率重加权公式证明了它的作用：$\gamma>1$ 不再严格采样原始 conditional distribution，而是在采样一个更偏好高分类器置信度的 tilted distribution。

噪声形式中的负号来自 score 与噪声方向相反。Classifier gradient 指向提高条件概率的方向，因此要从预测噪声中减去相应向量，反向均值才会朝条件概率更高的方向移动。

#### 直觉理解

无条件 score 像一名负责“画得像真实照片”的老师，分类器梯度像另一名只负责“更像猫”的老师。Guidance scale 决定第二名老师的音量。$\gamma$ 太小时，作品可能很自然却不符合主题；$\gamma$ 很大时，最典型的猫特征会被反复强调，但姿态、背景和风格的多样性可能减少。

#### 本章总结

**本章解决了什么问题：** 本章从 Bayes 公式推导出 conditional score 等于 unconditional score 加 classifier gradient，并证明 guidance scale 对应将分类器概率提升到 $\gamma$ 次幂的分布重加权。

**核心公式：** Classifier Guidance 使用 $s_{\text{guided}}=s_{\text{uncond}}+\gamma\nabla_{x_t}\log p_\phi(c\mid x_t,t)$，噪声形式为 $\epsilon_{\text{guided}}=\epsilon_{\text{uncond}}-\gamma\sqrt{1-\bar\alpha_t}\nabla_{x_t}\log p_\phi(c\mid x_t,t)$。

**需要记住的直觉：** Unconditional score 保证“像数据”，classifier gradient 保证“像条件”；放大后者会提高条件一致性，但会把概率集中到更少、更典型的样本上。

**与下一章的关系：** Classifier Guidance 的问题是必须额外训练一个能识别所有噪声级别的分类器。下一章将证明 conditional score 与 unconditional score 的差本身就等于 classifier gradient，从而完全移除外部分类器。

### 9.3 Classifier-Free Guidance：为什么 CFG 有效

#### 背景

Classifier-Free Guidance（CFG）的“classifier-free”是指采样时不需要独立分类器。它仍然需要同时获得两种预测：给定条件 $c$ 的 conditional prediction，以及使用空条件 $\varnothing$ 的 unconditional prediction。实践中通常不是训练两套完全独立的 Diffusion model，而是让同一个网络在训练时随机丢弃条件，从而同时学会两种行为。

CFG 的核心并不是经验性的向量插值。它来自上一章的 Bayes score 分解：classifier gradient 可以精确写成 conditional score 减 unconditional score。把这个差值放大，就能复现 Classifier Guidance 的方向。

#### 数学推导

##### 第一步：用两种 score 的差替代分类器

由上一章 $\nabla_{x_t}\log p_t(x_t\mid c)=\nabla_{x_t}\log p_t(x_t)+\nabla_{x_t}\log p_t(c\mid x_t)$，移项得到

$$
\nabla_{x_t}\log p_t(c\mid x_t)
=s_{\text{cond}}(x_t,t,c)-s_{\text{uncond}}(x_t,t).
$$

这个公式描述条件相对于无条件分布新增的方向。它从 Bayes 公式直接推导而来；若两种 score 都准确，它就等于理想分类器的输入梯度，不需要真的训练分类器。

把它代入 Classifier Guidance 的 $s_{\text{uncond}}+\gamma\nabla\log p(c\mid x_t)$：

$$
\begin{aligned}
s_{\text{CFG}}
&=s_{\text{uncond}}
+\gamma(s_{\text{cond}}-s_{\text{uncond}})\\
&=\gamma s_{\text{cond}}
+(1-\gamma)s_{\text{uncond}}.
\end{aligned}
$$

第一行更适合理解：从无条件方向出发，沿“条件与无条件的差”移动 $\gamma$ 倍。第二行更适合分析权重。$\gamma=0$ 时得到 unconditional score；$\gamma=1$ 时得到普通 conditional score；$\gamma>1$ 时不再是在两者之间插值，而是越过 conditional score、朝远离 unconditional score 的方向外推。

##### 第二步：证明 CFG 对应什么概率分布

把 score 写回对数密度梯度：

$s_{\text{CFG}}=\gamma\nabla\log p_t(x_t\mid c)+(1-\gamma)\nabla\log p_t(x_t)$。

利用梯度的线性和 $\log a^\gamma=\gamma\log a$，可以合并为

$$
s_{\text{CFG}}
=\nabla_{x_t}
\log\left[
p_t(x_t\mid c)^\gamma
p_t(x_t)^{1-\gamma}
\right].
$$

因此 CFG 对应的归一化目标分布为 $\tilde p_{\gamma,t}(x_t\mid c)\propto p_t(x_t\mid c)^\gamma p_t(x_t)^{1-\gamma}$。再用 Bayes 公式 $p_t(x_t\mid c)\propto p_t(x_t)p_t(c\mid x_t)$，可得 $\tilde p_{\gamma,t}(x_t\mid c)\propto p_t(x_t)p_t(c\mid x_t)^\gamma$，与上一章 Classifier Guidance 的 tilted distribution 相同。

这就是 CFG 有效的概率原因：conditional/unconditional score 的差估计了分类器梯度，$\gamma$ 次外推等价于提高条件似然的幂。这个结论在两种 score 完全一致、模型足够准确时严格成立；实际网络有近似误差，因此 CFG 是这一理想关系的近似实现。

##### 第三步：转换为实现中常见的噪声公式

对同一个 $t$，有 $s_{\text{cond}}=-\epsilon_{\text{cond}}/\sigma_t$ 和 $s_{\text{uncond}}=-\epsilon_{\text{uncond}}/\sigma_t$，其中 $\sigma_t=\sqrt{1-\bar\alpha_t}$。把它们代入 score CFG 并乘以 $-\sigma_t$，得到

$$
\epsilon_{\text{CFG}}
=\epsilon_{\text{uncond}}
+\gamma
\left(
\epsilon_{\text{cond}}-\epsilon_{\text{uncond}}
\right).
$$

$\epsilon_{\text{cond}}=\epsilon_\theta(x_t,t,c)$，$\epsilon_{\text{uncond}}=\epsilon_\theta(x_t,t,\varnothing)$。两次预测使用同一个 $x_t$ 和 $t$，只改变条件输入。得到 $\epsilon_{\text{CFG}}$ 后，把它当作普通噪声预测代入 DDPM 反向均值，再采样 $x_{t-1}$。

有些资料把公式写成 $\epsilon_{\text{cond}}+w(\epsilon_{\text{cond}}-\epsilon_{\text{uncond}})$。它与这里的记号关系是 $w=\gamma-1$。因此比较不同资料的 guidance scale 时，必须先看公式定义；本教程遵循论文的约定，使 $\gamma=1$ 表示 vanilla conditional model。

##### 第四步：一个网络如何同时学会 conditional 与 unconditional

训练时以概率 $p_{\text{drop}}$ 把真实条件替换为空条件：

$\tilde c=\varnothing$ 的概率为 $p_{\text{drop}}$，而 $\tilde c=c$ 的概率为 $1-p_{\text{drop}}$。

随后仍然使用同一个噪声 MSE：

$$
\mathcal L_{\text{CFG-train}}(\theta)=
\mathbb E_{(x_0,c),t,\epsilon,\tilde c}
\left[
\left\|
\epsilon-\epsilon_\theta(x_t,t,\tilde c)
\right\|_2^2
\right].
$$

当 $\tilde c=c$ 时，样本训练 conditional prediction；当 $\tilde c=\varnothing$ 时，条件信息被移除，同一网络必须只根据 $x_t,t$ 预测噪声，因而学习 unconditional prediction。空条件可以是专门的 null embedding 或固定常量，而不应与某个真实条件混淆。

##### 第五步：完整 CFG 采样流程

生成时先采样 $x_T\sim\mathcal N(0,I)$。对 $t=T,\ldots,1$，用同一个网络分别计算 $\epsilon_{\text{cond}}$ 与 $\epsilon_{\text{uncond}}$，按上式组合为 $\epsilon_{\text{CFG}}$，再计算

$\mu_{\text{CFG}}(x_t,t,c)=\frac1{\sqrt{\alpha_t}}\left(x_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\epsilon_{\text{CFG}}\right)$，并从相应反向 Gaussian 采样 $x_{t-1}$。当 $t=1$ 完成后，得到最终条件样本 $x_0$。

CFG 不需要外部分类器，但通常需要每个时间步做 conditional 和 unconditional 两次网络前向计算。较大的 $\gamma$ 会强化条件差异，却也可能带来颜色过饱和、细节不自然或多样性下降；这不是公式失效，而是 tilted distribution 和模型近似误差共同造成的结果。

#### 公式解释

Score 差公式描述 conditional model 相对 unconditional model 新增了什么局部方向。两者共享“怎样更像一般数据”的成分，相减后这部分大体抵消，剩下的主要是“怎样更符合条件 $c$”的成分。因此 CFG 虽然没有显式分类器，仍然能够近似 classifier gradient。

CFG score 公式中的 $\gamma$ 决定沿条件差方向移动多远。它在 $0$ 与 $1$ 之间是插值，在大于 $1$ 时是外推。概率密度公式进一步说明，外推不是任意的向量技巧，而是在理想情况下对应对条件似然进行幂次重加权。

CFG noise 公式是实际 DDPM 中最常见的形式。因为 conditional 和 unconditional score 具有相同的比例因子 $-1/\sigma_t$，它们的线性组合会原样转换成 noise prediction 的线性组合。组合后的噪声进入原有反向均值，因此采样器的其余部分不需要改变。

Condition dropout 训练公式让一个参数共享的网络同时估计两套 score。$p_{\text{drop}}$ 太小会导致 unconditional 分支训练不足，太大又会削弱 conditional 学习；它是训练超参数，而不是采样时的 guidance scale。CFG 的“free”指免去独立分类器，不代表没有额外计算成本。

#### 直觉理解

可以把 unconditional prediction 看作“在不看题目时会怎样画”，conditional prediction 看作“看见题目后会怎样画”。两者相减，得到题目本身带来的改变量。CFG 从不看题目的答案出发，把这份改变量放大后再加回去，所以主题更加鲜明。放大过度时，一切与题目最强相关的特征都会被夸张，而那些自然但不典型的变化会消失。

#### 本章总结

**本章解决了什么问题：** 本章从 Bayes score 恒等式完整推导了 CFG，证明 conditional/unconditional score 的差等于理想 classifier gradient，解释了 CFG 的概率重加权含义，并给出 condition dropout 训练与 DDPM 采样流程。

**核心公式：** CFG 使用 $s_{\text{CFG}}=s_{\text{uncond}}+\gamma(s_{\text{cond}}-s_{\text{uncond}})$，在噪声参数化下为 $\epsilon_{\text{CFG}}=\epsilon_{\text{uncond}}+\gamma(\epsilon_{\text{cond}}-\epsilon_{\text{uncond}})$。

**需要记住的直觉：** Conditional 与 unconditional 预测的差提取了条件专属方向；放大这个方向会提高条件一致性，但 $\gamma>1$ 采样的是更集中的重新加权分布，而不是原始 $p(x\mid c)$，所以多样性通常下降。

**与下一章的关系：** 本教程的主线到此完成：VAE 的 ELBO 导出了 Diffusion 的反向学习目标，noise prediction 与 score matching 给出了可训练形式，Guidance 则把无条件 score 扩展为可控生成。后续学习连续时间 SDE、DDIM、latent diffusion 或 flow matching 时，都可以把它们放回这条“分布—score—反向生成”的主线上理解。
