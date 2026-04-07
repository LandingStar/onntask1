## 1. 执行概况

代码逻辑已修正并验证通过。当前版本成功实现了基于单层 D2NN 将 **20 类输入图像聚合分类至 5 个探测器** 的任务目标。模型在训练中表现稳定，达到了预期的分类效果。

项目地址：[LandingStar/onntask1](https://github.com/LandingStar/onntask1)

## 2. 一些调整

- 在 `train.py` 中落实了 "4-to-1" 的分组策略。将 20 个类别每 4 个划分为一组，强制映射到对应探测器的中心位置，优先保证了组间分类的准确性。
- 去除了 train dataset 的 elastic transform，提高GPU利用率，提高了训练速度。
- 引入了一些优化，提升计算资源使用率，轻微提高训练速度。
- 减弱了对train数据预处理时的变换强度，原先的变换强度过高，train acc低于val acc约10%，影响了model的学习（最高仅能到90%的acc），调整后train acc和val acc的均值基本持平，val acc提高到98%, 在450个epoch下提高到99.5%以上。
- 在训练中加入了探测器抖动（$\pm 10\%$边长），以抵抗安装误差
- 加入了 phase\_mask相位偏移值5%的相对抖动，以抵抗加工误差

# 3. 实验结果

### 3.1 训练收敛性

模型在 40 个 Epoch 内快速收敛，并于150epoch内达到较高的准确率(99\%),在, 训练集与验证集 Loss 同步下降，train acc, val acc 相对接近。

![[onn training/task1/report/one_layer/450epoch/loss_acc.png]]
图中130epoch-end段的波动是因为acc已经达标，开始使用较为激进的权重优化光强分布。
### 3.2 分类性能

验证集的混淆矩阵显示清晰的块状对角线结构，证明 20 个细分类别已较为准确地归类到 5 个目标探测器。
![[onn training/task1/report/one_layer/450epoch/confusion_matrix.png]]
### 3.3 可视化样本

下图展示了不同输入类别在 5 个探测器上的响应。能量并没有主要落在探测器区域内。

<img src="onn training/task1/report/one_layer/450epoch/evaluation_samples.png" height = "800" alt="example_samples" align=center />


## 4. 两层网络的尝试

两层网络在理想对齐下性能极佳(50个epoch即可到达99%以上)，但是当引入对齐误差时性能急剧下降（150个epoch下 acc<95%)。但两层网络的能量集中度明显优于一层，且在引入对齐误差时更明显。

## 4.1 ideal assemble
理想对齐下model提升迅速
![[onn training/task1/report/two_layer/no_error/loss_acc.png]]
<img src="onn training/task1/report/two_layer/no_error/evaluation_samples.png" height = "800" alt="example_samples" align=center />


## 4.2 with error
然而当引入error时，情况就变得糟糕。（即使假设倾斜度为0，仅有平面内的平移/旋转）
图中展示的是无倾斜，300epoch下 151-300epoch的训练记录。
![[onn training/task1/report/two_layer/zero_tilt/loss_acc.png]]
但是光斑形态发生了很大的变化。（也就说明原来理想对齐的二层网络的光强分布仍有改进空间）
<img src="onn training/task1/report/two_layer/low_tilt/evaluation_samples.png" height = "800" alt="example_samples" align=center />
1，2组边缘处label8被整体错误分类。与一层的模型在较低epoch或是不合理的loss设计时的现象相似。
![[onn training/task1/report/two_layer/zero_tilt/confusion_matrix.png]]
同时model对倾斜度相对敏感，下面是倾斜为0.01度的训练记录
![[onn training/task1/report/two_layer/low_tilt/loss_acc.png]]
倾斜度为0.1度（可以发现此时train acc 已经完全超过val acc，推测在0.1度的倾斜下model基本不能完成任务）
![[onn training/task1/report/two_layer/high_tilt/loss_acc.png]]