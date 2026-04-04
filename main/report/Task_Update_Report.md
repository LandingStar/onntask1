## 1. 执行概况

代码逻辑已修正并验证通过。当前版本成功实现了基于单层 D2NN 将 **20 类输入图像聚合分类至 5 个探测器** 的任务目标。模型在训练中表现稳定，达到了预期的分类效果。

项目地址：[LandingStar/onntask1](https://github.com/LandingStar/onntask1)
## 2. 一些调整

-  在 `train.py` 中落实了 "4-to-1" 的分组策略。将 20 个类别每 4 个划分为一组，强制映射到对应探测器的中心位置，优先保证了组间分类的准确性。
- 去除了 train dataset 的 elastic transform，提高GPU利用率，提高了训练速度。
- 减弱了对train数据预处理时的变换强度，原先的变换强度过高，train acc低于val acc约10%，影响了model的学习（最高仅能到90%的acc），调整后train acc和val acc的均值基本持平，val acc提高到98%。
- 在训练中加入了探测器抖动（$\pm 10\%$边长），以抵抗安装误差
- 加入了 phase_mask相位偏移值5%的相对抖动，以抵抗加工误差

# 3. 实验结果

### 3.1 训练收敛性

模型在 40 个 Epoch 内快速收敛，并于100epoch内达到较高的准确率（98%）, 训练集与验证集 Loss 同步下降，acc 相对接近。

![[onn training/task1/ai_refined_5_detectors/report/one_layer/loss_acc.png]]
### 3.2 分类性能

验证集的混淆矩阵显示清晰的块状对角线结构，证明 20 个细分类别已较为准确地归类到 5 个目标探测器。
![[onn training/task1/ai_refined_5_detectors/report/one_layer/confusion_matrix.png]]

### 3.3 可视化样本

下图展示了不同输入类别在 5 个探测器上的响应。因为训练中对总体光强分布的约束较弱，能量并没有主要落在探测器区域内。
![[onn training/task1/ai_refined_5_detectors/report/one_layer/evaluation_samples.png]]
## 4. 下一步
调整光强分布约束，在不影响model现有性能的基础上，尽量提高detector内的光强。
另外尝试两层的网络的性能。