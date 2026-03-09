# FlowSwitch 架构文档

本文档详细描述了当前 FlowSwitch 模型的端到端架构，涵盖数据预处理、特征工程、模型结构及训练流程。

## 1. 数据预处理与特征工程 (Data Preprocessing & Feature Engineering)

### 1.1 输入数据
- **原始输入**: PCAP 格式的加密流量数据。
- **处理单元**: 单条流 (Flow)，由一系列数据包 (Packets) 组成。
- **截断策略**: 仅使用每条流的前 $K$ 个数据包 (默认 $K=20$)。

### 1.2 特征提取
对于每条流，提取两类特征：

#### A. 序列特征 (Sequence Features) - 逐包提取
对前 $K$ 个数据包中的每一个包 $p_i$，提取 5 维特征向量：
1. **Packet Length**: 包长度 (Bytes)
2. **Direction**: 方向 (0=Down, 1=Up)
3. **Inter-arrival Time ($\Delta t$)**: 距离上一个包的时间间隔
4. **Log Length**: $\ln(1 + \text{Length})$
5. **Burst Flag**: 突发标志 (若 $\Delta t < 0.05s$ 则为 1，否则为 0)

若流长度不足 $K$，则进行零填充。输出维度: $[B, K, 5]$。

#### B. 统计特征 (Statistical Features) - 全局提取
基于整条流的所有包（不仅是前 $K$ 个），计算全局统计特征。
总计计算 55 个有效特征 (原实现为64维，后9维为填充，现已被截断)：
1. **基础统计 (10维)**: 包总数, 上/下行包数, 持续时间, 均值/标准差/最大/最小长度, 均值/标准差间隔时间。
2. **方向统计 (6维)**: 上行比例, 方向切换次数, 最大连续上/下行包数, 熵, 前10个上行包长度和。
3. **长度直方图 (16维)**: 包长度分布 (0-100, ..., 1500+)。
4. **时间直方图 (16维)**: 包到达间隔分布。
5. **高阶统计 (7维)**: 长度/间隔的偏度(Skewness)与峰度(Kurtosis), 包速率, 突发比率, 活跃/空闲比(A/I Ratio)。

在模型输入阶段，这 55 维特征会被复制扩充到 $K$ 个时间步。输出维度: $[B, K, 55]$。

### 1.3 模型输入构建
对于每个样本 (Flow)，将序列特征与统计特征拼接，构成最终的输入张量：
$$ X \in \mathbb{R}^{B \times K \times D_{in}} $$
其中 $D_{in} = 55 (\text{Stat}) + 5 (\text{Seq}) = 60$。

---

## 2. 模型架构 (Model Architecture)

FlowSwitch 模型由 **FlowEmbedding** 模块（特征融合与序列编码）和 **SwitchTab** 核心组件（双流处理与分类）组成。

### 2.1 FlowEmbedding 模块
该模块负责将原始的包级特征序列转换为固定维度的流嵌入 (Flow Embedding)。

1. **输入归一化 (Min-Max Scaling)**:
   - 对输入张量 $X$ 进行样本级的 Min-Max 归一化，防止数值不稳定。
   
2. **特征投影 (Linear Projection)**:
   - $Linear(60 \to 128)$
   - 输出: $[B, K, 128]$

3. **CLS Token 追加**:
   - 在序列头部追加一个可学习的 `[CLS]` token。
   - 输出: $[B, K+1, 128]$

4. **位置编码 (Positional Embedding)**:
   - 加入可学习的位置编码 $PE \in \mathbb{R}^{1 \times (K+1) \times 128}$。
   - 注意：此前显式的“相对位置特征”已被移除，由 Transformer 内部的位置编码隐式处理位置信息。

5. **Transformer Encoder**:
   - 层数: 2
   - 注意力头数: 4
   - 前馈层维度: 256
   - Dropout: 0.1
   - 输出: $[B, K+1, 128]$

6. **特征提取**:
   - 取出 `[CLS]` token 对应的输出作为流的最终表示。
   - 输出: **Flow Embedding** $Z \in \mathbb{R}^{B \times 128}$

### 2.2 SwitchTab 核心组件
基于 SwitchTab 架构，用于增强特征表示并进行分类。

1. **Encoder**:
   - 3 层 Transformer Encoder (Head=2, Dim=128)。
   - 对 Flow Embedding 进行进一步编码。

2. **Projectors (Salient & Mutual)**:
   - 将编码后的特征分别映射到“显著特征空间” (Salient) 和“互信息空间” (Mutual)。
   - $Z \to S$ (Salient), $Z \to M$ (Mutual)。

3. **Switch & Reconstruct (Decoder)**:
   - **自重构**: 使用自身的 $S$ 和 $M$ 重构原始特征。
   - **交换重构 (Switching)**: 使用样本 A 的 $M_A$ 和样本 B 的 $S_B$ 组合，试图重构样本 A'。
   - Decoder 为单层线性层 + Sigmoid。

4. **Predictor (Classifier)**:
   - 线性分类头: $Linear(128 \to \text{NumClasses})$。

---

## 3. 训练过程 (Training Process)

### 3.1 损失函数 (Loss Function)
总体损失由两部分组成：
$$ L = L_{recon} + \alpha \cdot L_{cls} $$

1. **分类损失 ($L_{cls}$)**:
   - 标准交叉熵损失 (CrossEntropy)。
   
2. **重构损失 ($L_{recon}$)**:
   - 均方误差 (MSE)。
   - 包含四项：
     - $MSE(x_1, \text{rec}(m_1, s_1))$
     - $MSE(x_2, \text{rec}(m_2, s_2))$
     - $MSE(x_1, \text{switch}(m_2, s_1))$
     - $MSE(x_2, \text{switch}(m_1, s_2))$

### 3.2 优化配置
- **Optimzier**: Adam
- **Learning Rate**: 0.001
- **Epochs**: 50
- **Batch Size**: 128
- **Alpha ($\alpha$)**: 0.3 (平衡分类与重构权重的超参数)

### 3.3 训练逻辑
- 输入成对样本 $(x_1, x_2)$ 进入模型。
- 前向传播同时计算分类对数概率 (Logits) 和重构损失。
- 反向传播更新所有模块参数（FlowEmbedding 与 SwitchTab 联合训练）。


## 4. 存在的问题及下一步规划
### 4.1 加密流量的嵌入-Embedding
- 问题：加密流量的每条流选用哪些特征组成特征空间，包括序列特征、包头特征、统计特征等等？现有的特征多为转化为Token，用于生成式的NLP架构，这种方案迁移到表格模型上是否还能保持良好的性能？
- 计划：阅读加密流量分析相关论文，进一步思考加密流量嵌入的独特性
### 4.2 加密流量数据表示-Representation
- 问题：在流级加密流量分析任务中，每条流由K个数据包组成，每个数据包可以表示为一行表格数据，则每条流为K行表格数据组成的矩阵。这与普通的表格数据任务不同，如何表示加密流量的一条流，使其既能够保持流中的数据包的时序性、可区分性特征，又能够含有流的全局特征？
- 计划：阅读加密流量分析相关论文，学习相关论文中的加密流量表示方法和类型。目前准确先阅读Pcap-Encoder和Traffic Former。
### 4.3  FlowSwitch的局限
- 现状：经过几天的架构调整，最高准确率只达到了64%。SwitchTab的架构非常轻量级，是不是难以处理加密流量的高度复杂、难以区分的特征空间？且训练过程中的Loss已经降到了0.7，损失函数是否需要进一步设计？
- 计划：用真正的MLP替换简单的线性分类头，尝试重新设计损失函数。
