# SwitchTab Project TODO & Notes

## ✅ 已完成 (Completed)

### 1. 模型架构重构 (Model Architecture Refactoring)
- [x] **移除冗余 Encoder**: 移除了独立的 `FlowSequenceEncoder` 和 `StatisticalFeatureExtractor`。
- [x] **特征重组**: 在 `FlowSwitch.extract_features` 中直接构建包特征向量。
- [x] **FlowEmbedding 模块**: 实现了新的流编码器 `FlowEmbedding`。
    - 输入: `[Batch, K, 60]` (60维包特征)
    - 结构: Linear(60->128) -> Concat([CLS]) -> Add(PosEmbed) -> TransformerEncoder(2 layers) -> Extract([CLS])
    - 输出: `[Batch, 128]` (流 Embedding)

### 2. 特征工程优化 (Feature Engineering)
- [x] **特征维度精简**: 去除了原有的 3 个零填充特征。
- [x] **当前包特征构成 (60维)**:
    - **统计特征 (55维)**: 全局流统计信息 (去除原本最后9个且只取前55个有效特征)。
    * **序列特征 (5维)**: 长度(length), 方向(dir), 时间间隔(dt), 对数长度(log_len), 突发标志(burst)。
    - **[removed] 相对位置**: 已移除。

### 3. 配置与训练 (Config & Training)
- [x] **参数集中化**: 所有超参数 (维度、层数、Epochs、LR等) 统一移至 `config.py`。
- [x] **日志系统**: 在 `train.py` 中集成了 `logging` 模块，输出日志到 `./log/training.log`。
- [x] **评价指标**: 增加了 Accuracy 和 F1 Score (Macro) 的计算与记录。
- [x] **测试流程**: 训练结束后自动在测试集上评估。

---

## ℹ️ 架构说明 (Architecture Notes)

### Feature Switching 机制
- **问题**: Feature Switching的时候，m1和s1的大小是原来z1的一半吗，还是相同？
- **实现状态**: 是的，是一半。
    - 代码实现: `half_feature_size = feature_size // 2`
    - `z1` (Encoder输出): 128维
    - `m1` (Mutual特征) / `s1` (Salient特征): 各 64维
- **实验验证**：m1和s1是原来的一半时，实验准确率提升2%

### 数据预处理特征详情 (Data Preprocessing)
**每个流表示**: `[K, 61]` 张量 (K=20)
**单包特征向量 (61维)**:

1.  **统计特征 (Statistical Features) - 55维** (所有包共享)
    *   **基础统计 (10)**: 总包数, 上/下行包数, 持续时间, 包长(均/标/大/小), 间隔(均/标)。
    *   **方向统计 (6)**: 上行比, 切换数, 最大连续上/下, 熵, 前10上行和。
    *   **长度直方图 (16)**: 0-1500+ 分布。
    *   **时间直方图 (16)**: 0-1.0s 分布。
    *   **高阶统计 (7)**: 偏度/峰度(长/间隔), 速率, 突发比, 活跃比。
    *   *(注: 原有的9个填充位已被截断，只取前55个有效位)*

2.  **序列特征 (Sequence Features) - 5维** (每包独立)
    *   Packet Length, Direction, Delta Time, Log Length, Burst Flag

3.  **位置特征 (Positional Feature) - 1维** (每包独立)
    *   Relative Position (index / max_len)

### MLP层使用真的MLP还是像现在一样只用一个线性层
---

## 📝 待办 / 计划 (Future Work)
- [ ] **预训练 (Pre-training)**: 实现基于 SwitchTab 的自监督预训练逻辑 (目前 Config 中已有占位参数)。
- [ ] **数据增强**: 考虑在预处理阶段加入流级别的数据增强。
- [ ] **模型保存**: 在 `train.py` 中添加保存最佳模型权重的逻辑 (目前已注释)。
