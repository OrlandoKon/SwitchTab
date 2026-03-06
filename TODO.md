TODO：
1. Feature Switching的时候，m1和s1的大小是原来z1的一半吗，还是相同？

2. 数据预处理中包含的特征：
    根据代码 feature_preprocess.py 中的逻辑，数据预处理阶段总共提取了两类特征，具体细节如下：

    1. 序列特征 (Sequence Features)
    这些特征用于模型的第一阶段输入，针对流中的前 K 个报文提取。每个报文包含 5 个特征：

    Packet Length (length): 报文长度。
    Direction (direction): 报文方向（0 表示发送方/Forward，1 表示接收方/Backward）。
    Inter-arrival Time (delta_t): 这一个报文与前一个报文的时间间隔。
    Log Length (log_length): 报文长度的对数值 (ln(1+length))。
    Burst Flag (burst_flag): 突发标志（如果 delta_t < 0.05s 则为 1.0，否则为 0.0）。
    2. 统计特征 (Statistical Features)
    这些特征用于模型的第二阶段输入，是一个 64维 的向量，从整个流（或截断部分）中统计得出：

    基础统计 (Basic Stats) - 10维

    总包数 (total_packets)
    上行包数 (up_packets)、下行包数 (down_packets)
    流持续时间 (duration)
    包长度的均值 (mean_len)、标准差 (std_len)、最大值 (max_len)、最小值 (min_len)
    包间隔时间的均值 (mean_dt)、标准差 (std_dt)
    方向统计 (Direction Stats) - 6维

    上行包比例 (up_ratio)
    方向切换次数 (switches)
    最大连续上行包数 (max_cons_up)
    最大连续下行包数 (max_cons_down)
    方向分布的熵 (entropy)
    前10个上行包的长度总和 (first_10_up_sum)
    长度直方图 (Length Histogram) - 16维

    将包长度分为 16 个区间（0-100, 100-200, ..., 1500+）进行统计并归一化。
    时间直方图 (Time Histogram) - 16维

    将包间隔时间 (delta_t) 分为 16 个线性区间（0-1.0s，步长约 0.0625s）进行统计并归一化。
    高阶统计 (High Order Stats) - 16维

    包含 7个 计算出的特征：
    包长度的偏度 (skew_len)、峰度 (kurt_len)
    包间隔时间的偏度 (skew_dt)、峰度 (kurt_dt)
    发包速率 (packet_rate)
    突发包比例 (burst_ratio，即 delta_t < 0.05 的比例)
    活跃/空闲时间比 (ai_ratio)
    剩余 9个 维度用 0 填充。