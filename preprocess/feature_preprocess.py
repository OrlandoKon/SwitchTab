import numpy as np
import torch
import math

class FeaturePreprocessor:
    def __init__(self):
        # Constants
        self.K = 20
        self.MAX_LEN = 1500 # For histogram
        self.DELTA_T_MAX = 1.0 # For histogram, assumed
        
    def process_flow(self, flow):
        """
        Input: flow dict
        Output: 
            sequence: [K, 5]
            stats: [64]
        """
        packets = flow.get('packets', [])
        
        # 1. Sequence Features (Stage 1 input)
        sequence_feature = self._extract_sequence(packets)
        
        # 2. Statistical Features (Stage 2 input)
        statistical_feature = self._extract_statistics(packets)
        
        return sequence_feature, statistical_feature

    def _extract_sequence(self, packets):
        """
        Extract top K packets features: [length, direction, delta_t, log_length, burst_flag]
        """
        seq = []
        prev_time = packets[0]['timestamp'] if packets else 0
        
        for i, pkt in enumerate(packets[:self.K]):
            length = pkt['length']
            direction = pkt.get('direction', 0)
            timestamp = pkt['timestamp']
            
            delta_t = timestamp - prev_time
            prev_time = timestamp
            
            log_length = math.log(1 + length)
            burst_flag = 1.0 if delta_t < 0.05 else 0.0 # Threshold assumed 0.05s
            
            seq.append([length, direction, delta_t, log_length, burst_flag])
            
        # Padding
        if len(seq) < self.K:
            pad_len = self.K - len(seq)
            for _ in range(pad_len):
                seq.append([0.0, 0.0, 0.0, 0.0, 0.0])
                
        return np.array(seq, dtype=np.float32)

    def _extract_statistics(self, packets):
        """
        Extract 64-dim statistical features
        """
        if not packets:
            return np.zeros(64, dtype=np.float32)
            
        lengths = np.array([p['length'] for p in packets])
        directions = np.array([p.get('direction', 0) for p in packets])
        timestamps = np.array([p['timestamp'] for p in packets])
        
        if len(timestamps) > 1:
            delta_ts = np.diff(timestamps)
        else:
            delta_ts = np.array([0.0])

        features = []
        
        # 4.1 Basic Stats (10)
        total_packets = len(packets)
        up_packets = np.sum(directions == 1)
        down_packets = np.sum(directions == 0)
        duration = timestamps[-1] - timestamps[0] if len(timestamps) > 0 else 0
        mean_len = np.mean(lengths)
        std_len = np.std(lengths)
        max_len = np.max(lengths)
        min_len = np.min(lengths)
        mean_dt = np.mean(delta_ts)
        std_dt = np.std(delta_ts)
        
        features.extend([total_packets, up_packets, down_packets, duration, 
                         mean_len, std_len, max_len, min_len, mean_dt, std_dt])
        
        # 4.2 Direction Stats (6)
        up_ratio = up_packets / total_packets if total_packets > 0 else 0
        
        # Direction switches
        switches = 0
        if len(directions) > 1:
            switches = np.sum(directions[1:] != directions[:-1])
            
        # Max consecutive
        max_cons_up = 0
        max_cons_down = 0
        current_up = 0
        current_down = 0
        
        for d in directions:
            if d == 1:
                current_up += 1
                current_down = 0
                max_cons_up = max(max_cons_up, current_up)
            else:
                current_down += 1
                current_up = 0
                max_cons_down = max(max_cons_down, current_down)
                
        # Entropy
        p_up = up_ratio
        p_down = 1 - up_ratio
        entropy = 0
        if p_up > 0: entropy -= p_up * math.log(p_up)
        if p_down > 0: entropy -= p_down * math.log(p_down)
        
        # First 10 up sum
        first_10_up_sum = 0
        count = 0
        for i, d in enumerate(directions):
            if d == 1:
                first_10_up_sum += lengths[i]
                count += 1
                if count >= 10: break
                
        features.extend([up_ratio, switches, max_cons_up, max_cons_down, entropy, first_10_up_sum])
        
        # 4.3 Length Hist (16 bins)
        # Bins: 0-100, 100-200, ... 1500+
        len_hist = np.zeros(16)
        for l in lengths:
            idx = min(int(l / 100), 15)
            len_hist[idx] += 1
        len_hist = len_hist / total_packets if total_packets > 0 else len_hist
        features.extend(len_hist)
        
        # 4.4 Time Hist (16 bins)
        # Log bins for time or linear? 
        # Requirement: "delta_t 16 bin". Usually delta_t is small. 
        # Assume 0-1s divided into 15 bins, last > 1s, or log scale.
        # User said "Time Histogram (16 dims)". No specific bins range given.
        # I will use log-space bins from 0.00001 to 1.0
        dt_hist = np.zeros(16)
        for dt in delta_ts:
            if dt <= 0: 
                idx = 0
            else:
                # Simple linear binning 0-1.0s?
                idx = min(int(dt / 0.0625), 15) # 1.0 / 16 = 0.0625
            dt_hist[idx] += 1
        dt_hist = dt_hist / len(delta_ts) if len(delta_ts) > 0 else dt_hist
        features.extend(dt_hist)
        
        # 4.5 High Order (16)
        # Skew, Kurtosis for Length and DT
        def moment(arr, k):
            if len(arr) == 0: return 0
            m = np.mean(arr)
            s = np.std(arr)
            if s == 0: return 0
            return np.mean(((arr - m)/s)**k)
            
        skew_len = moment(lengths, 3)
        kurt_len = moment(lengths, 4)
        skew_dt = moment(delta_ts, 3)
        kurt_dt = moment(delta_ts, 4)
        
        packet_rate = total_packets / duration if duration > 0 else 0
        # Burst ratio: packets in bursts / total
        burst_pkts = 0
        for dt in delta_ts:
             if dt < 0.05: burst_pkts += 1
        burst_ratio = burst_pkts / len(delta_ts) if len(delta_ts) > 0 else 0
        
        # A/I Ratio: active time / idle time? 
        # Let's say active if dt < 0.05
        active_time = np.sum(delta_ts[delta_ts < 0.05])
        idle_time = np.sum(delta_ts[delta_ts >= 0.05])
        ai_ratio = active_time / (idle_time + 1e-9)
        
        features.extend([skew_len, kurt_len, skew_dt, kurt_dt, 
                         packet_rate, burst_ratio, ai_ratio])
                         
        # Padding to 16 dims (9 more needed)
        curr_len = len(features)
        target_len = 10 + 6 + 16 + 16 + 16 # = 64
        needed = target_len - curr_len
        features.extend([0.0] * needed)
        
        return np.array(features, dtype=np.float32)

