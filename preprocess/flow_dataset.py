import torch
from torch.utils.data import Dataset
import numpy as np
import random
from preprocess.feature_preprocess import FeaturePreprocessor

# Dataset implementation
class FlowDataset(Dataset):
    def __init__(self, data=None, is_train=True):
        self.is_train = is_train
        self.preprocessor = FeaturePreprocessor()
        
        # If no data provided, generate mock encrypted flow data
        if data is None:
            self.data = self._generate_mock_data(1000 if is_train else 200)
        else:
            self.data = data
            
    def _generate_mock_data(self, count):
        data = []
        for _ in range(count):
            # Simulate a simplified flow structure
            n_packets = random.randint(5, 50)
            packets = []
            start_time = 0.0
            for _ in range(n_packets):
                pkt = {
                    'timestamp': start_time,
                    'length': random.randint(40, 1500),
                    'direction': random.choice([0, 1])
                }
                packets.append(pkt)
                start_time += random.uniform(0.001, 0.1) # Inter-arrival time
            
            label = random.randint(0, 1)
            data.append({'packets': packets, 'label': label})
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        flow = self.data[idx]
        packets = flow['packets']
        label = flow['label']
        
        # Preprocess features using the defined logic
        # 1. Sequence Feature [K, 5]
        # 2. Statistical Feature [64]
        # Note: process_flow returns sequence, stats
        # We need to ensure process_flow handles single dict input correctly
        
        # Wrapping in dict as process_flow expects flow dict with 'packets' key
        # Our flow IS the dict {'packets': ..., 'label': ...}
        
        seq_feat, stat_feat = self.preprocessor.process_flow(flow)
        
        # Convert to tensors
        seq_tensor = torch.tensor(seq_feat, dtype=torch.float32)
        stat_tensor = torch.tensor(stat_feat, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return seq_tensor, stat_tensor, label_tensor
