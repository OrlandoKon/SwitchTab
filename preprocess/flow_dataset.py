import torch
from torch.utils.data import Dataset
import numpy as np
import random
import os
from preprocess.feature_preprocess import FeaturePreprocessor
from .pcap_loader import load_encrypted_traffic_dataset

# Dataset implementation
class FlowDataset(Dataset):
    def __init__(self, cfg=None, split='train', data_dir=None):
        self.split = split
        self.preprocessor = FeaturePreprocessor(cfg)
        self.cfg = cfg
        self.label_map = {}
        
        # If data directory is provided, load from disk
        if data_dir is not None and os.path.exists(data_dir):
            print(f"Loading {split} data from {data_dir}...")
            # Load real pcap data or cached .pt files based on split
            self.data, self.label_map = load_encrypted_traffic_dataset(data_dir, split=split)
            
            # Update num_classes in config if possible
            if self.label_map and cfg is not None:
                 cfg.num_classes = len(self.label_map)
                 print(f"Updated num_classes to {cfg.num_classes}")

        # Fallback to mock data
        else:
            print("No data provided found. Generating mock data.")
            self.data = self._generate_mock_data(1000 if split == 'train' else 200)
            
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
