import torch
import torch.nn as nn
import math

class FlowSequenceEncoder(nn.Module):
    def __init__(self, input_dim=5, embed_dim=64, num_heads=4, num_layers=2, ffn_dim=128, dropout=0.1, max_len=20):
        super(FlowSequenceEncoder, self).__init__()
        self.max_len = max_len
        self.embed_dim = embed_dim
        # No encoder layers needed as we are just replacing features

    def forward(self, x, stats):
        """
        x: [batch, seq_len, 5]
        stats: [batch, 64] containing 55 valid features and 9 padded zeros
        """
        batch_size, seq_len, _ = x.size()
        
        # Expand stats to match sequence length: [batch, seq_len, 64]
        seq_stats = stats.unsqueeze(1).expand(-1, seq_len, -1)
        
        # Generate relative positions: [batch, seq_len, 1]
        positions = torch.arange(0, seq_len, dtype=torch.float32, device=x.device)
        rel_positions = positions / self.max_len
        rel_positions = rel_positions.unsqueeze(0).unsqueeze(2).expand(batch_size, -1, 1)
        
        # Reconstruct the 64-dim vector for each packet
        # 0-54: Original stats
        # 55-59: Sequence features (from x)
        # 60: Relative position
        # 61-63: Padding (zeros)
        
        part_stats = seq_stats[:, :, :55] # [batch, seq_len, 55]
        part_pad = torch.zeros(batch_size, seq_len, 3, device=x.device)
        
        combined = torch.cat([part_stats, x, rel_positions, part_pad], dim=-1) # [batch, seq_len, 64]
        
        # Mean pooling to get flow representation [batch, 64]
        flow_embedding = torch.mean(combined, dim=1)
        
        return flow_embedding
