import torch
import torch.nn as nn
import math

class FlowPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=20, dropout=0.1):
        super(FlowPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class FlowSequenceEncoder(nn.Module):
    def __init__(self, input_dim=5, embed_dim=64, num_heads=4, num_layers=2, ffn_dim=128, dropout=0.1, max_len=20):
        super(FlowSequenceEncoder, self).__init__()
        
        # Linear projection: 5 -> 64
        self.input_projection = nn.Linear(input_dim, embed_dim)
        
        # Positional Encoding
        self.pos_encoder = FlowPositionalEncoding(embed_dim, max_len=max_len, dropout=dropout)
        
        # Transformer Encoder Layers
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=ffn_dim, 
            dropout=dropout
        )
        self.sequence_encoder_layers = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        self.embed_dim = embed_dim

    def forward(self, x):
        """
        x: [batch, seq_len, input_dim] = [batch, 20, 5]
        """
        # Transpose for Transformer: [seq_len, batch, input_dim]
        if x.dim() == 3:
            x = x.transpose(0, 1)
            
        # Linear Projection
        x = self.input_projection(x) # [seq_len, batch, embed_dim]
        
        # Add PE
        x = self.pos_encoder(x)
        
        # Transformer Encoder
        output = self.sequence_encoder_layers(x) # [seq_len, batch, embed_dim]
        
        # Mean Pooling
        # Transpose back: [batch, seq_len, embed_dim]
        output = output.transpose(0, 1)
        
        # Simple mean pooling over sequence dimension
        sequence_embedding = torch.mean(output, dim=1) # [batch, embed_dim]
        
        return sequence_embedding
