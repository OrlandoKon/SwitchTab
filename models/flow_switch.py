import torch
import torch.nn as nn
# from .sequence_encoder import FlowSequenceEncoder
# from .statistical_extractor import StatisticalFeatureExtractor

# --- New Module ---
class FlowEmbedding(nn.Module):
    def __init__(self, input_dim=60, embed_dim=128, token_dim=64):
        super(FlowEmbedding, self).__init__()
        
        # --- Feature Tokenizer ---
        # W_j for each feature j (input_dim features)
        self.feature_weights = nn.Parameter(torch.Tensor(input_dim, token_dim))
        # b_j for each feature j
        self.feature_biases = nn.Parameter(torch.Tensor(input_dim, token_dim))
        
        # Initialize weights and biases
        nn.init.xavier_uniform_(self.feature_weights)
        nn.init.zeros_(self.feature_biases)

        # Final projection: Linear(d -> embed_dim)
        # Note: Aggregate over feature dimension, so input to linear is d (token_dim)
        self.input_proj = nn.Linear(token_dim, embed_dim)
        
    def forward(self, x):
        """
        x: [B, K, 60]
        Output: [B, K, 128] (Sequence embeddings)
        """
        B, K, F = x.shape
        
        # 1. Feature Tokenization
        # T_j = b_j + x_j * W_j
        # x shape: [B, K, F] -> [B, K, F, 1] for broadcasting
        x_expanded = x.unsqueeze(-1)
        
        # Weights/Biases: [F, d] -> [1, 1, F, d] for broadcasting
        W = self.feature_weights.view(1, 1, F, -1)
        b = self.feature_biases.view(1, 1, F, -1)
        
        # Compute tokens: [B, K, F, d]
        tokens = b + x_expanded * W
        
        # 2. Aggregate Tokens
        # Mean over feature dimension F: [B, K, F, d] -> [B, K, d]
        packet_emb = tokens.mean(dim=2)
        
        # 3. Project to Transformer Dimension
        # [B, K, d] -> [B, K, embed_dim]
        x = self.input_proj(packet_emb)
        
        return x

# --- SwitchTab Sub-modules copied/adapted for FlowSwitch ---

class Encoder(nn.Module):
    def __init__(self, feature_size, max_len, num_heads=2, num_layers=3, dropout=0.1):
        super(Encoder, self).__init__()
        self.feature_size = feature_size
        
        # Learnable CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, feature_size))
        
        # Position Embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len + 1, feature_size)) # +1 for CLS
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.02)

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=num_heads, batch_first=True, dropout=dropout)
        self.transformer_layers = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        # x: [B, K, D]
        B, K, D = x.shape
        
        # Expand CLS token: [B, 1, D]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        
        # Concatenate: [B, K+1, D]
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Add Pos Embedding
        seq_len = x.size(1)
        # Safe slicing of pos_embed to match sequence length
        # Assuming max_len covers typical usage, or simple truncation of pos_embed
        if seq_len <= self.pos_embed.size(1):
             x = x + self.pos_embed[:, :seq_len, :]
        else:
             # Fallback if sequence is longer than max_len (though cfg should prevent this)
             x = x + self.pos_embed[:, :self.pos_embed.size(1), :] 
        
        # Transform
        x = self.transformer_layers(x)
        
        # Return CLS
        return x[:, 0, :]

class Projector(nn.Module):
    def __init__(self, feature_size, output_size):
        super(Projector, self).__init__()
        self.linear = nn.Linear(feature_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.linear(x))

class Decoder(nn.Module):
    def __init__(self, input_feature_size, output_feature_size):
        super(Decoder, self).__init__()
        self.linear = nn.Linear(input_feature_size, output_feature_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.linear(x))

class Predictor(nn.Module):
    def __init__(self, feature_size, num_classes):
        super(Predictor, self).__init__()
        self.linear = nn.Linear(feature_size, num_classes)

    def forward(self, x):
        return self.linear(x)

# --- FlowSwitch Model ---

class FlowSwitch(nn.Module):
    def __init__(self, cfg):
        super(FlowSwitch, self).__init__()
        
        self.cfg = cfg
        self.max_len = cfg.K
        
        # --- Section 1: Feature Extraction ---
        # FlowEmbedding integration
        self.flow_embedding = FlowEmbedding(
            input_dim=cfg.packet_input_dim, # 61 for current config
            embed_dim=cfg.flow_embed_dim,    # 128
            token_dim=64 # Assuming a default token_dim or add to config
        )
        
        self.fusion_dropout = nn.Dropout(cfg.dropout)
        
        # --- Section 2: SwitchTab Components (Flattened) ---
        # Feature size matches FlowEmbedding output (128)
        feature_size = cfg.flow_embed_dim
        num_classes = cfg.num_classes
        num_heads = cfg.switchtab_num_heads
        
        # Encoder
        self.encoder = Encoder(feature_size, max_len=cfg.K, num_heads=num_heads, num_layers=cfg.flow_num_layers, dropout=cfg.dropout)
        
        # Projectors
        half_feature_size = feature_size // 2
        self.projector_s = Projector(feature_size, half_feature_size)
        self.projector_m = Projector(feature_size, half_feature_size)
        
        # Decoder
        self.decoder = Decoder(feature_size, feature_size)
        
        # Predictor
        self.predictor = Predictor(feature_size, num_classes)

        # Loss function
        self.mse_loss = nn.MSELoss()

    def extract_features(self, sequence_input, stat_input):
        """
        Integrates logic from FlowSequenceEncoder directly.
        sequence_input: [B, K, 5]
        stat_input: [B, 64]
        """
        batch_size, seq_len, _ = sequence_input.size()
        
        # Expand stats to match sequence length: [B, K, 64]
        seq_stats = stat_input.unsqueeze(1).expand(-1, seq_len, -1)
        
        # Reconstruct the vector for each packet using config dimensions
        # 0 : cfg.stat_feature_dim  -> Original stats 
        # +-> cfg.seq_feature_dim   -> Sequence features
        
        stat_dim = self.cfg.stat_feature_dim
        part_stats = seq_stats[:, :, :stat_dim] # [B, K, 55]
        
        # Concatenate: 55 + 5 = 60 (or from cfg)
        combined = torch.cat([part_stats, sequence_input], dim=-1) # [B, K, 60]
        
        # Min-Max Scaling (Per-sample)
        min_val, _ = torch.min(combined, dim=-1, keepdim=True)
        max_val, _ = torch.max(combined, dim=-1, keepdim=True)
        combined = (combined - min_val) / (max_val - min_val + 1e-8)
        
        # Pass through FlowEmbedding to get [B, 128]
        flow_embedding = self.flow_embedding(combined)
        
        # Apply dropout
        x = self.fusion_dropout(flow_embedding)
        return x

    def forward(self, sequence_input1, stat_input1, sequence_input2=None, stat_input2=None):
        """
        Forward pass with optional second sample for training logic.
        """
        x1 = self.extract_features(sequence_input1, stat_input1)
        z1_encoded = self.encoder(x1)
        
        if sequence_input2 is not None and stat_input2 is not None:
             x2 = self.extract_features(sequence_input2, stat_input2)
             z2_encoded = self.encoder(x2)
             
             # Calculate Projectors
             s1_salient = self.projector_s(z1_encoded)
             m1_mutual = self.projector_m(z1_encoded)
             
             s2_salient = self.projector_s(z2_encoded)
             m2_mutual = self.projector_m(z2_encoded)
             
             # Reconstruct (Own Mutual + Own Salient)
             x1_rec = self.decoder(torch.cat((m1_mutual, s1_salient), dim=1))
             x2_rec = self.decoder(torch.cat((m2_mutual, s2_salient), dim=1))
             
             # Switch (Other Mutual + Own Salient)
             x1_switched = self.decoder(torch.cat((m2_mutual, s1_salient), dim=1))
             x2_switched = self.decoder(torch.cat((m1_mutual, s2_salient), dim=1))
             
             # Predict
             logits1 = self.predictor(z1_encoded)
             logits2 = self.predictor(z2_encoded)
             
             # Recon Loss
             loss_rec = self.mse_loss(x1, x1_rec) + self.mse_loss(x2, x2_rec) + \
                        self.mse_loss(x1, x1_switched) + self.mse_loss(x2, x2_switched)
             
             return {
                 "logits1": logits1,
                 "logits2": logits2,
                 "recon_loss": loss_rec
             }
        else:
             logits = self.predictor(z1_encoded)
             return {"logits": logits}

