import torch
import torch.nn as nn
from .sequence_encoder import FlowSequenceEncoder
from .statistical_extractor import StatisticalFeatureExtractor

# --- SwitchTab Sub-modules copied/adapted for FlowSwitch ---

class Encoder(nn.Module):
    def __init__(self, feature_size, num_heads=2):
        super(Encoder, self).__init__()
        self.transformer_layers = nn.Sequential(
            nn.TransformerEncoderLayer(d_model=feature_size, nhead=num_heads, batch_first=True),
            nn.TransformerEncoderLayer(d_model=feature_size, nhead=num_heads, batch_first=True),
            nn.TransformerEncoderLayer(d_model=feature_size, nhead=num_heads, batch_first=True)
        )

    def forward(self, x):
        # x: [B, D] -> [B, 1, D] for Transformer (if batch_first=True)
        x_unsqueezed = x.unsqueeze(1)
        return self.transformer_layers(x_unsqueezed).squeeze(1)

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
        
        # --- Section 1: Feature Extraction ---
        self.sequence_encoder = FlowSequenceEncoder(
            input_dim=cfg.packet_feature_dim,
            embed_dim=cfg.embed_dim,
            num_heads=cfg.trans_num_heads,
            num_layers=cfg.trans_num_layers,
            ffn_dim=cfg.trans_ffn_dim,
            dropout=cfg.dropout,
            max_len=cfg.K
        )
        
        self.stat_extractor = StatisticalFeatureExtractor(input_dim=cfg.stat_feature_dim, output_dim=cfg.stat_feature_dim)
        
        # self.fusion_norm = nn.LayerNorm(cfg.fusion_output_dim) # Replaced with Min-Max Scaling
        self.fusion_dropout = nn.Dropout(cfg.dropout)
        
        # --- Section 2: SwitchTab Components (Flattened) ---
        feature_size = cfg.switchtab_input_dim
        num_classes = cfg.num_classes
        num_heads = cfg.switchtab_num_heads
        
        # Encoder
        self.encoder = Encoder(feature_size, num_heads)
        
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
        seq_emb = self.sequence_encoder(sequence_input)
        stat_emb = self.stat_extractor(stat_input)
        combined = torch.cat([seq_emb, stat_emb], dim=1)
        
        # Min-Max Scaling (Per-sample)
        min_val, _ = torch.min(combined, dim=1, keepdim=True)
        max_val, _ = torch.max(combined, dim=1, keepdim=True)
        x = (combined - min_val) / (max_val - min_val + 1e-8)
        
        # x = self.fusion_norm(combined) 
        x = self.fusion_dropout(x)
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

