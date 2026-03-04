import torch

class Config:
    def __init__(self):
        # --- Data Preprocessing ---
        self.num_classes = 2 # Binary classification for demo or adjust as needed
        
        # Stage 1: Sequence Features
        self.K = 20  # Max packets per flow
        self.packet_feature_dim = 5  # [length, direction, delta_t, log_length, burst_flag]
        
        # Stage 2: Statistical Features
        self.stat_feature_dim = 64  # Output dimension of statistical extractor
        
        # Histograms
        self.hist_bins_len = 16
        self.hist_bins_time = 16
        
        # --- Model Architecture ---
        # Transformer (Stage 1)
        self.embed_dim = 64
        self.trans_num_heads = 4
        self.trans_num_layers = 2
        self.trans_ffn_dim = 128
        self.dropout = 0.1
        
        # Fusion
        self.fusion_output_dim = 128  # Dimension after fusing Seq + Stat
        
        # SwitchTab Core
        self.switchtab_input_dim = 128 # Must match fusion_output_dim
        self.switchtab_num_heads = 2   # For SwitchTab's internal Encoder
        
        # --- Training Hyperparameters ---
        self.batch_size = 128
        self.epochs = 50   # Training epochs
        self.learning_rate = 0.001
        self.loss_alpha = 0.5   # Weight for reconstruction loss (lambda_recon in spec)
        
        # Device configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')