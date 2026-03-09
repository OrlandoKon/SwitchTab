import torch

class Config:
    def __init__(self):
        # Logging
        self.log_dir = './log'

        # --- Data Preprocessing ---
        self.num_classes = 16 # Placeholder, will be updated by dataset
        self.data_dir = '/root/Demo/SwitchTab/data/ISCX-VPN-2016'
        
        # Flow Parameters
        self.K = 20  # Max packets per flow
        
        # Packet Feature Construction
        self.seq_feature_dim = 5  # [length, direction, delta_t, log_length, burst_flag]
        self.stat_feature_dim = 55 # First 55 features from statistical extractor (64 total - 9 padding)
        # self.pos_feature_dim = 1   # Relative position feature (removed)
        
        # Total packet feature dimension input to FlowEmbedding: 5 + 55 = 60
        self.packet_input_dim = self.seq_feature_dim + self.stat_feature_dim
        
        # --- Model Architecture ---
        # FlowEmbedding (Stage 1)
        self.flow_embed_dim = 128
        self.flow_num_heads = 4
        self.flow_num_layers = 2
        self.flow_ffn_dim = 256
        self.dropout = 0.1
        
        # SwitchTab Core (Feature size matches FlowEmbedding output)
        self.switchtab_input_dim = 128 
        self.switchtab_num_heads = 2   # For SwitchTab's internal Encoder
        
        # --- Training Hyperparameters ---
        self.batch_size = 128
        self.epochs = 50   # Fine-tuning stage max epochs
        self.learning_rate = 0.001 # Adam learning rate for fine-tuning
        self.loss_alpha = 0.3   # Weight for classification loss (L_total = L_recon + alpha * L_cls)
        
        # Pre-training settings (Optional, if implementing pre-training later)
        self.pretrain_epochs = 1000
        self.pretrain_lr = 0.0003
        
        # Device configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')