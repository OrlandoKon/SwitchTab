import torch

class Config:
    def __init__(self):
        # Data settings
        self.data_path = '/root/Demo/SwitchTab/data/Tatanic/train.csv'
        self.num_samples = 1000
        self.feature_size = 20
        self.num_classes = 3
        
        # Training settings
        self.batch_size = 64
        self.epochs = 100
        self.learning_rate = 0.001
        
        # SwitchTab Model settings
        self.model_dim = 128
        self.output_dim = 64
        self.num_attention_heads = 4
        
        # Device configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')