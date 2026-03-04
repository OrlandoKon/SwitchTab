import torch.nn as nn

class StatisticalFeatureExtractor(nn.Module):
    def __init__(self, input_dim=64, output_dim=64):
        super(StatisticalFeatureExtractor, self).__init__()
        # As per spec, this module handles the statistical features.
        # Since the features are pre-calculated (64 dims), this module 
        # acts as a transformation layer if needed.
        # Spec says "Output: stat_feature [64]".
        # We can just use Identity or a Linear layer for flexibility.
        # Given "Resource < 3M", adding parameters here is fine but
        # usually statistical features are just inputs.
        
        self.net = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, output_dim),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.net(x)
