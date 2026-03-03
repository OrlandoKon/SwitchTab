import torch

# Feature corruption function
def feature_corruption(x, corruption_ratio=0.3):
    # We sample a mask of the features to be zeroed out
    corruption_mask = torch.bernoulli(torch.full(x.shape, 1-corruption_ratio)).to(x.device)
    return x * corruption_mask
