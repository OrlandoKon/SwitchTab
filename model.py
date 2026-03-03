import torch
from torch import nn

# Encoder network with a three-layer transformer
class Encoder(nn.Module):
    def __init__(self, feature_size, num_heads=2):
        super(Encoder, self).__init__()
        self.transformer_layers = nn.Sequential(
            nn.TransformerEncoderLayer(d_model=feature_size, nhead=num_heads),
            nn.TransformerEncoderLayer(d_model=feature_size, nhead=num_heads),
            nn.TransformerEncoderLayer(d_model=feature_size, nhead=num_heads)
        )

    def forward(self, x):
        # Since Transformer expects seq_length x batch x features, we assume x is already shaped correctly
        return self.transformer_layers(x)

# Projector network
class Projector(nn.Module):
    def __init__(self, feature_size, output_size):
        super(Projector, self).__init__()
        self.linear = nn.Linear(feature_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.linear(x))

# Decoder network
class Decoder(nn.Module):
    def __init__(self, input_feature_size, output_feature_size):
        super(Decoder, self).__init__()
        self.linear = nn.Linear(input_feature_size, output_feature_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.linear(x))

# Prediction network for pre-training
class Predictor(nn.Module):
    def __init__(self, feature_size, num_classes):
        super(Predictor, self).__init__()
        self.linear = nn.Linear(feature_size, num_classes)

    def forward(self, x):
        return self.linear(x)

class SwitchTabModel(nn.Module):
    def __init__(self, feature_size, num_classes, num_heads=2):
        super(SwitchTabModel, self).__init__()
        self.encoder = Encoder(feature_size, num_heads)
        half_feature_size = feature_size // 2
        self.projector_s = Projector(feature_size, half_feature_size)
        self.projector_m = Projector(feature_size, half_feature_size)
        self.decoder = Decoder(feature_size, feature_size)  # Assuming concatenation of two half-sized embeddings
        self.predictor = Predictor(feature_size, num_classes)

    def forward(self, x1, x2):
        # Feature corruption is not included in the model itself and should be applied to the data beforehand
        z1_encoded = self.encoder(x1)
        z2_encoded = self.encoder(x2)

        s1_salient = self.projector_s(z1_encoded)
        m1_mutual = self.projector_m(z1_encoded)
        s2_salient = self.projector_s(z2_encoded)
        m2_mutual = self.projector_m(z2_encoded)

        x1_reconstructed = self.decoder(torch.cat((m1_mutual, s1_salient), dim=1))
        x2_reconstructed = self.decoder(torch.cat((m2_mutual, s2_salient), dim=1))
        x1_switched = self.decoder(torch.cat((m2_mutual, s1_salient), dim=1))
        x2_switched = self.decoder(torch.cat((m1_mutual, s2_salient), dim=1))

        return x1_reconstructed, x2_reconstructed, x1_switched, x2_switched

    def get_salient_embeddings(self, x):
        z_encoded = self.encoder(x)
        s_salient = self.projector_s(z_encoded)
        return s_salient
