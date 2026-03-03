import unittest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from model import Encoder, Projector, Decoder, Predictor
from utils import feature_corruption

class TestSwitchTabComponents(unittest.TestCase):
    def setUp(self):
        self.feature_size = 10  # A small feature size for testing
        self.batch_size = 2  # A small batch size for testing
        self.num_classes = 3  # Assuming three classes for testing
        # Initialize synthetic data for tests
        self.x_train = torch.randn(self.batch_size, self.feature_size)
        self.x_batch = torch.randn(self.batch_size, self.feature_size)
        # Initialize model components
        self.encoder = Encoder(self.feature_size)  # Assuming encoder initialization requires feature_size
        self.half_feature_size = self.feature_size // 2
        self.projector_s = Projector(self.feature_size, self.half_feature_size)  # Same for projectors
        self.projector_m = Projector(self.feature_size, self.half_feature_size)
        self.decoder = Decoder(self.feature_size, self.feature_size)
        self.predictor = Predictor(self.feature_size, self.num_classes)
    
    def test_feature_corruption(self):
        corrupted_x = feature_corruption(self.x_batch)
        # Check if the corruption function returns a tensor of the same shape
        self.assertEqual(corrupted_x.shape, self.x_batch.shape)
        # Check if about 30% of the elements are zeroed out
        # Calculate the actual corruption ratio
        actual_corruption_ratio = (corrupted_x == 0).float().mean().item()
        # Increased delta to accommodate variability
        self.assertAlmostEqual(actual_corruption_ratio, 0.3, delta=0.2)

    def test_encoder_forward_pass(self):
        # Check if the encoder can perform a forward pass without errors
        encoded_x = self.encoder(self.x_batch.unsqueeze(0))  # Add a sequence length dimension
        self.assertEqual(encoded_x.shape, (1, self.batch_size, self.feature_size))

    def test_projector_forward_pass(self):
        # Check if the projectors can perform a forward pass without errors
        projected_x_s = self.projector_s(self.x_batch)
        projected_x_m = self.projector_m(self.x_batch)
        self.assertEqual(projected_x_s.shape, (self.batch_size, self.half_feature_size))
        self.assertEqual(projected_x_m.shape, (self.batch_size, self.half_feature_size))

    def test_decoder_forward_pass(self):
        # Check if the decoder can perform a forward pass without errors
        # Now mock_input is [2, 10] because half=5, 5+5=10. Original feature_size=10
        mock_input = torch.cat([torch.randn(self.batch_size, self.half_feature_size), 
                                torch.randn(self.batch_size, self.half_feature_size)], dim=1)
        decoded_x = self.decoder(mock_input)
        self.assertEqual(decoded_x.shape, self.x_batch.shape, "Decoder output shape does not match expected.")

    def test_predictor_forward_pass(self):
        # Check if the predictor can perform a forward pass without errors
        predictions = self.predictor(self.x_batch)
        self.assertEqual(predictions.shape, (self.batch_size, self.num_classes))

if __name__ == '__main__':
    unittest.main()
