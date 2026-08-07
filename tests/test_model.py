import torch
import unittest
from src.models.sg_tcn_lstm import MultiHorizonTCNLSTM
from src.models.bilstm_attention import BiLSTMAttention

class TestModels(unittest.TestCase):
    def setUp(self):
        self.batch_size = 4
        self.seq_len = 30
        self.input_dim = 4
        self.hidden_dim = 16
        self.num_layers = 1
        self.output_dim = 3
        
    def test_sg_tcn_lstm_shape(self):
        model = MultiHorizonTCNLSTM(
            input_dim=self.input_dim, 
            hidden_dim=self.hidden_dim, 
            num_layers=self.num_layers, 
            output_dim=self.output_dim
        )
        dummy_input = torch.randn(self.batch_size, self.seq_len, self.input_dim)
        output = model(dummy_input)
        self.assertEqual(output.shape, (self.batch_size, self.output_dim))

    def test_bilstm_attention_shape(self):
        model = BiLSTMAttention(
            input_dim=self.input_dim, 
            hidden_dim=self.hidden_dim, 
            num_layers=self.num_layers, 
            output_dim=self.output_dim
        )
        dummy_input = torch.randn(self.batch_size, self.seq_len, self.input_dim)
        output = model(dummy_input)
        self.assertEqual(output.shape, (self.batch_size, self.output_dim))

if __name__ == '__main__':
    unittest.main()
