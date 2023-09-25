from joeynmt.encoders import CNNEncoder
import unittest
import torch
from torch import nn
from math import sqrt

class TestCNNEncoderLayer(unittest.TestCase):
    def setUp(self):
        self.emb_size = 512
        self.num_layers = 4
        self.kernel_width = 3
        self.convolutions ={"layer 1": {"output_channels":512,"kernel_width":5,"residual":True},
                            "layer 2": {"output_channels":512,"kernel_width":5,"residual":False},
        }
        seed = 42
        torch.manual_seed(seed)
        self.encoder = CNNEncoder(self.emb_size,self.num_layers,self.convolutions)
        self.dropout = 0.1

    def test_layers_expansion(self):
        four_layer_convolutions = [ {"output_channels":512,"kernel_width":5,"residual":True},
                                    {"output_channels":512,"kernel_width":5,"residual":False},
                                    {"output_channels":512,"kernel_width":3,"residual":True}, 
                                    {"output_channels":512,"kernel_width":3,"residual":True},    
        ]

        assert self.encoder.num_layers ==4
        assert len(self.encoder.convs) ==4
        assert self.encoder.convs == four_layer_convolutions

    def init_encoder(self):
        conv_linear_layers_name = ["map_to_conv_dim","map_residual_to_output"]
        for name,p in self.encoder.named_parameters():
            if name in conv_linear_layers_name:
                nn.init.normal_(p, mean=0, std=sqrt((1 - self.dropout) / self.emb_size))
            elif "embed" in name:
                nn.init.normal_(p,mean=0, std=0.1)
            elif "bias" in name:
                nn.init.constant_(p, 0)
            else:
                std =sqrt((4 * (1.0 - self.dropout)) / (self.kernel_width * self.emb_size))
                nn.init.normal_(p, mean=0, std=std)

    def test_cnn_encoder_output_size(self):
        # first two layer: 128+2*2-(5-1)-1+1 = 128 last three layers: 128+2-(3-1)+1-1 = 128
        # formula taken from https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html 
        expected_output_shape = torch.Size([10,128,512])
        # batch size x src_length x emb_size
        test_input = torch.randn(10,128,self.emb_size)
        last_encoder_output,_ = self.encoder(test_input)
        assert last_encoder_output.shape == expected_output_shape
