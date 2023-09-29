from joeynmt.encoders import CNNEncoder
import unittest
import torch
from torch import nn
from math import sqrt

class TestCNNEncoderLayer(unittest.TestCase):
    def setUp(self):
        self.emb_size =512
        torch.manual_seed(42)

    def get_encoder(self,
                    emb_size = 512,
                    num_layers:int = 4,
                    convolutions:dict[str,dict[str,int]]=None
                    ):

        return CNNEncoder(convolutions,emb_size,num_layers)

    def test_layers_expansion(self):
        convolutions = {"layer 1": {"output_channels":512,"kernel_width":5,"residual":True},
                        "layer 2": {"output_channels":512,"kernel_width":5,"residual":False}}

        encoder = self.get_encoder(convolutions=convolutions)     

        four_layer_convolutions = [ {"output_channels":512,"kernel_width":5,"residual":True},
                                    {"output_channels":512,"kernel_width":5,"residual":False},
                                    {"output_channels":512,"kernel_width":3,"residual":True},
                                    {"output_channels":512,"kernel_width":3,"residual":True},    
        ]

        assert encoder.num_layers ==4
        assert len(encoder.convs) ==4
        assert encoder.convs == four_layer_convolutions


    def test_cnn_encoder_output_size(self):
        convolutions = {"layer 1": {"output_channels":512,"kernel_width":5,"residual":True},
                        "layer 2": {"output_channels":512,"kernel_width":5,"residual":True},
                        "layer 3": {"output_channels":512,"kernel_width":3,"residual":True},
                        "layer 4": {"output_channels":512,"kernel_width":3,"residual":True}}

        # output shape batch x output_channels x trgt_len
        # output channels: first two layer: 128+2*2-(5-1)-1+1 = 128 last three layers: 128+2-(3-1)+1-1 = 128
        # taken from https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
        expected_output_shape = torch.Size([10,128,512])

        encoder = self.get_encoder(convolutions=convolutions)
        # batch size x src_length x emb_size
        test_input = torch.randn(10,128,self.emb_size)
        last_encoder_output,_ = encoder(test_input)
        assert last_encoder_output.shape == expected_output_shape
        self.assertFalse(torch.any(torch.eq(test_input,last_encoder_output)))

    def test_cnn_encoder_with_different_output_channels(self):
        convolutions = {"layer 1": {"output_channels":512,"kernel_width":5,"residual":True},
                        "layer 2": {"output_channels":768,"kernel_width":5,"residual":True},
                        "layer 3": {"output_channels":512,"kernel_width":5,"residual":True}}

        expected_output_shape = torch.Size([10,128,512])
        encoder = self.get_encoder(num_layers=3,convolutions=convolutions)
        test_input = torch.randn(10,128,self.emb_size)
        last_encoder_output,_ = encoder(test_input)
        assert last_encoder_output.shape == expected_output_shape
        self.assertFalse(torch.any(torch.eq(test_input,last_encoder_output)))

   