import unittest
from math import sqrt
import torch 
from torch import nn 
from joeynmt.decoders import CNNDecoder
class TestCNNDecoder(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.src_len = 4
        self.emb_size = 512
        self.kernel_width = 3
        self.dropout = 0.1
        self.vocab_size= 7
        self.seed = 42

    def _init_decoder_state(self,decoder):
        conv_linear_layers_name = ["map_to_conv_dim","map_residual_to_output"]
        for name,p in decoder.named_parameters():
            if name in conv_linear_layers_name:
                nn.init.normal_(p, mean=0, std=sqrt((1 - self.dropout) / self.emb_size))
            elif "embed" in name:
                nn.init.normal_(p,mean=0, std=0.1)
            elif "bias" in name:
                nn.init.constant_(p, 0)
            else:
                std =sqrt((4 * (1.0 - self.dropout)) / (self.kernel_width * self.emb_size))
                nn.init.normal_(p, mean=0, std=std)
        return decoder 
    
    def test_decoder_output(self):
        torch.manual_seed(self.seed)
        # batch x src_len x embedding 
        target_embedding = torch.randn(self.batch_size,self.src_len,self.emb_size)
        src_mask = torch.ones(size=(self.batch_size, 1, self.emb_size)) == 1
        encoder_output = torch.randn(self.batch_size,self.src_len,self.emb_size)
        encoder_attention_value = torch.randn(self.batch_size,self.src_len,self.emb_size)
        # use the default values, 512 out_channels,kernel_width=3 and use residual connection
        # for the convolution
        decoder = CNNDecoder(self.emb_size,num_layers=4,vocab_size=self.vocab_size)
        decoder = self._init_decoder_state(decoder)
        decoder(trg_embed=target_embedding,
                encoder_output=encoder_output,
                encoder_hidden=encoder_attention_value,
                src_mask=src_mask)

        