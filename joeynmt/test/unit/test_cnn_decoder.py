import unittest
from math import sqrt
import torch 
from torch import nn 
from joeynmt.decoders import CNNDecoder
class TestCNNDecoder(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.tgt_len = 4
        self.emb_size = 512
        self.kernel_width = 3
        self.dropout = 0.1
        self.vocab_size= 7
    
    def test_decoder_output(self):
        target_embedding = torch.randn(self.batch_size,self.tgt_len,self.emb_size)
        src_mask = torch.ones(size=(self.batch_size, 1, self.tgt_len)) == 1
        encoder_output = torch.randn(self.batch_size,self.tgt_len,self.emb_size)
        decoder_hidden = None  # unused
        trg_mask = None  # unused
        unroll_steps = None  # unused
        encoder_attention_value = torch.randn(self.batch_size,self.tgt_len,self.emb_size)
        convolutions = {"layer 1": {"output_channels":512,"kernel_width":5,"residual":True},
                        "layer 2": {"output_channels":512,"kernel_width":5,"residual":False}}

        decoder = CNNDecoder(convolutions,self.emb_size,num_layers=4,vocab_size=self.vocab_size)
        decoder_output,x,*res = decoder(target_embedding,
                                        encoder_output,
                                        encoder_attention_value,
                                        src_mask,
                                        unroll_steps,
                                        decoder_hidden,
                                        trg_mask)
        expected_decoder_state_shape = torch.Size([self.batch_size,self.tgt_len,self.emb_size])
        expeted_output_shape          = torch.Size([self.batch_size,self.tgt_len,self.vocab_size])
        test_input = torch.randn(self.batch_size,self.tgt_len,self.emb_size)
        assert x.shape == expected_decoder_state_shape
        self.assertFalse(torch.any(torch.eq(test_input,x)))
        assert decoder_output.shape == expeted_output_shape

       