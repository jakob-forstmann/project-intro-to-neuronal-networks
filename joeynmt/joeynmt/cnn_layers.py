import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

from joeynmt.attention import ConvolutionalAttention

from math import sqrt

class CNNEncoderLayer(nn.Module):
    """ implements the one CNN layer consisting of a 1DConv
    followed by a GLU and if specified residual connections
    Also the layer is weight normalized https://arxiv.org/pdf/1602.07868.pdf 
    """
    def __init__(self,
                in_channels:int,
                out_channels:int=512,
                kernel_width:int=3,
                residual:int=1,
                dropout:int=0.1
                ) -> None: 
        super().__init__()
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.kernel_width = kernel_width
        self.has_residual_connection = residual
        self.map_residual_to_output = None
        padding = self._add_padding()
        self.dropout = nn.Dropout(dropout)
        self.conv1D = weight_norm(nn.Conv1d(self.in_channels,self.out_channels*2,self.kernel_width,padding=padding))

    def forward(self,x):
        """
        :param x (batch x embed_size x src_len)
        :return feature map (batch x output_channels x src_len)
        """
        inital_input = x
        x = self.dropout(x)
        x = self.conv1D(x)

        # batch x 2*output_channels x src_len -> batch x output_channels x src_len
        x = F.glu(x,dim=1)
        if self.has_residual_connection:
            if self.map_residual_to_output is not None:
                residual = self.map_residual_to_output(inital_input)
                x = (x + residual) *sqrt(0.5)       
        return x
    
    def _add_padding(self):
        """ 
        Formula for zero padding is P = (F-1)//2 taken from https://cs231n.github.io/convolutional-networks/ 
        where F is the kernel_width,this formula is only valid if the stride is 1 which is the default value
        for the original implementation """
        if self.kernel_width%2== 1:   
            return (self.kernel_width-1)//2
        raise RuntimeError("only odd sized kernel are supported")
   

    
class CNNDecoderLayer(nn.Module):
    def __init__(
                self,
                embd_dim:int,
                in_channels:int,
                out_channels:int=512,
                kernel_width:int=3,
                residual:int=1,
                dropout:int=0.1
                
                ) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.kernel_width = kernel_width
        self.has_residual_connection = residual
        self.map_to_conv_dim = weight_norm(nn.Linear(embd_dim,in_channels))
        self.map_residual_to_output = None
        self.dropout = nn.Dropout(dropout)
        self.conv1D = weight_norm(nn.Conv1d(self.in_channels,
                                            self.out_channels*2,
                                            self.kernel_width,
                                            padding=self.add_padding_()))
        self.encoder_decoder_attention = ConvolutionalAttention(self.out_channels,embd_dim)
        
    def forward(
        self,
        x:Tensor,
        trg_embed: Tensor,
        encoder_output: Tensor,
        encoder_attention_value:Tensor,
        src_mask: Tensor,
        **kwargs,
    ):
        """
        :param x: positional embedded input to the current layer, batch x trg_size x output_channels
        :param trg_embed: embedded targets batch x trg_size x embd_size 
        :param encoder_output: last encoder state batch x trg_len x embed_size
        :param src_mask: to mask out source paddings
        :param kwargs:
        """
        # (batch x trgt_size x emb_size) -> (batch x emb_size x src_len)
        x = x.transpose(1,2)
        inital_input = x
        x = self.dropout(x)
        x = self.conv1D(x)
        x = F.glu(x,dim=1)
        # remove padded elements from the end of the output
        x = x[:,:,:-(self.kernel_width-1)]
        encoder_output =self.transpose_encoder_output(encoder_output)
        x,att_score = self.encoder_decoder_attention(encoder_output,
                                           encoder_attention_value,
                                           current_decoder_state=x,
                                           padding_mask_encoder=src_mask,
                                           target_embedding=trg_embed)
        if self.has_residual_connection:
            if self.map_residual_to_output is not None:
                residual = self.map_residual_to_output(inital_input)
                x = (x + residual) *sqrt(0.5)
        return x,att_score

    def transpose_encoder_output(self,encoder_output):
        """ batch x src_len x emb_size -> batch x emb_size x src_len
        neccessary for calculating the attention score"""
        return encoder_output.transpose(2,1)

    def add_padding_(self):
        if self.kernel_width%2== 1:   
            return self.kernel_width-1
        raise RuntimeError("only odd sized kernel are supported")
   