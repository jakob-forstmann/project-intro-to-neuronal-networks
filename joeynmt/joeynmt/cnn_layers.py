import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

from joeynmt.attention import ConvolutionalAttention

from math import sqrt

"""CNN Encoder and Decoder Layer"""

class CNNEncoderLayer(nn.Module):
    """ implements the one CNN layer consisting of a 1DConv
    followed by a GLU and if specified residual connections
    Also the layer is weight normalized https://arxiv.org/pdf/1602.07868.pdf 
    """
    def __init__(self,
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

        # batch x 2*out_channels x src_len -> src_len x batch x 2*out_channels
        x = x.view(x.shape[2],x.shape[0],-1)
        # Note: no need for masking padding in the input is already handled in file batch.py
        x = F.glu(x,dim=2)

        if self.has_residual_connection:
            if self.map_residual_to_output is not None:
                residual = self.map_residual_to_output(inital_input)
                x = (x + residual) *sqrt(0.5)
        
        # output_channels x batch x src_len -> batch x output_channels x src_len
        x = x.view(x.shape[1],x.shape[2],-1)
        return x
    
    def _add_padding(self):
        """ 
        Formula for zero padding is P = (F-1)//2 taken from https://cs231n.github.io/convolutional-networks/ 
        where F is the kernel_width,this formula is only valid if the stride is 1 
        which is the case for the  original implementation of Conv Seq2Seq 
        """
        if self.kernel_width%2== 1:   
            return (self.kernel_width-1)//2
        else:
            raise RuntimeError("only odd sized kernel are supported")
   
    def add_residual_connections(self):
        if self.residual:
            # map from in_channels to output_channels b.c. otherwise the sum of 
            # the output of a layer and its input would be impossible
            if self.in_channels != self.out_channels:
                self.map_residual_to_output = nn.Linear(self.in_channels,self.out_channels)
    
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
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.kernel_width = kernel_width
        self.has_residual_connection = residual
        self.map_to_conv_dim = weight_norm(nn.Linear(embd_dim,in_channels))
        self.map_residual_to_output = None
        padding = self._add_padding()
        self.dropout = nn.Dropout(dropout)
        self.conv1D = weight_norm(nn.Conv1d(self.in_channels,self.out_channels*2,self.kernel_width,padding=padding))
        self.encoder_decoder_attention = ConvolutionalAttention(self.out_channels,embd_dim)
        out_channels = self.convs[-1]["output_channels"]
        self.map_to_emb_dim = weight_norm(nn.Linear(out_channels,self.emb_size))

        
    def forward(
        self,
        trg_embed: Tensor,
        encoder_output: Tensor,
        encoder_hidden: Tensor,
        src_mask: Tensor,
        **kwargs,
    ):
        """
        :param trg_embed: embedded targets 
        :param encoder_output: last encoder state, batch x src_len x embed_size
        :param src_mask: to mask out source paddings
        :param kwargs:
        :return: 
        """
        last_encoder_state = encoder_output[0]
        encoder_attention_value = encoder_output[1]
        x = self.dropout(trg_embed)
        x = self.conv1D(x)
        inital_input = x
        # batch x 2*output_channels x src_len -> src_len x batch x 2*output_channels
        x = x.view(x.shape[2],x.shape[0],-1)        
        x = F.glu(x,dim=2)
        if self.has_residual_connection:
            if self.map_residual_to_output is not None:
                residual = self.map_residual_to_output(inital_input)
                x = (x + residual) *sqrt(0.5)
        # output_channels x batch x src_len -> batch x output_channels x src_len
        x = x.view(x.shape[1],x.shape[2],-1)

        x = self.encoder_decoder_attention(last_encoder_state,x,encoder_attention_value,src_mask,trg_embed)
        return x


    def _add_padding(self):
        if self.kernel_width%2== 1:   
            return (self.kernel_width-1)
        raise RuntimeError("only odd sized kernel are supported")
   
    def add_residual_connections(self):
        if self.residual:
            # map from in_channels to output_channels since otherwise the sum of 
            # the output of a layer and its input would be impossible
            if self.in_channels != self.out_channels:
                self.map_residual_to_output = nn.Linear(self.in_channels,self.out_channels)