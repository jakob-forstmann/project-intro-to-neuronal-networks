# coding: utf-8
"""
Various encoders
"""
from typing import Tuple
from math import sqrt
import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.utils import weight_norm

from joeynmt.helpers import freeze_params
from joeynmt.transformer_layers import PositionalEncoding, TransformerEncoderLayer
from joeynmt.cnn_layers import CNNEncoderLayer

class Encoder(nn.Module):
    """
    Base encoder class
    """

    # pylint: disable=abstract-method
    @property
    def output_size(self):
        """
        Return the output size

        :return:
        """
        return self._output_size


class RecurrentEncoder(Encoder):
    """Encodes a sequence of word embeddings"""

    # pylint: disable=unused-argument
    def __init__(
        self,
        rnn_type: str = "gru",
        hidden_size: int = 1,
        emb_size: int = 1,
        num_layers: int = 1,
        dropout: float = 0.0,
        emb_dropout: float = 0.0,
        bidirectional: bool = True,
        freeze: bool = False,
        **kwargs,
    ) -> None:
        """
        Create a new recurrent encoder.

        :param rnn_type: RNN type: `gru` or `lstm`.
        :param hidden_size: Size of each RNN.
        :param emb_size: Size of the word embeddings.
        :param num_layers: Number of encoder RNN layers.
        :param dropout:  Is applied between RNN layers.
        :param emb_dropout: Is applied to the RNN input (word embeddings).
        :param bidirectional: Use a bi-directional RNN.
        :param freeze: freeze the parameters of the encoder during training
        :param kwargs:
        """
        super().__init__()

        self.emb_dropout = torch.nn.Dropout(p=emb_dropout, inplace=False)
        self.type = rnn_type
        self.emb_size = emb_size

        rnn = nn.GRU if rnn_type == "gru" else nn.LSTM

        self.rnn = rnn(
            emb_size,
            hidden_size,
            num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self._output_size = 2 * hidden_size if bidirectional else hidden_size

        if freeze:
            freeze_params(self)

    def _check_shapes_input_forward(self, src_embed: Tensor, src_length: Tensor,
                                    mask: Tensor) -> None:
        """
        Make sure the shape of the inputs to `self.forward` are correct.
        Same input semantics as `self.forward`.

        :param src_embed: embedded source tokens
        :param src_length: source length
        :param mask: source mask
        """
        # pylint: disable=unused-argument
        assert src_embed.shape[0] == src_length.shape[0]
        assert src_embed.shape[2] == self.emb_size
        # assert mask.shape == src_embed.shape
        assert len(src_length.shape) == 1

    def forward(self, src_embed: Tensor, src_length: Tensor, mask: Tensor,
                **kwargs) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Applies a bidirectional RNN to sequence of embeddings x.
        The input mini-batch x needs to be sorted by src length.
        x and mask should have the same dimensions [batch, time, dim].

        :param src_embed: embedded src inputs,
            shape (batch_size, src_len, embed_size)
        :param src_length: length of src inputs
            (counting tokens before padding), shape (batch_size)
        :param mask: indicates padding areas (zeros where padding), shape
            (batch_size, src_len, embed_size)
        :param kwargs:
        :return:
            - output: hidden states with
                shape (batch_size, max_length, directions*hidden),
            - hidden_concat: last hidden state with
                shape (batch_size, directions*hidden)
        """
        self._check_shapes_input_forward(src_embed=src_embed,
                                         src_length=src_length,
                                         mask=mask)
        total_length = src_embed.size(1)

        # apply dropout to the rnn input
        src_embed = self.emb_dropout(src_embed)

        packed = pack_padded_sequence(src_embed, src_length.cpu(), batch_first=True)
        output, hidden = self.rnn(packed)

        if isinstance(hidden, tuple):
            hidden, memory_cell = hidden  # pylint: disable=unused-variable

        output, _ = pad_packed_sequence(output,
                                        batch_first=True,
                                        total_length=total_length)
        # hidden: dir*layers x batch x hidden
        # output: batch x max_length x directions*hidden
        batch_size = hidden.size()[1]
        # separate final hidden states by layer and direction
        hidden_layerwise = hidden.view(
            self.rnn.num_layers,
            2 if self.rnn.bidirectional else 1,
            batch_size,
            self.rnn.hidden_size,
        )
        # final_layers: layers x directions x batch x hidden

        # concatenate the final states of the last layer for each directions
        # thanks to pack_padded_sequence final states don't include padding
        fwd_hidden_last = hidden_layerwise[-1:, 0]
        bwd_hidden_last = hidden_layerwise[-1:, 1]

        # only feed the final state of the top-most layer to the decoder
        # pylint: disable=no-member
        hidden_concat = torch.cat([fwd_hidden_last, bwd_hidden_last], dim=2).squeeze(0)
        # final: batch x directions*hidden

        assert hidden_concat.size(0) == output.size(0), (
            hidden_concat.size(),
            output.size(),
        )
        return output, hidden_concat

    def __repr__(self):
        return f"{self.__class__.__name__}(rnn={self.rnn})"


class TransformerEncoder(Encoder):
    """
    Transformer Encoder
    """

    def __init__(
        self,
        hidden_size: int = 512,
        ff_size: int = 2048,
        num_layers: int = 8,
        num_heads: int = 4,
        dropout: float = 0.1,
        emb_dropout: float = 0.1,
        freeze: bool = False,
        **kwargs,
    ):
        """
        Initializes the Transformer.
        :param hidden_size: hidden size and size of embeddings
        :param ff_size: position-wise feed-forward layer size.
          (Typically this is 2*hidden_size.)
        :param num_layers: number of layers
        :param num_heads: number of heads for multi-headed attention
        :param dropout: dropout probability for Transformer layers
        :param emb_dropout: Is applied to the input (word embeddings).
        :param freeze: freeze the parameters of the encoder during training
        :param kwargs:
        """
        super().__init__()

        self._output_size = hidden_size

        # build all (num_layers) layers
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                size=hidden_size,
                ff_size=ff_size,
                num_heads=num_heads,
                dropout=dropout,
                alpha=kwargs.get("alpha", 1.0),
                layer_norm=kwargs.get("layer_norm", "pre"),
                activation=kwargs.get("activation", "relu"),
            ) for _ in range(num_layers)
        ])

        self.pe = PositionalEncoding(hidden_size)
        self.emb_dropout = nn.Dropout(p=emb_dropout)

        self.layer_norm = (nn.LayerNorm(hidden_size, eps=1e-6) if kwargs.get(
            "layer_norm", "post") == "pre" else None)

        if freeze:
            freeze_params(self)

    def forward(
        self,
        src_embed: Tensor,
        src_length: Tensor,  # unused
        mask: Tensor = None,
        **kwargs,
    ) -> Tuple[Tensor, Tensor]:
        """
        Pass the input (and mask) through each layer in turn.
        Applies a Transformer encoder to sequence of embeddings x.
        The input mini-batch x needs to be sorted by src length.
        x and mask should have the same dimensions [batch, time, dim].

        :param src_embed: embedded src inputs,
            shape (batch_size, src_len, embed_size)
        :param src_length: length of src inputs
            (counting tokens before padding), shape (batch_size)
        :param mask: indicates padding areas (zeros where padding), shape
            (batch_size, 1, src_len)
        :param kwargs:
        :return:
            - output: hidden states with shape (batch_size, max_length, hidden)
            - None
        """
        # pylint: disable=unused-argument
        x = self.pe(src_embed)  # add position encoding to word embeddings
        x = self.emb_dropout(x)

        for layer in self.layers:
            x = layer(x, mask)

        if self.layer_norm is not None:
            x = self.layer_norm(x)
        return x, None

    def __repr__(self):
        return (f"{self.__class__.__name__}(num_layers={len(self.layers)}, "
                f"num_heads={self.layers[0].src_src_att.num_heads}, "
                f"alpha={self.layers[0].alpha}, "
                f'layer_norm="{self.layers[0]._layer_norm_position}", '
                f"activation={self.layers[0].feed_forward.pwff_layer[1]})")

class CNNEncoder(Encoder):
    """ implements the Encoder from Convolutional Sequence to Sequence Learning"""
    def __init__(self, 
                emb_size:int,
                num_layers:int=1,
                layers:dict[str,dict[str,int]]={"layer 1": {"output_channels":512,"kernel_width":3,"residual":True}},
                dropout:float = 0.1,
                emb_dropout:float=0.1,):
        """
        initialize the CNN Encoder 
        :param num_layers: number of layers each layer contains of a 1D convolutional followed by a GLU  
        :param layers: a dict of layer name(just for convience) and the layers output_channels,kernel_width 
        and wether a residual connection should be used 
        :param dropout probality for dropout 
        """
        super().__init__()
        self.num_layers = num_layers
        self.layers = layers
        self.convs:list[dict[str,int]] = [*self.layers.values()]
        if len(self.layers)!=self.num_layers:
                self._use_default_settings()
        self.in_channels_first_layer = self.convs[0]["output_channels"]
        out_channel_last_layer = self.convs[-1]["output_channels"]
        self.emb_size = emb_size
        self.pse = PositionalEncoding(self.in_channels_first_layer)
        self.emb_dropout = nn.Dropout(p=emb_dropout)
        self.dropout = dropout
        self.map_to_conv_dim = weight_norm(nn.Linear(self.emb_size,self.in_channels_first_layer))
        self.map_to_emb_dim = weight_norm(nn.Linear(out_channel_last_layer,self.emb_size))
        self.conv_layers= nn.ModuleList([])
        self._build_layers()

    def forward(self,src_embed: Tensor):
        """
        pass the embedded input tensor to each layer of the CNN Encoder
        Each Layer consists of an 1D Convolutional followed by a GLU 
        :param src_embed (batch x src_len x embed_size)
        :return 
            - output of the last encoder layer with shape (batch x src_len x embed_size)
            -  attention value vector (batch x src_len x embed_size)
        """
        x = self.pse(src_embed) # add positional encoding
        x = self.emb_dropout(x)
        inital_input = x
        # project to dim of first conv layer
        x = self.map_to_conv_dim(x)
        # (batch x src_len x emb_size) -> (batch x emb_size x src_len)
        x = x.transpose(1,2)
        for layer in self.conv_layers:
            x = layer(x)
        # (batch x output_channels x src_len) -> (batch x src_len x output_channels)
        x = x.transpose(2,1)
        # project to embedding dim to calculate the attention value
        x = self.map_to_emb_dim(x)
        # for attention add current embedded element to the output of the last encoder layer
        attention_values = (x+inital_input) * sqrt(0.5)
        return (x,attention_values)

    def _build_layers(self):       
        in_channels = self.in_channels_first_layer
        for conv in self.convs:
            self.conv_layers.append(CNNEncoderLayer(
                                    self.emb_size,in_channels,
                                    conv["output_channels"],
                                    conv["kernel_width"],
                                    conv["residual"],
                                    self.dropout))
            in_channels = conv["output_channels"]

    def _use_default_settings(self):
        """ fills convs with values up to num_layers with the standard values 
        convs now contains num_layers many entries with the values for each layer"""    
        self.convs = self.convs + [ {"output_channels":512,"kernel_width":3,"residual":True} 
                     for i in range((len(self.layers)),self.num_layers)]

   