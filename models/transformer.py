"""
Transformer modules and build function.
"""
import copy
from typing import Optional

import torch
from torch import nn, Tensor
import torch.nn.functional as F


class Transformer(nn.Module):
    """
    Class implementing the Transformer module.

    Attributes:
        feature_dim (int): Feature size.
        num_heads (int): Number of attention heads.
        encoder (TransformerEncoder): Global multi-head self-attention encoder.
    """

    def __init__(self, num_encoder_layers=6, num_group_layers=6,
                 feature_dim=256, num_heads=8, mha_dropout=0.1,
                 ffn_hidden_dim=2048, ffn_dropout=0.1,
                 return_intermediate=False):
        """
        Initializes the Transformer module.

        Args:
            num_encoder_layers (int): Number of encoder layers.
            num_group_layers (int): Mumber of group layers.
            feature_dim (int): Feature size.
            num_heads (int): Number of attention heads.
            mha_dropout (float): Dropout used during multi-head attention (MHA).
            ffn_hidden_dim (int): Hidden dimension of feedforward network (FFN).
            ffn_dropout (float): Dropout used during feedforward network (FFN).
            return_intermediate (bool): Return intermediate group layer outputs.
        """

        super().__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads

        encoder_layer = TransformerEncoderLayer(feature_dim, num_heads, mha_dropout, ffn_hidden_dim, ffn_dropout)
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers)

        self.reset_parameters()

    def reset_parameters(self):
        """
        Resets module parameters to random initial values.

        All multi-dimensional parameters are being reset following the uniform xavier initialization.
        """

        for parameter in self.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)

    def forward(self, in_features, padding_mask, pos_encoding):
        """
        Forward method of Transformer module.

        Args:
            in_features (Tensor): Input features of shape [batch_size x feature_dim x H x W].
            padding_mask (Tensor): Boolean mask with padding information of shape [batch_size x H x W].
            pos_encoding (Tensor): Position encoding of shape [batch_size x feature_dim x H x W].

        Returns:
            out_features (Tensor): Output features of shape [batch_size x feature_dim x H x W].
        """

        # flatten from NxCxHxW to HWxNxC
        in_features = in_features.flatten(2).permute(2, 0, 1)
        pos_encoding = pos_encoding.flatten(2).permute(2, 0, 1)

        # flatten from NxHxW to NxHW
        padding_mask = padding_mask.flatten(1)

        # compute encoder features with shape HWxNxC
        encoder_features = self.encoder(in_features, feature_mask=padding_mask, pos=pos_encoding)

        # out_features = None.transpose(1, 2)

        return encoder_features


class TransformerEncoder(nn.Module):
    """
    Concatenation of encoder layers.

    Attributes:
        layers (nn.ModulesList): List of encoder layers being concatenated.
        num_layers (int): Number of concatenated encoder layers.
    """

    def __init__(self, encoder_layer, num_layers):
        """
        Initializes the TransformerEncoder module.

        Args:
            encoder_layer (TransformerEncoderLayer): Encoder layer module to be concatenated.
            num_layers (int): Number of concatenated encoder layers.
        """

        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers

    def forward(self, in_features: Tensor, feature_mask: Optional[Tensor] = None, pos: Optional[Tensor] = None):
        """
        Forward method of TransformerEncoder module.

        Args:
            in_features (Tensor): Input features of shape [num_features x batch_size x feature_dim].
            feature_mask (Tensor): Boolean mask encoding inactive features of shape [batch_size, num_features].
            pos (Tensor): Position encoding of shape [num_features x batch_size x feature_dim].

        Returns:
            out_features (Tensor): Output features of shape [num_features x batch_size x feature_dim].
        """

        out_features = in_features
        for layer in self.layers:
            out_features = layer(out_features, src_mask=feature_mask, pos=pos)

        return out_features


class TransformerEncoderLayer(nn.Module):
    """
    Encoder layer with global multi-head self-attention and FFN.

    Attributes:
        self_attn (nn.MultiheadAtttenion): Multi-head attetion (MHA) module.
        drop1 (nn.Dropout): Dropout module after global MHA.
        norm1 (nn.LayerNorm): Layernorm after MHA skip connection.
        linear1 (nn.Linear): First FFN linear layer.
        dropout (nn.Dropout): Dropout module after first FFN layer.
        linear2 (nn.Linear): Second FFN linear layer.
        dropout2 (nn.Dropout): Dropout module after second FFN layer.
        norm2 (nn.LayerNorm): Layernorm after FFN skip connection.
    """

    def __init__(self, feature_dim, num_heads, mha_dropout=0.1, ffn_hidden_dim=2048, ffn_dropout=0.1):
        super().__init__()

        # Implementation of global multi-head self-attention
        self.self_attn = nn.MultiheadAttention(feature_dim, num_heads, dropout=mha_dropout)
        self.dropout1 = nn.Dropout(mha_dropout)
        self.norm1 = nn.LayerNorm(feature_dim)

        # Implementation of feedforward network (FFN)
        self.linear1 = nn.Linear(feature_dim, ffn_hidden_dim)
        self.dropout = nn.Dropout(ffn_dropout)
        self.linear2 = nn.Linear(ffn_hidden_dim, feature_dim)
        self.dropout2 = nn.Dropout(ffn_dropout)
        self.norm2 = nn.LayerNorm(feature_dim)

    def with_pos_embed(self, tensor: Tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, src, src_mask: Optional[Tensor] = None, pos: Optional[Tensor] = None):
        # Global multi-head self-attention with position encoding
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, key_padding_mask=src_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # Feedforward network (FFN)
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src


def build_transformer(args):
    """
    Build transformer from command-line arguments.

    Args:
        args (argparse.Namespace): Command-line arguments.

    Returns:
        Transformer module.
    """

    return Transformer(
        num_encoder_layers=args.num_encoder_layers,
        num_group_layers=args.num_group_layers,
        feature_dim=args.feature_dim,
        mha_dropout=args.mha_dropout,
        num_heads=args.num_heads,
        ffn_dropout=args.ffn_dropout,
        ffn_hidden_dim=args.ffn_hidden_dim,
        return_intermediate=True,
    )


if __name__ == '__main__':
    test1 = False

    if test1:
        from main import get_parser
        args = get_parser().parse_args()
        transformer = build_transformer(args)

        detr_url = 'https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth'
        detr_state_dict = torch.hub.load_state_dict_from_url(detr_url)['model']

        from collections import OrderedDict
        encoder_state_dict = OrderedDict()

        encoder_identifier = 'transformer.encoder.'
        identifier_length = len(encoder_identifier)

        for detr_name, detr_state in detr_state_dict.items():
            if encoder_identifier in detr_name:
                new_name = detr_name[identifier_length:]
                encoder_state_dict[new_name] = detr_state

        transformer.encoder.load_state_dict(encoder_state_dict)
