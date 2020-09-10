"""
Encoder modules and build function.
"""
import copy

import torch
from torch import nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """
    Class implementing the Encoder module.

    Attributes:
        feat_dim (int): Feature dimension used in the encoder.
        layers (nn.ModulesList): List of encoder layers being concatenated.
        num_layers (int): Number of concatenated encoder layers.
    """

    def __init__(self, encoder_layer, feat_dim, num_layers):
        """
        Initializes the Encoder module.

        Args:
            encoder_layer (nn.Module): Encoder layer module to be concatenated.
            feat_dim (int): Feature dimension used in the encoder.
            num_layers (int): Number of concatenated encoder layers.
        """

        super().__init__()
        self.feat_dim = feat_dim
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers

    def reset_parameters(self):
        """
        Resets all multi-dimensional module parameters according to the uniform xavier initialization.
        """

        for parameter in self.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)

    def forward(self, features, feature_masks, pos_encodings):
        """
        Forward method of the Encoder module.

        Args:
            features (FloatTensor): Features of shape [H*W, batch_size, feat_dim].
            feature_masks (BoolTensor): Boolean masks encoding inactive features of shape [batch_size, H, W].
            pos_encodings (FloatTensor): Position encodings of shape [H*W, batch_size, feat_dim].

        Returns:
            features (FloatTensor): Features of shape [H*W, batch_size, feat_dim].
        """

        # Reshape feature_masks as src_masks
        batch_size, H, W = feature_masks.shape
        src_masks = feature_masks.view(batch_size, H*W)

        # Loop over different encoder layers
        for layer in self.layers:
            features = layer(features, src_masks, pos_encodings)

        return features


class SelfAttentionEncoderLayer(nn.Module):
    """
    Class implementing the SelfAttentionEncoderLayer.

    Encoder layer with global multi-head self-attention, followed by a FFN.

    Attributes:
        self_attn (nn.MultiheadAtttenion): Multi-head attention (MHA) module.
        drop1 (nn.Dropout): Dropout module after global MHA.
        norm1 (nn.LayerNorm): Layernorm after MHA skip connection.
        linear1 (nn.Linear): First FFN linear layer.
        dropout (nn.Dropout): Dropout module after first FFN layer.
        linear2 (nn.Linear): Second FFN linear layer.
        dropout2 (nn.Dropout): Dropout module after second FFN layer.
        norm2 (nn.LayerNorm): Layernorm after FFN skip connection.
    """

    def __init__(self, feat_dim, mha_dict, ffn_dict):
        """
        Initializes the SelfAttentionEncoderLayer module.

        Args:
            feat_dim (int): feat_dim (int): Feature dimension used in the encoder layer.
            mha_dict (Dict): Dict containing parameters of the MultiheadAttention module:
                - num_heads (int): number of attention heads;
                - dropout (float): dropout probability used throughout the MultiheadAttention module.
            ffn_dict (Dict): Dict containing parameters of the FFN module:
                - hidden_dim (int): number of hidden dimensions in the FFN hidden layers;
                - dropout (float): dropout probability used throughout the FFN module.
        """

        # Intialization of default nn.Module
        super().__init__()

        # Initialization of multi-head attention module
        num_heads = mha_dict['num_heads']
        mha_dropout = mha_dict['dropout']

        self.self_attn = nn.MultiheadAttention(feat_dim, num_heads, dropout=mha_dropout)
        self.dropout1 = nn.Dropout(mha_dropout)
        self.norm1 = nn.LayerNorm(feat_dim)

        # Initialization of feedforward network (FFN) module
        ffn_hidden_dim = ffn_dict['hidden_dim']
        ffn_dropout = ffn_dict['dropout']

        self.linear1 = nn.Linear(feat_dim, ffn_hidden_dim)
        self.dropout = nn.Dropout(ffn_dropout)
        self.linear2 = nn.Linear(ffn_hidden_dim, feat_dim)
        self.dropout2 = nn.Dropout(ffn_dropout)
        self.norm2 = nn.LayerNorm(feat_dim)

    def forward(self, features, feature_masks, pos_encodings):
        """
        Forward method of the SelfAttentionEncoderLayer module.

        Args:
            features (FloatTensor): Features of shape [H*W, batch_size, feat_dim].
            feature_masks (BoolTensor): Boolean masks encoding inactive features of shape [batch_size, H*W].
            pos_encodings (FloatTensor): Position encodings of shape [H*W, batch_size, feat_dim].

        Returns:
            features (FloatTensor): Features of shape [H*W, batch_size, feat_dim].
        """

        # Global multi-head self-attention with position encoding
        queries = features + pos_encodings
        keys = features + pos_encodings
        values = features

        delta_features = self.self_attn(queries, keys, values, key_padding_mask=feature_masks, need_weights=False)[0]
        features = features + self.dropout1(delta_features)
        features = self.norm1(features)

        # Feedforward network (FFN)
        delta_features = self.linear2(self.dropout(F.relu(self.linear1(features))))
        features = features + self.dropout2(delta_features)
        features = self.norm2(features)

        return features


def build_encoder(args):
    """
    Build Encoder module from command-line arguments.

    Args:
        args (argparse.Namespace): Command-line arguments.

    Returns:
        encoder (Encoder): The specified Encoder module.
    """

    mha_dict = {'num_heads': args.num_heads, 'dropout': args.mha_dropout}
    ffn_dict = {'hidden_dim': args.ffn_hidden_dim, 'dropout': args.ffn_dropout}

    encoder_layer = SelfAttentionEncoderLayer(args.feat_dim, mha_dict, ffn_dict)
    encoder = Encoder(encoder_layer, args.feat_dim, args.num_encoder_layers)

    return encoder


if __name__ == '__main__':
    test1 = True

    if test1:
        from main import get_parser
        args = get_parser().parse_args()
        encoder = build_encoder(args)

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

        encoder.load_state_dict(encoder_state_dict)
