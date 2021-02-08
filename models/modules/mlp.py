"""
Collection of MLP-like modules.
"""
from copy import deepcopy

from torch import nn
import torch.nn.functional as F


class FFN(nn.Module):
    """
    Class implementing the FFN module.

    Attributes:
        layer_norm (nn.LayerNorm): Pre-activaion layer normalization module.
        in_proj (nn.Linear): Linear input projection module projecting normalized features to hidden features.
        out_proj (nn.Linear): Linear output projection module projecting hidden features to delta output features.
    """

    def __init__(self, feat_size, hidden_size):
        """
        Initializes the FFN module.

        Args:
            feat_size (int): Integer containing the input and output feature size.
            hidden_size (int): Integer containing the hidden feature size.
        """

        # Initialization of default nn.Module
        super().__init__()

        # Initialize layer normalization module
        self.layer_norm = nn.LayerNorm(feat_size)

        # Initialize input and output projection modules
        self.in_proj = nn.Linear(feat_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, feat_size)

    def forward(self, in_feat_list):
        """
        Forward method of the FFN module.

        Args:
            in_feat_list (List): List [num_feat_sets] containing input features of shape [num_features, feat_size].

        Returns:
            out_feat_list (List): List [num_feat_sets] containing output features of shape [num_features, feat_size].
        """

        # Initialize empty output features list
        out_feat_list = []

        # Perform FFN-operation on every set of input features
        for in_feats in in_feat_list:

            # Get FFN delta features
            norm_feats = self.layer_norm(in_feats)
            hidden_feats = self.in_proj(F.relu(norm_feats, inplace=True))
            delta_feats = self.out_proj(F.relu(hidden_feats, inplace=True))

            # Get output features
            out_feats = in_feats + delta_feats
            out_feat_list.append(out_feats)

        return out_feat_list


class MLP(nn.Module):
    """
    Class implementing the MLP module.

    Attributes:
        head (nn.Sequential): Sequence of activation and projection modules.
    """

    def __init__(self, in_size, hidden_size, out_size, num_hidden_layers):
        """
        Initializes the MLP module.

        Args:
            in_size (int): Integer containing the input feature size.
            hidden_size (int): Integer containing the hidden feature size.
            out_size (int): Integer containing the output feature size.
            num_hidden_layers (int): Integer containing the number of hidden layers.

        Raises:
            ValueError: Raised when the requested number of hidden layers is negative.
        """

        # Initialization of default nn.Module
        super().__init__()

        # Get MLP head depending on the number of hidden layers
        if num_hidden_layers == 0:
            self.head = nn.Sequential(nn.ReLU(inplace=True), nn.Linear(in_size, out_size))

        elif num_hidden_layers > 0:
            input_block = nn.Sequential(nn.ReLU(inplace=True), nn.Linear(in_size, hidden_size))
            hidden_block = nn.Sequential(nn.ReLU(inplace=True), nn.Linear(hidden_size, hidden_size))
            output_block = nn.Sequential(nn.ReLU(inplace=True), nn.Linear(hidden_size, out_size))

            hidden_blocks = [deepcopy(hidden_block) for _ in range(num_hidden_layers-1)]
            self.head = nn.Sequential(input_block, *hidden_blocks, output_block)

        else:
            raise ValueError(f"The number of hidden layers must be non-negative (got {num_hidden_layers}).")

    def forward(self, in_feat_list):
        """
        Forward method of the MLP module.

        Args:
            in_feat_list (List): List [num_feat_sets] containing input features of shape [num_features, feat_size].

        Returns:
            out_feat_list (List): List [num_feat_sets] containing output features of shape [num_features, feat_size].
        """

        # Perform MLP-operation on every set of input features
        out_feat_list = [self.head(in_feats) for in_feats in in_feat_list]

        return out_feat_list
