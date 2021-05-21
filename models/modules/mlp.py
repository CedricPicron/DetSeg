"""
Multi-layer perceptron (MLP) module.
"""
from copy import deepcopy

from torch import nn


class MLP(nn.Module):
    """
    Class implementing the MLP module.

    Attributes:
        mlp (nn.Sequential): Sequence of activation and projection modules.
    """

    def __init__(self, in_size, hidden_size, out_size, layers, **kwargs):
        """
        Initializes the MLP module.

        Args:
            in_size (int): Integer containing the input feature size.
            hidden_size (int): Integer containing the hidden feature size.
            out_size (int): Integer containing the output feature size.
            layers (int): Integer containing the number of hidden layers.

        Raises:
            ValueError: Raised when the requested number of hidden layers is negative.
        """

        # Initialization of default nn.Module
        super().__init__()

        # Get MLP head depending on the number of hidden layers
        if layers == 0:
            self.mlp = nn.Sequenatial(nn.Sequential(nn.ReLU(inplace=False), nn.Linear(in_size, out_size)))

        elif layers > 0:
            input_block = nn.Sequential(nn.ReLU(inplace=False), nn.Linear(in_size, hidden_size))
            hidden_block = nn.Sequential(nn.ReLU(inplace=True), nn.Linear(hidden_size, hidden_size))
            output_block = nn.Sequential(nn.ReLU(inplace=True), nn.Linear(hidden_size, out_size))

            hidden_blocks = [deepcopy(hidden_block) for _ in range(layers-1)]
            self.mlp = nn.Sequential(input_block, *hidden_blocks, output_block)

        else:
            raise ValueError(f"The number of hidden layers must be non-negative (got {layers}).")

    def forward(self, in_feat_list, **kwargs):
        """
        Forward method of the MLP module.

        Args:
            in_feat_list (List): List [num_feat_sets] containing input features of shape [num_features, feat_size].

        Returns:
            out_feat_list (List): List [num_feat_sets] containing output features of shape [num_features, feat_size].
        """

        # Perform MLP-operation on every set of input features
        out_feat_list = [self.mlp(in_feats) for in_feats in in_feat_list]

        return out_feat_list
