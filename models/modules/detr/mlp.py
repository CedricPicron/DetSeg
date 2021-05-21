"""
DETR multi-layer perceptron (MLP) module.
"""

from torch import nn
import torch.nn.functional as F


class MLP(nn.Module):
    """
    Module implementing a simple multi-layer perceptron (MLP).

    Attributes:
        layers (nn.ModuleList): List of nn.Linear modules, which will be split by ReLU activation functions.
        num_layers (int): Number of linear layers in the multi-layer perceptron.
    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        """
        Initializes the MLP module.

        Args:
            input_dim (int): Expected dimension of input features.
            hidden_dim (int): Feature dimension in hidden layers.
            output_dim (int): Dimension of output features.
            num_layers (int): Number of linear layers in the multi-layer perceptron.
        """

        super().__init__()
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.num_layers = num_layers

    def forward(self, x):
        """
        Forward method of the MLP module.
        """

        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers-1 else layer(x)

        return x
