"""
Collection of container-type modules.
"""

from torch import nn


class Sequential(nn.Sequential):
    """
    Class implementing the enhanced Sequential module allowing the use of keyword arguments.
    """

    def forward(self, input, **kwargs):
        """
        Forward method of the enhanced Sequential module.

        Args:
            input (Any): Input of the forward method.
            kwargs (Dict): Dictionary of keyword arguments.
        """

        for module in self:
            input = module(input, **kwargs)

        return input
