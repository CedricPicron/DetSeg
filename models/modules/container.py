"""
Collection of container-type modules.
"""
from inspect import signature

from torch import nn

from models.build import MODELS


@MODELS.register_module()
class Sequential(nn.Sequential):
    """
    Class implementing the enhanced Sequential module.

    It adds following two features to the forward method:
        1) It allows the use of keyword arguments to be passed to the sub-modules depending on their forward signature.
        2) It adds the possibility to return all intermediate outputs from each of the sub-modules.
    """

    def forward(self, input, return_intermediate=False, **kwargs):
        """
        Forward method of the enhanced Sequential module.

        Args:
            input (Any): Input of the forward method.
            return_intermediate (bool): Whether intermediate outputs should be returned (default=False).
            kwargs (Dict): Dictionary of keyword arguments to be passed to each of the sub-modules.

        Returns:
            * If return_intermediate is False:
                output (Any): Output from the final sub-module.

            * If return_intermediate is True:
                output (List): List of size [num_sub_modules] containing the outputs from each of the sub-modules.
        """

        # Initialize empty list of outputs
        output = []

        # Iterate over all sub-modules
        for module in self:
            module_kwargs = {name: kwargs[name] for name in signature(module.forward).parameters if name in kwargs}
            input = module(input, **module_kwargs)
            output.append(input)

        # Select output from final sub-module if requested
        if not return_intermediate:
            output = output[-1]

        return output
