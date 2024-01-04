"""
Collection of modules computing ground-truth targets.
"""

from torch import nn

from models.build import MODELS


@MODELS.register_module()
class AmbiguityTargets(nn.Module):
    """
    Class implementing the AmbiguityTargets module.

    Attributes:
        in_key (str): String with key to retrieve input tensor from storage dictionary.
        out_key (str): String with key to store output ambiguity targets in storage dictionary.
        low_bnd (float): Value containing the ambiguity lower bound.
        up_bnd (float): Value containing the ambiguity upper bound.
    """

    def __init__(self, in_key, out_key, low_bnd=0.0, up_bnd=1.0):
        """
        Initializes the AmbiguityTargets module.

        Args:
            in_key (str): String with key to retrieve input tensor from storage dictionary.
            out_key (str): String with key to store output ambiguity targets in storage dictionary.
            low_bnd (float): Value containing the ambiguity lower bound (default=0.0).
            up_bnd (float): Value containing the ambiguity upper bound (default=1.0).
        """

        # Initialization of default nn.Module
        super().__init__()

        # Set additional attributes
        self.in_key = in_key
        self.out_key = out_key
        self.low_bnd = low_bnd
        self.up_bnd = up_bnd

    def forward(self, storage_dict, **kwargs):
        """
        Forward method of the AmbiguityTargets module.

        Args:
            storage_dict (Dict): Storage dictionary containing at least following key:
                - {self.in_key} (FloatTensor): input tensor from which to compute ambiguity targets of shape [*].

            kwargs (Dict): Dictionary of unused keyword arguments.

        Returns:
            storage_dict (Dict): Storage dictionary containing following additional key:
                - {self.out_key} (FloatTensor): output tensor with ambiguity targets of shape [*].
        """

        # Retrieve input tensor from storage dictionary
        in_tensor = storage_dict[self.in_key]

        # Get output tensor with ambiguity targets
        out_tensor = (in_tensor > self.low_bnd) & (in_tensor < self.up_bnd)
        out_tensor = out_tensor.float()

        # Store output tensor in storage dictionary
        storage_dict[self.out_key] = out_tensor

        return storage_dict


@MODELS.register_module()
class BinaryTargets(nn.Module):
    """
    Class implementing the BinaryTargets module.

    Attributes:
        in_key (str): String with key to retrieve input tensor from storage dictionary.
        out_key (str): String with key to store binarized output tensor in storage dictionary.
        threshold (float): Threshold used to binarize the input tensor.
    """

    def __init__(self, in_key, out_key, threshold=0.5):
        """
        Initializes the BinaryTargets module.

        Args:
            in_key (str): String with key to retrieve input tensor from storage dictionary.
            out_key (str): String with key to store binarized output tensor in storage dictionary.
            threshold (float): Threshold used to binarize the input tensor (default=0.5).
        """

        # Initialization of default nn.Module
        super().__init__()

        # Set additional attributes
        self.in_key = in_key
        self.out_key = out_key
        self.threshold = threshold

    def forward(self, storage_dict, **kwargs):
        """
        Forward method of the BinaryTargets module.

        Args:
            storage_dict (Dict): Storage dictionary containing at least following key:
                - {self.in_key} (FloatTensor): input tensor to be binarized of shape [*].

            kwargs (Dict): Dictionary of unused keyword arguments.

        Returns:
            storage_dict (Dict): Storage dictionary containing following additional key:
                - {self.out_key} (FloatTensor): binarized output tensor of shape [*].
        """

        # Retrieve input tensor from storage dictionary
        in_tensor = storage_dict[self.in_key]

        # Get binarized output tensor
        out_tensor = in_tensor > self.threshold
        out_tensor = out_tensor.float()

        # Store output tensor in storage dictionary
        storage_dict[self.out_key] = out_tensor

        return storage_dict
