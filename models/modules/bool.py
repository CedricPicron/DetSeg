"""
Collection of modules computing booleans.
"""

from torch import nn

from models.build import MODELS


@MODELS.register_module()
class PointRendBool(nn.Module):
    """
    Class implementing the PointRendBool module.

    Attributes:
        in_key (str): String with key to retrieve input map from storage dictionary.
        force_key (str): String with key to retrieve force boolean from storage dictionary (or None).
        out_key (str): String with key to store output boolean in storage dictionary.
        num_updates (int): Integer containing the number of updated 2D map locations.
        scale_factor (float): Value scaling the input map shape to get map shape of interest.
    """

    def __init__(self, in_key, out_key, num_updates, force_key=None, scale_factor=1.0):
        """
        Initializes the PointRendBool module.

        Args:
            in_key (str): String with key to retrieve input map from storage dictionary.
            out_key (str): String with key to store output boolean in storage dictionary.
            num_updates (int): Integer containing the number of updated 2D map locations.
            force_key (str): String with key to retrieve force boolean from storage dictionary (default=None).
            scale_factor (float): Value scaling the input map shape to get map shape of interest (default=1.0).
        """

        # Initialization of default nn.Module
        super().__init__()

        # Set additional attributes
        self.in_key = in_key
        self.force_key = force_key
        self.out_key = out_key
        self.num_updates = num_updates
        self.scale_factor = scale_factor

    def forward(self, storage_dict, **kwargs):
        """
        Forward method of the PointRendBool module.

        Args:
            storage_dict (Dict): Storage dictionary (possibly) containing following keys:
                - {self.in_key} (FloatTensor): input map of shape [*, mH, mW];
                - {self.force_key} (bool): boolean indicating whether to force sparse computation.

            kwargs (Dict): Dictionary of unused keyword arguments.

        Returns:
            storage_dict (Dict): Storage dictionary containing following additional key:
                - {self.out_key} (bool): output boolean indicating whether to perform sparse computation.
        """

        # Retrieve desired items from storage dictionary
        in_map = storage_dict[self.in_key]
        force_bool = storage_dict[self.force_key] if self.force_key is not None else False

        # Get output boolean
        mH, mW = [int(self.scale_factor * size) for size in in_map.size()[-2:]]
        out_bool = (self.num_updates < (mH * mW)) or force_bool

        # Store output boolean in storage dictionary
        storage_dict[self.out_key] = out_bool

        return storage_dict
