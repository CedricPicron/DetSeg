"""
Collection of select-related modules.
"""

from torch import nn

from models.build import MODELS


@MODELS.register_module()
class GridSelect2d(nn.Module):
    """
    Class implementing the GridSelect2d module.

    Attributes:
        in_key (str): String with key to retrieve input feature map from storage dictionary.
        grp_key (str): String with key to retrieve group indices from storage dictionary.
        grid_key (str): String with key to retrieve grid indices from storage dictionary.
        out_key (str): String with key to store selected output features in storage dictionary.
    """

    def __init__(self, in_key, grp_key, grid_key, out_key):
        """
        Initializes the GridInsert2d module.

        Args:
            in_key (str): String with key to retrieve input feature map from storage dictionary.
            grp_key (str): String with key to retrieve group indices from storage dictionary.
            grid_key (str): String with key to retrieve grid indices from storage dictionary.
            out_key (str): String with key to store selected output features in storage dictionary.
        """

        # Initialization of default nn.Module
        super().__init__()

        # Set additional attributes
        self.in_key = in_key
        self.grp_key = grp_key
        self.grid_key = grid_key
        self.out_key = out_key

    def forward(self, storage_dict, **kwargs):
        """
        Forward method of the GridSelect2d module.

        Args:
            storage_dict (Dict): Storage dictionary containing at least following keys:
                - {self.in_key} (FloatTensor): input feature map of shape [num_groups, feat_size, fH, fW];
                - {self.grp_key} (LongTensor): group indices of selects of shape [num_selects];
                - {self.grid_key} (FloatTensor): grid indices of selects of shape [num_selects, 2].

            kwargs (Dict): Dictionary of unused keyword arguments.

        Returns:
            storage_dict (Dict): Storage dictionary containing following additional key:
                - {self.out_key} (FloatTensor): selected output features of shape [num_selects, feat_size].
        """

        # Retrieve desired items from storage dictionary
        feat_map = storage_dict[self.in_key]
        grp_ids = storage_dict[self.grp_key]
        grid_ids = storage_dict[self.grid_key]

        # Get selected output features
        out_feats = feat_map[grp_ids, :, grid_ids[:, 1], grid_ids[:, 0]]

        # Store output features in storage dictionary
        storage_dict[self.out_key] = out_feats

        return storage_dict
