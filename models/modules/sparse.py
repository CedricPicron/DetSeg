"""
Collection of sparsity-related modules.
"""

from torch import nn

from models.build import MODELS


@MODELS.register_module()
class SparseBool2d(nn.Module):
    """
    Class implementing the SparseBool2d module.

    Attributes:
        in_key (str): String with key to retrieve input map from storage dictionary.
        force_key (str): String with key to retrieve force boolean from storage dictionary (or None).
        out_key (str): String with key to store output boolean in storage dictionary.
        num_updates (int): Integer containing the number of updated 2D map locations.
        scale_factor (float): Value scaling the input map shape to get map shape of interest.
    """

    def __init__(self, in_key, out_key, num_updates, force_key=None, scale_factor=1.0):
        """
        Initializes the SparseBool2d module.

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
        Forward method of the SparseBool2d module.

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


@MODELS.register_module()
class SparseInsert2d(nn.Module):
    """
    Class implementing the SparseInsert2d module.

    Attributes:
        in_key (str): String with key to retrieve input feature map from storage dictionary.
        ins_ids_key (str): String with key to retrieve 2D-flattened insert indices from storage dictionary.
        ins_feats_key (str): String with key to retrieve insert features from storage dictionary.
        out_key (str): String with key to store output feature map in storage dictionary (or None).
    """

    def __init__(self, in_key, ins_ids_key, ins_feats_key, out_key=None):
        """
        Initializes the SparseInsert2d module.

        Args:
            in_key (str): String with key to retrieve input feature map from storage dictionary.
            ins_ids_key (str): String with key to retrieve 2D-flattened insert indices from storage dictionary.
            ins_feats_key (str): String with key to retrieve insert features from storage dictionary.
            out_key (str): String with key to store output feature map in storage dictionary (default=None).
        """

        # Initialization of default nn.Module
        super().__init__()

        # Set additional attributes
        self.in_key = in_key
        self.ins_ids_key = ins_ids_key
        self.ins_feats_key = ins_feats_key
        self.out_key = out_key

    def forward(self, storage_dict, **kwargs):
        """
        Forward method of the SparseInsert2d module.

        Args:
            storage_dict (Dict): Storage dictionary containing at least following keys:
                - {self.in_key} (FloatTensor): input feature map of shape [batch_size, feat_size, fH, fW];
                - {self.ins_ids_key} (LongTensor): 2D-flattened insert indices of shape [batch_size, num_inserts];
                - {self.ins_feats_key} (FloatTensor): insert features of shape [batch_size, num_inserts, feat_size].

            kwargs (Dict): Dictionary of unused keyword arguments.

        Returns:
            storage_dict (Dict): Storage dictionary possibly containing following additional key:
                - {self.out_key} (FloatTensor): output feature map of shape [batch_size, feat_size, fH, fW].
        """

        # Retrieve desired items from storage dictionary
        feat_map = storage_dict[self.in_key]
        ins_ids = storage_dict[self.ins_ids_key]
        ins_feats = storage_dict[self.ins_feats_key]

        # Clone feature map if needed
        if self.out_key is not None:
            feat_map = feat_map.clone()

        # Update input map by inserting features
        batch_size, feat_size, fH, fW = feat_map.size()
        ins_ids = ins_ids[:, None, :].expand(-1, feat_size, -1)
        ins_feats = ins_feats.transpose(1, 2)

        feat_map = feat_map.view(batch_size, feat_size, fH*fW)
        feat_map.scatter_(dim=2, index=ins_ids, src=ins_feats)
        feat_map = feat_map.view(batch_size, feat_size, fH, fW)

        # Store resulting map in storage dictionary if needed
        if self.out_key is not None:
            storage_dict[self.out_key] = feat_map

        return storage_dict
