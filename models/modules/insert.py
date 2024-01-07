"""
Collection of insert-related modules.
"""

from torch import nn

from models.build import MODELS


@MODELS.register_module()
class GridInsert2d(nn.Module):
    """
    Class implementing the GridInsert2d module.

    Attributes:
        in_key (str): String with key to retrieve input feature map from storage dictionary.
        grp_key (str): String with key to retrieve group indices from storage dictionary.
        grid_key (str): String with key to retrieve grid indices from storage dictionary.
        feats_key (str): String with key to retrieve insert features from storage dictionary.
        out_key (str): String with key to store output feature map in storage dictionary (or None).
    """

    def __init__(self, in_key, grp_key, grid_key, feats_key, out_key=None):
        """
        Initializes the GridInsert2d module.

        Args:
            in_key (str): String with key to retrieve input feature map from storage dictionary.
            grp_key (str): String with key to retrieve group indices from storage dictionary.
            grid_key (str): String with key to retrieve grid indices from storage dictionary.
            feats_key (str): String with key to retrieve insert features from storage dictionary.
            out_key (str): String with key to store output feature map in storage dictionary (default=None).
        """

        # Initialization of default nn.Module
        super().__init__()

        # Set additional attributes
        self.in_key = in_key
        self.grp_key = grp_key
        self.grid_key = grid_key
        self.feats_key = feats_key
        self.out_key = out_key

    def forward(self, storage_dict, **kwargs):
        """
        Forward method of the GridInsert2d module.

        Args:
            storage_dict (Dict): Storage dictionary containing at least following keys:
                - {self.in_key} (FloatTensor): input feature map of shape [num_groups, feat_size, fH, fW];
                - {self.grp_key} (LongTensor): group indices of insertions of shape [num_inserts];
                - {self.grid_key} (FloatTensor): grid indices of insertions of shape [num_inserts, 2];
                - {self.feats_key} (FloatTensor): features to be inserted of shape [num_inserts, feat_size].

            kwargs (Dict): Dictionary of unused keyword arguments.

        Returns:
            storage_dict (Dict): Storage dictionary possibly containing following additional key:
                - {self.out_key} (FloatTensor): output feature map of shape [num_groups, feat_size, fH, fW].
        """

        # Retrieve desired items from storage dictionary
        feat_map = storage_dict[self.in_key]
        grp_ids = storage_dict[self.grp_key]
        grid_ids = storage_dict[self.grid_key]
        ins_feats = storage_dict[self.feats_key]

        # Clone feature map if needed
        if self.out_key is not None:
            feat_map = feat_map.clone()

        # Update feature map by inserting features
        feat_map[grp_ids, :, grid_ids[:, 1], grid_ids[:, 0]] = ins_feats

        # Store feature map in storage dictionary if needed
        if self.out_key is not None:
            storage_dict[self.out_key] = feat_map

        return storage_dict


@MODELS.register_module()
class PointRendGridInsert2d(nn.Module):
    """
    Class implementing the PointRendGridInsert2d module.

    Attributes:
        in_key (str): String with key to retrieve input feature map from storage dictionary.
        ins_ids_key (str): String with key to retrieve 2D-flattened insert indices from storage dictionary.
        ins_feats_key (str): String with key to retrieve insert features from storage dictionary.
        out_key (str): String with key to store output feature map in storage dictionary (or None).
    """

    def __init__(self, in_key, ins_ids_key, ins_feats_key, out_key=None):
        """
        Initializes the PointRendGridInsert2d module.

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
        Forward method of the PointRendGridInsert2d module.

        Args:
            storage_dict (Dict): Storage dictionary containing at least following keys:
                - {self.in_key} (FloatTensor): input feature map of shape [num_groups, feat_size, fH, fW];
                - {self.ins_ids_key} (LongTensor): 2D-flattened insert indices of shape [num_groups, num_pts];
                - {self.ins_feats_key} (FloatTensor): insert features of shape [num_groups, num_pts, feat_size].

            kwargs (Dict): Dictionary of unused keyword arguments.

        Returns:
            storage_dict (Dict): Storage dictionary possibly containing following additional key:
                - {self.out_key} (FloatTensor): output feature map of shape [num_groups, feat_size, fH, fW].
        """

        # Retrieve desired items from storage dictionary
        feat_map = storage_dict[self.in_key]
        ins_ids = storage_dict[self.ins_ids_key]
        ins_feats = storage_dict[self.ins_feats_key]

        # Clone feature map if needed
        if self.out_key is not None:
            feat_map = feat_map.clone()

        # Update feature map by inserting features
        num_groups, feat_size, fH, fW = feat_map.size()
        ins_ids = ins_ids[:, None, :].expand(-1, feat_size, -1)
        ins_feats = ins_feats.transpose(1, 2)

        feat_map = feat_map.view(num_groups, feat_size, fH*fW)
        feat_map.scatter_(dim=2, index=ins_ids, src=ins_feats)
        feat_map = feat_map.view(num_groups, feat_size, fH, fW)

        # Store feature map in storage dictionary if needed
        if self.out_key is not None:
            storage_dict[self.out_key] = feat_map

        return storage_dict
