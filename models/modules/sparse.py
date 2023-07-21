"""
Collection of sparsity-based modules.
"""

import torch
from torch import nn

from models.build import build_model, MODELS
from models.functional.utils import maps_to_seq, seq_to_maps


@MODELS.register_module()
class IdBase2d(nn.Module):
    """
    Class implementing the IdBase2d module.

    Attributes:
        act_mask_key (str): String with key to retrieve the active mask from the storage dictionary.
        id (nn.Module): Module performing the 2D index-based processing.
    """

    def __init__(self, act_mask_key, id_cfg):
        """
        Initializes the IdBase2d module.

        Args:
            act_mask_key (str): String with key to retrieve the active mask from the storage dictionary.
            id_cfg (Dict): Configuration dictionary specifying the 2D index-based processing module.
        """

        # Initialization of default nn.Module
        super().__init__()

        # Set active mask key attribute
        self.act_mask_key = act_mask_key

        # Build index-based processing module
        self.id = build_model(id_cfg)

    def forward(self, in_feat_map, storage_dict, **kwargs):
        """
        Forward method of the IdBase2d module.

        Args:
            in_feat_map (FloatTensor): Input feature map of shape [batch_size, feat_size, fH, fW].

            storage_dict (Dict): Storage dictionary containing at least following key:
                - {self.act_mask_key} (BoolTensor): active mask of shape [batch_size, 1, fH, fW].

            kwargs (Dict): Dictionary of unused keyword arguments.

        Returns:
            out_feat_map (FloatTensor): Output feature map of shape [batch_size, feat_size, fH, fW].
        """

        # Get active and passive mask
        act_mask = storage_dict[self.act_mask_key].squeeze(dim=1)
        pas_mask = ~act_mask

        # Get desired items for index-based processing
        batch_ids, y_ids, x_ids = pas_mask.nonzero(as_tuple=True)
        pas_feats = in_feat_map[batch_ids, :, y_ids, x_ids]

        batch_ids, y_ids, x_ids = act_mask.nonzero(as_tuple=True)
        act_feats = in_feat_map[batch_ids, :, y_ids, x_ids]
        pos_ids = torch.stack([x_ids, y_ids], dim=1)

        device = in_feat_map.device
        num_act_feats = len(act_feats)
        num_feats = num_act_feats + len(pas_feats)

        id_map = torch.zeros_like(act_mask, dtype=torch.int64)
        id_map[act_mask] = torch.arange(num_act_feats, device=device)
        id_map[pas_mask] = torch.arange(num_act_feats, num_feats, device=device)

        # Update active features with index-based processing
        id_kwargs = {'aux_feats': pas_feats, 'id_map': id_map, 'roi_ids': batch_ids, 'pos_ids': pos_ids}
        act_feats = self.id(act_feats, **id_kwargs, **kwargs)

        # Get output feature map
        out_feat_map = in_feat_map.clone()
        out_feat_map[batch_ids, :, y_ids, x_ids] = act_feats

        return out_feat_map


@MODELS.register_module()
class Sparse3d(nn.Module):
    """
    Class implementing the Sparse3d module.

    Attributes:
        seq_feats_key (str): String used to retrieve sequential features from storage dictionary (or None).
        act_map_ids (List): List [num_act_maps] with map indices to get active features from (or None).
        act_mask_key (str): String used to retrieve active mask from storage dictionary (or None).
        pos_feats_key (str): String used to retrieve position features from storage dictionary (or None).
        get_act_batch_ids (bool): Boolean indicating whether to get active batch indices.
        get_pas_feats (bool): Boolean indicating whether to get passive features.
        get_id_maps (bool): Boolean indicating whether to get list of index maps.
        sparse (nn.Module): Underlying sparse module updating the active features.
    """

    def __init__(self, sparse_cfg, seq_feats_key=None, act_map_ids=None, act_mask_key=None, pos_feats_key=None,
                 get_act_batch_ids=False, get_pas_feats=True, get_id_maps=True):
        """
        Initializes the Sparse3d module.

        Args:
            sparse_cfg (Dict): Configuration dictionary specifying the underlying sparse module.
            seq_feats_key (str): String used to retrieve sequential features from storage dictionary (default=None).
            act_map_ids (List): List [num_act_maps] with map indices to get active features from (default=None).
            act_mask_key (str): String used to retrieve active mask from storage dictionary (default=None).
            pos_feats_key (str): String used to retrieve position features from storage dictionary (default=None).
            get_act_batch_ids (bool): Boolean indicating whether to get active batch indices (default=False).
            get_pas_feats (bool): Boolean indicating whether to get passive features (default=True).
            get_id_maps (bool): Boolean indicating whether to get list of index maps (default=True).
        """

        # Initialization of default nn.Module
        super().__init__()

        # Build underlying sparse module
        self.sparse = build_model(sparse_cfg)

        # Set additional attributes
        self.seq_feats_key = seq_feats_key
        self.act_map_ids = act_map_ids
        self.act_mask_key = act_mask_key
        self.pos_feats_key = pos_feats_key
        self.get_act_batch_ids = get_act_batch_ids
        self.get_pas_feats = get_pas_feats
        self.get_id_maps = get_id_maps

    def forward(self, in_feat_maps, storage_dict, **kwargs):
        """
        Forward method of the Sparse3d module.

        Args:
            in_feat_maps (List): List [num_maps] of input feature maps of shape [batch_size, feat_size, fH, fW].

            storage_dict (Dict): Storage dictionary (possibly) containing following keys:
                - {self.seq_feats_key} (FloatTensor): sequential features of shape [batch_size, sum(fH*fW), feat_size];
                - {self.act_mask_key} (BoolTensor): mask indicating active features of shape [batch_size, sum(fH*fW)];
                - {self.pos_feats_key} (FloatTensor): position features of shape [batch_size, sum(fH*fW), feat_size];
                - map_shapes (LongTensor): feature map shapes in (H, W) format of shape [num_maps, 2].

            kwargs (Dict): Dictionary of keyword arguments passed to the underlying sparse module.

        Returns:
            out_feat_maps (List): List [num_maps] of output feature maps of shape [batch_size, feat_size, fH, fW].

        Raises:
            ValueError: Error when neither the 'act_map_ids' attribute nor the 'act_mask_key' attribute is provided.
        """

        # Get batch size and device
        batch_size = len(in_feat_maps[0])
        device = in_feat_maps[0].device

        # Get map shapes if needed
        if 'map_shapes' not in storage_dict:
            map_shapes = [feat_map.shape[-2:] for feat_map in in_feat_maps]
            storage_dict['map_shapes'] = torch.tensor(map_shapes, device=device)

        # Get input sequantial features
        if self.seq_feats_key is None:
            seq_feats = maps_to_seq(in_feat_maps)

        elif self.seq_feats_key not in storage_dict:
            seq_feats = maps_to_seq(in_feat_maps)
            storage_dict[self.seq_feats_key] = seq_feats

        else:
            seq_feats = storage_dict[self.seq_feats_key]

        # Get active mask
        if self.act_map_ids is not None:
            act_masks = []

            for map_id, feat_map in enumerate(in_feat_maps):
                fH, fW = feat_map.size()[-2:]

                if map_id in self.act_map_ids:
                    act_mask = torch.ones(batch_size, fH*fW, dtype=torch.bool, device=device)
                else:
                    act_mask = torch.zeros(batch_size, fH*fW, dtype=torch.bool, device=device)

                act_masks.append(act_mask)

            act_mask = torch.cat(act_masks, dim=1)

        elif self.act_mask_key is not None:
            act_mask = storage_dict[self.act_mask_key]

        else:
            error_msg = "Either the 'act_map_ids' or the 'act_mask_key' attribute must be provided (both are None)."
            raise ValueError(error_msg)

        # Get active features
        act_feats = seq_feats[act_mask]
        storage_dict['act_feats'] = act_feats

        # Get active position features if needed
        if self.pos_feats_key is not None:
            pos_feats = storage_dict[self.pos_feats_key]
            act_pos_feats = pos_feats[act_mask]
            storage_dict['act_pos_feats'] = act_pos_feats

        # Get active batch indices if needed
        if self.get_act_batch_ids:
            act_batch_ids = act_mask.nonzero(as_tuple=True)[0]
            storage_dict['act_batch_ids'] = act_batch_ids

        # Get passive features
        if self.get_pas_feats:
            pas_mask = ~act_mask
            pas_feats = seq_feats[pas_mask]
            storage_dict['pas_feats'] = pas_feats

        # Get index maps
        if self.get_id_maps:
            num_act_feats = len(act_feats)
            num_pas_feats = len(pas_feats)

            act_ids = torch.arange(num_act_feats, device=device)
            pas_ids = torch.arange(num_pas_feats, device=device)

            ids = torch.empty_like(act_mask, dtype=torch.int64)
            ids[act_mask] = act_ids
            ids[pas_mask] = pas_ids

            ids = ids.unsqueeze(dim=2)
            id_maps = seq_to_maps(ids, storage_dict['map_shapes'])
            storage_dict['id_maps'] = id_maps

        # Apply underlying sparse module
        act_feats = self.sparse(act_feats, storage_dict=storage_dict, **kwargs)

        # Get output sequential features
        seq_feats[act_mask] = act_feats

        # Add output sequential feature to storage dictionary if needed
        if self.seq_feats_key is not None:
            storage_dict[self.seq_feats_key] = seq_feats

        # Get output feature maps
        out_feat_maps = seq_to_maps(seq_feats, storage_dict['map_shapes'])

        return out_feat_maps
