"""
Collection of sparsity-based modules.
"""

import torch
from torch import nn

from models.build import build_model, MODELS


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
