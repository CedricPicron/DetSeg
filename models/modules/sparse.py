"""
Collection of sparsity-based modules.
"""

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from models.build import build_model, MODELS
from models.functional.sparse import id_scale_attn
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
class IdScaleAttn(nn.Module):
    """
    Class implementing the IdScaleAttn module.

    Attributes:
        scale_embed (Parameter): Parameter containing the scale embeddings of shape [num_maps, feat_size].
        attn_weights (nn.Linear): Module computing the unnormalized attention weights.
        val_proj (nn.Linear): Module computing the value features.
        out_proj (nn.Linear): Module computing the output features.
        num_heads (int): Integer containing the number of attention heads.
        num_maps (int): Integer containing the number of feature maps.
    """

    def __init__(self, feat_size, num_maps, num_heads=8):
        """
        Initializes the IdScaleAttn module.

        Args:
            feat_size (int): Integer containing the feature size.
            num_maps (int): Integer containing the number of feature maps.
            num_heads (int): Integer containing the number of attention heads (default=8).

        Raises:
            ValueError: Error when the feature size does not divide the number of attention heads.
        """

        # Check divisibility feature size by number of heads
        if feat_size % num_heads != 0:
            error_msg = f"The feature size ({feat_size}) must divide the number of attention heads ({num_heads})."
            raise ValueError(error_msg)

        # Initialization of default nn.Module
        super().__init__()

        # Initialize scale embedding
        self.scale_embed = Parameter(torch.empty(num_maps, feat_size), requires_grad=True)
        nn.init.zeros_(self.scale_embed)

        # Initialize module computing the unnormalized attention weights
        self.attn_weights = nn.Linear(feat_size, num_heads * num_maps)
        nn.init.zeros_(self.attn_weights.weight)
        nn.init.zeros_(self.attn_weights.bias)

        # Initialize module computing the value features
        self.val_proj = nn.Linear(feat_size, feat_size)
        nn.init.xavier_uniform_(self.val_proj.weight)
        nn.init.zeros_(self.val_proj.bias)

        # Initialize module computing the output features
        self.out_proj = nn.Linear(feat_size, feat_size)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

        # Set remaining attributes
        self.num_heads = num_heads
        self.num_maps = num_maps

    def forward(self, in_act_feats, storage_dict, **kwargs):
        """
        Forward method of the IdScaleAttn module.

        Args:
            in_act_feats (FloatTensor): Input active features of shape [num_act_feats, feat_size].

            storage_dict (Dict): Storage dictionary containing at least following keys:
                - act_batch_ids (LongTensor): batch indices of active features of shape [num_act_feats];
                - act_map_ids (LongTensor): map indices of active features of shape [num_act_feats];
                - act_xy_ids (LongTensor): (X, Y) location indices of active features of shape [num_act_feats, 2];
                - map_shapes (LongTensor): feature map shapes in (H, W) format of shape [num_maps, 2];
                - pas_feats (FloatTensor): passive features of shape [num_pas_feats, feat_size];
                - id_maps (List): list [num_maps] with feature indices of shape [batch_size, 1, fH, fW].

            kwargs (Dict): Dictionary of unused keyword arguments.

        Returns:
            out_act_feats (FloatTensor): Output active features of shape [num_act_feats, feat_size].
        """

        # Retrieve desired items from storage dictionary
        act_batch_ids = storage_dict['act_batch_ids']
        act_map_ids = storage_dict['act_map_ids']
        act_xy_ids = storage_dict['act_xy_ids']
        map_shapes = storage_dict['map_shapes']
        pas_feats = storage_dict['pas_feats']
        id_maps = storage_dict['id_maps']

        # Get device and number of active features
        device = in_act_feats.device
        num_act_feats = len(in_act_feats)

        # Add scale embedding to input active features
        act_feats = in_act_feats + self.scale_embed[act_map_ids]

        # Get normalized attention weights
        attn_weights = self.attn_weights(act_feats).view(num_act_feats, self.num_heads, self.num_maps)
        attn_weights = F.softmax(attn_weights, dim=2)

        # Get sample locations
        map_shapes = map_shapes.fliplr()
        sample_xy = (act_xy_ids + 0.5) / map_shapes[act_map_ids]

        sample_xy = sample_xy[:, None, :] * map_shapes[None, :, :]
        sample_xy = sample_xy[:, :, None, :] - 0.5

        # Get sample indices
        sample_offs = torch.tensor([[0, 0], [1, 0], [0, 1], [1, 1]], device=device)
        sample_ids = sample_xy.floor().long() + sample_offs

        # Get sample weights
        sample_weights = 1 - (sample_xy - sample_ids).abs()
        sample_weights = sample_weights.prod(dim=3)

        # Get feature indices
        act_batch_ids = act_batch_ids[:, None].expand(-1, 4)
        feat_ids = sample_ids.new_empty([num_act_feats, self.num_maps, 4])

        for map_id, id_map in enumerate(id_maps):
            fH, fW = id_map.size()[-2:]

            sample_ids_x = sample_ids[:, map_id, :, 0].clamp_(min=0, max=fW-1)
            sample_ids_y = sample_ids[:, map_id, :, 1].clamp_(min=0, max=fH-1)

            feat_ids[:, map_id, :] = id_map[act_batch_ids, 0, sample_ids_y, sample_ids_x]

        # Get feature weights
        feat_weights = attn_weights[:, :, :, None] * sample_weights[:, None, :, :]

        # Get weighted value features
        weight = self.val_proj.weight
        bias = self.val_proj.bias
        val_feats = id_scale_attn(in_act_feats, pas_feats, feat_ids, feat_weights, weight, bias)

        # Get output active features
        out_act_feats = self.out_proj(val_feats)

        return out_act_feats


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
        get_act_map_ids (bool): Boolean indicating whether to get active map indices.
        get_act_xy_ids (bool): Boolean indicating whether to get active location indices.
        get_pas_feats (bool): Boolean indicating whether to get passive features.
        get_id_maps (bool): Boolean indicating whether to get list of index maps.
        sparse (nn.Module): Underlying sparse module updating the active features.
    """

    def __init__(self, sparse_cfg, seq_feats_key=None, act_map_ids=None, act_mask_key=None, pos_feats_key=None,
                 get_act_batch_ids=False, get_act_map_ids=False, get_act_xy_ids=False, get_pas_feats=True,
                 get_id_maps=True):
        """
        Initializes the Sparse3d module.

        Args:
            sparse_cfg (Dict): Configuration dictionary specifying the underlying sparse module.
            seq_feats_key (str): String used to retrieve sequential features from storage dictionary (default=None).
            act_map_ids (List): List [num_act_maps] with map indices to get active features from (default=None).
            act_mask_key (str): String used to retrieve active mask from storage dictionary (default=None).
            pos_feats_key (str): String used to retrieve position features from storage dictionary (default=None).
            get_act_batch_ids (bool): Boolean indicating whether to get active batch indices (default=False).
            get_act_map_ids (bool): Boolean indicating whether to get active map indices (default=False).
            get_act_xy_ids (bool): Boolean indicating whether to get active location indices (default=False).
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
        self.get_act_map_ids = get_act_map_ids
        self.get_act_xy_ids = get_act_xy_ids
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
                - map_shapes (LongTensor): feature map shapes in (H, W) format of shape [num_maps, 2];
                - map_offs (LongTensor): cumulative number of features per feature map of shape [num_maps+1].

            kwargs (Dict): Dictionary of keyword arguments passed to the underlying sparse module.

        Returns:
            out_feat_maps (List): List [num_maps] of output feature maps of shape [batch_size, feat_size, fH, fW].

        Raises:
            ValueError: Error when neither the 'act_map_ids' attribute nor the 'act_mask_key' attribute is provided.
        """

        # Get batch size and device
        batch_size = len(in_feat_maps[0])
        device = in_feat_maps[0].device

        # Add map shapes to storage dictionary if needed
        if 'map_shapes' not in storage_dict:
            map_shapes = [feat_map.shape[-2:] for feat_map in in_feat_maps]
            storage_dict['map_shapes'] = torch.tensor(map_shapes, device=device)

        # Add map offsets to storage dictionary if needed
        if 'map_offs' not in storage_dict:
            map_offs = storage_dict['map_shapes'].prod(dim=1).cumsum(dim=0)
            storage_dict['map_offs'] = torch.cat([map_offs.new_zeros([1]), map_offs], dim=0)

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

        # Get active map indices if needed
        if self.get_act_map_ids:
            act_feat_ids = act_mask.nonzero(as_tuple=True)[1]
            act_map_ids = act_feat_ids[:, None] - storage_dict['map_offs'][None, 1:-1]

            act_map_ids = (act_map_ids >= 0).sum(dim=1)
            storage_dict['act_map_ids'] = act_map_ids

        # Get active location indices if needed
        if self.get_act_xy_ids:
            act_masks = seq_to_maps(act_mask.unsqueeze(dim=2), storage_dict['map_shapes'])
            act_xy_ids = [act_mask.nonzero()[:, 2:].fliplr() for act_mask in act_masks]

            act_xy_ids = torch.cat(act_xy_ids, dim=0)
            storage_dict['act_xy_ids'] = act_xy_ids

        # Get passive features
        if self.get_pas_feats:
            pas_mask = ~act_mask
            pas_feats = seq_feats[pas_mask]
            storage_dict['pas_feats'] = pas_feats

        # Get index maps
        if self.get_id_maps:
            num_act_feats = len(act_feats)
            num_feats = num_act_feats + len(pas_feats)

            act_ids = torch.arange(num_act_feats, device=device)
            pas_ids = torch.arange(num_act_feats, num_feats, device=device)

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
