"""
Collection of fusion modules.
"""

import torch
from torch import nn

from models.build import build_model, MODELS


@MODELS.register_module()
class AttnFusion(nn.Module):
    """
    Class implementing the AttnFusion module.

    Attributes:
        attn (nn.MultiheadAttention): Module performing the multi-head attention operation.
    """

    def __init__(self, **kwargs):
        """
        Initializes the AttnFusion module.

        Args:
            kwargs (Dict): Dictionary passed to the __init__ method of the underlying attention module.
        """

        # Initialization of default nn.Module
        super().__init__()

        # Get attention module
        kwargs.pop('batch_first', None)
        self.attn = nn.MultiheadAttention(batch_first=True, **kwargs)

    def forward(self, in_feats, feats_list, **kwargs):
        """
        Forward method of the AttnFusion module.

        Args:
            in_feats (FloatTensor): Input features of shape [num_groups, feat_size].
            feats_list (List): List of features to be fused of shape [num_groups, *, feat_size].
            kwargs (Dict): Dictionary of unused keyword arguments.

        Returns:
            out_feats (FloatTensor): Output features of shape [num_groups, feat_size].
        """

        # Get query features
        qry_feats = in_feats[:, None, :]

        # Get key-value features
        kv_feats_list = []

        for feats_i in feats_list:
            if feats_i.dim() == 2:
                kv_feats_list.append(feats_i[:, None, :])
            else:
                kv_feats_list.append(feats_i)

        kv_feats = torch.cat(kv_feats_list, dim=1)

        # Get output features
        out_feats = self.attn(qry_feats, kv_feats, kv_feats)[0]
        out_feats = out_feats.squeeze(dim=1)

        return out_feats


@MODELS.register_module()
class FusionInit(nn.Module):
    """
    Class implementing the FusionInit module.

    Attributes:
        init_type (str): String containing the fusion initialization type.
        init_id (int): Integer with the initialization index for the 'index' initialization type.
    """

    def __init__(self, init_type, init_id=None):
        """
        Initializes the FusionInit module.

        Args:
            init_type (str): String containing the fusion initialization type.
            init_id (int): Integer with the initialization index for the 'index' initialization type (default=None).
        """

        # Initialization of default nn.Module
        super().__init__()

        # Set attributes
        self.init_type = init_type
        self.init_id = init_id

    def forward(self, in_feats_list, **kwargs):
        """
        Forward method of the FusionInit module.

        Args:
            in_feats_list (List): List of input features to be fused of shape [num_groups, *, feat_size].
            kwargs (Dict): Dictionary of unused keyword arguments.

        Returns:
            init_feats (FloatTensor): Tensor containing the initial fusion features of shape [num_groups, feat_size].

        Raises:
            ValueError: Error when an invalid initialization type is provided.
        """

        # Get initial fusion features
        if self.init_type == 'index':
            init_feats = in_feats_list[self.init_id]

            if init_feats.dim() == 3:
                init_feats = init_feats.mean(dim=1)

        elif self.init_type == 'mean':
            init_feats_list = []

            for feats_i in in_feats_list:
                if feats_i.dim() == 2:
                    init_feats_list.append(feats_i[:, None, :])
                else:
                    init_feats_list.append(feats_i)

            init_feats = torch.cat(init_feats_list, dim=1)
            init_feats = init_feats.mean(dim=1)

        elif self.init_type == 'zero':
            num_groups = in_feats_list[0].size(dim=0)
            feat_size = in_feats_list[0].size(dim=-1)
            device = in_feats_list[0].device
            init_feats = torch.zeros(num_groups, feat_size, device=device)

        else:
            error_msg = f"Invalid initialization type in FusionInit module (got '{self.init_type}')."
            raise ValueError(error_msg)

        return init_feats


@MODELS.register_module()
class PosMapFusion(nn.Module):
    """
    Class implementing the PosMapFusion module.

    Attributes:
        in_key (str): String with key to retrieve input feature map from storage dictionary.
        pos (nn.Module): Module containing the position encoder.
        out_key (str): String with key to store output feature map in storage dictionary.
    """

    def __init__(self, in_key, pos_cfg, out_key):
        """
        Initializes the PosMapFusion module.

        Args:
            in_key (str): String with key to retrieve input feature map from storage dictionary.
            pos_cfg (Dict): Configuration dictionary specifying the position encoder module.
            out_key (str): String with key to store output feature map in storage dictionary.
        """

        # Initialization of default nn.Module
        super().__init__()

        # Build position encoder module
        self.pos = build_model(pos_cfg)

        # Set additional attributes
        self.in_key = in_key
        self.out_key = out_key

    def forward(self, storage_dict, **kwargs):
        """
        Forward method of the PosMapFusion module.

        Args:
            storage_dict (Dict): Storage dictionary containing at least following key:
                - {in_key} (FloatTensor): input feature map of shape [batch_size, feat_size, fH, fW].

            kwargs (Dict): Dictionary of unused keyword arguments.

        Returns:
            storage_dict (Dict): Storage dictionary containing following additional key:
                - {out_key} (FloatTensor): output feature map of shape [batch_size, feat_size, fH, fW].
        """

        # Retrieve input feature map from storage dictionary
        in_feat_map = storage_dict[self.in_key]

        # Get position features
        feat_size, fH, fW = in_feat_map.size()[1:]
        device = in_feat_map.device

        pts_x = torch.linspace(0.5/fW, 1-0.5/fW, steps=fW, device=device)
        pts_y = torch.linspace(0.5/fH, 1-0.5/fH, steps=fH, device=device)

        norm_xy = torch.meshgrid(pts_x, pts_y, indexing='xy')
        norm_xy = torch.stack(norm_xy, dim=2).flatten(0, 1)
        pos_feats = self.pos(norm_xy)

        # Fuse position features
        out_feat_map = in_feat_map + pos_feats.t().view(1, feat_size, fH, fW)

        # Store output feature map in storage dictionary
        storage_dict[self.out_key] = out_feat_map

        return storage_dict


@MODELS.register_module()
class QryMapFusion(nn.Module):
    """
    Class implementing the QryMapFusion module.

    Attributes:
        in_key (str): String with key to retrieve input feature map from storage dictionary.
        qry_key (str): String with key to retrieve query features from storage dictionary.
        pre_cat (nn.Module): Module updating the query features before concatenation (or None).
        post_cat (nn.Module): Module updating the concatenated features.
        out_key (str): String with key to store output feature map in storage dictionary.
    """

    def __init__(self, in_key, qry_key, post_cat_cfg, out_key, pre_cat_cfg=None):
        """
        Initializes the QryMapFusion module.

        Args:
            in_key (str): String with key to retrieve input feature map from storage dictionary.
            qry_key (str): String with key to retrieve query features from storage dictionary.
            post_cat_cfg (Dict): Configuration dictionary specifying the post-concatenation module.
            out_key (str): String with key to store output feature map in storage dictionary.
            pre_cat_cfg (Dict): Configuration dictionary specifying the pre-concatenation module (default=None).
        """

        # Initialization of default nn.Module
        super().__init__()

        # Build underlying modules
        self.pre_cat = build_model(pre_cat_cfg) if pre_cat_cfg is not None else None
        self.post_cat = build_model(post_cat_cfg)

        # Set additional attributes
        self.in_key = in_key
        self.qry_key = qry_key
        self.out_key = out_key

    def forward(self, storage_dict, **kwargs):
        """
        Forward method of the QryMapFusion module.

        Args:
            storage_dict (Dict): Storage dictionary containing at least following keys:
                - {in_key} (FloatTensor): input feature map of shape [num_qrys, feat_size, fH, fW];
                - {qry_key} (FloatTensor): query features of shape [num_qrys, feat_size].

            kwargs (Dict): Dictionary of unused keyword arguments.

        Returns:
            storage_dict (Dict): Storage dictionary containing following additional key:
                - {out_key} (FloatTensor): output feature map of shape [num_qrys, feat_size, fH, fW].
        """

        # Retrieve desired items from storage dictionary
        in_feat_map = storage_dict[self.in_key]
        qry_feats = storage_dict[self.qry_key]

        # Fuse query features
        if self.pre_cat is not None:
            qry_feats = self.pre_cat(qry_feats)

        qry_feats = qry_feats[:, :, None, None].expand_as(in_feat_map)
        cat_feats = torch.cat([in_feat_map, qry_feats], dim=1)
        out_feat_map = in_feat_map + self.post_cat(cat_feats)

        # Store output feature map in storage dictionary
        storage_dict[self.out_key] = out_feat_map

        return storage_dict
