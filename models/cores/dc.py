"""
Deformable core.
"""

import torch
from torch import nn

from structures.boxes import get_anchors
from models.modules.attention import DeformableAttn


class DeformableCore(nn.Module):
    """
    Class implementing the DeformableCore module.

    Attributes:
        in_projs (nn.ModuleList): List [num_maps] with modules obtaining the initial feature pyramid from input maps.
        layers (nn.ModuleList): List [num_layers] of DeformableAttn layers updating their input feature pyramid.
        scale_encs (nn.Parameter): Optional parameter containing the scale encodings of shape [num_maps, feat_size].

        map_ids (List): List [num_maps] containing the indices (i.e. downsampling exponents) of the feature maps.
        prior_type (str): String containing the type of used sample priors.
        prior_factor (float): Factor scaling the sample priors of type 'box'.

        out_ids (List): List [num_out_maps] containing the indices of the output feature maps.
        out_sizes (List): List [num_out_maps] containing the feature sizes of the output feature maps.
    """

    def __init__(self, in_ids, in_sizes, core_ids, feat_size, num_layers, da_dict, num_groups=8, prior_type='location',
                 prior_factor=2.0, scale_encs=False):
        """
        Initializes the DeformableCore module.

        Args:
            in_ids (List): List [num_in_maps] containing the indices of the input feature maps.
            in_sizes (List): List [num_in_maps] containing the feature sizes of the input feature maps.
            core_ids (List): List [num_maps] containing the indices of the core feature maps.
            feat_size (int): Integer containing the feature size of the feature maps processed by this module.
            num_layers (int): Integer containing the number of consecutive DeformableAttn layers.

            da_dict: Deformable attention (DA) network dictionary containing following keys:
                - norm (str): string containing the type of normalization of the DA network;
                - act_fn (str): string containing the type of activation function of the DA network;
                - skip (bool): boolean indicating whether layers of the DA network contain skip connections;
                - version (int): integer containing the version of the DA network;
                - num_heads (int): integer containing the number of attention heads of the DA network;
                - num_points (int): integer containing the number of points of the DA network;
                - rad_pts (int): integer containing the number of radial points of the DA network;
                - ang_pts (int): integer containing the number of angular points of the DA network;
                - lvl_pts (int): integer containing the number of level points of the DA network;
                - dup_pts (int): integer containing the number of duplicate points of the DA network;
                - qk_size (int): query and key feature size of the DA network;
                - val_size (int): value feature size of the DA network;
                - val_with_pos (bool): boolean indicating whether position info is added to DA value features;
                - norm_z (float): factor normalizing the DA sample offsets in the Z-direction.

            num_groups (int): Integer with the number of group normalization groups of bottom-up layers (default=8).
            prior_type (str): String containing the type of used sample priors (default='location').
            prior_factor (float): Factor scaling the sample priors of type 'box' (default=2.0).
            scale_encs (bool): Boolean indicating whether to use scale encodings (default=False).

        Raises:
            ValueError: Error when the 'in_ids' length and the 'in_sizes' length do not match.
            ValueError: Error when the 'in_ids' list does not match the first elements of the 'core_ids' list.
        """

        # Check inputs
        if len(in_ids) != len(in_sizes):
            error_msg = f"The 'in_ids' length ({len(in_ids)}) must match the 'in_sizes' length ({len(in_sizes)})."
            raise ValueError(error_msg)

        if in_ids != core_ids[:len(in_ids)]:
            error_msg = f"The 'in_ids' list ({in_ids}) must match the first elements of 'core_ids' list ({core_ids})."
            raise ValueError(error_msg)

        # Initialization of default nn.Module
        super().__init__()

        # Initialization of input projection layers
        conv_kwargs = {'kernel_size': 1, 'stride': 1, 'padding': 0}
        self.in_projs = nn.ModuleList([nn.Conv2d(in_size, feat_size, **conv_kwargs) for in_size in in_sizes])

        conv_kwargs = {'kernel_size': 3, 'stride': 2, 'padding': 1}
        num_bottom_up_layers = len(core_ids) - len(in_ids)

        for i in range(num_bottom_up_layers):
            if i == 0:
                in_proj = nn.Conv2d(in_sizes[-1], feat_size, **conv_kwargs)

            else:
                in_proj = nn.Sequential()
                in_proj.add_module('norm', nn.GroupNorm(num_groups, feat_size))
                in_proj.add_module('act', nn.ReLU(inplace=True))
                in_proj.add_module('conv', nn.Conv2d(feat_size, feat_size, **conv_kwargs))

            self.in_projs.append(in_proj)

        # Initialization of DeformableAttn layers
        num_maps = len(self.in_projs)
        da_dict['num_levels'] = num_maps
        self.layers = nn.ModuleList([DeformableAttn(feat_size, feat_size, **da_dict) for _ in range(num_layers)])

        # Initialization of scale encodings if needed
        if scale_encs:
            self.scale_encs = nn.Parameter(torch.zeros(num_maps, feat_size))

        # Set attributes related to sample priors
        self.map_ids = core_ids
        self.prior_type = prior_type
        self.prior_factor = prior_factor

        # Set attributes related to output feature maps
        self.out_ids = core_ids
        self.out_sizes = [feat_size] * num_maps

    def forward(self, in_feat_maps, images=None, **kwargs):
        """
        Forward method of the DeformableCore module.

        Args:
            in_feat_maps (List): Input feature maps [num_in_maps] of shape [batch_size, feat_size, fH, fW].
            images (Images): Images structure containing the batched images (default=None).
            kwargs (Dict): Dictionary of keyword arguments not used by this module.

        Returns:
            out_feat_maps (List): Output feature maps [num_out_maps] of shape [batch_size, feat_size, fH, fW].

        Raises:
            ValueError: Error when no Images structure is provided.
            ValueError: Error when prior type is provided.
        """

        # Get initial feature pyramid
        feat_maps = [self.in_projs[i](in_feat_map) for i, in_feat_map in enumerate(in_feat_maps)]
        bu_feat_map = in_feat_maps[-1]

        for in_proj in self.in_projs[len(in_feat_maps):]:
            bu_feat_map = in_proj(bu_feat_map)
            feat_maps.append(bu_feat_map)

        # Check whether Images structure is provided
        if images is None:
            error_msg = "An Images structure containing the batched images must be provided."
            raise ValueError(error_msg)

        # Prepare for deformable attention layers
        feats = torch.cat([feat_map.flatten(2).permute(0, 2, 1) for feat_map in feat_maps], dim=1)

        if self.prior_type == 'box':
            sample_priors = get_anchors(feat_maps, self.map_ids, scale_factor=self.prior_factor)
            sample_priors = sample_priors.normalize(images[0]).to_format('cxcywh').boxes

        elif self.prior_type == 'location':
            sample_priors = get_anchors(feat_maps, self.map_ids)
            sample_priors = sample_priors.normalize(images[0]).to_format('cxcywh').boxes[:, :2]

        else:
            error_msg = f"Invalid prior type '{self.prior_type}'."
            raise ValueError(error_msg)

        batch_size = len(in_feat_maps[0])
        sample_priors = sample_priors[None, :, :].expand(batch_size, -1, -1)

        sample_map_shapes = torch.tensor([feat_map.shape[-2:] for feat_map in feat_maps], device=feats.device)
        feats_per_map = sample_map_shapes.prod(dim=1)

        sample_map_start_ids = feats_per_map.cumsum(dim=0)[:-1]
        sample_map_start_ids = torch.cat([sample_map_shapes.new_zeros((1,)), sample_map_start_ids], dim=0)

        num_maps = len(feat_maps)
        map_ids = torch.arange(num_maps, device=feats.device)
        map_ids = map_ids.repeat_interleave(feats_per_map, dim=0)
        map_ids = map_ids[None, :].expand(batch_size, -1)

        da_kwargs = {'sample_priors': sample_priors, 'sample_map_shapes': sample_map_shapes}
        da_kwargs = {**da_kwargs, 'sample_map_start_ids': sample_map_start_ids, 'map_ids': map_ids}

        if hasattr(self, 'scale_encs'):
            da_kwargs['add_encs'] = self.scale_encs.repeat_interleave(feats_per_map, dim=0)

        # Apply deformable attention layers
        for layer in self.layers:
            da_kwargs['sample_feats'] = feats
            feats = layer(feats, **da_kwargs)

        # Get output feature pyramid
        out_feat_maps = feats.split(feats_per_map.tolist(), dim=1)
        out_feat_maps = [out_feat_maps[i].transpose(1, 2).view_as(feat_maps[i]) for i in range(num_maps)]

        return out_feat_maps
