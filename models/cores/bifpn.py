"""
BiFPN (Bidirectional FPN) core.
"""

import torch
from torch import nn
import torch.nn.functional as F


class BiFPN(nn.Module):
    """
    Class implementing the BiFPN (Bidirectional FPN) module.

    Attributes:
        in_projs (nn.ModuleList): List [num_maps] with modules obtaining the initial feature pyramid from input maps.
        layers (nn.Sequential): Sequence of BiFPN layers updating their input feature pyramid.

        out_ids (List): List [num_out_maps] containing the indices of the output feature maps.
        out_sizes (List): List [num_out_maps] containing the feature sizes of the output feature maps.
    """

    def __init__(self, in_ids, in_sizes, core_ids, feat_size, num_layers, norm_type='batch', separable_conv=False):
        """
        Initializes the BiFPN module.

        Args:
            in_ids (List): List [num_in_maps] containing the indices of the input feature maps.
            in_sizes (List): List [num_in_maps] containing the feature sizes of the input feature maps.
            core_ids (List): List [num_maps] containing the indices of the core feature maps.
            feat_size (int): Integer containing the feature size of the feature maps processed by this module.
            num_layers (int): Integer containing the number of consecutive BiFPN layers.
            norm_type (str): String containing the type of BiFPN normalization layer (default='batch').
            separable_conv (bool): Boolean indicating whether separable convolutions should be used (default=False).

        Raises:
            ValueError: Error when the 'in_ids' length and the 'in_sizes' length do not match.
            ValueError: Error when the 'in_ids' list does not match the first elements of the 'core_ids' list.
            ValueError: Error when invalid normalization layer is provided.
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

        # Get normalization module and corresponding keyword arguments for initialization
        if norm_type == 'batch':
            norm_layer = nn.BatchNorm2d
            norm_kwargs = {'num_features': feat_size, 'eps': 1e-3, 'momentum': 1e-2}

        elif norm_type == 'group':
            norm_layer = nn.GroupNorm
            norm_kwargs = {'num_groups': 8, 'num_channels': feat_size}

        else:
            error_msg = f"Invalid type of normalization layer (got '{norm_type}')."
            raise ValueError(error_msg)

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
                in_proj.add_module('norm', norm_layer(**norm_kwargs))
                in_proj.add_module('act', nn.ReLU(inplace=True))
                in_proj.add_module('conv', nn.Conv2d(feat_size, feat_size, **conv_kwargs))

            self.in_projs.append(in_proj)

        # Initialization of BiFPN layers
        num_maps = len(self.in_projs)
        bifpn_layer_args = (feat_size, num_maps, norm_layer, norm_kwargs, separable_conv)
        self.layers = nn.Sequential(*[BiFPNLayer(*bifpn_layer_args) for _ in range(num_layers)])

        # Set attributes related to output feature maps
        self.out_ids = core_ids
        self.out_sizes = [feat_size] * num_maps

    def forward(self, in_feat_maps, **kwargs):
        """
        Forward method of the BiFPN module.

        Args:
            in_feat_maps (List): Input feature maps [num_in_maps] of shape [batch_size, feat_size, fH, fW].
            kwargs (Dict): Dictionary of keyword arguments not used by this module.

        Returns:
            out_feat_maps (List): Output feature maps [num_out_maps] of shape [batch_size, feat_size, fH, fW].
        """

        # Get initial feature pyramid
        init_feat_maps = [self.in_projs[i](feat_map) for i, feat_map in enumerate(in_feat_maps)]
        bu_feat_map = in_feat_maps[-1]

        for in_proj in self.in_projs[len(in_feat_maps):]:
            bu_feat_map = in_proj(bu_feat_map)
            init_feat_maps.append(bu_feat_map)

        # Get output feature pyramid
        out_feat_maps = self.layers(init_feat_maps)

        return out_feat_maps


class BiFPNLayer(nn.Module):
    """
    Class implementing the BiFPNLayer module.

    Attributes:
        td_layers (nn.ModuleList): List [num_maps-1] of top-down projection layers.
        bu_layers (nn.ModuleList): List [num_maps-1] of bottom-up projection layers.
        td_weights (nn.Parameter): Factors weighting feature maps during top-down processing of shape [num_maps-1, 2].
        bu_weights (nn.Parameter): Factors weighting feature maps during bottom-up processing of shape [num_maps-1, 3].
        eps (float): Value added to denominator of weight normalization for numerical stability.
    """

    def __init__(self, feat_size, num_maps, norm_layer=nn.BatchNorm2d, norm_kwargs={}, separable_conv=False, eps=1e-4):
        """
        Initializes the BiFPNLayer module.

        Args:
            feat_size (int): Integer containing the feature size of the feature maps processed by this module.
            num_maps (int): Number of feature maps to process.
            norm_layer (nn.Module): Module implementing the normalization layer (default=nn.BatchNorm2d).
            norm_kwargs (dict): Dictionary with keyword arguments to initialize the normalization layer (default={}).
            separable_conv (bool): Boolean indicating whether separable convolutions should be used (default=False).
            eps (float): Value added to denominator of weight normalization for numerical stability (default=1e-4).
        """

        # Initialization of default nn.Module
        super().__init__()

        # Initialization of top-down projection layers
        self.td_layers = nn.ModuleList()

        for _ in range(num_maps-1):
            td_layer = nn.Sequential()
            td_layer.add_module('act', nn.ReLU(inplace=True))

            if separable_conv:
                conv_kwargs = {'kernel_size': 3, 'padding': 1, 'groups': feat_size, 'bias': True}
                td_layer.add_module('depth_conv', nn.Conv2d(feat_size, feat_size, **conv_kwargs))

                conv_kwargs = {'kernel_size': 1, 'padding': 0, 'groups': 1, 'bias': False}
                td_layer.add_module('point_conv', nn.Conv2d(feat_size, feat_size, **conv_kwargs))

            else:
                conv_kwargs = {'kernel_size': 3, 'padding': 1, 'groups': 1, 'bias': False}
                td_layer.add_module('conv', nn.Conv2d(feat_size, feat_size, **conv_kwargs))

            td_layer.add_module('norm', norm_layer(**norm_kwargs))
            self.td_layers.append(td_layer)

        # Initialization of bottom-up projection layers
        self.bu_layers = nn.ModuleList()

        for _ in range(num_maps-1):
            bu_layer = nn.Sequential()
            bu_layer.add_module('act', nn.ReLU(inplace=True))

            if separable_conv:
                conv_kwargs = {'kernel_size': 3, 'padding': 1, 'groups': feat_size, 'bias': True}
                bu_layer.add_module('depth_conv', nn.Conv2d(feat_size, feat_size, **conv_kwargs))

                conv_kwargs = {'kernel_size': 1, 'padding': 0, 'groups': 1, 'bias': False}
                bu_layer.add_module('point_conv', nn.Conv2d(feat_size, feat_size, **conv_kwargs))

            else:
                conv_kwargs = {'kernel_size': 3, 'padding': 1, 'groups': 1, 'bias': False}
                bu_layer.add_module('conv', nn.Conv2d(feat_size, feat_size, **conv_kwargs))

            bu_layer.add_module('norm', norm_layer(**norm_kwargs))
            self.bu_layers.append(bu_layer)

        # Initialization of weight parameters
        self.td_weights = nn.Parameter(torch.ones(num_maps-1, 2))
        self.bu_weights = nn.Parameter(torch.ones(num_maps-1, 3))

        # Set epsilon attribute
        self.eps = eps

    def forward(self, in_maps, **kwargs):
        """
        Forward method of the BiFPNLayer module.

        Args:
            in_maps (List): Input feature maps [num_maps] of shape [batch_size, feat_size, fH, fW].
            kwargs (Dict): Dictionary of keyword arguments not used by this module.

        Returns:
            bu_maps (List): Output feature maps [num_maps] of shape [batch_size, feat_size, fH, fW].
        """

        # Get number of maps
        num_maps = len(in_maps)

        # Get normalized top-down and bottom-up weights
        td_weights = F.relu(self.td_weights)
        td_weights = td_weights / (td_weights.sum(dim=1, keepdim=True) + self.eps)

        bu_weights = F.relu(self.bu_weights)
        bu_weights = bu_weights / (bu_weights.sum(dim=1, keepdim=True) + self.eps)

        # Perform top-down processing
        td_maps = [in_maps[-1]]

        for i in range(num_maps-2, -1, -1):
            up_map = F.interpolate(td_maps[0], size=in_maps[i].shape[-2:], mode='nearest')
            td_map = td_weights[i, 0] * in_maps[i] + td_weights[i, 1] * up_map

            td_map = self.td_layers[i](td_map)
            td_maps.insert(0, td_map)

        # Perform bottom-up processing
        bu_maps = [td_maps[0]]

        for i in range(0, num_maps-1):
            down_map = F.interpolate(bu_maps[-1], size=td_maps[i+1].shape[-2:], mode='nearest')
            bu_map = bu_weights[i, 0] * in_maps[i+1] + bu_weights[i, 1] * td_maps[i+1] + bu_weights[i, 2] * down_map

            bu_map = self.bu_layers[i](bu_map)
            bu_maps.append(bu_map)

        return bu_maps
