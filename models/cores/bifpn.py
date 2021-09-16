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
        feat_sizes (List): List [num_out_maps] containing the feature sizes of the output feature maps.
    """

    def __init__(self, in_feat_sizes, in_bot_layers, feat_size, num_layers, separable_conv=False):
        """
        Initializes the BiFPN module.

        Args:
            in_feat_sizes (List): List [num_in_maps] containing the feature sizes of the input feature maps.
            in_bot_layers (int): Integer containing the number of bottom-up layers applied on last input feature map.
            feat_size (int): Integer containing the feature size of the feature maps processed by this module.
            num_layers (int): Integer containing the number of consecutive BiFPN layers.
            separable_conv (bool): Boolean indicating whether separable convolutions should be used (default=False).
        """

        # Initialization of default nn.Module
        super().__init__()

        # Initialization of input projection layers
        conv_kwargs = {'kernel_size': 1, 'stride': 1, 'padding': 0}
        self.in_projs = nn.ModuleList([nn.Conv2d(in_size, feat_size, **conv_kwargs) for in_size in in_feat_sizes])

        if in_bot_layers > 0:
            conv_kwargs = {'kernel_size': 3, 'stride': 2, 'padding': 1}

            for i in range(in_bot_layers):
                if i == 0:
                    in_proj = nn.Conv2d(in_feat_sizes[-1], feat_size, **conv_kwargs)
                    self.in_projs.append(in_proj)
                    continue

                in_proj = nn.Sequential()
                in_proj.add_module('norm', nn.BatchNorm2d(feat_size, momentum=1e-2, eps=1e-3))
                in_proj.add_module('act', nn.ReLU(inplace=True))
                in_proj.add_module('conv', nn.Conv2d(feat_size, feat_size, **conv_kwargs))
                self.in_projs.append(in_proj)

        # Initialization of BiFPN layers
        num_maps = len(self.in_projs)
        self.layers = nn.Sequential(*[BiFPNLayer(feat_size, num_maps, separable_conv) for _ in range(num_layers)])

        # Set feature sizes attribute
        self.feat_sizes = [feat_size for _ in range(num_maps)]

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
        epsilon (float): Value added to denominator of weight normalization for numerical stability
    """

    def __init__(self, feat_size, num_maps, separable_conv=False, epsilon=1e-4):
        """
        Initializes the BiFPNLayer module.

        Args:
            feat_size (int): Integer containing the feature size of the feature maps processed by this module.
            num_maps (int): Number of feature maps to process.
            separable_conv (bool): Boolean indicating whether separable convolutions should be used (default=False).
            epsilon (float): Value added to denominator of weight normalization for numerical stability (default=1e-4).
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

            td_layer.add_module('norm', nn.BatchNorm2d(feat_size, momentum=1e-2, eps=1e-3))
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

            bu_layer.add_module('norm', nn.BatchNorm2d(feat_size, momentum=1e-2, eps=1e-3))
            self.bu_layers.append(bu_layer)

        # Initialization of weight parameters
        self.td_weights = nn.Parameter(torch.ones(num_maps-1, 2))
        self.bu_weights = nn.Parameter(torch.ones(num_maps-1, 3))

        # Set epsilon attribute
        self.epsilon = epsilon

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
        td_weights /= td_weights.sum(dim=1, keepdim=True) + self.epsilon

        bu_weights = F.relu(self.bu_weights)
        bu_weights /= bu_weights.sum(dim=1, keepdim=True) + self.epsilon

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
