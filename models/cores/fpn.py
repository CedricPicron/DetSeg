"""
FPN (Feature Pyramid Network) core.
"""

from torch import nn
import torch.nn.functional as F

from models.modules.projector import Projector


class FPN(nn.Module):
    """
    Class implementing the FPN (Feature Pyramid Network) module.

    Attributes:
        lat_proj (Projector): Module computing lateral feature maps from input feature maps.
        out_proj (Projector): Module computing output feature maps from fused feature maps.
        bot_layers (ModuleList): Optional list [num_bottom_up_layers] with modules computing additional output maps.

        fuse_type (str): String containing the FPN fuse operation chosen from {'avg', 'sum'}.

        out_ids (List): List [num_out_maps] containing the indices of the output feature maps.
        out_sizes (List): List [num_out_maps] containing the feature sizes of the output feature maps.
    """

    def __init__(self, in_ids, in_sizes, core_ids, feat_size, fuse_type='sum'):
        """
        Initializes the FPN module.

        Args:
            in_ids (List): List [num_in_maps] containing the indices of the input feature maps.
            in_sizes (List): List [num_in_maps] containing the feature sizes of the input feature maps.
            core_ids (List): List [num_maps] containing the indices of the core feature maps.
            feat_size (int): Integer containing the feature size of the feature maps processed by this module.
            fuse_type (str): String containing the FPN fuse operation (default='sum').

        Raises:
            ValueError: Error when the 'in_ids' length and the 'in_sizes' length do not match.
            ValueError: Error when the 'in_ids' list does not match the first elements of the 'core_ids' list.
            ValueError: Error when the fuse operation is not chosen from {'avg', 'sum'}.
        """

        # Check inputs
        if len(in_ids) != len(in_sizes):
            error_msg = f"The 'in_ids' length ({len(in_ids)}) must match the 'in_sizes' length ({len(in_sizes)})."
            raise ValueError(error_msg)

        if in_ids != core_ids[:len(in_ids)]:
            error_msg = f"The 'in_ids' list ({in_ids}) must match the first elements of 'core_ids' list ({core_ids})."
            raise ValueError(error_msg)

        if fuse_type not in ('avg', 'sum'):
            error_msg = f"The fuse operation must be chosen from {{'avg', 'sum'}}, but got '{fuse_type}'."
            raise ValueError(error_msg)

        # Initialization of default nn.Module
        super().__init__()

        # Initialize lateral projector
        fixed_settings = {'out_feat_size': feat_size, 'proj_type': 'conv1', 'conv_stride': 1}
        proj_dicts = [{'in_map_id': i, **fixed_settings} for i in range(len(in_ids))]
        self.lat_proj = Projector(in_sizes, proj_dicts)

        # Initialize output projector
        fixed_settings = {'out_feat_size': feat_size, 'proj_type': 'conv3', 'conv_stride': 1}
        proj_dicts = [{'in_map_id': i, **fixed_settings} for i in range(len(in_ids))]
        self.out_proj = Projector([feat_size] * len(in_ids), proj_dicts)

        # Initialize bottom-up layers
        conv_kwargs = {'kernel_size': 3, 'stride': 2, 'padding': 1}
        num_bottom_up_layers = len(core_ids) - len(in_ids)
        self.bot_layers = nn.ModuleList() if num_bottom_up_layers > 0 else None

        for i in range(num_bottom_up_layers):
            if i == 0:
                bot_layer = nn.Conv2d(in_sizes[-1], feat_size, **conv_kwargs)

            else:
                bot_layer = nn.Sequential()
                bot_layer.add_module('act', nn.ReLU(inplace=False))
                bot_layer.add_module('conv', nn.Conv2d(feat_size, feat_size, **conv_kwargs))

            self.bot_layers.append(bot_layer)

        # Set fuse type attribute
        self.fuse_type = fuse_type

        # Set attributes related to output feature maps
        self.out_ids = core_ids
        self.out_sizes = [feat_size] * len(core_ids)

        # Set default initial values of module parameters
        self.reset_parameters()

    def reset_parameters(self):
        """
        Resets module parameters to default initial values.
        """

        for name, parameter in self.named_parameters():
            if 'weight' in name:
                nn.init.kaiming_uniform_(parameter, a=1)

            elif 'bias' in name:
                nn.init.zeros_(parameter)

    def forward(self, in_feat_maps, **kwargs):
        """
        Forward method of the FPN module.

        Args:
            in_feat_maps (List): Input feature maps [num_in_maps] of shape [batch_size, feat_size, fH, fW].

        Returns:
            out_feat_maps (List): Output feature maps [num_out_maps] of shape [batch_size, feat_size, fH, fW].
        """

        # Get lateral feature maps
        lat_feat_maps = self.lat_proj(in_feat_maps)

        # Get fused feature maps
        interpolation_kwargs = {'mode': 'nearest'}
        fused_feat_map = lat_feat_maps[-1]
        fused_feat_maps = [fused_feat_map]

        for lat_feat_map in lat_feat_maps[:-1][::-1]:
            fused_feat_map = F.interpolate(fused_feat_map, size=lat_feat_map.shape[-2:], **interpolation_kwargs)
            fused_feat_map = fused_feat_map + lat_feat_map
            fused_feat_map = fused_feat_map / 2 if self.fuse_type == 'avg' else fused_feat_map
            fused_feat_maps.insert(0, fused_feat_map)

        # Get sideways output feature maps
        out_feat_maps = self.out_proj(fused_feat_maps)

        # Get additional bottom-up output maps
        if hasattr(self, 'bottom_up'):
            bottom_up_feat_map = in_feat_maps[-1]

            for module in self.bottom_up:
                bottom_up_feat_map = module(bottom_up_feat_map)
                out_feat_maps.append(bottom_up_feat_map)

        return out_feat_maps
