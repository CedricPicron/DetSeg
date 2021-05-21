"""
FPN module and build function.
"""

from torch import nn
import torch.nn.functional as F

from models.modules.projector import Projector


class FPN(nn.Module):
    """
    Class implementing the FPN (Feature Pyriamid Network) module.

    Attributes:
        lat_proj (Projector): Module computing lateral feature maps from input feature maps.
        out_proj (Projector): Module computing output feature maps from fused feature maps.
        fuse_type (str): String of lateral with top-down fuse operation chosen from {'avg', 'sum'}.
        bottom_up (ModuleList): List of size [num_bottom_up_layers] computing additional bottom-up output maps.
        feat_sizes (List): List of size [num_out_maps] containing the feature size of each output map.
    """

    def __init__(self, in_feat_sizes, out_feat_sizes, fuse_type, bottom_up_dict=None):
        """
        Initializes the FPN module.

        Args:
            in_feat_sizes (List): List of size [num_in_maps] containing the feature size of each input map.
            out_feat_sizes (List): List of size [num_in_maps] containing the feature size of sideways output maps.
            fuse_type (str): String of lateral with top-down fuse operation chosen from {'avg', 'sum'}.

            bottom_up_dict (Dict): Optional bottom-up dictionary containing following key:
                - feat_sizes (List): bottom-up output feature sizes of size [num_bottom_up_layers].
        """

        # Check whether fuse type input is valid
        allowed_fuse_types = {'avg', 'sum'}
        check = fuse_type in allowed_fuse_types
        assert check, f"We support fuse types {allowed_fuse_types}, but got {fuse_type}."

        # Initialization of default nn.Module
        super().__init__()

        # Initialize lateral projector
        fixed_settings = {'proj_type': 'conv1', 'conv_stride': 1}
        proj_dicts = []

        for in_map_id, out_feat_size in enumerate(out_feat_sizes):
            proj_dict = {'in_map_id': in_map_id, 'out_feat_size': out_feat_size, **fixed_settings}
            proj_dicts.append(proj_dict)

        self.lat_proj = Projector(in_feat_sizes, proj_dicts)

        # Initialize output projector
        fixed_settings = {'proj_type': 'conv3', 'conv_stride': 1}
        proj_dicts = []

        for in_map_id, out_feat_size in enumerate(out_feat_sizes):
            proj_dict = {'in_map_id': in_map_id, 'out_feat_size': out_feat_size, **fixed_settings}
            proj_dicts.append(proj_dict)

        self.out_proj = Projector(out_feat_sizes, proj_dicts)

        # Set fuse type attribute
        self.fuse_type = fuse_type

        # Initialize bottom-up modules
        if bottom_up_dict:
            feat_sizes = [in_feat_sizes[-1], *bottom_up_dict['feat_sizes']]
            conv_kwargs = {'kernel_size': 3, 'stride': 2, 'padding': 1}
            self.bottom_up = nn.ModuleList([nn.Conv2d(feat_sizes[0], feat_sizes[1], **conv_kwargs)])

            for in_size, out_size in zip(feat_sizes[1:-1], feat_sizes[2:]):
                layers = [nn.ReLU(), nn.Conv2d(in_size, out_size, **conv_kwargs)]
                self.bottom_up.append(nn.Sequential(*layers))

        # Set feature sizes attribute
        self.feat_sizes = [*out_feat_sizes, *bottom_up_dict['feat_sizes']]

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
