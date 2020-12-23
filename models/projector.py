"""
Projector module.
"""

from torch import nn


class Projector(nn.Module):
    """
    Class implementing the Projector module.

    Attributes:
        in_map_ids (List): List of size [num_out_maps] containing the input map ids corresponding to each projection.
        projs (nn.ModuleList): List of size [num_out_maps] containing the projection modules.
    """

    def __init__(self, in_feat_sizes, proj_dicts):
        """
        Initializes the Projector module.

        Args:
            in_feat_sizes (List): List of size [num_in_maps] containing the feature size of each input map.

            proj_dicts (List): List of size [num_out_maps] with projection dictionaries containing following keys:
                - in_map_id (int): integer containing the map id of the input map to be projected;
                - out_feat_size (int): integer containing the output feature size after projection;
                - proj_type (str): string with projection type chosen from {'conv1', 'conv3'};
                - conv_stride (int): integer containing the convolution stride for projection types {'conv1', 'conv3'}.

        Raises:
            ValueError: Raised when an invalid projection type is provided in a projection dictionary.
        """

        # Initialization of default nn.Module
        super().__init__()

        # Intialize list of projection modules with corresponding input map ids
        allowed_proj_types = {'conv1', 'conv3'}
        self.in_map_ids = []
        self.projs = nn.ModuleList([])

        for proj_dict in proj_dicts:
            in_map_id = proj_dict['in_map_id']
            in_feat_size = in_feat_sizes[in_map_id]

            out_feat_size = proj_dict['out_feat_size']
            proj_type = proj_dict['proj_type']

            if proj_type == 'conv1':
                stride = proj_dict['conv_stride']
                proj = nn.Conv2d(in_feat_size, out_feat_size, kernel_size=1, stride=stride)

            elif proj_type == 'conv3':
                stride = proj_dict['conv_stride']
                proj = nn.Conv2d(in_feat_size, out_feat_size, kernel_size=3, stride=stride, padding=1)

            else:
                raise ValueError(f"We support {allowed_proj_types} as projection types, but got {proj_type}.")

            self.in_map_ids.append(in_map_id)
            self.projs.append(proj)

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

    def forward(self, in_feat_maps):
        """
        Forward method of the Projector module.

        Args:
            in_feat_maps (List): Input feature maps [num_in_maps] of shape [batch_size, feat_size, fH, fW].

        Returns:
            out_feat_maps (List): Projected feature maps [num_out_maps] of shape [batch_size, feat_size, fH, fW].
        """

        # Get projected feature maps
        out_feat_maps = [proj(in_feat_maps[in_map_id]) for in_map_id, proj in zip(self.in_map_ids, self.projs)]

        return out_feat_maps
