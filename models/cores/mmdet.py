"""
Module building cores from MMDetection.
"""

from mmcv import Config
from mmdet.models import build_neck as build_core
from torch import nn


class MMDetCore(nn.Module):
    """
    Class implementing the MMDetCore module.

    Attributes:
        out_ids (List): List [num_out_maps] containing the indices of the output feature maps.
        out_sizes (List): List [num_out_maps] containing the feature sizes of the output feature maps.

        body (nn.Module): Module computing the core output feature maps from the input feature maps.
    """

    def __init__(self, cfg_path):
        """
        Initializes the MMDetCore module.

        Args:
            cfg_path (str): Path to configuration file specifying the MMDetection core.
        """

        # Initialization of default nn.Module
        super().__init__()

        # Get config specifying the MMDetection core
        cfg = Config.fromfile(cfg_path)

        # Set attributes related to output feature maps
        self.out_ids = cfg.core.pop('out_ids')
        self.out_sizes = cfg.core.pop('out_sizes')

        # Get core body
        self.body = build_core(cfg.core)
        self.body.init_weights()

    def forward(self, in_feat_maps):
        """
        Forward method of the MMDetCore module.

        Args:
            in_feat_maps (List): Input feature maps [num_in_maps] of shape [batch_size, feat_size, fH, fW].

        Returns:
            out_feat_maps (List): Output feature maps [num_out_maps] of shape [batch_size, feat_size, fH, fW].
        """

        # Get output feature maps
        out_feat_maps = list(self.body(in_feat_maps))

        return out_feat_maps
