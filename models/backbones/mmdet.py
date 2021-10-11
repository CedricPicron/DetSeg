"""
Module building backbones from MMDetection.
"""

from mmcv import Config
from mmdet.models import build_backbone
import torch
from torch import nn


class MMDetBackbone(nn.Module):
    """
    Class implementing the MMDetBackbone module.

    Attributes:
        in_norm_mean (FloatTensor): Tensor (buffer) of shape [3] containing the input normalization means.
        in_norm_std (FloatTensor): Tensor (buffer) of shape [3] containing the input normalization standard deviations.
        feat_sizes (List): List of size [num_maps] containing the feature sizes of the returned feature maps.
        body (nn.Module): Module computing and returning the requested backbone feature maps.
    """

    def __init__(self, cfg_path):
        """
        Initializes the MMDetBackbone module.

        Args:
            cfg_path (str): Path to configuration file specifying the MMDetection backbone.
        """

        # Initialization of default nn.Module
        super().__init__()

        # Register input normalization buffers with values expected by torchvision pretrained backbones
        self.register_buffer('in_norm_mean', torch.tensor([0.485, 0.456, 0.406]))
        self.register_buffer('in_norm_std', torch.tensor([0.229, 0.224, 0.225]))

        # Get config specifying the MMDetection backbone
        cfg = Config.fromfile(cfg_path)

        # Get feature sizes of returned feature maps
        self.feat_sizes = cfg.backbone.pop('out_sizes')

        # Get backbone body
        self.body = build_backbone(cfg.backbone)

    def forward(self, images):
        """
        Forward method of the MMDetBackbone module.

        Args:
            images (Images): Images structure containing the batched images.

        Returns:
            feat_maps (List): List of size [num_maps] with feature maps of shape [batch_size, feat_size, fH, fW].
        """

        # Normalize input images
        norm_images = images.normalize(self.in_norm_mean, self.in_norm_std)

        # Compute backbone feature maps
        feat_maps = list(self.body(norm_images))

        return feat_maps
