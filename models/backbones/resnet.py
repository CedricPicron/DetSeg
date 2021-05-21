"""
ResNet backbone.
"""
from collections import OrderedDict

import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from torchvision.models._utils import IntermediateLayerGetter

from models.modules.normalization import FrozenBatchNorm2d


class ResNet(nn.Module):
    """
    Class implementing the ResNet module.

    Attributes:
        in_norm_mean (FloatTensor): Tensor (buffer) of shape [3] containing the input normalization means.
        in_norm_std (FloatTensor): Tensor (buffer) of shape [3] containing the input normalization standard deviations.
        body (IntermediateLayerGetter): Module computing and returning the requested ResNet feature maps.
        feat_sizes (List): List of size [num_maps] containing the feature size of each returned feature map.
    """

    def __init__(self, name, dilation, return_layers):
        """
        Initializes the ResNet module.

        Args:
            name (str): String containing the full name of the desired ResNet model.
            dilation (bool): Boolean indicating whether to use dilation instead of stride for the last ResNet layer.
            return_layers (Dict): Dictionary mapping names of ResNet layers to be returned to new names.
        """

        # Initialization of default nn.Module
        super().__init__()

        # Register input normalization buffers with values expected by torchvision pretrained backbones
        self.register_buffer('in_norm_mean', torch.tensor([0.485, 0.456, 0.406]))
        self.register_buffer('in_norm_std', torch.tensor([0.229, 0.224, 0.225]))

        # Load ImageNet pretrained ResNet from torchvision
        resnet_kwargs = {'replace_stride_with_dilation': [False, False, dilation], 'pretrained': True}
        resnet_kwargs = {**resnet_kwargs, 'norm_layer': FrozenBatchNorm2d}
        resnet = getattr(torchvision.models, name)(**resnet_kwargs)

        # Determine which backbone parameters should be trained
        for name, parameter in resnet.named_parameters():
            if 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)

        # Get body module computing and returning the requested ResNet feature maps
        self.body = IntermediateLayerGetter(resnet, return_layers=return_layers)

        # Get feature sizes of returned feature maps
        feat_sizes = [64, 128, 256, 512] if name in ['resnet18', 'resnet34'] else [256, 512, 1024, 2048]
        feat_sizes = {f'layer{i+1}': feat_sizes[i] for i in range(4)}
        self.feat_sizes = [feat_sizes[layer_name] for layer_name in return_layers]

    def load_from_original_detr(self, fb_detr_state_dict):
        """
        Loads backbone from state dictionary of an original Facebook DETR model.

        fb_detr_state_dict (Dict): Dictionary containing Facebook DETR model parameters and persistent buffers.
        """

        backbone_identifier = 'backbone.0.'
        identifier_length = len(backbone_identifier)
        backbone_state_dict = OrderedDict()

        for original_name, state in fb_detr_state_dict.items():
            if backbone_identifier in original_name:
                new_name = original_name[identifier_length:]
                backbone_state_dict[new_name] = state

        self.load_state_dict(backbone_state_dict)

    def _load_from_state_dict(self, state_dict, prefix, *args):
        """
        Copies parameters and buffers from given state dictionary into only this module, but not its descendants.

        state_dict (Dict): Dictionary containing model parameters and persistent buffers.
        prefix (str): String containing this module's prefix in the given state dictionary.
        args (Tuple): Tuple containing additional arguments used by the default loading method.
        """

        # Set default input normalization buffers if missing
        state_dict.setdefault(f'{prefix}in_norm_mean', torch.tensor([0.485, 0.456, 0.406]))
        state_dict.setdefault(f'{prefix}in_norm_std', torch.tensor([0.229, 0.224, 0.225]))

        # Continue with default loading
        super()._load_from_state_dict(state_dict, prefix, *args)

    def forward(self, images, return_masks=False):
        """
        Forward method of ResNet module.

        Args:
            images (Images): Images structure containing the batched images.
            return_masks (bool): Boolean indicating whether to return feature masks or not (default=False).

        Returns:
            feat_maps (List): List of size [num_maps] with feature maps of shape [batch_size, feat_size, fH, fW].

            If 'return_masks' is True:
                feat_masks (List): List of size [num_maps] with masks of active features of shape [batch_size, fH, fW].
        """

        # Normalize input images
        norm_images = images.normalize(self.in_norm_mean, self.in_norm_std)

        # Compute backbone feature maps
        feat_maps = list(self.body(norm_images).values())

        # Only return feature maps if desired
        if not return_masks:
            return feat_maps

        # Compute feature masks
        feat_masks = images.masks[None].float()
        feat_masks = [F.interpolate(feat_masks, size=feat_map.shape[-2:]).to(torch.bool)[0] for feat_map in feat_maps]

        return feat_maps, feat_masks
