"""
Backbone modules and build function.
"""
from collections import OrderedDict

import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.misc import FrozenBatchNorm2d


class FrozenBatchNorm2d (FrozenBatchNorm2d):
    """
    Two-dimensional batch normalization layer with frozen statistics.

    Copy from torchvision, but with default eps of 1e-5.

    Attributes:
        num_features (int): Expected number of 2D input feature maps.
        eps (float): Value added to the denominator for numerical stability (defaults to 1e-5).
    """

    def __init__(self, num_features, eps=1e-5):
        super().__init__(num_features, eps=eps)


class Backbone(nn.Module):
    """
    Class implementing the Backbone module.

    Attributes:
        in_norm_mean (FloatTensor): Tensor (buffer) of shape [3] containing the input normalization means.
        in_norm_std (FloatTensor): Tensor (buffer) of shape [3] containing the input normalization standard deviations.
        body (IntermediateLayerGetter): Module computing feature maps of different resolutions.
        feat_sizes (List): List of size [num_maps] containing the feature size of each returned backbone feature map.
        trained (bool): Bool indicating whether backbone is trained or not.
    """

    def __init__(self, name, dilation, return_layers, train_backbone):
        """
        Initializes the Backbone module.

        Args:
            name (str): Name of backbone model to build (resnet only).
            dilation (bool): Bool indicating whether to use dilation in last layer or not.
            return_layers (Dict): Dictionary mapping names of backbone layers that will be returned to new names.
            train_backbone (bool): Bool indicating whether backbone should be trained or not.
        """

        # Initialization of default nn.Module
        super().__init__()

        # Register input normalization buffers with values expected by torchvision pretrained backbones
        self.register_buffer('in_norm_mean', torch.tensor([0.485, 0.456, 0.406]))
        self.register_buffer('in_norm_std', torch.tensor([0.229, 0.224, 0.225]))

        # Load ImageNet pretrained backbone from torchvision
        backbone_kwargs = {'replace_stride_with_dilation': [False, False, dilation], 'pretrained': True}
        backbone_kwargs = {**backbone_kwargs, 'norm_layer': FrozenBatchNorm2d}
        backbone = getattr(torchvision.models, name)(**backbone_kwargs)

        # Determine which backbone parameters should be trained
        self.trained = train_backbone
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)

        # Get module performing backbone computations and returning intermediate layers
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)

        # Get feature sizes of different feature maps
        default_feat_sizes = [64, 128, 256, 512] if name in ['resnet18', 'resnet34'] else [256, 512, 1024, 2048]
        self.feat_sizes = default_feat_sizes[-len(return_layers):]

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

    def forward(self, images):
        """
        Forward method of Backbone module.

        Args:
            images (Images): Images structure containing the batched images.

        Returns:
            feat_maps (List): List of size [num_maps] with feature maps of shape [batch_size, feat_size, fH, fW].
            masks (List): List of size [num_maps] with boolean masks of active pixels of shape [batch_size, fH, fW].
        """

        # Normalize input images
        norm_images = images.normalize(self.in_norm_mean, self.in_norm_std)

        # Compute backbone feature maps
        feat_maps = list(self.body(norm_images).values())

        # Compute downsampled masks of active pixels
        masks = images.masks[None].float()
        masks = [F.interpolate(masks, size=feat_map.shape[-2:]).to(torch.bool)[0] for feat_map in feat_maps]

        return feat_maps, masks


def build_backbone(args):
    """
    Build backbone from command-line arguments.

    Args:
        args (argsparse.Namespace): Command-line arguments.

    Returns:
        backbone (Backbone): The specified Backbone module.
    """

    # Find out whether backbone should be trained or not
    train_backbone = args.lr_backbone > 0

    # Build desired backbone module
    if args.meta_arch == 'BiViNet':
        assert not args.dilation, "'--dilation' is not allowed for meta-architecture BiViNet"

        map_ids = range(args.min_downsampling, args.max_downsampling+1)
        return_layers = {f'layer{i-1}': str(i) for i in map_ids if i >= 2 and i <= 5}
        backbone = Backbone(args.backbone, args.dilation, return_layers, train_backbone)

    elif args.meta_arch == 'DETR':
        return_layers = {'layer4': '0'}
        backbone = Backbone(args.backbone, args.dilation, return_layers, train_backbone)

    else:
        raise ValueError(f"Unknown meta-architecture type '{args.meta_arch}' was provided.")

    return backbone
