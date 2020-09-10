"""
Backbone modules and build function.
"""
from typing import List

import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.misc import FrozenBatchNorm2d

from utils.data import NestedTensor
from utils.distributed import is_main_process


class Backbone(nn.Module):
    """
    Class implementing the Backbone module.

    Attributes:
        body (IntermediateLayerGetter): Module computing the feature maps.
        num_channels (int): Number of channels of last feature map.
    """

    def __init__(self, name: str, train_backbone: bool, dilation: bool):
        """
        Initializes the Backbone module.

        Args:
            name (str): Name of backbone model to build (resnet only).
            train_backbone (bool): Whether backbone should be trained or not.
            dilation (bool): Whehter to use dilation in last layer or not.
        """

        super().__init__()
        backbone = getattr(torchvision.models, name)(replace_stride_with_dilation=[False, False, dilation],
                                                     pretrained=is_main_process(), norm_layer=FrozenBatchNorm2d)

        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)

        return_layers = {'layer4': '0'}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = 512 if name in ('resnet18', 'resnet34') else 2048

    def forward(self, images: NestedTensor) -> List[NestedTensor]:
        """
        Forward method of Backbone module.

        Args:
            images (NestedTensor): NestedTensor which consists of:
               - images.tensors (FloatTensor): batched images of shape [batch_size, 3, H, W];
               - images.mask (BoolTensor): boolean masks encoding inactive pixels of shape [batch_size, H, W].

        Returns:
            out (List[NestedTensor]): List of feature maps, with each feature map a NestedTensor.
        """

        conv_feature_maps = self.body(images.tensors)

        original_mask = images.mask
        assert original_mask is not None, 'No mask specified in NestedTensor'
        original_mask = original_mask[None].float()

        out: List[NestedTensor] = []
        for conv_feature_map in conv_feature_maps.values():
            map_size = conv_feature_map.shape[-2:]
            mask = F.interpolate(original_mask, size=map_size).to(torch.bool)[0]
            out.append(NestedTensor(conv_feature_map, mask))

        return out


def build_backbone(args):
    """
    Build backbone from command-line arguments.

    Args:
        args (argsparse.Namespace): Command-line arguments.

    Returns:
        backbone (Backbone): The specified Backbone module.
    """

    train_backbone = args.lr_backbone > 0
    backbone = Backbone(args.backbone, train_backbone, args.dilation)

    return backbone


if __name__ == '__main__':
    from main import get_parser
    args = get_parser().parse_args()
    backbone = build_backbone(args)

    detr_url = 'https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth'
    detr_state_dict = torch.hub.load_state_dict_from_url(detr_url)['model']

    from collections import OrderedDict
    backbone_state_dict = OrderedDict()

    backbone_identifier = 'backbone.0.'
    identifier_length = len(backbone_identifier)

    for detr_name, detr_state in detr_state_dict.items():
        if backbone_identifier in detr_name:
            new_name = detr_name[identifier_length:]
            backbone_state_dict[new_name] = detr_state

    backbone.load_state_dict(backbone_state_dict)
