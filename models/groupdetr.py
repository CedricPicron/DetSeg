"""
GroupDETR modules and build function.
"""

import torch
from torch import nn
import torch.nn.functional as F

from .backbone import build_backbone
from .position import build_position_encoder
from .transformer import build_transformer
from utils.data import NestedTensor, nested_tensor_from_tensor_list


class GroupDETR(nn.Module):
    """
    Class implementing the GroupDETR module.

    Attributes:
        backbone (Backbone): Torch module of the backbone architecture.
        position_encoder (PositionEncodingSine): Torch module for the position encoding.
        projector (nn.Conv2d): Torch module projecting conv. features to transformer features.
        transformer (Transformer): Torch module of the transformer architecture.
        class_embed (nn.Linear): Torch module projecting transformer features to class logits.
        bbox_embed (MLP): Torch module projecting transformer features to bbox "logits" (i.e. before sigmoid).
        aux_loss (bool): Apply loss at every group layer if True, else only after last group layer.
    """

    def __init__(self, backbone, position_encoder, transformer, num_classes, aux_loss=False):
        """
        Initializes the GroupDETR module.

        Args:
            backbone (Backbone): Torch module of the backbone architecture.
            position_encoder (PositionEncodingSine): Torch module for the position encoding.
            projector (nn.Conv2d): Torch module projecting conv. features to transformer features.
            transformer (Transformer): Torch module of the transformer architecture.
            num_classes (int): Number of object classes (without background class).
            aux_loss (bool): Apply loss at every group layer if True, else only after last group layer.
        """

        super().__init__()
        self.backbone = backbone

        self.position_encoder = position_encoder
        self.projector = nn.Conv2d(backbone.num_channels, transformer.feature_dim, kernel_size=1)
        self.transformer = transformer

        self.class_embed = nn.Linear(transformer.feature_dim, num_classes + 1)
        self.bbox_embed = MLP(transformer.feature_dim, transformer.feature_dim, 4, 3)
        self.aux_loss = aux_loss

    def forward(self, images: NestedTensor):
        """
        Forward method of GroupDETR module.

        Args:
            images (NestedTensor): NestedTensor which consists of:
               - images.tensors: batched images of shape [batch_size x 3 x H x W];
               - images.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels.

        Returns:
            List of dicts, with each dict consisting of:
               - "pred_logits": the classification logits (including background) for all groups, represented as
                                [batch_size x num_groups x (num_classes + 1)];
               - "pred_boxes": the normalized box coordinates (center_x, center_y, height, width) w.r.t. padded images
                               for all groups, represented as [batch_size x num_groups x 4];
            The list length equals the number of grouping layers in the transformer, unless aux_loss is False,
            which a list length of 1 instead (only features from last group layer are returned).
        """

        if isinstance(images, (list, torch.Tensor)):
            images = nested_tensor_from_tensor_list(images)

        masked_conv_features = self.backbone(images)[-1]
        pos_encoding = self.position_encoder(masked_conv_features)

        conv_features, padding_mask = masked_conv_features.decompose()
        proj_features = self.projector(conv_features)
        final_features = self.transformer(proj_features, padding_mask, pos_encoding, return_intermediate=self.aux_loss)

        class_logits = self.class_embed(final_features)
        bbox_coord = self.bbox_embed(final_features).sigmoid()

        return [{'pred_logits': a, 'pred_boxes': b} for a, b in zip(class_logits, bbox_coord)]


class MLP(nn.Module):
    """
    Simple multi-layer perceptron (MLP).
    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build_model(args):
    """
    Build GroupDETR module from command-line arguments.

    Args:
        args (argparse.Namespace): Command-line arguments.

    Returns:
        GroupDETR module.
    """

    backbone = build_backbone(args)
    position_encoder = build_position_encoder(args)
    transformer = build_transformer(args)

    return GroupDETR(backbone, position_encoder, transformer, args.num_classes, args.aux_loss)
