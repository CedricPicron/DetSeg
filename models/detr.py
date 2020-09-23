"""
DETR modules and build function.
"""

import torch
from torch import nn

from .backbone import build_backbone
from .position import build_position_encoder
from .encoder import build_encoder
from .decoder import build_decoder
from .utils import MLP
from utils.data import NestedTensor, nested_tensor_from_tensor_list


class DETR(nn.Module):
    """
    Class implementing the DETR module.

    Attributes:
        backbone (nn.Module): Module implementing the DETR backbone.
        position_encoder (nn.Module): Module implementing the position encoding.
        projector (nn.Conv2d): Module projecting conv. features to initial encoder features.
        encoder (nn.Module): Module implementing the DETR encoder.
        decoder (nn.Module): Module implementing the DETR decoder.
        class_embed (nn.Linear): Module projecting decoder features to class logits.
        bbox_embed (MLP): Module projecting decoder features to bbox "logits" (i.e. before sigmoid).
    """

    def __init__(self, backbone, position_encoder, encoder, decoder, num_classes):
        """
        Initializes the DETR module.

        Args:
            backbone (nn.Module): Module implementing the DETR backbone.
            position_encoder (nn.Module): Module implementing the position encoding.
            projector (nn.Conv2d): Module projecting conv. features to initial encoder features.
            encoder (nn.Module): Module implementing the DETR encoder.
            decoder (nn.Module): Module implementing the DETR decoder.
            num_classes (int): Number of object classes (without background class).
        """

        super().__init__()
        self.backbone = backbone

        self.position_encoder = position_encoder
        self.projector = nn.Conv2d(backbone.num_channels, encoder.feat_dim, kernel_size=1)

        self.encoder = encoder
        self.decoder = decoder

        self.class_embed = nn.Linear(decoder.feat_dim, num_classes + 1)
        self.bbox_embed = MLP(decoder.feat_dim, decoder.feat_dim, 4, 3)

    def forward(self, images: NestedTensor):
        """
        Forward method of DETR module.

        Args:
            images (NestedTensor): NestedTensor which consists of:
                - images.tensors (FloatTensor): batched images of shape [batch_size, 3, H, W];
                - images.mask (BoolTensor): boolean masks encoding inactive pixels of shape [batch_size, H, W].

        Returns:
            pred_list (List): List of predictions, where each entry is a dict containing the key:
                - logits (FloatTensor): class logits (with background) of shape [num_slots_total, (num_classes + 1)];
                - boxes (FloatTensor): normalized box coordinates (center_x, center_y, height, width) within non-padded
                                       regions, of shape [num_slots_total, 4];
                - batch_idx (IntTensor): batch indices of slots (sorted in ascending order) of shape [num_slots_total];
                - layer_id (int): integer corresponding to the decoder layer producing the predictions;
                - iter_id (int): integer corresponding to the iteration of the decoder layer producing the predictions.

            The list size is [1] or [num_decoder_layers*num_decoder_iterations] depending on args.aux_loss.
        """

        if isinstance(images, (list, torch.Tensor)):
            images = nested_tensor_from_tensor_list(images)

        masked_conv_features = self.backbone(images)[-1]
        conv_features, feature_masks = masked_conv_features.decompose()
        proj_features = self.projector(conv_features)
        pos_encodings = self.position_encoder(proj_features, feature_masks)

        batch_size, feat_dim, H, W = proj_features.shape
        proj_features = proj_features.view(batch_size, feat_dim, H*W).permute(2, 0, 1)
        pos_encodings = pos_encodings.view(batch_size, feat_dim, H*W).permute(2, 0, 1)

        encoder_features = self.encoder(proj_features, feature_masks, pos_encodings)
        slots, batch_idx, seg_maps = self.decoder(encoder_features, feature_masks, pos_encodings)

        class_logits = self.class_embed(slots)
        bbox_coord = self.bbox_embed(slots).sigmoid()

        num_layers = self.decoder.num_layers
        iters = self.decoder.num_iterations
        pred_list = [{'logits': a, 'boxes': b, 'batch_idx': c} for a, b, c in zip(class_logits, bbox_coord, batch_idx)]
        [pred_dict.update({'layer_id': num_layers-i, 'iter_id': i//iters+1}) for i, pred_dict in enumerate(pred_list)]

        return pred_list


def build_detr(args):
    """
    Build DETR module from command-line arguments.

    Args:
        args (argparse.Namespace): Command-line arguments.

    Returns:
        detr (DETR): The specified DETR module.
    """

    backbone = build_backbone(args)
    position_encoder = build_position_encoder(args)
    encoder = build_encoder(args)
    decoder = build_decoder(args)

    # Temporary hack
    args.num_classes = 91

    detr = DETR(backbone, position_encoder, encoder, decoder, args.num_classes)

    return detr
