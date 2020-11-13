"""
DETR modules and build function.
"""
from collections import OrderedDict

import torch
from torch import nn

from .backbone import build_backbone
from .position import build_position_encoder
from .encoder import build_encoder
from .decoder import build_decoder, GlobalDecoder
from .utils import MLP


class DETR(nn.Module):
    """
    Class implementing the DETR module.

    Attributes:
        backbone (nn.Module): Module implementing the DETR backbone.
        position_encoder (nn.Module): Module implementing the position encoding.
        projector (nn.Conv2d): Module projecting conv. features to initial encoder features.
        encoder (nn.Module): Module implementing the DETR encoder.
        decoder (nn.Module): Module implementing the DETR decoder.
        class_head (nn.Linear): Module projecting decoder features to class logits.
        bbox_head (MLP): Module projecting decoder features to bounding box logits (i.e. values before sigmoid).
        train_dict (Dict): Dictionary of booleans indicating whether projector and heads should be trained or not.
    """

    def __init__(self, backbone, position_encoder, encoder, decoder, num_classes, train_dict):
        """
        Initializes the DETR module.

        Args:
            backbone (nn.Module): Module implementing the DETR backbone.
            position_encoder (nn.Module): Module implementing the position encoding.
            projector (nn.Conv2d): Module projecting conv. features to initial encoder features.
            encoder (nn.Module): Module implementing the DETR encoder.
            decoder (nn.Module): Module implementing the DETR decoder.
            num_classes (int): Number of object classes (without background class).
            train_dict (Dict): Dictionary of booleans indicating whether projector and heads should be trained or not.
        """

        super().__init__()
        self.backbone = backbone
        self.position_encoder = position_encoder

        self.projector = nn.Conv2d(backbone.feat_sizes[-1], encoder.feat_dim, kernel_size=1)
        self.projector.requires_grad_(train_dict['projector'])

        self.encoder = encoder
        self.decoder = decoder

        self.class_head = nn.Linear(decoder.feat_dim, num_classes+1)
        self.class_head.requires_grad_(train_dict['class_head'])

        self.bbox_head = MLP(decoder.feat_dim, decoder.feat_dim, 4, 3)
        self.bbox_head.requires_grad_(train_dict['bbox_head'])
        self.train_dict = train_dict

    def load_state_dict(self, state_dict):
        """
        Loads DETR module from state dictionary.

        It removes shared parameter duplicates in the state dictionary if present.

        Args:
            state_dict (Dict): Dictionary containing the whole state of a DETR module.
        """

        # Identify shared parameters
        shared_param_names = [name for key in state_dict.keys() for name in key.split('.') if 'shared' in name]

        # Remove shared parameter duplicates
        for shared_param_name in shared_param_names:
            attn_prefix = shared_param_name.split('_')[1]
            suffix = '_'.join(shared_param_name.split('_')[2:])
            duplicate_identifier = f'{attn_prefix}_attention.{suffix}'

            for param_name in list(state_dict.keys()).copy():
                if duplicate_identifier in param_name:
                    del state_dict[param_name]

        # Load DETR module from resulting state dictionary
        super().load_state_dict(state_dict)

    def load_from_original_detr(self):
        """
        Loads backbone, projector, encoder, global decoder and heads from an original Facebook DETR model if untrained.
        """

        original_detr_url = 'https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth'
        state_dict = torch.hub.load_state_dict_from_url(original_detr_url)['model']
        is_global_decoder = isinstance(self.decoder, GlobalDecoder)

        self.backbone.load_from_original_detr(state_dict) if not self.backbone.trained else None
        self.load_projector_from_original_detr(state_dict) if not self.train_dict['projector'] else None
        self.encoder.load_from_original_detr(state_dict) if not self.encoder.trained else None
        self.decoder.load_from_original_detr(state_dict) if not self.decoder.trained and is_global_decoder else None
        self.load_class_head_from_original_detr(state_dict) if not self.train_dict['class_head'] else None
        self.load_bbox_head_from_original_detr(state_dict) if not self.train_dict['bbox_head'] else None

    def load_projector_from_original_detr(self, state_dict):
        """
        Loads projector from state_dict of an original Facebook DETR model.

        Args:
            state_dict (Dict): Dictionary containing Facebook's model parameters and persistent buffers.
        """

        projector_identifier = 'input_proj.'
        identifier_length = len(projector_identifier)
        projector_state_dict = OrderedDict()

        for original_name, state in state_dict.items():
            if projector_identifier in original_name:
                new_name = original_name[identifier_length:]
                projector_state_dict[new_name] = state

        self.projector.load_state_dict(projector_state_dict)

    def load_class_head_from_original_detr(self, state_dict):
        """
        Loads classification head from state_dict of an original Facebook DETR model.

        Args:
            state_dict (Dict): Dictionary containing Facebook's model parameters and persistent buffers.
        """

        class_head_identifier = 'class_embed.'
        identifier_length = len(class_head_identifier)
        class_head_state_dict = OrderedDict()

        for original_name, state in state_dict.items():
            if class_head_identifier in original_name:
                new_name = original_name[identifier_length:]
                class_head_state_dict[new_name] = state

        self.class_head.load_state_dict(class_head_state_dict)

    def load_bbox_head_from_original_detr(self, state_dict):
        """
        Loads bounding box head from state_dict of an original Facebook DETR model.

        Args:
            state_dict (Dict): Dictionary containing Facebook's model parameters and persistent buffers.
        """

        bbox_head_identifier = 'bbox_embed.'
        identifier_length = len(bbox_head_identifier)
        bbox_head_state_dict = OrderedDict()

        for original_name, state in state_dict.items():
            if bbox_head_identifier in original_name:
                new_name = original_name[identifier_length:]
                bbox_head_state_dict[new_name] = state

        self.bbox_head.load_state_dict(bbox_head_state_dict)

    @staticmethod
    def get_parameter_families():
        """
        Method returning the DETR parameter families.

        Returns:
            List of strings containing the DETR parameter families.
        """

        return ['backbone', 'projector', 'encoder', 'decoder', 'class_head', 'bbox_head']

    @staticmethod
    def get_sizes(batch_idx, batch_size):
        """
        Computes the cumulative number of predictions across batch entries.

        Args:
            batch_idx (IntTensor): Batch indices (sorted in ascending order) of shape [num_pred_sets, num_slots_total].
            batch_size (int): Total number of batch entries.

        Returns:
            sizes (IntTensor): Cumulative sizes of batch entries of shape [num_pred_sets, batch_size+1].
        """

        num_pred_sets, num_slots_total = batch_idx.shape
        sizes = torch.zeros(num_pred_sets, batch_size+1, dtype=torch.int)

        for i in range(num_pred_sets):
            prev_idx = 0

            for j, curr_idx in enumerate(batch_idx[i]):
                if curr_idx != prev_idx:
                    sizes[i, prev_idx+1:curr_idx+1] = j
                    prev_idx = curr_idx

            sizes[i, curr_idx+1:] = num_slots_total

        return sizes

    def forward(self, images):
        """
        Forward method of DETR module.

        Args:
            images (NestedTensor): NestedTensor consisting of:
                - images.tensor (FloatTensor): padded images of shape [batch_size, 3, H, W];
                - images.mask (BoolTensor): boolean masks encoding inactive pixels of shape [batch_size, H, W].

        Returns:
            pred_list (List): List of predictions, where each entry is a dict containing following keys:
                - logits (FloatTensor): class logits (with background) of shape [num_slots_total, (num_classes + 1)];
                - boxes (FloatTensor): normalized box coordinates (center_x, center_y, height, width) within non-padded
                                       regions, of shape [num_slots_total, 4];
                - batch_idx (IntTensor): batch indices of slots (in ascending order) of shape [num_slots_total];
                - sizes (IntTensor): cumulative number of predictions across batch entries of shape [batch_size+1];
                - layer_id (int): integer corresponding to the decoder layer producing the predictions;
                - curio_loss (FloatTensor): optional curiosity based losses of shape [1] from a sample decoder.

            The list size is [1] or [num_decoder_layers] depending on args.aux_loss.
        """

        conv_features, feature_masks = self.backbone(images)
        conv_features, feature_masks = (conv_features[-1], feature_masks[-1])
        proj_features = self.projector(conv_features)
        pos_encodings = self.position_encoder(proj_features, feature_masks)

        batch_size, feat_dim, H, W = proj_features.shape
        proj_features = proj_features.view(batch_size, feat_dim, H*W).permute(2, 0, 1)
        pos_encodings = pos_encodings.view(batch_size, feat_dim, H*W).permute(2, 0, 1)

        encoder_features = self.encoder(proj_features, feature_masks, pos_encodings)
        decoder_output_dict = self.decoder(encoder_features, feature_masks, pos_encodings)

        batch_idx = decoder_output_dict['batch_idx']
        sizes = self.get_sizes(batch_idx, batch_size)

        slots = decoder_output_dict['slots']
        class_logits = self.class_head(slots)
        bbox_coord = self.bbox_head(slots).sigmoid()

        pred_list = [{'logits': logits, 'boxes': boxes} for logits, boxes in zip(class_logits, bbox_coord)]
        [pred_dict.update({'batch_idx': batch_idx[i], 'sizes': sizes[i]}) for i, pred_dict in enumerate(pred_list)]
        [pred_dict.update({'layer_id': self.decoder.num_layers-i}) for i, pred_dict in enumerate(pred_list)]

        if 'curio_losses' in decoder_output_dict.keys():
            curio_losses = decoder_output_dict['curio_losses']
            [pred_dict.update({'curio_loss': curio_losses[i]}) for i, pred_dict in enumerate(pred_list)]

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

    train_projector = args.lr_projector > 0
    train_class_head = args.lr_class_head > 0
    train_bbox_head = args.lr_bbox_head > 0

    train_dict = {'projector': train_projector, 'class_head': train_class_head, 'bbox_head': train_bbox_head}
    detr = DETR(backbone, position_encoder, encoder, decoder, args.num_classes, train_dict)

    return detr
