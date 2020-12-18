"""
DETR modules and build function.
"""
from collections import OrderedDict

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

from .backbone import build_backbone
from .criterion import build_criterion
from .decoder import build_decoder, GlobalDecoder
from .encoder import build_encoder
from .position import build_position_encoder
from .utils import MLP
from utils.boxes import Boxes


class DETR(nn.Module):
    """
    Class implementing the DETR module.

    Attributes:
        backbone (nn.Module): Module implementing the DETR backbone.
        position_encoder (nn.Module): Module implementing the position encoder.
        projector (nn.Conv2d): Module projecting conv. features to initial encoder features.
        encoder (nn.Module): Module implementing the DETR encoder.
        decoder (nn.Module): Module implementing the DETR decoder.
        criterion (nn.Module): Module comparing predictions with targets.
        class_head (nn.Linear): Module projecting decoder features to class logits.
        bbox_head (MLP): Module projecting decoder features to bounding box logits (i.e. values before sigmoid).
        train_dict (Dict): Dictionary of booleans indicating whether projector and heads should be trained or not.
    """

    def __init__(self, backbone, position_encoder, encoder, decoder, criterion, num_classes, train_dict, metadata):
        """
        Initializes the DETR module.

        Args:
            backbone (nn.Module): Module implementing the DETR backbone.
            position_encoder (nn.Module): Module implementing the position encoder.
            projector (nn.Conv2d): Module projecting conv. features to initial encoder features.
            encoder (nn.Module): Module implementing the DETR encoder.
            decoder (nn.Module): Module implementing the DETR decoder.
            criterion (nn.Module): Module comparing predictions with targets.
            num_classes (int): Number of object classes (without background class).
            train_dict (Dict): Dictionary of booleans indicating whether projector and heads should be trained or not.
            metadata (detectron2.data.Metadata, optional): Metadata instance containing additional dataset information.
        """

        super().__init__()
        self.backbone = backbone
        self.position_encoder = position_encoder

        self.projector = nn.Conv2d(backbone.feat_sizes[-1], encoder.feat_dim, kernel_size=1)
        self.projector.requires_grad_(train_dict['projector'])

        self.encoder = encoder
        self.decoder = decoder
        self.criterion = criterion

        self.class_head = nn.Linear(decoder.feat_dim, num_classes+1)
        self.class_head.requires_grad_(train_dict['class_head'])

        self.bbox_head = MLP(decoder.feat_dim, decoder.feat_dim, 4, 3)
        self.bbox_head.requires_grad_(train_dict['bbox_head'])

        self.train_dict = train_dict
        self.metadata = metadata

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

    def load_projector_from_original_detr(self, fb_detr_state_dict):
        """
        Loads projector from state dictionary of an original Facebook DETR model.

        Args:
            fb_detr_state_dict (Dict): Dictionary containing Facebook DETR model parameters and persistent buffers.
        """

        projector_identifier = 'input_proj.'
        identifier_length = len(projector_identifier)
        projector_state_dict = OrderedDict()

        for original_name, state in fb_detr_state_dict.items():
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

        original_ids = [*self.metadata.thing_dataset_id_to_contiguous_id.keys(), 91]
        class_head_state_dict['weight'] = class_head_state_dict['weight'][original_ids]
        class_head_state_dict['bias'] = class_head_state_dict['bias'][original_ids]

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
    def get_param_families():
        """
        Method returning the DETR parameter families.

        Returns:
            List of strings containing the DETR parameter families.
        """

        return ['backbone', 'projector', 'encoder', 'decoder', 'class_head', 'bbox_head']

    @staticmethod
    def get_sizes(batch_ids, batch_size):
        """
        Computes the cumulative number of predictions across batch entries.

        Args:
            batch_ids (LongTensor): Batch indices (sorted in ascending order) of shape [num_pred_sets, num_slots_total].
            batch_size (int): Total number of batch entries.

        Returns:
            sizes (LongTensor): Cumulative sizes of batch entries of shape [num_pred_sets, batch_size+1].
        """

        num_pred_sets, num_slots_total = batch_ids.shape
        sizes = torch.zeros(num_pred_sets, batch_size+1).to(batch_ids)

        for i in range(num_pred_sets):
            prev_idx = 0

            for j, curr_idx in enumerate(batch_ids[i]):
                if curr_idx != prev_idx:
                    sizes[i, prev_idx+1:curr_idx+1] = j
                    prev_idx = curr_idx

            sizes[i, curr_idx+1:] = num_slots_total

        return sizes

    def make_predictions(self, out_dict):
        """
        Make classified bounding box predictions based on given DETR decoder layer output dictionary.

        Args:
            out_dict (Dict): Output dictionary containing at least following keys:
                - logits (FloatTensor): class logits (with background) of shape [num_slots_total, (num_classes+1)];
                - boxes (Boxes): axis-aligned bounding boxes of size [num_slots_total];
                - batch_ids (LongTensor): batch indices of slots (in ascending order) of shape [num_slots_total].

        Returns:
            pred_dict (Dict): Prediction dictionary containing following keys:
                - labels (LongTensor): predicted class indices of shape [num_slots_total];
                - boxes (Boxes): axis-aligned bounding boxes of size [num_slots_total];
                - scores (FloatTensor): normalized prediction scores of shape [num_slots_total];
                - batch_ids (LongTensor): batch indices of predictions of shape [num_slots_total].
        """

        # Get predicted object scores and labels
        probs = F.softmax(out_dict['logits'], dim=-1)
        scores, labels = probs[:, :-1].max(dim=-1)

        # Place predictions into prediction dictionary
        pred_dict = {}
        pred_dict['labels'] = labels
        pred_dict['boxes'] = out_dict['boxes']
        pred_dict['scores'] = scores
        pred_dict['batch_ids'] = out_dict['batch_ids']

        return pred_dict

    def forward(self, images, tgt_dict=None, optimizer=None, **kwargs):
        """
        Forward method of the DETR module.

        Args:
            images (NestedTensor): NestedTensor consisting of:
                - images.tensor (FloatTensor): padded images of shape [batch_size, 3, max_iH, max_iW];
                - images.mask (BoolTensor): masks encoding padded pixels of shape [batch_size, max_iH, max_iW].

            tgt_dict (Dict): Optional target dictionary used during training and validation containing following keys:
                - labels (LongTensor): tensor of shape [num_targets_total] containing the class indices;
                - boxes (Boxes): axis-aligned bounding boxes of size [num_targets_total];
                - sizes (LongTensor): tensor of shape [batch_size+1] with the cumulative target sizes of batch entries.

            optimizer (torch.optim.Optimizer): Optional optimizer updating the DETR model parameters during training.

            kwargs(Dict): Dictionary of keyword arguments, potentially containing following keys:
                - max_grad_norm (float): maximum norm of optimizer update during training (clipped if larger).

        Returns:
            * If tgt_dict is not None and optimizer is not None (i.e. during training):
                loss_dict (Dict): Dictionary of different weighted loss terms used for backpropagation during training.
                analysis_dict (Dict): Dictionary of different analyses used for logging purposes only.

            * If tgt_dict is not None and optimizer is None (i.e. during validation):
                pred_dict (Dict): Prediction dictionary from the last decoder layer containing following keys:
                    - labels (LongTensor): predicted class indices of shape [num_slots_total];
                    - boxes (Boxes): axis-aligned bounding boxes of size [num_slots_total];
                    - scores (FloatTensor): normalized prediction scores of shape [num_slots_total];
                    - batch_ids (LongTensor): batch indices of predictions of shape [num_slots_total].

                loss_dict (Dict): Dictionary of different weighted loss terms used for backpropagation during training.
                analysis_dict (Dict): Dictionary of different analyses used for logging purposes only.

            * If tgt_dict is None (i.e. during testing):
                pred_dict (Dict): Prediction dictionary from the last decoder layer (see above for more information).
        """

        # Get projected backbone features and position encodings
        conv_features, feature_masks = self.backbone(images)
        conv_features, feature_masks = (conv_features[-1], feature_masks[-1])
        proj_features = self.projector(conv_features)
        pos_encodings = self.position_encoder(proj_features, feature_masks)

        # Reshape features and position encodings for encoder and decoder modules
        batch_size, feat_dim, fH, fW = proj_features.shape
        proj_features = proj_features.view(batch_size, feat_dim, fH*fW).permute(2, 0, 1)
        pos_encodings = pos_encodings.view(batch_size, feat_dim, fH*fW).permute(2, 0, 1)

        # Compute slots through feature encoding and decoding
        encoder_features = self.encoder(proj_features, feature_masks, pos_encodings)
        decoder_output_dict = self.decoder(encoder_features, feature_masks, pos_encodings)

        # Compute the number of predictions (i.e. slots) per batch entry
        batch_ids = decoder_output_dict['batch_ids']
        sizes = self.get_sizes(batch_ids, batch_size)

        # Predict classification logits and bounding box coordinates
        slots = decoder_output_dict['slots']
        logits_list = self.class_head(slots)

        boxes_per_image = sizes[1:] - sizes[:-1]
        kwargs = {'boxes_per_image': boxes_per_image, 'image_scales': 0, 'scales': 0}

        boxes_list = self.bbox_head(slots).sigmoid()
        boxes_list = [Boxes(boxes, format='cxcywh', **kwargs) for boxes in boxes_list]

        # Build list of output dictionaries
        out_list = [{'logits': logits, 'boxes': boxes} for logits, boxes in zip(logits_list, boxes_list)]
        [out_dict.update({'batch_ids': batch_ids[i], 'sizes': sizes[i]}) for i, out_dict in enumerate(out_list)]
        [out_dict.update({'layer_id': self.decoder.num_layers-i}) for i, out_dict in enumerate(out_list)]

        # Add curiosity losses to the output dictionaries if available
        if 'curio_losses' in decoder_output_dict.keys():
            curio_losses = decoder_output_dict['curio_losses']
            [out_dict.update({'curio_loss': curio_losses[i]}) for i, out_dict in enumerate(out_list)]

        # Convert target boxes and get loss and analysis dictionaries (trainval only)
        if tgt_dict is not None:

            # Convert target boxes to normalized (cx, cy, width, height) format (trainval only)
            heights, widths = ((~images.mask[:, :, 0]).sum(dim=1), (~images.mask[:, 0, :]).sum(dim=1))
            tgt_sizes = tgt_dict['sizes'][1:] - tgt_dict['sizes'][:-1]
            tgt_batch_ids = torch.repeat_interleave(torch.arange(batch_size).to(sizes), tgt_sizes)

            heights, widths = (heights[tgt_batch_ids], widths[tgt_batch_ids])
            scales = torch.stack([widths, heights, widths, heights], dim=1)
            tgt_dict['boxes'] = box_xyxy_to_cxcywh(tgt_dict['boxes']) / scales

            # Get loss and analysis dictionaries (trainval only)
            loss_dict, analysis_dict = self.criterion(out_list, tgt_dict)

        # Get prediction dictionary (validation/testing only)
        if optimizer is None:
            pred_dict = self.make_predictions(out_list[0], kwargs['image_sizes'])

        # Return prediction dictionary
        if tgt_dict is None:
            return pred_dict

        # Return prediction, loss and analysis dictionaries (validation only)
        if optimizer is None:
            return pred_dict, loss_dict, analysis_dict

        # Reset gradients of model parameters (training only)
        optimizer.zero_grad()

        # Backpropagate loss (training only)
        loss = sum(loss_dict.values())
        loss.backward()

        # Clip gradient when positive maximum norm is provided (training only)
        if 'max_grad_norm' in kwargs:
            if kwargs['max_grad_norm'] > 0:
                clip_grad_norm_(self.parameters(), kwargs['max_grad_norm'])

        # Update model parameters (training only)
        optimizer.step()

        return loss_dict, analysis_dict


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
    criterion = build_criterion(args)

    train_projector = args.lr_projector > 0
    train_class_head = args.lr_class_head > 0
    train_bbox_head = args.lr_bbox_head > 0

    train_dict = {'projector': train_projector, 'class_head': train_class_head, 'bbox_head': train_bbox_head}
    metadata = args.train_metadata if args.dataset == 'coco' else None
    detr = DETR(backbone, position_encoder, encoder, decoder, criterion, args.num_classes, train_dict, metadata)

    return detr
