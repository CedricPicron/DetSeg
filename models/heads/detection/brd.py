"""
Base Reinforced Detector (BRD) head.
"""
from copy import deepcopy

import torch
from torch import nn

from models.functional.position import sine_pos_encodings
from models.modules.attention import SelfAttn1d
from models.modules.mlp import FFN, MLP
from models.modules.policy import PolicyNet
from structures.boxes import Boxes, box_giou


class BRD(nn.Module):
    """
    Class implementing the Base Reinforced Detector (BRD) module.

    Attributes:
        policy (PolicyNet): Policy network computing action masks and initial action losses.
        decoder (nn.Sequential): Sequence of decoder layers, with each layer having a self-attention and FFN operation.
        cls_head (MLP): Module computing the classification logits from object features.
        box_head (MLP): Module computing the bounding box predictions from object features.
    """

    def __init__(self, feat_size, policy_dict, decoder_dict, head_dict):
        """
        Initializes the BRD module.

        Args:
            feat_size (int): Integer containing the feature size.

            policy_dict (Dict): Policy dictionary, potentially containing following keys:
                - num_hidden_layers (int): number of hidden layers of the policy head;
                - inference_samples (int): maximum number of samples during inference;
                - num_groups (int): number of groups used for group normalization.

            decoder_dict (Dict): Decoder dictionary containing following keys:
                - num_heads (int): number of attention heads used during the self-attention operation;
                - hidden_size (int): integer containing the hidden feature size used during the FFN operation;
                - num_layers (int): number of consecutive decoder layers.

            head_dict (Dict): Head dictionary containing following keys:
                - num_classes (int): integer containing the number of object classes (without background);
                - hidden_size (int): integer containing the hidden feature size used during the MLP operation;
                - num_hidden_layers (int): number of hidden layers of the MLP head.
        """

        # Initialization of default nn.Module
        super().__init__()

        # Initialize policy network
        self.policy = PolicyNet(feat_size, input_type='pyramid', **policy_dict)

        # Initialize decoder
        self_attn = SelfAttn1d(feat_size, decoder_dict['num_heads'])
        ffn = FFN(feat_size, decoder_dict['hidden_size'])
        decoder_layer = nn.Sequential(self_attn, ffn)

        num_decoder_layers = decoder_dict['num_layers']
        self.decoder = nn.Sequential(*[deepcopy(decoder_layer) for _ in range(num_decoder_layers)])

        # Initialize classification and bounding box heads
        num_classes = head_dict['num_classes']
        self.cls_head = MLP(feat_size, out_size=num_classes, **head_dict)
        self.box_head = MLP(feat_size, out_size=4, **head_dict)

    def forward_init(self, images, feat_maps, tgt_dict=None):
        """
        Forward initialization method of the BRD module.

        Args:
            images (Images): Images structure containing the batched images.
            feat_maps (List): List of size [num_maps] with feature maps of shape [batch_size, feat_size, fH, fW].

            tgt_dict (Dict): Optional target dictionary used during trainval containing at least following keys:
                - labels (LongTensor): tensor of shape [num_targets_total] containing the class indices;
                - boxes (Boxes): structure containing axis-aligned bounding boxes of size [num_targets_total];
                - sizes (LongTensor): tensor of shape [batch_size+1] with the cumulative target sizes of batch entries.

        Returns:
            * If tgt_dict is not None (i.e. during training and validation):
                tgt_dict (Dict): Updated target dictionary with following updated keys:
                    - labels (List): list of size [batch_size] with class indices of shape [num_targets];
                    - boxes (List): list of size [batch_size] with normalized Boxes structure of size [num_targets].

                attr_dict (Dict): Empty dictionary.
                buffer_dict (Dict): Empty dictionary.

            * If tgt_dict is None (i.e. during testing):
                tgt_dict (None): Contains the None value.
                attr_dict (Dict): Empty dictionary.
                buffer_dict (Dict): Empty dictionary.
        """

        # Return when no target dictionary is provided (testing only)
        if tgt_dict is None:
            return None, {}, {}

        # Get normalized bounding boxes
        norm_boxes = tgt_dict['boxes'].normalize(images)

        # Update target dictionary
        sizes = tgt_dict['sizes']
        tgt_dict['labels'] = [tgt_dict['labels'][i0:i1] for i0, i1 in zip(sizes[:-1], sizes[1:])]
        tgt_dict['boxes'] = [norm_boxes[i0:i1] for i0, i1 in zip(sizes[:-1], sizes[1:])]

        return tgt_dict, {}, {}

    def get_loss(self, action_losses, cls_logits, pred_boxes, tgt_labels, tgt_boxes):
        """
        Get BRD loss and its corresponding analysis.

        Args:
            action_losses (List): List of size [batch_size] with initial action losses of shape [train_samples].
            cls_logits (List): List of size [batch_size] with classification logits of shape [num_preds, num_classes].
            pred_boxes (List): List of size [batch_size] with bounding box predictions of shape [num_preds, 4].
            tgt_labels (List): List of size [batch_size] with class indices of shape [num_targets].
            tgt_boxes (List): List of size [batch_size] with normalized Boxes structure of size [num_targets].

        Returns:
            loss_dict (Dict): Loss dictionary.
            analysis_dict (Dict): Analysis dictionary.

        Raises:
            ValueError: Raised when target boxes are not normalized.
        """

        # Get batch size
        batch_size = len(cls_logits)

        # Get loss and analysis for every batch entry
        for i in range(batch_size):

            # Prepare boxes
            pred_boxes_i = Boxes(pred_boxes[i], format='cxcywh', normalized=True)
            tgt_boxes_i = tgt_boxes[i].to_format('cxcywh')

            # Check whether target boxes are normalized
            if not tgt_boxes.normalized:
                raise ValueError("Target boxes should be normalized when using BRD loss.")

            # Get matching cost matrix
            with torch.no_grad():

                # Get classification cost matrix
                cls_probs = cls_logits[i].sigmoid()
                cls_cost = -self.match_cls_weight * cls_probs[:, tgt_labels[i]]

                # Get L1 and GIOU bounding box cost matrices
                l1_cost = self.match_l1_weight * torch.cdist(pred_boxes_i.boxes, tgt_boxes_i.boxes, p=1)
                giou_cost = -self.match_giou_weight * box_giou(pred_boxes_i, tgt_boxes_i)

                # Get full cost matrix
                cost_matrix = cls_cost + l1_cost + giou_cost

                return cost_matrix

    @staticmethod
    def make_predictions(cls_logits, pred_boxes):
        """
        Make classified bounding box predictions based on given classification and bounding box predictions.

        Args:
            cls_logits (List): List of size [batch_size] with classification logits of shape [num_preds, num_classes].
            pred_boxes (List): List of size [batch_size] with bounding box predictions of shape [num_preds, 4].

        Returns:
            pred_dict (Dict): Prediction dictionary containing following keys:
                - labels (LongTensor): predicted class indices of shape [num_preds_total];
                - boxes (Boxes): structure containing axis-aligned bounding boxes of size [num_preds_total];
                - scores (FloatTensor): normalized prediction scores of shape [num_preds_total];
                - batch_ids (LongTensor): batch indices of predictions of shape [num_preds_total].
        """

        # Initialize prediction dictionary
        pred_dict = {}
        pred_dict['labels'] = []
        pred_dict['boxes'] = []
        pred_dict['scores'] = []
        pred_dict['batch_ids'] = []

        # Get batch size and general box information
        batch_size = len(cls_logits)
        box_kwargs = {'format': 'cxcywh', 'normalized': True}

        # Add predictions to prediction dictionary
        for i in range(batch_size):
            scores, labels = cls_logits[i].sigmoid().max(dim=1)
            boxes = Boxes(pred_boxes[i], **box_kwargs)
            batch_ids = torch.full_like(labels, i)

            pred_dict['labels'].append(labels)
            pred_dict['boxes'].append(boxes)
            pred_dict['scores'].append(scores)
            pred_dict['batch_ids'].append(batch_ids)

        # Concatenate predictions across batch entries
        pred_dict.update({k: torch.cat(v, dim=0) for k, v in pred_dict.items() if k != 'boxes'})
        pred_dict['boxes'] = Boxes.cat(pred_dict['boxes'])

        return pred_dict

    def forward(self, feat_maps, feat_masks=None, tgt_dict=None, **kwargs):
        """
        Forward method of the BRD module.

        Args:
            feat_maps (List): List of size [num_maps] with feature maps of shape [batch_size, feat_size, fH, fW].
            feat_masks (List): Optional list [num_maps] with masks of active features of shape [batch_size, fH, fW].

            tgt_dict (Dict): Optional target dictionary containing at least following keys:
                - labels (List): list of size [batch_size] with class indices of shape [num_targets];
                - boxes (List): list of size [batch_size] with normalized Boxes structure of size [num_targets].

        Returns:
            * If tgt_dict is not None (i.e. during training and validation):
                loss_dict (Dict): Loss dictionary.
                analysis_dict (Dict): Analysis dictionary.

            * If tgt_dict is None (i.e. during testing and possibly during validation):
                pred_dict (Dict): Prediction dictionary containing following keys:
                    - labels (LongTensor): predicted class indices of shape [num_preds_total];
                    - boxes (Boxes): structure containing axis-aligned bounding boxes of size [num_preds_total];
                    - scores (FloatTensor): normalized prediction scores of shape [num_preds_total];
                    - batch_ids (LongTensor): batch indices of predictions of shape [num_preds_total].
        """

        # Assume no padded regions when feature masks are missing
        if feat_masks is None:
            tensor_kwargs = {'dtype': torch.bool, 'device': feat_maps[0].device}
            feat_masks = [torch.ones(*feat_map[:, 0].shape, **tensor_kwargs) for feat_map in feat_maps]

        # Get position-augmented features
        pos_maps = sine_pos_encodings((feat_maps, feat_masks), input_type='pyramid')
        aug_maps = [feat_map+pos_map for feat_map, pos_map in zip(feat_maps, pos_maps)]
        aug_feats = torch.cat([aug_map.flatten(2).permute(0, 2, 1) for aug_map in aug_maps], dim=1)

        # Apply policy network to obtain object features
        if tgt_dict is not None:
            sample_masks, action_losses = self.policy(feat_maps, mode='training')
            obj_feats = [aug_feats[i][sample_masks[i]] for i in range(len(aug_feats))]

        else:
            sample_ids = self.policy(feat_maps, mode='inference')
            obj_feats = [aug_feats[i][sample_ids[i]] for i in range(aug_feats)]

        # Process object features with decoder
        obj_feats = self.decoder(obj_feats)

        # Get classification logits and bounding box predictions
        cls_logits = self.cls_head(obj_feats)
        pred_boxes = self.box_head(obj_feats)

        # Get loss and analysis dictionaries during trainval
        if tgt_dict is not None:
            tgt_labels = tgt_dict['labels']
            tgt_boxes = tgt_dict['boxes']
            loss_dict, analysis_dict = self.get_loss(action_losses, cls_logits, pred_boxes, tgt_labels, tgt_boxes)

            return loss_dict, analysis_dict

        # Get prediction dictionary validation/testing
        else:
            pred_dict = BRD.make_predictions(cls_logits, pred_boxes)

            return pred_dict
