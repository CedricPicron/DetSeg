"""
Criterion modules and build function.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.matcher import build_matcher
from utils.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from utils.distributed import get_world_size, is_dist_avail_and_initialized


class SetCriterion(nn.Module):
    """
    This class computes the loss for DETR-based models.

    The process happens in two steps:
        1) we compute the hungarian matching between the model predictions and ground-truth targets;
        2) we supervise each pair of matched model prediction and ground-truth target (both class and box).

    Attributes:
        num_classes (int): Number of object categories (without the no-object category).
        matcher (nn.Module): Module computing the matching between predictions and targets.
        weight_dict (Dict): Dict containing the weights or loss coefficients for the different loss terms.
        loss_functions (List): List of loss functions to be applied.
        analysis_names (List): List with names of analyses to be performed.
        class_weights (Tensor): Tensor of shape [num_classes + 1] with (relative) classification weights.
    """

    def __init__(self, num_classes, matcher, weight_dict, no_obj_weight, loss_names, analysis_names):
        """
        Initializes the SetCriterion module.

        Args:
            num_classes (int): Number of object categories (without the no-object category).
            matcher (nn.Module): Module computing the matching between predictions and targets.
            weight_dict (Dict): Dict containing the weights or loss coefficients for the different loss terms.
            no_obj_weight (float): Relative classification weight applied to the no-object category.
            loss_names (List): List with names of losses to be applied.
            analysis_names (List): List with names of analyses to be performed.
        """

        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict

        self.loss_functions = [getattr(self, f'loss_{name}') for name in loss_names]
        self.analysis_names = analysis_names

        class_weights = torch.ones(self.num_classes + 1)
        class_weights[-1] = no_obj_weight
        self.register_buffer('class_weights', class_weights)

    def loss_labels(self, pred_dict, tgt_dict, match_idx, *args):
        """
        Method computing the weighted cross-entropy classification loss.

        Args:
            pred_dict (Dict): Dictionary containing following keys:
                 - logits (FloatTensor): classification logits of shape [num_slots_total, num_classes];
                 - boxes (FloatTensor): normalized box coordinates of shape [num_slots_total, 4];
                 - batch_idx (IntTensor): batch indices of slots (in ascending order) of shape [num_slots_total];
                 - layer_id (int): integer corresponding to the decoder layer producing the predictions.

            tgt_dict (Dict): Dictionary containing following keys:
                 - labels (IntTensor): tensor of shape [num_target_boxes_total] (with num_target_boxes_total the total
                                       number of objects across batch entries) containing the target class indices;
                 - boxes (FloatTensor): tensor of shape [num_target_boxes_total, 4] with the target box coordinates;
                 - sizes (IntTensor): tensor of shape [batch_size+1] containing the cumulative sizes of batch entries.

            match_idx (Tuple): Tuple of (pred_idx, tgt_idx) with:
                - pred_idx (IntTensor): Chosen predictions of shape [sum(min(num_slots_batch, num_targets_batch))];
                - tgt_idx (IntTensor): Matching targets of shape [sum(min(num_slots_batch, num_targets_batch))].

        Returns:
            loss_dict (Dict): Dictionary containing the weighted cross-entropy classification loss.
            analysis_dict (Dict): Dictionary containing classification related analyses.
        """

        # Some renaming for code readability
        pred_logits = pred_dict['logits']
        num_slots_total = pred_logits.shape[0]
        device = pred_logits.device
        tgt_labels = tgt_dict['labels']
        pred_idx, tgt_idx = match_idx
        layer_id = pred_dict['layer_id']

        # Compute target classes
        tgt_classes = torch.full((num_slots_total,), self.num_classes, dtype=torch.int64, device=device)
        tgt_classes[pred_idx] = tgt_labels[tgt_idx]

        # Compute cross-entropy loss
        loss_class = F.cross_entropy(pred_logits, tgt_classes, self.class_weights)
        loss_dict = {f'loss_class_{layer_id}': self.weight_dict['class']*loss_class}

        # Perform classification related analyses
        with torch.no_grad():
            analysis_dict = {}

            # Compute predicted classes if required
            if 'accuracy' or 'cardinality' in self.analysis_names:
                pred_classes = torch.argmax(pred_logits, dim=-1)

            # Perform accuracy analysis if requested
            if 'accuracy' in self.analysis_names:
                correct_predictions = torch.eq(pred_classes, tgt_classes)
                accuracy = correct_predictions.sum().item()/len(correct_predictions)
                analysis_dict[f'accuracy_{layer_id}'] = 100*accuracy

            # Perform cardinality analysis if requested
            if 'cardinality' in self.analysis_names:
                pred_cardinality = (pred_classes != self.num_classes).sum().item()
                tgt_cardinality = len(tgt_labels)
                cardinality_error = abs(pred_cardinality-tgt_cardinality)
                analysis_dict[f'card_error_{layer_id}'] = cardinality_error

        return loss_dict, analysis_dict

    def loss_boxes(self, pred_dict, tgt_dict, match_idx, num_boxes):
        """
        Method computing the weighted L1 and GIoU bounding box losses.

        Args:
            pred_dict (Dict): Dictionary containing following keys:
                 - logits (FloatTensor): classification logits of shape [num_slots_total, num_classes];
                 - boxes (FloatTensor): normalized box coordinates of shape [num_slots_total, 4];
                 - batch_idx (IntTensor): batch indices of slots (in ascending order) of shape [num_slots_total];
                 - layer_id (int): integer corresponding to the decoder layer producing the predictions.

            tgt_dict (Dict): Dictionary containing following keys:
                 - labels (IntTensor): tensor of shape [num_target_boxes_total] (with num_target_boxes_total the total
                                       number of objects across batch entries) containing the target class indices;
                 - boxes (FloatTensor): tensor of shape [num_target_boxes_total, 4] with the target box coordinates;
                 - sizes (IntTensor): tensor of shape [batch_size+1] containing the cumulative sizes of batch entries.

            match_idx (Tuple): Tuple of (pred_idx, tgt_idx) with:
                - pred_idx (IntTensor): Chosen predictions of shape [sum(min(num_slots_batch, num_targets_batch))];
                - tgt_idx (IntTensor): Matching targets of shape [sum(min(num_slots_batch, num_targets_batch))].

            num_boxes (float): Average number of target boxes across all nodes.

        Returns:
            loss_dict (Dict): Dictionary containing the weighted L1 and GIoU bounding box losses.
            analysis_dict (Dict): Dictionary containing bounding box related analyses.
        """

        # Get the predicted and target boxes
        pred_idx, tgt_idx = match_idx
        pred_boxes = pred_dict['boxes'][pred_idx, :]
        tgt_boxes = tgt_dict['boxes'][tgt_idx, :]

        # Compute the L1 and GIoU bounding box losses
        loss_l1 = F.l1_loss(pred_boxes, tgt_boxes, reduction='sum')
        loss_giou = 1 - torch.diag(generalized_box_iou(box_cxcywh_to_xyxy(pred_boxes), box_cxcywh_to_xyxy(tgt_boxes)))

        # Populate loss_dict with the weighted bounding box losses
        loss_dict = {}
        layer_id = pred_dict['layer_id']
        loss_dict[f'loss_l1_{layer_id}'] = self.weight_dict['l1'] * (loss_l1 / num_boxes)
        loss_dict[f'loss_giou_{layer_id}'] = self.weight_dict['giou'] * (loss_giou.sum() / num_boxes)

        # Perform bounding box related analyses
        with torch.no_grad():
            analysis_dict = {}

        return loss_dict, analysis_dict

    @staticmethod
    def get_num_boxes(tgt_dict):
        """
        Computes the average number of target boxes across all nodes, for normalization purposes.

        Args:
            tgt_dict (Dict): Dictionary containing following keys:
                 - labels (IntTensor): tensor of shape [num_target_boxes_total] (with num_target_boxes_total the total
                                       number of objects across batch entries) containing the target class indices;
                 - boxes (FloatTensor): tensor of shape [num_target_boxes_total, 4] with the target box coordinates;
                 - sizes (IntTensor): tensor of shape [batch_size+1] containing the cumulative sizes of batch entries.

        Returns:
            num_boxes (float): Average number of target boxes across all nodes.
        """

        num_boxes = len(tgt_dict['labels'])
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=tgt_dict['labels'].device)

        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)

        world_size = get_world_size()
        num_boxes = torch.clamp(num_boxes/world_size, min=1).item()

        return num_boxes

    def forward(self, pred_list, tgt_dict):
        """
        Forward method of the SetCriterion module. Performs the loss computation.

        Args:
             pred_list (List): List of predictions, where each entry is a dict containing the keys:
                - logits (FloatTensor): class logits (with background) of shape [num_slots_total, (num_classes + 1)];
                - boxes (FloatTensor): normalized box coordinates (center_x, center_y, height, width) within non-padded
                                       regions, of shape [num_slots_total, 4];
                - batch_idx (IntTensor): batch indices of slots (sorted in ascending order) of shape [num_slots_total];
                - layer_id (int): integer corresponding to the decoder layer producing the predictions.

             tgt_dict (Dict): Dictionary containing following keys:
                 - labels (IntTensor): tensor of shape [num_target_boxes_total] (with num_target_boxes_total the total
                                       number of objects across batch entries) containing the target class indices;
                 - boxes (FloatTensor): tensor of shape [num_target_boxes_total, 4] with the target box coordinates;
                 - sizes (IntTensor): tensor of shape [batch_size+1] containing the cumulative sizes of batch entries.

        Returns:
            full_loss_dict (Dict): Dict of different weighted losses, at different layers, used for backpropagation.
            full_analysis_dict (Dict): Dict of different analyses, at different layers, used for logging purposes only.
        """

        full_loss_dict = {}
        full_analysis_dict = {}

        for layer_id, pred_dict in enumerate(pred_list):
            match_idx = self.matcher(pred_dict, tgt_dict)
            num_boxes = self.get_num_boxes(tgt_dict)

            for loss_function in self.loss_functions:
                loss_dict, analysis_dict = loss_function(pred_dict, tgt_dict, match_idx, num_boxes)
                full_loss_dict.update(loss_dict)
                full_analysis_dict.update(analysis_dict)

        return full_loss_dict, full_analysis_dict


def build_criterion(args):
    """
    Build criterion from command-line arguments.

    Args:
        args (argparse.Namespace): Command-line arguments.

    Returns:
        criterion (SetCriterion): The specified SetCriterion module.
    """

    matcher = build_matcher(args)
    weight_dict = {'class': args.loss_coef_class, 'l1': args.loss_coef_l1, 'giou': args.loss_coef_giou}
    loss_names = ['labels', 'boxes']
    analysis_names = ['accuracy', 'cardinality']
    criterion = SetCriterion(args.num_classes, matcher, weight_dict, args.no_obj_weight, loss_names, analysis_names)

    return criterion
