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
        losses (List): List with names of losses to be applied. See get_loss for list of available losses.
        class_weights (Tensor): Tensor of shape [num_classes + 1] with (relative) classification weights.
    """

    def __init__(self, num_classes, matcher, weight_dict, no_obj_weight, losses):
        """
        Initializes the SetCriterion module.

        Args:
            num_classes (int): Number of object categories (without the no-object category).
            matcher (nn.Module): Module computing the matching between predictions and targets.
            weight_dict (Dict): Dict containing the weights or loss coefficients for the different loss terms.
            no_obj_weight (float): Relative classification weight applied to the no-object category.
            losses (List[str]): List with names of losses to be applied. See get_loss for list of available losses.
        """

        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses

        class_weights = torch.ones(self.num_classes + 1)
        class_weights[-1] = self.no_obj_weight
        self.register_buffer('class_weights', class_weights)

    @staticmethod
    @torch.no_grad()
    def accuracy(output, target, topk=(1,)):
        """Computes the precision@k for the specified values of k"""

        if target.numel() == 0:
            return [torch.zeros([], device=output.device)]

        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))

        return res

    def loss_labels(self, outputs, targets, indices, num_boxes, layer_id, log=True):
        """
        Classification loss (NLL).

        Targets dicts must contain the key "labels" containing a tensor of shape [num_target_boxes].
        """

        assert 'logits' in outputs
        pred_logits = outputs['logits']

        batch_idx, pred_idx = self._get_src_permutation_idx(indices)
        tgt_classes_o = torch.cat([t['labels'][J] for t, (_, J) in zip(targets, indices)])

        tgt_classes = torch.full(pred_logits.shape[:2], self.num_classes, dtype=torch.int64, device=pred_logits.device)
        tgt_classes[batch_idx, pred_idx] = tgt_classes_o

        loss_ce = F.cross_entropy(pred_logits.transpose(1, 2), tgt_classes, self.class_weights)
        losses = {f'loss_class_{layer_id}': self.weight_dict['class']*loss_ce}
        losses[f'class_error_{layer_id}'] = 100 - self.accuracy(pred_logits[batch_idx, pred_idx], tgt_classes_o)[0]

        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes, layer_id):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {f'cardinality_error_{layer_id}': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes, layer_id):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        losses = {}
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        losses[f'loss_bbox_{layer_id}'] = self.weight_dict['bbox'] * (loss_bbox.sum() / num_boxes)

        loss_giou = 1 - torch.diag(generalized_box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes)))
        losses[f'loss_giou_{layer_id}'] = self.weight_dict['giou'] * (loss_giou.sum() / num_boxes)

        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, layer_id, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, layer_id, **kwargs)

    @staticmethod
    def get_num_boxes(tgt_list):
        """
        Computes the average number of target boxes across all nodes, for normalization purposes.

        Args:
            tgt_list (List): List of targets of shape[batch_size], where each entry is a dict containing the keys:
                - labels (IntTensor): tensor of dim [num_target_boxes] (where num_target_boxes is the number of
                                      ground-truth objects in the target) containing the ground-truth class indices;
                - boxes (FloatTensor): tensor of dim [num_target_boxes, 4] containing the target box coordinates.

        Returns:
            num_boxes (float): Average number of target boxes across all nodes.
        """

        num_boxes = sum(len(t['labels']) for t in tgt_list)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=tgt_list[0]['boxes'].device)

        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)

        world_size = get_world_size()
        num_boxes = torch.clamp(num_boxes/world_size, min=1).item()

        return num_boxes

    def forward(self, pred_list, tgt_list):
        """
        Forward method of the SetCriterion module. Performs the loss computation.

        Args:
             pred_list (List): List of predictions, where each entry is a dict containing the key:
                 - logits (FloatTensor): the class logits (with background) of shape [num_slots, (num_classes + 1)];
                 - boxes (FloatTensor): the normalized box coordinates (center_x, center_y, height, width) within
                                        padded images, of shape [num_slots, 4];
                 - batch_idx (IntTensor): batch indices of slots (sorted in ascending order) of shape [num_slots].
             tgt_list (List): List of targets of shape[batch_size], where each entry is a dict containing the keys:
                 - labels (IntTensor): tensor of dim [num_target_boxes] (where num_target_boxes is the number of
                                       ground-truth objects in the target) containing the ground-truth class indices;
                 - boxes (FloatTensor): tensor of dim [num_target_boxes, 4] containing the target box coordinates.

        Returns:
            loss_dict (Dict): Dict of different weighted losses, at different layers.
        """

        loss_dict = {}
        for layer_id, pred_dict in enumerate(pred_list):
            idx = self.matcher(pred_dict, tgt_list)
            num_boxes = self.get_num_boxes(tgt_list)

            for loss in self.losses:
                loss_dict.update(self.get_loss(loss, pred_dict, tgt_list, idx, num_boxes, layer_id))

        return loss_dict


def build_criterion(args):
    """
    Build criterion from command-line arguments.

    Args:
        args (argparse.Namespace): Command-line arguments.

    Returns:
        criterion (SetCriterion): The specified SetCriterion module.
    """

    matcher = build_matcher(args)
    weight_dict = {'class': args.loss_coef_class, 'bbox': args.loss_coef_bbox, 'giou': args.loss_coef_giou}
    losses = ['labels', 'boxes', 'cardinality']
    criterion = SetCriterion(args.num_classes, matcher, weight_dict, args.no_obj_weight, losses)

    return criterion
