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
        1) we compute the hungarian matching between ground-truth boxes and the outputs of the model;
        2) we supervise each pair of matched ground-truth / prediction (both class and box).

    Attributes:
        num_classes (int): Number of object categories (without the no-object category).
        matcher (nn.Module): Module able to compute a matching between targets and proposals.
        weight_dict (Dict[str, float]): Dict mapping loss names to their relative weights.
        no_obj_weight (float): Relative classification weight applied to the no-object category.
        losses (List[str]): List with names of losses to be applied. See get_loss for list of available losses.
        class_weights (Tensor): Tensor of shape [num_classes + 1] with (relative) classification weights.
    """

    def __init__(self, num_classes, matcher, weight_dict, no_obj_weight, losses):
        """
        Initializes the SetCriterion module.

        Args:
            num_classes (int): Number of object categories (without the no-object category).
            matcher (nn.Module): Module able to compute a matching between targets and proposals.
            weight_dict (Dict[str, float]): Dict mapping loss names to their relative weights.
            no_obj_weight (float): Relative classification weight applied to the no-object category.
            losses (List[str]): List with names of losses to be applied. See get_loss for list of available losses.
        """

        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.no_obj_weight = no_obj_weight
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

        assert 'pred_logits' in outputs
        pred_logits = outputs['pred_logits']

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
        assert 'pred_boxes' in outputs
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
    def get_num_boxes(outputs, targets):
        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        return num_boxes

    def forward(self, outputs_list, targets):
        """
        Forward method of the SetCriterion module. Performs the loss computation.

        Args:
             outputs_list (List[Dict]): List of dicts, see the output specification of the model for the format.
             targets (List[Dict]): List of dicts, such that len(targets) == batch_size. The expected keys
                                   in each dict depends on the losses applied, see each loss doc.

        Returns:
            loss_dict (Dict[float]): Dict of different weighted losses, at different layers.
        """

        loss_dict = {}
        for layer_id, outputs in enumerate(outputs_list):
            indices = self.matcher(outputs, targets)
            num_boxes = self.get_num_boxes(outputs, targets)

            for loss in self.losses:
                loss_dict.update(self.get_loss(loss, outputs, targets, indices, num_boxes, layer_id))

        return loss_dict


def build_criterion(args):
    """
    Build criterion from command-line arguments.

    Args:
        args (argparse.Namespace): Command-line arguments.

    Returns:
        SetCriterion module.
    """

    matcher = build_matcher(args)
    weight_dict = {'class': args.loss_coeff_class, 'bbox': args.loss_coef_bbox, 'giou': args.loss_coef_giou}
    losses = ['labels', 'boxes', 'cardinality']

    return SetCriterion(args.num_classes, matcher, weight_dict, args.no_obj_weight, losses)
