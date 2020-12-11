"""
Detection head modules and build function.
"""

from detectron2.layers import batched_nms
from detectron2.modeling.anchor_generator import DefaultAnchorGenerator
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.matcher import Matcher
from detectron2.modeling.meta_arch.retinanet import RetinaNetHead
from detectron2.structures.boxes import Boxes, pairwise_iou
from fvcore.nn import sigmoid_focal_loss, smooth_l1_loss
import torch
from torch import nn
import torch.nn.functional as F

from utils.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_xywh


class RetinaHead(nn.Module):
    """
    Implements the RetinaHead module.
    """

    def __init__(self, in_feat_sizes, num_classes, pred_head_dict, matcher_dict, loss_dict, inference_dict):
        """
        Initializes the RetinaHead module.
        """

        # Initialization of default nn.Module
        super().__init__()

        # Initialization of anchor generator
        anchor_sizes = [[2**(i+2), 2**(i+2) * 2**(1.0/3), 2**(i+2) * 2**(2.0/3)] for i in in_feat_sizes.keys()]
        anchor_aspect_ratios = [[0.5, 1.0, 2.0]]
        anchor_strides = [2**i for i in in_feat_sizes.keys()]
        self.anchor_generator = DefaultAnchorGenerator(anchor_sizes, anchor_aspect_ratios, anchor_strides)

        # Initialization of anchor matcher
        matcher_thresholds = [0.4, 0.5]
        matcher_labels = [0, -1, 1]
        self.anchor_matcher = Matcher(matcher_thresholds, matcher_labels, allow_low_quality_matches=True)

        # Initialization of box to box transform object
        self.box_to_box = Box2BoxTransform(weights=(1.0, 1.0, 1.0, 1.0))

        # Initialization of linear input projection modules
        self.in_projs = nn.ModuleList(nn.Linear(f, pred_head_dict['feat_size']) for f in in_feat_sizes.values())

        # Initialization of RetinaNet prediction head module
        num_anchors = self.anchor_generator.num_cell_anchors[0]
        conv_dims = [pred_head_dict['feat_size']] * pred_head_dict['num_convs']
        self.pred_head = RetinaNetHead(num_classes=num_classes, num_anchors=num_anchors, conv_dims=conv_dims)

        # Initialization of loss attributes
        self.focal_alpha = loss_dict['focal_alpha']
        self.focal_gamma = loss_dict['focal_gamma']
        self.smooth_l1_beta = loss_dict['smooth_l1_beta']

        self.loss_normalizer = loss_dict['normalizer']
        self.loss_momentum = loss_dict['momentum']
        self.loss_weight = loss_dict['weight']

        # Initialization of inference attributes
        self.test_score_threshold = inference_dict['score_threshold']
        self.test_num_candidates = inference_dict['num_candidates']
        self.test_nms_threshold = inference_dict['nms_threshold']
        self.test_max_detections = inference_dict['max_detections']

    @staticmethod
    def reshape_pred_map(pred_map, pred_size):
        """
        Reshapes a prediction map from the module's prediction head.

        Args:
            pred_map (FloatTensor): Prediction map of shape [batch_size, num_anchors*pred_size, fH, fW].
            pred_size (int): Size of a single prediction (i.e. corresponding to a single anchor and map position).

        Returns:
            pred_map (FloatTensor): Reshaped prediction map of shape [batch_size, fH*fW*num_anchors, pred_size].
        """

        batch_size, _, fH, fW = pred_map.shape
        pred_map = pred_map.view(batch_size, -1, pred_size, fH, fW).permute(0, 3, 4, 1, 2)
        pred_map = pred_map.reshape(batch_size, -1, pred_size)

        return pred_map

    @torch.no_grad()
    def forward_init(self, feat_maps, tgt_dict=None):
        """
        Forward initialization method of the RetinaHead module.

        It sets the 'anchors' attribute, which is used when making predictions (see 'make_predictions' method).
        It also updates the 'loss_normalizer' attribute when 'tgt_dict' is not None (i.e. during training/validation).

        Args:
            feat_maps (List): List of size [num_maps] with feature maps of shape [batch_size, fH, fW, feat_size].
            tgt_dict (Dict): Optional target dictionary used during trainval containing at least following keys:
                - labels (LongTensor): tensor of shape [num_targets_total] containing the class indices;
                - boxes (FloatTensor): boxes of shape [num_targets_total, 4] in (cx, cy, width, height) format;
                - sizes (LongTensor): tensor of shape [batch_size+1] with the cumulative target sizes of batch entries.

        Returns:
            * If tgt_dict is not None (i.e. during training and validation):
                tgt_dict (Dict): Updated target dictionary containing following additional keys:
                    - anchor_labels (LongTensor): tensor of class indices of shape [batch_size, num_anchors_total];
                    - anchor_deltas (FloatTensor): anchor to box deltas of shape [batch_size, num_anchors_total, 4].

            * If tgt_dict is None (i.e. during testing), it returns None (after setting the 'anchors' attribute).
        """

        # Generate anchors and set anchors attribute
        feat_map_views = [feat_map.permute(0, 3, 1, 2) for feat_map in feat_maps]
        self.anchors = self.anchor_generator(feat_map_views)

        # Return when no target dictionary is provided (validation/testing only)
        if tgt_dict is None:
            return

        # Some preparation before anchor labeling (trainval only)
        anchors = Boxes.cat(self.anchors)
        tgt_sizes = tgt_dict['sizes']

        anchor_labels_list = []
        anchor_deltas_list = []

        # Label anchors for every batch entry (trainval only)
        for i0, i1 in zip(tgt_sizes[:-1], tgt_sizes[1:]):
            tgt_labels = tgt_dict['labels'][i0:i1]
            tgt_boxes = Boxes(box_cxcywh_to_xyxy(tgt_dict['boxes'][i0:i1]))
            matched_ids, match_labels = self.anchor_matcher(pairwise_iou(tgt_boxes, anchors))

            if len(tgt_labels) > 0:
                anchor_labels = tgt_labels[matched_ids]
                anchor_labels[match_labels == 0] = self.num_classes
                anchor_labels[match_labels == -1] = -1
                anchor_labels_list.append(anchor_labels)

                anchor_deltas = self.box_to_box.get_deltas(anchors.tensor, tgt_boxes.tensor[matched_ids])
                anchor_deltas_list.append(anchor_deltas)

            else:
                anchor_labels_list.append(torch.full_like(matched_ids, self.num_classes))
                anchor_deltas_list.append(torch.zeros_like(anchors.tensor))

        # Get batched anchor labels and deltas (trainval only)
        anchor_labels = torch.stack(anchor_labels_list)
        anchor_deltas = torch.stack(anchor_deltas_list)

        # Update loss normalizer attribute (trainval only)
        num_pos_anchors = ((anchor_labels >= 0) & (anchor_labels != self.num_classes)).sum().item()
        self.loss_normalizer *= self.loss_momentum
        self.loss_normalizer += (1 - self.loss_momentum) * max(num_pos_anchors, 1)

        # Add batched anchor labels and deltas to target dictionary
        tgt_dict['anchor_labels'] = anchor_labels
        tgt_dict['anchor_deltas'] = anchor_deltas

        return tgt_dict

    def make_predictions(self, logit_maps, delta_maps):
        """
        Make classified bounding box predictions based on given logit and delta maps.

        Args:
            logit_maps (List): Maps [num_maps] with logits of shape [batch_size, fH*fW*num_anchors, num_classes].
            delta_maps (List): Maps [num_maps] with box deltas shape [batch_size, fH*fW*num_anchors, 4].

        Returns:
            pred_dict (Dict): Prediction dictionary containing following keys:
                - labels (LongTensor): predicted class indices of shape [num_preds_total];
                - boxes (FloatTensor): predicted boxes of shape [num_preds_total, 4] in (left, top, width, height);
                - scores (FloatTensor): normalized prediction scores of shape [num_preds_total];
                - batch_ids (LongTensor): batch indices of predictions of shape [num_preds_total].
        """

        # Initialize list of predicted labels and boxes
        batch_size = len(logit_maps[0])
        pred_labels = [[] for _ in range(batch_size)]
        pred_boxes = [[] for _ in range(batch_size)]
        pred_scores = [[] for _ in range(batch_size)]

        # Iterate over maps and images
        for logit_map, delta_map, map_anchors in zip(logit_maps, delta_maps, self.anchors):
            for i, img_logit_map, img_delta_map in zip(range(batch_size), logit_map, delta_map):

                # Flatten image logit and delta_maps
                logits = img_logit_map.flatten()
                deltas = img_delta_map.flatten()

                # Get class probabilities
                probs = logits.sigmoid_()

                # Filter predictions: Absolute (only keep predictions of sufficient confidence)
                abs_keep = probs > self.test_score_thresh
                probs = probs[abs_keep]

                # Filter predictions: Relative (only keep top predictions)
                num_preds_kept = min(self.test_topk_candidates, len(probs))
                probs, sort_ids = probs.sort(descending=True)
                top_scores = probs[:num_preds_kept]

                # Get ids of top predictions
                top_ids = torch.nonzero(abs_keep, as_tuple=True)[0]
                top_ids = top_ids[sort_ids[:num_preds_kept]]

                # Get labels of top predictions
                top_labels = top_ids % self.num_classes

                # Get boxes of top predictions
                anchor_ids = top_ids // self.num_classes
                top_deltas = deltas[anchor_ids]
                top_anchors = map_anchors.tensor[anchor_ids]
                top_boxes = self.box_to_box.apply_deltas(top_deltas, top_anchors)

                # Add labels, boxes and scores of top predictions to their respective lists
                pred_labels[i].append(top_labels)
                pred_boxes[i].append(top_boxes)
                pred_scores[i].append(top_scores)

        # Initialize prediction dictionary
        pred_dict = {}
        pred_dict['labels'] = []
        pred_dict['boxes'] = []
        pred_dict['scores'] = []
        pred_dict['batch_ids'] = []

        # Get final predictions for every image
        for i in range(batch_size):

            # Concatenate predictions from different feature maps
            labels = torch.cat(pred_labels[i], dim=0)
            boxes = torch.cat(pred_boxes[i], dim=0)
            scores = torch.cat(pred_scores[i], dim=0)

            # Keep best predictions after non-maxima suppression (NMS)
            keep = batched_nms(boxes, scores, labels, iou_threshold=self.test_nms_thresholds)
            keep = keep[:self.test_max_detections]

            # Add final predictions to the prediction dictionary
            pred_dict['labels'].append(labels[keep])
            pred_dict['boxes'].append(box_xyxy_to_xywh(boxes[keep]))
            pred_dict['scores'].append(scores[keep])
            pred_dict['batch_ids'].append(torch.full((len(keep),), i, device=labels.device))

        # Concatenate different image predictions into single tensor
        pred_dict = {k: torch.cat(v, dim=0) for k, v in pred_dict.items()}

        return pred_dict

    @staticmethod
    def get_accuracy(preds, targets):
        """
        Method returning the accuracy of the given predictions compared to the given targets.

        Args:
            preds (LongTensor): Tensor of shape [*] (same as targets) containing the predicted class indices.
            targets (LongTensor): Tensor of shape [*] (same as preds) containing the target class indices.

        Returns:
            accuracy (FloatTensor): Tensor of shape [] containing the prediction accuracy (between 0 and 1).
        """

        # Get boolean tensor indicating correct and incorrect predictions
        pred_correctness = torch.eq(preds, targets)

        # Compute accuracy
        if pred_correctness.numel() > 0:
            accuracy = pred_correctness.sum() / float(pred_correctness.numel())
        else:
            accuracy = torch.tensor(1.0).to(pred_correctness)

        return accuracy

    def perform_accuracy_analyses(self, preds, targets):
        """
        Method performing accuracy-related analyses.

        Args:
            preds (LongTensor): Tensor of shape [*] (same as targets) containing the predicted class indices.
            targets (LongTensor): Tensor of shape [*] (same as preds) containing the target class indices.

        Returns:
            analysis_dict (Dict): Dictionary of accuracy-related analyses containing following keys:
                - ret_cls_acc (FloatTensor): accuracy of the retina head classification of shape [];
                - ret_cls_acc_bg (FloatTensor): background accuracy of the retina head classification of shape [];
                - ret_cls_acc_obj (FloatTensor): object accuracy of the retina head classification of shape [].
        """

        # Compute general accuracy and place it into analysis dictionary
        accuracy = RetinaHead.get_accuracy(preds, targets)
        analysis_dict = {'ret_cls_acc': 100*accuracy}

        # Compute background accuracy and place it into analysis dictionary
        bg_mask = targets == self.num_classes
        bg_accuracy = RetinaHead.get_accuracy(preds[bg_mask], targets[bg_mask])
        analysis_dict['ret_cls_acc_bg'] = 100*bg_accuracy

        # Compute object accuracy and place it into analysis dictionary
        obj_mask = targets < self.num_classes
        obj_accuracy = RetinaHead.get_accuracy(preds[obj_mask], targets[obj_mask])
        analysis_dict['ret_cls_acc_obj'] = 100*obj_accuracy

        return analysis_dict

    def forward(self, feat_maps, feat_masks=None, tgt_dict=None, **kwargs):
        """
        Forward method of the RetinaHead module.

        Args:
            feat_maps (List): List of size [num_maps] with feature maps of shape [batch_size, fH, fW, feat_size].
            feat_masks (List): Optional list of size [num_maps] with padding masks of shape [batch_size, fH, fW].

            tgt_dict (Dict): Optional target dictionary containing at least following keys:
                - anchor_labels (LongTensor): tensor of class indices of shape [batch_size, num_anchors_total];
                - anchor_deltas (FloatTensor): anchor to box deltas of shape [batch_size, num_anchors_total, 4].

            kwargs(Dict): Dictionary of keyword arguments, potentially containing following key:
                - extended_analysis (bool): boolean indicating whether to perform extended analyses or not.

        Returns:
            * If tgt_dict is not None (i.e. during training and validation):
                loss_dict (Dict): Loss dictionary containing following keys:
                    - ret_cls_loss (FloatTensor): weighted classification loss of shape [];
                    - ret_box_loss (FloatTensor): weighted bounding box regression loss of shape [].

                analysis_dict (Dict): Analysis dictionary at least containing following keys:
                    - ret_cls_acc (FloatTensor): accuracy of the retina head classification of shape [];
                    - ret_cls_acc_bg (FloatTensor): background accuracy of the retina head classification of shape [];
                    - ret_cls_acc_obj (FloatTensor): object accuracy of the retina head classification of shape [].

            * If tgt_dict is None (i.e. during testing and possibly during validation):
                pred_dict (Dict): Prediction dictionary containing following keys:
                    - labels (LongTensor): predicted class indices of shape [num_preds_total];
                    - boxes (FloatTensor): predicted boxes of shape [num_preds_total, 4] in (left, top, width, height);
                    - scores (FloatTensor): normalized prediction scores of shape [num_preds_total];
                    - batch_ids (LongTensor): batch indices of predictions of shape [num_preds_total].
        """

        # Assume no padded regions when feature masks are missing (trainval only)
        if feat_masks is None and tgt_dict is not None:
            tensor_kwargs = {'dtype': torch.bool, 'device': feat_maps[0].device}
            feat_masks = [torch.zeros(*feat_map.shape[:-1], **tensor_kwargs) for feat_map in feat_maps]

        # Project feature maps to common feature space and permute to convolution format
        feat_maps = [in_proj(feat_map).permute(0, 3, 1, 2) for feat_map, in_proj in zip(feat_maps, self.in_projs)]

        # Predict logits and anchor regression deltas
        logit_maps, delta_maps = self.pred_head(feat_maps)
        logit_maps = [RetinaHead.reshape_pred_map(logit_map, self.num_classes) for logit_map in logit_maps]
        delta_maps = [RetinaHead.reshape_pred_map(delta_map, 4) for delta_map in delta_maps]

        # Compute and return prediction dictionary (validation/testing only)
        if tgt_dict is None:
            pred_dict = self.make_predictions(logit_maps, delta_maps)
            return pred_dict

        # Compute weighted classification loss (trainval only)
        anchor_labels = tgt_dict['anchor_labels']
        class_mask = anchor_labels >= 0

        logits = torch.cat(logit_maps, dim=1)
        class_logits = logits[class_mask]

        class_targets = F.one_hot(anchor_labels[class_mask], num_classes=self.num_classes+1)
        class_targets = class_targets[:, :-1].to(class_logits.dtype)

        class_kwargs = {'alpha': self.focal_alpha, 'gamma': self.focal_gamma, 'reduction': 'sum'}
        class_loss = sigmoid_focal_loss(class_logits, class_targets, **class_kwargs)
        class_loss = self.loss_weight * class_loss / self.loss_normalizer

        # Compute weighted bounding box regression loss (trainval only)
        box_mask = class_mask & (anchor_labels != self.num_classes)
        box_preds = torch.cat(delta_maps, dim=1)[box_mask]
        box_targets = tgt_dict['anchor_deltas'][box_mask]

        box_kwargs = {'beta': self.smooth_l1_beta, 'reduction': 'sum'}
        box_loss = smooth_l1_loss(box_preds, box_targets, **box_kwargs)
        box_loss = self.loss_weight * box_loss / self.loss_normalizer

        # Place weighted losses into loss dictionary (trainval only)
        loss_dict = {'ret_cls_loss': class_loss, 'ret_box_loss': box_loss}

        # Perform analyses (trainval only)
        with torch.no_grad():

            # Perform classification accuracy analyses (trainval only)
            num_anchors = self.anchor_generator.num_cell_anchors[0]
            feat_masks = [feat_mask.expand(-1, -1, -1, num_anchors).flatten(1) for feat_mask in feat_masks]
            padding_mask = torch.cat(feat_masks, dim=1)

            acc_mask = ~padding_mask & class_mask
            class_preds = torch.argmax(logits, dim=-1)
            analysis_dict = self.perform_accuracy_analyses(class_preds[acc_mask], anchor_labels[acc_mask])

            # If requested, perform extended analyses (trainval only)
            if kwargs.setdefault('extended_analysis', False):

                # Perform map-specific accuracy analyses (trainval only)
                map_sizes = [logit_map.shape[1] for logit_map in logit_maps]
                indices = torch.cumsum(torch.tensor([0, *map_sizes], device=logits.device), dim=0)

                for i, i0, i1 in zip(range(len(logit_maps)), indices[:-1], indices[1:]):
                    map_acc_mask = acc_mask[:, i0:i1]
                    map_preds = class_preds[:, i0:i1][map_acc_mask]
                    map_targets = anchor_labels[:, i0:i1][map_acc_mask]

                    map_analysis_dict = self.perform_accuracy_analyses(map_preds, map_targets)
                    analysis_dict.update({f'{k}_f{i}': v for k, v in map_analysis_dict.items()})

        return loss_dict, analysis_dict


def build_det_heads(args):
    """
    Build detection head modules from command-line arguments.

    Args:
        args (argparse.Namespace): Command-line arguments.

    Returns:
        det_heads (List): List of specified detection head modules.

    Raises:
        ValueError: Error when unknown detection head type was provided.
    """

    # Get dictionary of feature sizes
    map_ids = range(args.min_resolution_id, args.max_resolution_id+1)
    feat_sizes = {i: min((args.base_feat_size * 2**i, args.max_feat_size)) for i in map_ids}

    # Initialize empty list of detection head modules
    det_heads = []

    # Build desired detection head modules
    for det_head_type in args.det_heads:
        if det_head_type == 'retina':
            retina_head = RetinaHead(feat_sizes)
            det_heads.append(retina_head)

        else:
            raise ValueError(f"Unknown detection head type '{det_head_type}' was provided.")

    return det_heads
