"""
Detection head modules and build function.
"""

from detectron2.modeling.anchor_generator import DefaultAnchorGenerator
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.matcher import Matcher
from detectron2.modeling.meta_arch.retinanet import RetinaNetHead
from detectron2.structures.boxes import Boxes, pairwise_iou
from fvcore.nn import sigmoid_focal_loss, smooth_l1_loss
import torch
from torch import nn
import torch.nn.functional as F

from utils.box_ops import box_cxcywh_to_xyxy


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

        It updates the 'loss_normalizer' attribute when 'tgt_dict' is not None (i.e. during training and validation).

        Args:
            feat_maps (List): List of size [num_maps] with feature maps of shape [batch_size, fH, fW, feat_size].
            tgt_dict (Dict): Optional target dictionary containing at least following keys:
                - labels (IntTensor): tensor of shape [num_targets_total] containing the class indices;
                - boxes (FloatTensor): boxes of shape [num_targets_total, 4] in (cx, cy, width, height) format;
                - sizes (IntTensor): tensor of shape [batch_size+1] with the cumulative target sizes of batch entries.

        Returns:
            * If tgt_dict is not None (i.e. during training and validation):
                tgt_dict (Dict): Updated target dictionary containing following additional keys:
                    - anchor_labels (IntTensor): tensor of class indices of shape [batch_size, num_anchors_total];
                    - anchor_deltas (FloatTensor): anchor to box deltas of shape [batch_size, num_anchors_total, 4].
        """

        # Generate anchors and set anchors attribute
        feat_map_views = [feat_map.permute(0, 3, 1, 2) for feat_map in feat_maps]
        self.anchors = self.anchor_generator(feat_map_views)

        # Return when no target dictionary is provided (validation/testing only)
        if tgt_dict is None:
            return

        # Some preparation before anchor labeling (trainval only)
        anchors = Boxes.cat(self.anchors)
        box_to_box = Box2BoxTransform(weights=(1.0, 1.0, 1.0, 1.0))
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

                anchor_deltas = box_to_box.get_deltas(anchors.tensor, tgt_boxes.tensor[matched_ids])
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

    def forward(self, feat_maps, tgt_dict=None, **kwargs):
        """
        Forward method of the RetinaHead module.

        Args:
            feat_maps (List): List of size [num_maps] with feature maps of shape [batch_size, fH, fW, feat_size].
            tgt_dict (Dict): Optional target dictionary containing at least following keys:
                - anchor_labels (IntTensor): tensor of class indices of shape [batch_size, num_anchors_total];
                - anchor_deltas (FloatTensor): anchor to box deltas of shape [batch_size, num_anchors_total, 4].

        Returns:
            * If tgt_dict is not None (i.e. during training and validation):
                loss_dict (Dictionary): Loss dictionary containing following keys:
                    - ret_cls_loss (FloatTensor): weighted classification loss of shape [];
                    - ret_box_loss (FloatTensor): weighted bounding box regression loss of shape [].

                analysis_dict (Dictionary): Empty analysis dictionary.
        """

        # Project feature maps to common feature space and permute to convolution format
        feat_maps = [in_proj(feat_map).permute(0, 3, 1, 2) for feat_map, in_proj in zip(feat_maps, self.in_projs)]

        # Predict logits and anchor regression deltas
        logit_maps, delta_maps = self.pred_head(feat_maps)
        logit_maps = [RetinaHead.reshape_pred_map(logit_map, self.num_classes) for logit_map in logit_maps]
        delta_maps = [RetinaHead.reshape_pred_map(delta_map, 4) for delta_map in delta_maps]

        # Compute and return prediction dictionary (validation/testing only)
        if tgt_dict is None:
            return

        # Compute weighted classification loss (trainval only)
        anchor_labels = tgt_dict['anchor_labels']
        class_mask = anchor_labels >= 0

        class_preds = torch.cat(logit_maps, dim=1)[class_mask]
        class_targets = F.one_hot(anchor_labels[class_mask], num_classes=self.num_classes+1)
        class_targets = class_targets[:, :-1].to(class_preds.dtype)

        class_kwargs = {'alpha': self.focal_alpha, 'gamma': self.focal_gamma, 'reduction': 'sum'}
        class_loss = sigmoid_focal_loss(class_preds, class_targets, **class_kwargs)
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
            analysis_dict = {}

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
