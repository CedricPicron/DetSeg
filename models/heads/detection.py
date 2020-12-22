"""
Detection head modules and build function.
"""
import math

from detectron2.layers import batched_nms
from detectron2.modeling.anchor_generator import DefaultAnchorGenerator
from detectron2.modeling.matcher import Matcher
from fvcore.nn import sigmoid_focal_loss, smooth_l1_loss
import torch
from torch import nn
import torch.nn.functional as F

from structures.boxes import Boxes, apply_box_deltas, box_iou, get_box_deltas
from utils.distributed import is_dist_avail_and_initialized, get_world_size


class RetinaHead(nn.Module):
    """
    Class implementing the RetinaHead module.

    Attributes:
        num_classes (int): Integer containing the number of object classes (without background).

        anchor_generator (DefaultAnchorGenerator): Module generating anchors for a given list of feature maps.
        anchor_matcher (Matcher): Module matching target boxes with anchors via their pairwise IoU-matrix.
        pred_head (RetinaPredHead): Module computing logits and box deltas for every anchor-position combination.

        focal_alpha (float): Alpha value of the sigmoid focal loss used during classification.
        focal_gamma (float): Gamma value of the sigmoid focal loss used during classification.
        smooth_l1_beta (float): Beta value of the smooth L1 loss used during bounding box regression.

        loss_normalizers (FloatTensor): Buffer of shape [num_maps] containing the loss normalizers for each map.
        loss_momentum (float): Momentum factor used during the loss normalizer update.

        cls_loss_weight (float): Factor weighting the classification loss originating from this head.
        box_loss_weight (float): Factor weighting the bounding box regression loss originating from this head.

        test_score_threshold (float): Threshold used to remove detections before non-maxima suppression (NMS).
        test_max_candidates (int): Maximum number of candidate detections to keep per map of an image for NMS.
        test_nms_threshold (float): Threshold used during NMS to remove duplicate detections.
        test_max_detections (int): Maximum number of detections to keep per image after NMS.
    """

    def __init__(self, num_classes, map_ids, in_feat_sizes, pred_head_dict, loss_dict, test_dict):
        """
        Initializes the RetinaHead module.

        Args:
            num_classes (int): Integer containing the number of object classes (without background).
            map_ids (List): List of size [num_maps] containing the map ids (i.e. downsampling exponents) of each map.
            in_feat_sizes (List): List of size [num_maps] containing the feature sizes of each input feature map.

            pred_head_dict (Dict): Dictionary containing prediction head hyperparameters:
                - in_projs (bool): boolean indicating whether to perform linear projections on the input feature maps;
                - feat_size (int): the feature size used internally by the prediction head;
                - num_convs (int): the number of internal convolutions in the prediction head before prediction.

            loss_dict (Dict): Dictionary containing hyperparameters related the head's loss:
                - focal_alpha (float): alpha value of the sigmoid focal loss used during classification;
                - focal_gamma (float): gamma value of the sigmoid focal loss used during classification;
                - smooth_l1_beta (float): beta value of the smooth L1 loss used during bounding box regression;
                - normalizer (float): initial loss normalizer value (estimates expected number of positive anchors);
                - momentum (float): momentum factor used during the loss normalizer update;
                - cls_weight (float): factor weighting the classification loss originating from this head;
                - box_weight (float): factor weighting the boundig box regression loss originating from this head.

            test_dict (Dict): Dictionary containing testing (i.e. inference) hyperparameters:
                - score_threshold (float): threshold used to remove detections before non-maxima suppression (NMS);
                - max_candidates (int): maximum number of candidate detections to keep per map of an image for NMS;
                - nms_threshold (float): threshold used during NMS to remove duplicate detections;
                - max_detections (int): maximum number of detections to keep per image after NMS.
        """

        # Initialization of default nn.Module
        super().__init__()

        # Set number of classes attribute
        self.num_classes = num_classes

        # Initialization of anchor generator
        anchor_sizes = [[2**(i+2), 2**(i+2) * 2**(1.0/3), 2**(i+2) * 2**(2.0/3)] for i in map_ids]
        anchor_aspect_ratios = [[0.5, 1.0, 2.0]]
        anchor_strides = [2**i for i in map_ids]

        kwargs = {'sizes': anchor_sizes, 'aspect_ratios': anchor_aspect_ratios, 'strides': anchor_strides}
        self.anchor_generator = DefaultAnchorGenerator(**kwargs)

        # Initialization of anchor matcher
        matcher_thresholds = [0.4, 0.5]
        matcher_labels = [0, -1, 1]
        self.anchor_matcher = Matcher(matcher_thresholds, matcher_labels, allow_low_quality_matches=True)

        # Initialization of prediction head module
        input_dict = {'in_projs': pred_head_dict['in_projs'], 'in_feat_sizes': in_feat_sizes}
        conv_dict = {'feat_size': pred_head_dict['feat_size'], 'num_convs': pred_head_dict['num_convs']}
        num_anchors = self.anchor_generator.num_cell_anchors[0]
        self.pred_head = RetinaPredHead(input_dict, conv_dict, num_anchors, num_classes)

        # Initialization of loss attributes
        self.focal_alpha = loss_dict['focal_alpha']
        self.focal_gamma = loss_dict['focal_gamma']
        self.smooth_l1_beta = loss_dict['smooth_l1_beta']

        loss_normalizers = torch.full((len(map_ids),), loss_dict['normalizer'], dtype=torch.float)
        self.register_buffer('loss_normalizers', loss_normalizers)
        self.loss_momentum = loss_dict['momentum']

        self.cls_loss_weight = loss_dict['cls_weight']
        self.box_loss_weight = loss_dict['box_weight']

        # Initialization of test attributes
        self.test_score_threshold = test_dict['score_threshold']
        self.test_max_candidates = test_dict['max_candidates']
        self.test_nms_threshold = test_dict['nms_threshold']
        self.test_max_detections = test_dict['max_detections']

    @torch.no_grad()
    def forward_init(self, feat_maps, tgt_dict=None):
        """
        Forward initialization method of the RetinaHead module.

        Args:
            feat_maps (List): List of size [num_maps] with feature maps of shape [batch_size, fH, fW, feat_size].

            tgt_dict (Dict): Optional target dictionary used during trainval containing at least following keys:
                - labels (LongTensor): tensor of shape [num_targets_total] containing the class indices;
                - boxes (Boxes): structure containing axis-aligned bounding boxes of size [num_targets_total];
                - sizes (LongTensor): tensor of shape [batch_size+1] with the cumulative target sizes of batch entries.

        Returns:
            * If tgt_dict is not None (i.e. during training and validation):
                tgt_dict (Dict): Updated target dictionary containing following additional keys:
                    - anchor_labels (LongTensor): tensor of class indices of shape [batch_size, num_anchors_total];
                    - anchor_deltas (FloatTensor): anchor to box deltas of shape [batch_size, num_anchors_total, 4].

                attr_dict (Dict): Dictionary of attributes to be set and shared between the different head copies:
                    - anchors (List): list of size [num_maps] containing the anchor boxes corresponding to each map.

                buffer_dict (Dict): Dictionary containing updated buffer tensors:
                    - loss_normalizers (FloatTensor): updated loss normalizers of shape [num_maps].

            * If tgt_dict is None (i.e. during testing):
                tgt_dict (None): Contains the None value.

                attr_dict (Dict): Dictionary of attributes to be set and shared between the different head copies:
                    - anchors (List): list of size [num_maps] containing the anchor boxes corresponding to each map.

                buffer_dict (Dict): Empty dictionary.
        """

        # Generate anchors and place them into attribute dictionary
        feat_map_views = [feat_map.permute(0, 3, 1, 2) for feat_map in feat_maps]
        anchors = self.anchor_generator(feat_map_views)

        anchors = [Boxes(map_anchors.tensor, format='xyxy') for map_anchors in anchors]
        attr_dict = {'anchors': anchors}

        # Return when no target dictionary is provided (validation/testing only)
        if tgt_dict is None:
            return None, attr_dict, {}

        # Some preparation before anchor labeling (trainval only)
        map_sizes = [0] + [len(map_anchors) for map_anchors in anchors]
        map_sizes = torch.tensor(map_sizes).cumsum(dim=0)

        anchors = Boxes.cat(anchors)
        tgt_sizes = tgt_dict['sizes']

        anchor_labels_list = []
        anchor_deltas_list = []

        # Label anchors for every batch entry (trainval only)
        for i0, i1 in zip(tgt_sizes[:-1], tgt_sizes[1:]):
            tgt_labels = tgt_dict['labels'][i0:i1]
            tgt_boxes = tgt_dict['boxes'][i0:i1]
            matched_ids, match_labels = self.anchor_matcher(box_iou(tgt_boxes, anchors)[0])

            if len(tgt_labels) > 0:
                anchor_labels = tgt_labels[matched_ids]
                anchor_labels[match_labels == 0] = self.num_classes
                anchor_labels[match_labels == -1] = -1
                anchor_labels_list.append(anchor_labels)

                anchor_deltas = get_box_deltas(anchors, tgt_boxes[matched_ids])
                anchor_deltas_list.append(anchor_deltas)

            else:
                anchor_labels_list.append(torch.full_like(matched_ids, self.num_classes))
                anchor_deltas_list.append(torch.zeros_like(anchors.boxes))

        # Get batched anchor labels and deltas (trainval only)
        anchor_labels = torch.stack(anchor_labels_list)
        anchor_deltas = torch.stack(anchor_deltas_list)

        # Get updated loss normalizers and place them into buffer dictionary (trainval only)
        num_pos_anchors = torch.zeros(len(feat_maps), device=feat_maps[0].device)

        for i, i0, i1 in zip(range(len(feat_maps)), map_sizes[:-1], map_sizes[1:]):
            map_anchor_labels = anchor_labels[:, i0:i1]
            num_pos_anchors[i] = ((map_anchor_labels >= 0) & (map_anchor_labels != self.num_classes)).sum()

        if is_dist_avail_and_initialized():
            num_pos_anchors = torch.distributed.all_reduce(num_pos_anchors) / get_world_size()

        num_pos_anchors = num_pos_anchors.clamp(min=1)
        loss_normalizers = self.loss_momentum * self.loss_normalizers
        loss_normalizers += (1 - self.loss_momentum) * num_pos_anchors
        buffer_dict = {'loss_normalizers': loss_normalizers}

        # Add batched anchor labels and deltas to target dictionary
        tgt_dict['anchor_labels'] = anchor_labels
        tgt_dict['anchor_deltas'] = anchor_deltas

        return tgt_dict, attr_dict, buffer_dict

    def make_predictions(self, logit_maps, delta_maps):
        """
        Make classified bounding box predictions based on given logit and delta maps.

        Args:
            logit_maps (List): Maps [num_maps] with logits of shape [batch_size, fH*fW*num_anchors, num_classes].
            delta_maps (List): Maps [num_maps] with box deltas shape [batch_size, fH*fW*num_anchors, 4].

        Returns:
            pred_dict (Dict): Prediction dictionary containing following keys:
                - labels (LongTensor): predicted class indices of shape [num_preds_total];
                - boxes (Boxes): structure containing axis-aligned bounding boxes of size [num_preds_total];
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

                # Flatten image logit maps and get class probabilities
                logits = img_logit_map.flatten()
                probs = logits.sigmoid_()

                # Filter predictions: Absolute (only keep predictions of sufficient confidence)
                abs_keep = probs > self.test_score_threshold
                probs = probs[abs_keep]

                # Filter predictions: Relative (only keep top predictions)
                num_preds_kept = min(self.test_max_candidates, len(probs))
                probs, sort_ids = probs.sort(descending=True)
                top_scores = probs[:num_preds_kept]

                # Get ids of top predictions
                top_ids = torch.nonzero(abs_keep, as_tuple=True)[0]
                top_ids = top_ids[sort_ids[:num_preds_kept]]

                # Get labels of top predictions
                top_labels = top_ids % self.num_classes

                # Get boxes of top predictions
                anchor_ids = top_ids // self.num_classes
                top_deltas = img_delta_map[anchor_ids]
                top_anchors = map_anchors[anchor_ids]
                top_boxes = apply_box_deltas(top_deltas, top_anchors)

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
            labels = torch.cat(pred_labels[i])
            boxes = Boxes.cat(pred_boxes[i], same_image=True)
            scores = torch.cat(pred_scores[i])

            # Keep best predictions after non-maxima suppression (NMS)
            boxes = boxes.to_format('xyxy')
            keep = batched_nms(boxes.boxes, scores, labels, iou_threshold=self.test_nms_threshold)
            keep = keep[:self.test_max_detections]

            # Add final predictions to the prediction dictionary
            pred_dict['labels'].append(labels[keep])
            pred_dict['boxes'].append(boxes[keep])
            pred_dict['scores'].append(scores[keep])
            pred_dict['batch_ids'].append(torch.full_like(keep, i, dtype=torch.int64))

        # Concatenate different image predictions
        pred_dict = {k: torch.cat(v, dim=0) for k, v in pred_dict.items() if k != 'boxes'}
        pred_dict['boxes'] = Boxes.cat(pred_dict['boxes'])

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
                - ret_cls_acc_obj (FloatTensor): object accuracy of the retina head classification of shape [].
        """

        # Compute object accuracy and place it into analysis dictionary
        obj_mask = targets < self.num_classes
        obj_accuracy = RetinaHead.get_accuracy(preds[obj_mask], targets[obj_mask])
        analysis_dict = {'ret_cls_acc_obj': 100*obj_accuracy}

        return analysis_dict

    def forward(self, feat_maps, feat_masks=None, tgt_dict=None, **kwargs):
        """
        Forward method of the RetinaHead module.

        Args:
            feat_maps (List): List of size [num_maps] with feature maps of shape [batch_size, fH, fW, feat_size].
            feat_masks (List): Optional list [num_maps] with masks of active features of shape [batch_size, fH, fW].

            tgt_dict (Dict): Optional target dictionary containing at least following keys:
                - anchor_labels (LongTensor): tensor of class indices of shape [batch_size, num_anchors_total];
                - anchor_deltas (FloatTensor): anchor to box deltas of shape [batch_size, num_anchors_total, 4].

            kwargs (Dict): Dictionary of keyword arguments, potentially containing following key:
                - extended_analysis (bool): boolean indicating whether to perform extended analyses or not.

        Returns:
            * If tgt_dict is not None (i.e. during training and validation):
                loss_dict (Dict): Loss dictionary containing following keys:
                    - ret_cls_loss (FloatTensor): weighted classification loss of shape [];
                    - ret_box_loss (FloatTensor): weighted bounding box regression loss of shape [].

                analysis_dict (Dict): Analysis dictionary at least containing following keys:
                    - ret_cls_acc_obj (FloatTensor): object accuracy of the retina head classification of shape [].

            * If tgt_dict is None (i.e. during testing and possibly during validation):
                pred_dict (Dict): Prediction dictionary containing following keys:
                    - labels (LongTensor): predicted class indices of shape [num_preds_total];
                    - boxes (Boxes): structure containing axis-aligned bounding boxes of size [num_preds_total];
                    - scores (FloatTensor): normalized prediction scores of shape [num_preds_total];
                    - batch_ids (LongTensor): batch indices of predictions of shape [num_preds_total].
        """

        # Compute predicted logit and delta maps
        logit_maps, delta_maps = self.pred_head(feat_maps)
        logit_maps = [logit_map.view(len(logit_map), -1, self.num_classes) for logit_map in logit_maps]
        delta_maps = [delta_map.view(len(delta_map), -1, 4) for delta_map in delta_maps]

        # Compute and return prediction dictionary (validation/testing only)
        if tgt_dict is None:
            pred_dict = self.make_predictions(logit_maps, delta_maps)
            return pred_dict

        # Get map ids
        map_ids = [torch.full_like(logit_map[:, :, 0], i, dtype=torch.uint8) for i, logit_map in enumerate(logit_maps)]
        map_ids = torch.cat(map_ids, dim=1)

        # Compute weighted classification loss (trainval only)
        anchor_labels = tgt_dict['anchor_labels']
        cls_mask = anchor_labels >= 0

        logits = torch.cat(logit_maps, dim=1)
        cls_logits = logits[cls_mask]

        cls_targets = F.one_hot(anchor_labels[cls_mask], num_classes=self.num_classes+1)
        cls_targets = cls_targets[:, :-1].to(cls_logits.dtype)

        cls_kwargs = {'alpha': self.focal_alpha, 'gamma': self.focal_gamma, 'reduction': 'none'}
        cls_losses = sigmoid_focal_loss(cls_logits, cls_targets, **cls_kwargs)

        cls_map_ids = map_ids[cls_mask]
        cls_map_losses = [cls_losses[cls_map_ids == i].sum() / self.loss_normalizers[i] for i in range(len(feat_maps))]
        cls_loss = self.cls_loss_weight * sum(cls_map_losses)

        # Compute weighted bounding box regression loss (trainval only)
        box_mask = cls_mask & (anchor_labels != self.num_classes)
        box_preds = torch.cat(delta_maps, dim=1)[box_mask]
        box_targets = tgt_dict['anchor_deltas'][box_mask]

        box_kwargs = {'beta': self.smooth_l1_beta, 'reduction': 'none'}
        box_losses = smooth_l1_loss(box_preds, box_targets, **box_kwargs)

        box_map_ids = map_ids[box_mask]
        box_map_losses = [box_losses[box_map_ids == i].sum() / self.loss_normalizers[i] for i in range(len(feat_maps))]
        box_loss = self.box_loss_weight * sum(box_map_losses)

        # Place weighted losses into loss dictionary (trainval only)
        loss_dict = {'ret_cls_loss': cls_loss, 'ret_box_loss': box_loss}

        # Perform analyses (trainval only)
        with torch.no_grad():

            # Assume no padded regions when feature masks are missing (trainval only)
            if feat_masks is None:
                tensor_kwargs = {'dtype': torch.bool, 'device': feat_maps[0].device}
                feat_masks = [torch.ones(*feat_map.shape[:-1], **tensor_kwargs) for feat_map in feat_maps]

            # Perform classification accuracy analyses (trainval only)
            num_anchors = self.anchor_generator.num_cell_anchors[0]
            feat_masks = [mask[:, :, :, None].expand(-1, -1, -1, num_anchors).flatten(1) for mask in feat_masks]
            active_mask = torch.cat(feat_masks, dim=1)
            acc_mask = active_mask & cls_mask

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


class RetinaPredHead(nn.Module):
    """
    Class implementing the RetinaPredHead module.

    Attributes:
        in_projs (nn.ModuleList, optional): List of size [num_maps] with input projection modules.
        cls_subnet (nn.Sequential): Sequence of modules computing the final features before classification.
        bbox_subnet (nn.Sequential): Sequence of modules computing the final features before bounding box regression.
        cls_score (nn.Linear): Linear module computing the classification logits from its final features.
        bbox_pred (nn.Linear): Linear module computing the anchor to box deltas from its final features.
    """

    def __init__(self, input_dict, conv_dict, num_anchors, num_classes):
        """
        Initializes the RetinaPredHead module.

        Args:
            input_dict (Dict): Dictionary with information for a possible input projection containing following keys:
                - in_projs (bool): boolean indicating whether to perform linear projections on the input feature maps;
                - in_feat_sizes (List): list of size [num_maps] containing the feature sizes of each input feature map.

            conv_dict (Dict): Dictionary with subnet convolutions information containing following keys:
                - feat_size (int): the feature size used internally by the subnet convolution layers;
                - num_convs (int): integer containing the number of subnet convolution layers.

            num_anchors (int): Integer containing the number of different anchors per spatial position.
            num_classes (int): Integer containing the number of object classes (without background).
        """

        # Initialization of default nn.Module
        super().__init__()

        # Initialization of linear input projection modules (if requested)
        if input_dict['in_projs']:
            self.in_projs = nn.ModuleList(nn.Linear(f, conv_dict['feat_size']) for f in input_dict['in_feat_sizes'])

        # Initialize classification and bounding box subnets
        in_size = conv_dict['feat_size'] if input_dict['in_projs'] else input_dict['in_feat_sizes'][0]
        conv_sizes = [in_size] + [conv_dict['feat_size']] * conv_dict['num_convs']

        cls_subnet = []
        bbox_subnet = []

        for in_size, out_size in zip(conv_sizes[:-1], conv_sizes[1:]):
            cls_subnet.append(nn.Conv2d(in_size, out_size, kernel_size=3, stride=1, padding=1))
            bbox_subnet.append(nn.Conv2d(in_size, out_size, kernel_size=3, stride=1, padding=1))

            cls_subnet.append(nn.ReLU())
            bbox_subnet.append(nn.ReLU())

        self.cls_subnet = nn.Sequential(*cls_subnet)
        self.bbox_subnet = nn.Sequential(*bbox_subnet)

        # Initialize final linear classification and bounding box layers
        self.cls_score = nn.Linear(conv_sizes[-1], num_anchors * num_classes)
        self.bbox_pred = nn.Linear(conv_sizes[-1], num_anchors * 4)

        # Set default initial values of module parameters
        self.reset_parameters()

    def reset_parameters(self, prior_cls_prob=0.01):
        """
        Resets module parameters to default initial values.

        Args:
            prior_cls_prob (float): Prior class probability from which class bias is derived (defaults to 0.01).
        """

        # Set default parameters of convolution layers
        for modules in [self.cls_subnet, self.bbox_subnet]:
            for layer in modules.modules():
                if isinstance(layer, nn.Conv2d):
                    torch.nn.init.normal_(layer.weight, mean=0, std=0.01)
                    torch.nn.init.constant_(layer.bias, 0)

        # Set classification bias according to given prior classification probability
        bias_value = -(math.log((1 - prior_cls_prob) / prior_cls_prob))
        torch.nn.init.constant_(self.cls_score.bias, bias_value)

    def forward(self, feat_maps):
        """
        Forward method of the RetinaPredHead module.

        Args:
            feat_maps (List): List of size [num_maps] with feature maps of shape [batch_size, fH, fW, feat_size].

        Returns:
            logit_maps (List): Maps [num_maps] of class logits of shape [batch_size, fH, fW, num_anchors*num_classes].
            delta_maps (List): Maps [num_maps] of anchor to box deltas of shape [batch_size, fH, fW, num_anchors*4].
        """

        # Project feature maps to common feature space (if requested) and permute to convolution format
        if hasattr(self, 'in_projs'):
            feat_maps = [in_proj(feat_map).permute(0, 3, 1, 2) for feat_map, in_proj in zip(feat_maps, self.in_projs)]
        else:
            feat_maps = [feat_map.permute(0, 3, 1, 2) for feat_map in feat_maps]

        # Get logit and delta maps
        logit_maps = [self.cls_score(self.cls_subnet(feat_map).permute(0, 2, 3, 1)) for feat_map in feat_maps]
        delta_maps = [self.bbox_pred(self.bbox_subnet(feat_map).permute(0, 2, 3, 1)) for feat_map in feat_maps]

        return logit_maps, delta_maps


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
    map_ids = list(range(args.min_resolution_id, args.max_resolution_id+1))
    feat_sizes = [min(args.base_feat_size * 2**i, args.max_feat_size) for i in map_ids]

    # Initialize empty list of detection head modules
    det_heads = []

    # Build desired detection head modules
    for det_head_type in args.det_heads:
        if det_head_type == 'retina':
            pred_head_dict = {'in_projs': True, 'feat_size': args.ret_feat_size, 'num_convs': args.ret_num_convs}

            loss_dict = {'focal_alpha': args.ret_focal_alpha, 'focal_gamma': args.ret_focal_gamma}
            loss_dict = {**loss_dict, 'smooth_l1_beta': args.ret_smooth_l1_beta, 'normalizer': args.ret_normalizer}
            loss_dict = {**loss_dict, 'momentum': args.ret_momentum, 'cls_weight': args.ret_cls_weight}
            loss_dict = {**loss_dict, 'box_weight': args.ret_box_weight}

            test_dict = {'score_threshold': args.ret_score_threshold, 'max_candidates': args.ret_max_candidates}
            test_dict = {**test_dict, 'nms_threshold': args.ret_nms_threshold}
            test_dict = {**test_dict, 'max_detections': args.ret_max_detections}

            retina_head = RetinaHead(args.num_classes, map_ids, feat_sizes, pred_head_dict, loss_dict, test_dict)
            det_heads.append(retina_head)

        else:
            raise ValueError(f"Unknown detection head type '{det_head_type}' was provided.")

    return det_heads
