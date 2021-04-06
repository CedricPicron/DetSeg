"""
Duplicate-Free Detector (DFD) head.
"""
from collections import OrderedDict
from copy import deepcopy
import math

from detectron2.layers import batched_nms
from detectron2.structures.instances import Instances
from detectron2.utils.visualizer import Visualizer
from fvcore.nn import sigmoid_focal_loss, smooth_l1_loss
import torch
from torch import nn
import torch.nn.functional as F

from models.functional.position import sine_pos_encodings
from models.modules.convolution import BottleneckConv, ProjConv
from structures.boxes import apply_box_deltas, Boxes, box_intersection, box_iou, get_box_deltas


class DFD(nn.Module):
    """
    Class implementing the Duplicate-Free Detector (DFD) head.

    Attributes:
        cls_head (nn.Sequential): Module computing the classification logits.
        cls_focal_alpha (float): Alpha value of the sigmoid focal loss used by the classification head.
        cls_focal_gamma (float): Gamma value of the sigmoid focal loss used by the classification head.
        cls_weight (float): Factor weighting the classification loss.

        obj_head (nn.Sequential): Module computing the objectness logits.
        obj_focal_alpha (float): Alpha value of the sigmoid focal loss used by the objectness head.
        obj_focal_gamma (float): Gamma value of the sigmoid focal loss used by the objectness head.
        obj_weight (float): Factor weighting the objectness loss.

        box_head (nn.Sequential): Module computing the bounding box logits.
        box_sl1_beta (float): Beta value of the smooth L1 loss used by the bounding box head.
        box_weight (float): Factor weighting the bounding box loss.

        inf_nms_candidates (int): Maximum number of candidates retained for NMS during inference.
        inf_iou_threshold (float): Value determining the IoU threshold of NMS during inference.
        inf_max_detections (int): Maximum number of detections retained during inference.

        metadata (detectron2.data.Metadata): Metadata instance containing additional dataset information.
    """

    def __init__(self, in_feat_size, cls_dict, obj_dict, box_dict, inf_dict, metadata):
        """
        Initializes the DFD module.

        Args:
            in_feat_size (int): Integer containing the feature size of the input feature pyramid.

            cls_dict (Dict): Classification head dictionary containing following keys:
                - feat_size (int): hidden feature size of the classification head;
                - norm (str): string specifying the type of normalization used within the classification head;
                - num_classes (int): integer containing the number of object classes (without background);
                - prior_prob (float): prior class probability;
                - kernel_size (int): kernel size used by hidden layers of the classification head;
                - bottle_size (int): bottleneck size used by bottleneck layers of the classification head;
                - hidden_layers (int): number of hidden layers of the classification head;

                - focal_alpha (float): alpha value of the sigmoid focal loss used by the classification head;
                - focal_gamma (float): gamma value of the sigmoid focal loss used by the classification head;
                - weight (float): factor weighting the classification loss.

            obj_dict (Dict): Objectness head dictionary containing following keys:
                - feat_size (int): hidden feature size of the objectness head;
                - norm (str): string specifying the type of normalization used within the objectness head;
                - prior_prob (float): prior object probability;
                - kernel_size (int): kernel size used by hidden layers of the objectness head;
                - bottle_size (int): bottleneck size used by bottleneck layers of the objectness head;
                - hidden_layers (int): number of hidden layers of the objectness head;

                - focal_alpha (float): alpha value of the sigmoid focal loss used by the objectness head;
                - focal_gamma (float): gamma value of the sigmoid focal loss used by the objectness head;
                - weight (float): factor weighting the objectness loss.

            box_dict (Dict): Bounding box head dictionary containing following keys:
                - feat_size (int): hidden feature size of the bounding box head;
                - norm (str): string specifying the type of normalization used within the bounding box head;
                - kernel_size (int): kernel size used by hidden layers of the bounding box head;
                - bottle_size (int): bottleneck size used by bottleneck layers of the bounding box head;
                - hidden_layers (int): number of hidden layers of the bounding box head;

                - sl1_beta (float): beta value of the smooth L1 loss used by the bounding box head;
                - weight (float): factor weighting the bounding box loss.

            inf_dict (Dict): Inference dictionary containing following keys:
                - nms_candidates (int): maximum number of candidates retained for NMS during inference;
                - iou_threshold (float): value determining the IoU threshold of NMS during inference;
                - max_detections (int): maximum number of detections retained during inference.

            metadata (detectron2.data.Metadata): Metadata instance containing additional dataset information.
        """

        # Initialization of default nn.Module
        super().__init__()

        # Initialization of classification head
        cls_feat_size = cls_dict['feat_size']
        cls_norm = cls_dict['norm']
        num_classes = cls_dict['num_classes']

        cls_in_layer = ProjConv(in_feat_size, cls_feat_size, norm=cls_norm, skip=False)
        cls_out_layer = ProjConv(cls_feat_size, num_classes+1, norm=cls_norm, skip=False)

        cls_prior_prob = cls_dict['prior_prob']
        bias_value = -(math.log((1 - cls_prior_prob) / cls_prior_prob))
        torch.nn.init.constant_(cls_out_layer.conv.bias, bias_value)

        if cls_dict['kernel_size'] == 1:
            cls_hidden_layer = ProjConv(cls_feat_size, norm=cls_norm, skip=True)
        else:
            bottleneck_kwargs = {'kernel_size': cls_dict['kernel_size'], 'norm': cls_norm, 'skip': True}
            cls_hidden_layer = BottleneckConv(cls_feat_size, cls_dict['bottle_size'], **bottleneck_kwargs)

        cls_hidden_layers = nn.Sequential(*[deepcopy(cls_hidden_layer) for _ in range(cls_dict['hidden_layers'])])
        cls_head_dict = OrderedDict([('in', cls_in_layer), ('hidden', cls_hidden_layers), ('out', cls_out_layer)])
        self.cls_head = nn.Sequential(cls_head_dict)

        self.cls_focal_alpha = cls_dict['focal_alpha']
        self.cls_focal_gamma = cls_dict['focal_gamma']
        self.cls_weight = cls_dict['weight']

        # Initialization of objectness head
        obj_feat_size = obj_dict['feat_size']
        obj_norm = obj_dict['norm']

        obj_in_layer = ProjConv(in_feat_size, obj_feat_size, norm=obj_norm, skip=False)
        obj_out_layer = ProjConv(obj_feat_size, 2, norm=obj_norm, skip=False)

        obj_prior_prob = obj_dict['prior_prob']
        bias_value = -(math.log((1 - obj_prior_prob) / obj_prior_prob))
        torch.nn.init.constant_(obj_out_layer.conv.bias, bias_value)

        if obj_dict['kernel_size'] == 1:
            obj_hidden_layer = ProjConv(obj_feat_size, norm=obj_norm, skip=True)
        else:
            bottleneck_kwargs = {'kernel_size': obj_dict['kernel_size'], 'norm': obj_norm, 'skip': True}
            obj_hidden_layer = BottleneckConv(obj_feat_size, obj_dict['bottle_size'], **bottleneck_kwargs)

        obj_hidden_layers = nn.Sequential(*[deepcopy(obj_hidden_layer) for _ in range(obj_dict['hidden_layers'])])
        obj_head_dict = OrderedDict([('in', obj_in_layer), ('hidden', obj_hidden_layers), ('out', obj_out_layer)])
        self.obj_head = nn.Sequential(obj_head_dict)

        self.obj_focal_alpha = obj_dict['focal_alpha']
        self.obj_focal_gamma = obj_dict['focal_gamma']
        self.obj_weight = obj_dict['weight']

        # Initialization of bounding box head
        box_feat_size = box_dict['feat_size']
        box_norm = box_dict['norm']

        box_in_layer = ProjConv(in_feat_size, box_feat_size, norm=box_norm, skip=False)
        box_out_layer = ProjConv(box_feat_size, 4, norm=box_norm, skip=False)
        torch.nn.init.zeros_(box_out_layer.conv.bias)

        if box_dict['kernel_size'] == 1:
            box_hidden_layer = ProjConv(box_feat_size, norm=box_norm, skip=True)
        else:
            bottleneck_kwargs = {'kernel_size': box_dict['kernel_size'], 'norm': box_norm, 'skip': True}
            box_hidden_layer = BottleneckConv(box_feat_size, box_dict['bottle_size'], **bottleneck_kwargs)

        box_hidden_layers = nn.Sequential(*[deepcopy(box_hidden_layer) for _ in range(box_dict['hidden_layers'])])
        box_head_dict = OrderedDict([('in', box_in_layer), ('hidden', box_hidden_layers), ('out', box_out_layer)])
        self.box_head = nn.Sequential(box_head_dict)

        self.box_sl1_beta = box_dict['sl1_beta']
        self.box_weight = box_dict['weight']

        # Set inference attributes
        self.inf_nms_candidates = inf_dict['nms_candidates']
        self.inf_iou_threshold = inf_dict['iou_threshold']
        self.inf_max_detections = inf_dict['max_detections']

        # Set metadata attribute
        self.metadata = metadata

    @torch.no_grad()
    def forward_init(self, images, feat_maps, tgt_dict=None):
        """
        Forward initialization method of the DFD module.

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
        norm_boxes = tgt_dict['boxes'].normalize(images, with_padding=True)

        # Update target dictionary
        sizes = tgt_dict['sizes']
        tgt_dict['labels'] = [tgt_dict['labels'][i0:i1] for i0, i1 in zip(sizes[:-1], sizes[1:])]
        tgt_dict['boxes'] = [norm_boxes[i0:i1] for i0, i1 in zip(sizes[:-1], sizes[1:])]

        return tgt_dict, {}, {}

    def cls_loss(self, cls_logits, tgt_labels, inter_matrices):
        """
        Gets classification loss and corresponding classification analysis.

        Args:
            cls_logits (FloatTensor): Classification logits of shape [batch_size, num_preds, num_classes+1].
            tgt_labels (List): List [batch_size] with class indices of shape [num_targets].
            inter_matrices (List): List [batch_size] with intersection matrices of shape [num_preds, num_targets+1].

        Returns:
            cls_loss_dict (Dict): Classification loss dictionary containing following key:
                - dfd_cls_loss (FloatTensor): weighted classification loss of shape [1].

            cls_analysis_dict (Dict): Classification analysis dictionary containing following key:
                - dfd_cls_acc (FloatTensor): classification accuracy of shape [1].
        """

        # Initialize empty classification loss and analysis dictionaries
        tensor_kwargs = {'dtype': cls_logits.dtype, 'device': cls_logits.device}
        cls_loss_dict = {'dfd_cls_loss': torch.zeros(1, **tensor_kwargs)}
        cls_analysis_dict = {'dfd_cls_acc': torch.zeros(1, **tensor_kwargs)}

        # Get batch size and number of object classes with background
        batch_size = len(cls_logits)
        num_cls_labels = cls_logits.shape[-1]

        # Iterate over every batch entry
        for cls_logits_i, tgt_labels_i, inter_matrix in zip(cls_logits, tgt_labels, inter_matrices):

            # Append target with background label
            bg_label = torch.tensor([num_cls_labels-1]).to(tgt_labels_i)
            tgt_labels_i = torch.cat([tgt_labels_i, bg_label], dim=0)

            # Get classification targets
            inters, tgt_ids = torch.max(inter_matrix, dim=1)
            cls_targets_i = tgt_labels_i[tgt_ids]
            cls_targets_oh_i = F.one_hot(cls_targets_i, num_cls_labels).to(cls_logits_i.dtype)

            # Get unweighted classification losses
            cls_kwargs = {'alpha': self.cls_focal_alpha, 'gamma': self.cls_focal_gamma, 'reduction': 'none'}
            cls_losses = sigmoid_focal_loss(cls_logits_i, cls_targets_oh_i, **cls_kwargs)

            # Get weights normalized per target
            weights = torch.zeros_like(inters)

            for tgt_id in range(len(tgt_labels_i)):
                tgt_mask = tgt_ids == tgt_id
                weights[tgt_mask] = inters[tgt_mask] / inters[tgt_mask].sum()

            # Get weighted classification loss
            cls_losses = weights * cls_losses.sum(dim=1)
            cls_loss = self.cls_weight * cls_losses.sum()
            cls_loss_dict['dfd_cls_loss'] += cls_loss

            # Get classification accuracy
            with torch.no_grad():
                cls_preds_i = torch.argmax(cls_logits_i, dim=1)
                cls_acc = torch.eq(cls_preds_i, cls_targets_i).sum() / len(cls_preds_i)
                cls_analysis_dict['dfd_cls_acc'] += 100 * cls_acc / batch_size

        return cls_loss_dict, cls_analysis_dict

    def obj_loss(self, obj_logits, tgt_boxes, inter_matrices, prior_boxes):
        """
        Gets objectness loss and corresponding objectness analysis.

        It also returns a list with target indices, such that the subsequent 'box_loss' method should not recompute it.

        Args:
            obj_logits (FloatTensor): Objectness logits of shape [batch_size, num_preds, 2].
            tgt_boxes (List): List [batch_size] with normalized Boxes structure of size [num_targets].
            inter_matrices (List): List [batch_size] with intersection matrices of shape [num_preds, num_targets+1].
            prior_boxes (Boxes): Boxes structure containing prior boxes (i.e. anchors) of size [num_preds].

        Returns:
            obj_loss_dict (Dict): Objectness loss dictionary containing following key:
                - dfd_obj_loss (FloatTensor): weighted objectness loss of shape [1].

            obj_analysis_dict (Dict): Objectness analysis dictionary containing following key:
                - dfd_obj_acc (FloatTensor): objectness accuracy of shape [1].

            ids (List): List [batch_size] with target indices corresponding to each prediction of shape [num_preds].
        """

        # Initialize empty objectness loss and analysis dictionaries
        tensor_kwargs = {'dtype': obj_logits.dtype, 'device': obj_logits.device}
        obj_loss_dict = {'dfd_obj_loss': torch.zeros(1, **tensor_kwargs)}
        obj_analysis_dict = {'dfd_obj_acc': torch.zeros(1, **tensor_kwargs)}

        # Initialize empty ids list
        ids = []

        # Get batch size and areas of prior boxes
        batch_size = len(obj_logits)
        prior_areas = prior_boxes.area()

        # Get unique prior box sizes in (width, height) format
        prior_sizes = prior_boxes.to_format('cxcywh').boxes[:, 2:]
        prior_sizes, map_numel = torch.unique_consecutive(prior_sizes, return_counts=True, dim=0)

        # Get number of maps and number of predictions
        num_maps = len(prior_sizes)
        num_preds = len(prior_boxes)

        # Iterate over every batch entry
        for obj_logits_i, tgt_boxes_i, inter_matrix in zip(obj_logits, tgt_boxes, inter_matrices):

            # Get number of targets
            num_tgts = len(tgt_boxes_i)

            # Get map indices for positive maps corresponding to each target
            if num_tgts > 0:

                tgt_sizes_i = tgt_boxes_i.to_format('cxcywh').boxes[:, 2:]
                rel_sizes = prior_sizes[:, None] / tgt_sizes_i[None, :]
                rel_sizes, _ = torch.max(rel_sizes, dim=2)

                maxima, lower_map_ids = torch.where(rel_sizes <= 1, rel_sizes, torch.zeros_like(rel_sizes)).max(dim=0)
                lower_map_ids[maxima == 0] = -1
                upper_map_ids = lower_map_ids + 1

                map_ids = torch.stack([lower_map_ids, upper_map_ids], dim=1)
                map_ids = torch.clamp(map_ids, min=0, max=num_maps-1)

            # Get objectness mask
            obj_mask = torch.zeros(num_maps, num_tgts+1, dtype=torch.bool, device=obj_logits.device)

            if num_tgts > 0:
                obj_mask[map_ids, torch.arange(num_tgts)[:, None]] = True

            obj_mask = torch.repeat_interleave(obj_mask, map_numel, dim=0)

            # Get target ids and add them to ids list
            inters, tgt_ids = torch.max(inter_matrix, dim=1)
            pred_ids = torch.arange(num_preds).to(tgt_ids)
            tgt_ids = torch.where(obj_mask[pred_ids, tgt_ids], tgt_ids, num_tgts)
            ids.append(tgt_ids)

            # Get objectness targets
            obj_targets_i = torch.where(tgt_ids < num_tgts, 0, 1)
            obj_targets_oh_i = F.one_hot(obj_targets_i, 2).to(obj_logits_i.dtype)

            # Get unweighted objectness losses
            obj_kwargs = {'alpha': self.obj_focal_alpha, 'gamma': self.obj_focal_gamma, 'reduction': 'none'}
            obj_losses = sigmoid_focal_loss(obj_logits_i, obj_targets_oh_i, **obj_kwargs)

            # Get weights normalized per target
            inters[tgt_ids == num_tgts] = prior_areas[tgt_ids == num_tgts]
            weights = torch.zeros_like(inters)

            for tgt_id in range(num_tgts+1):
                tgt_mask = tgt_ids == tgt_id
                weights[tgt_mask] = inters[tgt_mask] / inters[tgt_mask].sum()

            # Get weighted objectness loss
            obj_losses = weights * obj_losses.sum(dim=1)
            obj_loss = self.obj_weight * obj_losses.sum()
            obj_loss_dict['dfd_obj_loss'] += obj_loss

            # Get objectness accuracy
            with torch.no_grad():
                obj_preds_i = torch.argmax(obj_logits_i, dim=1)
                obj_acc = torch.eq(obj_preds_i, obj_targets_i).sum() / len(obj_preds_i)
                obj_analysis_dict['dfd_obj_acc'] += 100 * obj_acc / batch_size

        return obj_loss_dict, obj_analysis_dict, ids

    def box_loss(self, box_logits, tgt_boxes, inter_matrices, prior_boxes, ids):
        """
        Gets bounding box loss and corresponding bounding box analysis.

        Args:
            box_logits (FloatTensor): Bounding box logits of shape [batch_size, num_preds, 4].
            tgt_boxes (List): List [batch_size] with normalized Boxes structure of size [num_targets].
            inter_matrices (List): List [batch_size] with intersection matrices of shape [num_preds, num_targets+1].
            prior_boxes (Boxes): Boxes structure containing prior boxes (i.e. anchors) of size [num_preds].
            ids (List): List [batch_size] with target indices corresponding to each prediction of shape [num_preds].

        Returns:
            box_loss_dict (Dict): Bounding box loss dictionary containing following key:
                - dfd_box_loss (FloatTensor): weighted bounding box loss of shape [1].

            box_analysis_dict (Dict): Bounding box analysis dictionary containing following key:
                - dfd_box_acc (FloatTensor): bounding box accuracy of shape [1].
        """

        # Initialize empty bounding box loss and analysis dictionaries
        tensor_kwargs = {'dtype': box_logits.dtype, 'device': box_logits.device}
        box_loss_dict = {'dfd_box_loss': torch.zeros(1, **tensor_kwargs)}
        box_analysis_dict = {'dfd_box_acc': torch.zeros(1, **tensor_kwargs)}

        # Get batch size
        batch_size = len(box_logits)

        # Iterate over every batch entry
        for box_logits_i, tgt_boxes_i, inter_matrix, tgt_ids in zip(box_logits, tgt_boxes, inter_matrices, ids):

            # Get number of targets
            num_tgts = len(tgt_boxes_i)

            # Skip batch entry if there are no targets
            if num_tgts == 0:
                box_loss_dict['dfd_box_loss'] += 0.0 * box_logits_i.sum()
                box_analysis_dict['dfd_box_acc'] += 100 / batch_size
                continue

            # Get selected bounding box logits
            box_logits_i = box_logits_i[tgt_ids < num_tgts]

            # Get corresponding bounding box targets
            prior_boxes_i = prior_boxes[tgt_ids < num_tgts]
            tgt_boxes_i = tgt_boxes_i[tgt_ids[tgt_ids < num_tgts]]
            box_targets_i = get_box_deltas(prior_boxes_i, tgt_boxes_i)

            # Get unweighted bounding box losses
            box_kwargs = {'beta': self.box_sl1_beta, 'reduction': 'none'}
            box_losses = smooth_l1_loss(box_logits_i, box_targets_i, **box_kwargs)

            # Get weights normalized per target
            inters, _ = torch.max(inter_matrix, dim=1)
            inters = inters[tgt_ids < num_tgts]
            weights = torch.zeros_like(inters)

            for tgt_id in range(num_tgts):
                tgt_mask = tgt_ids[tgt_ids < num_tgts] == tgt_id
                weights[tgt_mask] = inters[tgt_mask] / inters[tgt_mask].sum()

            # Get weighted bounding box loss
            box_losses = weights * box_losses.sum(dim=1)
            box_loss = self.box_weight * box_losses.sum()
            box_loss_dict['dfd_box_loss'] += box_loss

            # Get bounding box accuracy
            with torch.no_grad():
                pred_boxes_i = apply_box_deltas(box_logits_i, prior_boxes_i)
                box_acc = box_iou(pred_boxes_i, tgt_boxes_i).diag().mean()
                box_analysis_dict['dfd_box_acc'] += 100 * box_acc / batch_size

        return box_loss_dict, box_analysis_dict

    def forward(self, feat_maps, tgt_dict=None, **kwargs):
        """
        Forward method of the DFD module.

        Args:
            feat_maps (List): List of size [num_maps] with feature maps of shape [batch_size, feat_size, fH, fW].

            tgt_dict (Dict): Optional target dictionary containing at least following keys:
                - labels (List): list of size [batch_size] with class indices of shape [num_targets];
                - boxes (List): list of size [batch_size] with normalized Boxes structure of size [num_targets].

        Returns:
            * If tgt_dict is not None (i.e. during training and validation):
                loss_dict (Dict): Loss dictionary containing following keys:
                    - dfd_cls_loss (FloatTensor): weighted classification loss of shape [1];
                    - dfd_obj_loss (FloatTensor): weighted objectness loss of shape [1];
                    - dfd_box_loss (FloatTensor): weighted bounding box loss of shape [1].

                analysis_dict (Dict): Analysis dictionary containing following keys:
                    - dfd_cls_acc (FloatTensor): classification accuracy of shape [1];
                    - dfd_obj_acc (FloatTensor): objectness accuracy of shape [1];
                    - dfd_box_acc (FloatTensor): bounding box accuracy of shape [1].

            * If tgt_dict is None (i.e. during testing and possibly during validation):
                pred_dicts (List): List of prediction dictionaries with each dictionary containing following keys:
                    - labels (LongTensor): predicted class indices of shape [num_preds_total];
                    - boxes (Boxes): structure containing axis-aligned bounding boxes of size [num_preds_total];
                    - scores (FloatTensor): normalized prediction scores of shape [num_preds_total];
                    - batch_ids (LongTensor): batch indices of predictions of shape [num_preds_total].
        """

        # Get batch size, device and number of elements per map
        batch_size = len(feat_maps[0])
        device = feat_maps[0].device
        map_numel = torch.tensor([feat_map.flatten(2).shape[-1] for feat_map in feat_maps]).to(device)

        # Get position feature maps and position id maps
        pos_feat_maps, pos_id_maps = sine_pos_encodings(feat_maps, normalize=True)

        # Get prior boxes
        prior_cxcy = torch.cat([pos_id_map.flatten(1).t() for pos_id_map in pos_id_maps], dim=0)
        prior_wh = torch.tensor([[1/s for s in feat_map.shape[:1:-1]] for feat_map in feat_maps]).to(prior_cxcy)
        prior_wh = torch.repeat_interleave(prior_wh, map_numel, dim=0)

        prior_boxes = torch.cat([prior_cxcy, prior_wh], dim=1)
        prior_boxes = Boxes(prior_boxes, format='cxcywh', normalized='img_with_padding')

        # Get classification, objectness and bounding box logits
        cls_logits = torch.cat([cls_map.flatten(2).permute(0, 2, 1) for cls_map in self.cls_head(feat_maps)], dim=1)
        obj_logits = torch.cat([obj_map.flatten(2).permute(0, 2, 1) for obj_map in self.obj_head(feat_maps)], dim=1)
        box_logits = torch.cat([box_map.flatten(2).permute(0, 2, 1) for box_map in self.box_head(feat_maps)], dim=1)

        # Get loss and analysis dictionaries (trainval only)
        if tgt_dict is not None:

            # Get intersection matrices between prior boxes and target boxes
            inter_matrices = [box_intersection(prior_boxes, tgt_boxes_i) for tgt_boxes_i in tgt_dict['boxes']]

            # Add column with estimated background intersections (assuming no overlap between target boxes)
            prior_areas = prior_boxes.area()

            for i in range(batch_size):
                bg_inters = torch.clamp(prior_areas - inter_matrices[i].sum(dim=1), min=0)
                inter_matrices[i] = torch.cat([inter_matrices[i], bg_inters[:, None]], dim=1)

            # Initialize empty loss and analysis dictionaries
            loss_dict = {}
            analysis_dict = {}

            # Get classification loss and analysis
            tgt_labels = tgt_dict['labels']
            cls_loss_dict, cls_analysis_dict = self.cls_loss(cls_logits, tgt_labels, inter_matrices)
            loss_dict.update(cls_loss_dict)
            analysis_dict.update(cls_analysis_dict)

            # Get objectness loss and analysis
            tgt_boxes = tgt_dict['boxes']
            obj_loss_dict, obj_analysis_dict, ids = self.obj_loss(obj_logits, tgt_boxes, inter_matrices, prior_boxes)
            loss_dict.update(obj_loss_dict)
            analysis_dict.update(obj_analysis_dict)

            # Get bounding box loss and analysis
            box_loss_dict, box_analysis_dict = self.box_loss(box_logits, tgt_boxes, inter_matrices, prior_boxes, ids)
            loss_dict.update(box_loss_dict)
            analysis_dict.update(box_analysis_dict)

            return loss_dict, analysis_dict

        # Get list of prediction dictionaries (validation/testing)
        if tgt_dict is None:

            # Get predicted labels and corresponding classification scores
            cls_scores, labels = cls_logits[:, :, :-1].max(dim=2)
            cls_scores = cls_scores.sigmoid()

            # Get objectness scores
            obj_scores = obj_logits[:, :, 0].sigmoid()

            # Initialize list of prediciton dictionaries
            pred_dicts = []

            # Get prediction dictionaries corresponding to different scoring mechanisms
            for scores in [cls_scores, obj_scores]:

                # Initialize prediction dictionary
                pred_dict = {k: [] for k in ['labels', 'boxes', 'scores', 'batch_ids']}

                # Iterate over every batch entry
                for i in range(batch_size):

                    # Only keep best candidates for NMS
                    scores_i, sort_ids = scores[i].sort(descending=True)
                    candidate_ids = sort_ids[:self.inf_nms_candidates]

                    # Perform NMS and only keep the requested number of best remaining detections
                    boxes_i = apply_box_deltas(box_logits[i], prior_boxes)
                    boxes_i = boxes_i[candidate_ids].to_format('xyxy')

                    scores_i = scores_i[:self.inf_nms_candidates]
                    labels_i = labels[i][candidate_ids]

                    keep_ids = batched_nms(boxes_i.boxes, scores_i, labels_i, iou_threshold=self.inf_iou_threshold)
                    keep_ids = keep_ids[:self.inf_max_detections]

                    # Add final predictions to their corresponding lists
                    pred_dict['labels'].append(labels_i[keep_ids])
                    pred_dict['boxes'].append(boxes_i[keep_ids])
                    pred_dict['scores'].append(scores_i[keep_ids])
                    pred_dict['batch_ids'].append(torch.full_like(keep_ids, i, dtype=torch.int64))

                # Concatenate different batch entry predictions
                pred_dict.update({k: torch.cat(v, dim=0) for k, v in pred_dict.items() if k != 'boxes'})
                pred_dict['boxes'] = Boxes.cat(pred_dict['boxes'])

                # Add prediction dictionary to list of prediction dictionaries
                pred_dicts.append(pred_dict)

            return pred_dicts

    def visualize(self, images, pred_dicts, tgt_dict, score_treshold=0.4):
        """
        Draws predicted and target bounding boxes on given full-resolution images.

        Boxes must have a score of at least the score threshold to be drawn. Target boxes get a default 100% score.

        Args:
            images (Images): Images structure containing the batched images.

            pred_dicts (List): List of prediction dictionaries with each dictionary containing following keys:
                - labels (LongTensor): predicted class indices of shape [num_preds_total];
                - boxes (Boxes): structure containing axis-aligned bounding boxes of size [num_preds_total];
                - scores (FloatTensor): normalized prediction scores of shape [num_preds_total];
                - batch_ids (LongTensor): batch indices of predictions of shape [num_preds_total].

            tgt_dict (Dict): Target dictionary containing at least following keys:
                - labels (List): list of size [batch_size] with class indices of shape [num_targets];
                - boxes (List): list of size [batch_size] with normalized Boxes structure of size [num_targets];
                - sizes (LongTensor): tensor of shape [batch_size+1] with the cumulative target sizes of batch entries.

            score_threshold (float): Threshold indicating the minimum score for a box to be drawn (default=0.4).

        Returns:
            images_dict (Dict): Dictionary of images with drawn predicted and target bounding boxes.
        """

        # Get keys found in draw dictionaries
        draw_dict_keys = ['labels', 'boxes', 'scores', 'sizes']

        # Get draw dictionaries for predictions
        pred_draw_dicts = []

        for pred_dict in pred_dicts:
            pred_boxes = pred_dict['boxes'].to_img_scale(images).to_format('xyxy')
            well_defined = pred_boxes.well_defined()

            pred_scores = pred_dict['scores'][well_defined]
            sufficient_score = pred_scores >= score_treshold

            pred_labels = pred_dict['labels'][well_defined][sufficient_score]
            pred_boxes = pred_boxes.boxes[well_defined][sufficient_score]
            pred_scores = pred_scores[sufficient_score]
            pred_batch_ids = pred_dict['batch_ids'][well_defined][sufficient_score]

            pred_sizes = [0] + [(pred_batch_ids == i).sum() for i in range(len(images))]
            pred_sizes = torch.tensor(pred_sizes).cumsum(dim=0).to(tgt_dict['sizes'])

            draw_dict_values = [pred_labels, pred_boxes, pred_scores, pred_sizes]
            pred_draw_dict = {k: v for k, v in zip(draw_dict_keys, draw_dict_values)}
            pred_draw_dicts.append(pred_draw_dict)

        # Get draw dictionary for targets
        tgt_labels = torch.cat(tgt_dict['labels'])
        tgt_boxes = Boxes.cat(tgt_dict['boxes']).to_img_scale(images).to_format('xyxy').boxes
        tgt_scores = torch.ones_like(tgt_labels, dtype=torch.float)
        tgt_sizes = tgt_dict['sizes']

        draw_dict_values = [tgt_labels, tgt_boxes, tgt_scores, tgt_sizes]
        tgt_draw_dict = {k: v for k, v in zip(draw_dict_keys, draw_dict_values)}

        # Combine draw dicationaries and get corresponding dictionary names
        draw_dicts = [*pred_draw_dicts, tgt_draw_dict]
        dict_names = [f'pred_{i+1}'for i in range(len(pred_dicts))] + ['tgt']

        # Get image sizes without padding in (width, height) format
        img_sizes = images.size(with_padding=False)

        # Get and convert tensor with images
        images = images.images.clone().permute(0, 2, 3, 1)
        images = (images * 255).to(torch.uint8).cpu().numpy()

        # Get number of images and initialize images dictionary
        num_images = len(images)
        images_dict = {}

        # Draw bounding boxes on images and add them to images dictionary
        for dict_name, draw_dict in zip(dict_names, draw_dicts):
            sizes = draw_dict['sizes']

            for image_id, i0, i1 in zip(range(num_images), sizes[:-1], sizes[1:]):
                visualizer = Visualizer(images[image_id], metadata=self.metadata)

                img_size = img_sizes[image_id]
                img_size = (img_size[1], img_size[0])

                img_labels = draw_dict['labels'][i0:i1].cpu().numpy()
                img_boxes = draw_dict['boxes'][i0:i1].cpu().numpy()
                img_scores = draw_dict['scores'][i0:i1].cpu().numpy()

                instances = Instances(img_size, pred_classes=img_labels, pred_boxes=img_boxes, scores=img_scores)
                visualizer.draw_instance_predictions(instances)

                annotated_image = visualizer.output.get_image()
                images_dict[f'dfd_{dict_name}_{image_id}'] = annotated_image[:img_size[0], :img_size[1], :]

        return images_dict
