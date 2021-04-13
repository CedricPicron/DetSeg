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
from torch.nn.parameter import Parameter

from models.functional.position import sine_pos_encodings
from models.modules.convolution import BottleneckConv, ProjConv
from structures.boxes import apply_edge_dists, Boxes, box_iou, get_edge_dists, pts_inside_boxes


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

        pos_head (nn.Sequential): Module computing the position encodings.

        ins_head (nn.Sequential): Module computing the instance features.
        ins_bias (nn.Parameter): Bias value related to the probability of two features belonging to the same instance.
        ins_focal_alpha (float): Alpha value of the sigmoid focal loss used by the instance head.
        ins_focal_gamma (float): Gamma value of the sigmoid focal loss used by the instance head.
        ins_weight (float): Factor weighting the instance loss.

        inf_nms_candidates (int): Maximum number of candidates retained for NMS during inference.
        inf_nms_threshold (float): Value determining the IoU threshold of NMS during inference.
        inf_ins_candidates (int): Maximum number of candidates retained for instance head duplicate removal.
        inf_ins_threshold (float): Value determining whether two features are considered duplicates or not.
        inf_max_detections (int): Maximum number of detections retained during inference.

        metadata (detectron2.data.Metadata): Metadata instance containing additional dataset information.
    """

    def __init__(self, in_feat_size, cls_dict, obj_dict, box_dict, pos_dict, ins_dict, inf_dict, metadata):
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

            pos_dict (Dict): Position head dictionary containing following keys:
                - feat_size (int): hidden and output feature size of the position head;
                - norm (str): string specifying the type of normalization used within the position head;
                - kernel_size (int): kernel size used by the position head for both input and hidden layers;
                - bottle_size (int): bottleneck size used by bottleneck layers of the position head;
                - hidden_layers (int): number of hidden layers of the position head.

            ins_dict (Dict): Instance head dictionary containing following keys:
                - feat_size (int): hidden feature size of the instance head;
                - norm (str): string specifying the type of normalization used within the instance head;
                - prior_prob (float): prior instance probability;
                - kernel_size (int): kernel size used by hidden layers of the instance head;
                - bottle_size (int): bottleneck size used by bottleneck layers of the instance head;
                - hidden_layers (int): number of hidden layers of the instance head;
                - out_size (int): output feature size of the instance head;

                - focal_alpha (float): alpha value of the sigmoid focal loss used by the instance head;
                - focal_gamma (float): gamma value of the sigmoid focal loss used by the instance head;
                - weight (float): factor weighting the instance loss.

            inf_dict (Dict): Inference dictionary containing following keys:
                - nms_candidates (int): maximum number of candidates retained for NMS during inference;
                - nms_threshold (float): value determining the IoU threshold of NMS during inference;
                - ins_candidates (int): maximum number of candidates retained for instance head duplicate removal;
                - ins_threshold (float): value determining whether two features are considered duplicates or not;
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

        # Initialization of position head
        pos_feat_size = pos_dict['feat_size']
        pos_norm = pos_dict['norm']

        if pos_dict['kernel_size'] == 1:
            pos_hidden_layer = ProjConv(pos_feat_size, norm=pos_norm, skip=True)
        else:
            bottleneck_kwargs = {'kernel_size': pos_dict['kernel_size'], 'norm': pos_norm, 'skip': True}
            pos_hidden_layer = BottleneckConv(pos_feat_size, pos_dict['bottle_size'], **bottleneck_kwargs)

        pos_in_layer = ProjConv(2, pos_feat_size, pos_dict['kernel_size'], norm=pos_norm, skip=False)
        pos_hidden_layers = nn.Sequential(*[deepcopy(pos_hidden_layer) for _ in range(pos_dict['hidden_layers'])])
        pos_head_dict = OrderedDict([('in', pos_in_layer), ('hidden', pos_hidden_layers)])
        self.pos_head = nn.Sequential(pos_head_dict)

        # Initialization of instance head
        ins_feat_size = ins_dict['feat_size']
        ins_norm = ins_dict['norm']

        ins_in_size = in_feat_size + pos_feat_size
        ins_out_size = ins_dict['out_size']

        ins_in_layer = ProjConv(ins_in_size, ins_feat_size, norm=ins_norm, skip=False)
        ins_out_layer = ProjConv(ins_feat_size, ins_out_size, norm=ins_norm, skip=False)

        if ins_dict['kernel_size'] == 1:
            ins_hidden_layer = ProjConv(ins_feat_size, norm=ins_norm, skip=True)
        else:
            bottleneck_kwargs = {'kernel_size': ins_dict['kernel_size'], 'norm': ins_norm, 'skip': True}
            ins_hidden_layer = BottleneckConv(ins_feat_size, ins_dict['bottle_size'], **bottleneck_kwargs)

        ins_hidden_layers = nn.Sequential(*[deepcopy(ins_hidden_layer) for _ in range(ins_dict['hidden_layers'])])
        ins_head_dict = OrderedDict([('in', ins_in_layer), ('hidden', ins_hidden_layers), ('out', ins_out_layer)])
        self.ins_head = nn.Sequential(ins_head_dict)

        ins_prior_prob = ins_dict['prior_prob']
        bias_value = -(math.log((1 - ins_prior_prob) / ins_prior_prob))
        self.ins_bias = Parameter(torch.tensor([bias_value]))

        self.ins_focal_alpha = ins_dict['focal_alpha']
        self.ins_focal_gamma = ins_dict['focal_gamma']
        self.ins_weight = ins_dict['weight']

        # Set inference attributes
        self.inf_nms_candidates = inf_dict['nms_candidates']
        self.inf_nms_threshold = inf_dict['nms_threshold']
        self.inf_ins_candidates = inf_dict['ins_candidates']
        self.inf_ins_threshold = inf_dict['ins_threshold']
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

    def cls_loss(self, cls_logits, tgt_labels, tgt_ids, weights):
        """
        Gets classification loss and corresponding classification analysis.

        Args:
            cls_logits (FloatTensor): Classification logits of shape [batch_size, num_feats, num_classes+1].
            tgt_labels (List): List [batch_size] with class indices of shape [num_targets].
            tgt_ids (List): List [batch_size] with target indices corresponding to each feature of shape [num_feats].
            weights (List): List [batch_size] with weights corresponding to each feature of shape [num_feats].

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
        for cls_logits_i, tgt_labels_i, tgt_ids_i, weights_i in zip(cls_logits, tgt_labels, tgt_ids, weights):

            # Append target with background label and get number of targets
            bg_label = torch.tensor([num_cls_labels-1]).to(tgt_labels_i)
            tgt_labels_i = torch.cat([tgt_labels_i, bg_label], dim=0)
            num_tgts = len(tgt_labels_i)

            # Get classification targets
            cls_targets_i = tgt_labels_i[tgt_ids_i]
            cls_targets_oh_i = F.one_hot(cls_targets_i, num_cls_labels).to(cls_logits_i.dtype)

            # Get unweighted classification losses
            cls_kwargs = {'alpha': self.cls_focal_alpha, 'gamma': self.cls_focal_gamma, 'reduction': 'none'}
            cls_losses = sigmoid_focal_loss(cls_logits_i, cls_targets_oh_i, **cls_kwargs)

            # Get weights normalized per target
            norm_weights = torch.zeros_like(weights_i)

            for tgt_id in range(num_tgts):
                tgt_mask = tgt_ids_i == tgt_id
                norm_weights[tgt_mask] = weights_i[tgt_mask] / weights_i[tgt_mask].sum()

            # Get weighted classification loss
            cls_losses = norm_weights * cls_losses.sum(dim=1)
            cls_loss = self.cls_weight * cls_losses.sum()
            cls_loss_dict['dfd_cls_loss'] += cls_loss

            # Get classification accuracy
            with torch.no_grad():
                cls_preds_i = torch.argmax(cls_logits_i, dim=1)
                cls_acc = torch.eq(cls_preds_i, cls_targets_i).sum() / len(cls_preds_i)
                cls_analysis_dict['dfd_cls_acc'] += 100 * cls_acc / batch_size

        return cls_loss_dict, cls_analysis_dict

    def obj_loss(self, obj_logits, tgt_boxes, tgt_ids, weights, feat_wh):
        """
        Gets objectness loss and corresponding objectness analysis.

        It also returns a list with objectness indices corresponding to non-objects (0) and objects (1).

        Args:
            obj_logits (FloatTensor): Objectness logits of shape [batch_size, num_feats, 2].
            tgt_boxes (List): List [batch_size] with normalized Boxes structure of size [num_targets].
            tgt_ids (List): List [batch_size] with target indices corresponding to each feature of shape [num_feats].
            weights (List): List [batch_size] with weights corresponding to each feature of shape [num_feats].
            feat_wh (FloatTensor): Feature widths and heights of shape [num_feats, 2].

        Returns:
            obj_loss_dict (Dict): Objectness loss dictionary containing following key:
                - dfd_obj_loss (FloatTensor): weighted objectness loss of shape [1].

            obj_analysis_dict (Dict): Objectness analysis dictionary containing following key:
                - dfd_obj_acc (FloatTensor): objectness accuracy of shape [1].

            tgt_map_ids (List): List [batch_size] with map indices corresponding to each target of shape [num_targets].
            obj_targets (List): List [batch_size] with objectness target labels of shape [num_feats].
        """

        # Initialize empty objectness loss and analysis dictionaries
        tensor_kwargs = {'dtype': obj_logits.dtype, 'device': obj_logits.device}
        obj_loss_dict = {'dfd_obj_loss': torch.zeros(1, **tensor_kwargs)}
        obj_analysis_dict = {'dfd_obj_acc': torch.zeros(1, **tensor_kwargs)}

        # Initialize empty lists for target map ids and objectness targets
        tgt_map_ids = []
        obj_targets = []

        # Get batch size, feature indices and feature areas
        batch_size = len(obj_logits)
        feat_ids = torch.arange(len(feat_wh))
        feat_areas = feat_wh[:, 0] * feat_wh[:, 1]

        # Get unique feature widths and heights and get number of maps
        feat_wh, map_numel = torch.unique_consecutive(feat_wh, return_counts=True, dim=0)
        num_maps = len(feat_wh)

        # Iterate over every batch entry
        for obj_logits_i, tgt_boxes_i, tgt_ids_i, weights_i in zip(obj_logits, tgt_boxes, tgt_ids, weights):

            # Get number of targets
            num_tgts = len(tgt_boxes_i)

            # Get target map ids and objectness targets
            if num_tgts > 0:
                tgt_sizes_i = tgt_boxes_i.to_format('cxcywh').boxes[:, 2:]
                rel_sizes = feat_wh[:, None] / tgt_sizes_i[None, :]
                rel_sizes, _ = torch.max(rel_sizes, dim=2)

                rel_sizes[rel_sizes > 1] = 0
                tgt_map_ids_i = torch.argmax(rel_sizes, dim=0)

                obj_targets_i = torch.zeros(num_maps, num_tgts+1).to(tgt_ids_i)
                obj_targets_i[tgt_map_ids_i, torch.arange(num_tgts)] = 1
                obj_targets_i = torch.repeat_interleave(obj_targets_i, map_numel, dim=0)
                obj_targets_i = obj_targets_i[feat_ids, tgt_ids_i]

            else:
                tgt_map_ids_i = torch.zeros(0, device=tgt_ids_i.device)
                obj_targets_i = torch.zeros_like(tgt_ids_i)

            # Add target map ids and objectness targets to their corresponding lists
            tgt_map_ids.append(tgt_map_ids_i)
            obj_targets.append(obj_targets_i)

            # Get one-hot objectness targets
            obj_targets_oh_i = F.one_hot(obj_targets_i, 2).to(obj_logits_i.dtype)

            # Get unweighted objectness losses
            obj_kwargs = {'alpha': self.obj_focal_alpha, 'gamma': self.obj_focal_gamma, 'reduction': 'none'}
            obj_losses = sigmoid_focal_loss(obj_logits_i, obj_targets_oh_i, **obj_kwargs)

            # Get weights normalized per target
            obj_tgt_ids = torch.where(obj_targets_i == 1, tgt_ids_i, -1)
            norm_weights = torch.zeros_like(weights_i)

            for obj_tgt_id in range(num_tgts):
                obj_tgt_mask = obj_tgt_ids == obj_tgt_id
                norm_weights[obj_tgt_mask] = weights_i[obj_tgt_mask] / weights_i[obj_tgt_mask].sum()

            non_obj_mask = obj_tgt_ids == -1
            norm_weights[non_obj_mask] = feat_areas[non_obj_mask] / feat_areas[non_obj_mask].sum()

            # Get weighted objectness loss
            obj_losses = norm_weights * obj_losses.sum(dim=1)
            obj_loss = self.obj_weight * obj_losses.sum()
            obj_loss_dict['dfd_obj_loss'] += obj_loss

            # Get objectness accuracy
            with torch.no_grad():
                obj_preds_i = torch.argmax(obj_logits_i, dim=1)
                obj_acc = torch.eq(obj_preds_i, obj_targets_i).sum() / len(obj_preds_i)
                obj_analysis_dict['dfd_obj_acc'] += 100 * obj_acc / batch_size

        return obj_loss_dict, obj_analysis_dict, tgt_map_ids, obj_targets

    def box_loss(self, box_logits, tgt_boxes, tgt_ids, weights, feat_cts, feat_wh, obj_targets):
        """
        Gets bounding box loss and corresponding bounding box analysis.

        Args:
            box_logits (FloatTensor): Bounding box logits of shape [batch_size, num_feats, 4].
            tgt_boxes (List): List [batch_size] with normalized Boxes structure of size [num_targets].
            tgt_ids (List): List [batch_size] with target indices corresponding to each feature of shape [num_feats].
            weights (List): List [batch_size] with weights corresponding to each feature of shape [num_feats].
            feat_cts (FloatTensor): Feature centers in (x, y) format of shape [num_feats, 2].
            feat_wh (FloatTensor): Feature widths and heights of shape [num_feats, 2].
            obj_targets (List): List [batch_size] with objectness target labels of shape [num_feats].

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

        # Get batch size and zip tuple
        batch_size = len(box_logits)
        zip_tuple = (box_logits, tgt_boxes, tgt_ids, weights, obj_targets)

        # Iterate over every batch entry
        for box_logits_i, tgt_boxes_i, tgt_ids_i, weights_i, obj_targets_i in zip(*zip_tuple):

            # Get number of targets
            num_tgts = len(tgt_boxes_i)

            # Skip batch entry if there are no targets
            if num_tgts == 0:
                box_loss_dict['dfd_box_loss'] += 0.0 * box_logits_i.sum()
                box_analysis_dict['dfd_box_acc'] += 100 / batch_size
                continue

            # Get selected bounding box logits
            box_mask = obj_targets_i == 1
            box_logits_i = box_logits_i[box_mask]

            # Get corresponding bounding box targets
            box_tgt_ids = tgt_ids_i[box_mask]
            tgt_boxes_i = tgt_boxes_i[box_tgt_ids]
            box_targets_i = get_edge_dists(feat_cts[box_mask], tgt_boxes_i, scales=feat_wh[box_mask])

            # Get unweighted bounding box losses
            box_kwargs = {'beta': self.box_sl1_beta, 'reduction': 'none'}
            box_losses = smooth_l1_loss(box_logits_i, box_targets_i, **box_kwargs)

            # Get weights normalized per target
            weights_i = weights_i[box_mask]
            norm_weights = torch.zeros_like(weights_i)

            for tgt_id in range(num_tgts):
                tgt_mask = box_tgt_ids == tgt_id
                norm_weights[tgt_mask] = weights_i[tgt_mask] / weights_i[tgt_mask].sum()

            # Get weighted bounding box loss
            box_losses = norm_weights * box_losses.sum(dim=1)
            box_loss = self.box_weight * box_losses.sum()
            box_loss_dict['dfd_box_loss'] += box_loss

            # Get bounding box accuracy
            with torch.no_grad():
                if box_mask.sum() > 0:
                    pred_kwargs = {'scales': feat_wh[box_mask], 'normalized': 'img_with_padding'}
                    pred_boxes_i = apply_edge_dists(box_logits_i, feat_cts[box_mask], **pred_kwargs)
                    box_acc = box_iou(pred_boxes_i, tgt_boxes_i).diag().mean()
                    box_analysis_dict['dfd_box_acc'] += 100 * box_acc / batch_size

        return box_loss_dict, box_analysis_dict

    def ins_loss(self, ins_feats, tgt_boxes, tgt_ids, weights, feat_wh, tgt_map_ids):
        """
        Gets instance loss and corresponding instance analysis.

        Args:
            ins_feats (FloatTensor): Features for instance separation of shape [batch_size, num_feats, ins_feat_size].
            tgt_boxes (List): List [batch_size] with normalized Boxes structure of size [num_targets].
            tgt_ids (List): List [batch_size] with target indices corresponding to each feature of shape [num_feats].
            weights (List): List [batch_size] with weights corresponding to each feature of shape [num_feats].
            feat_wh (FloatTensor): Feature widths and heights of shape [num_feats, 2].
            tgt_map_ids (List): List [batch_size] with map indices corresponding to each target of shape [num_targets].

        Returns:
            ins_loss_dict (Dict): Instance loss dictionary containing following key:
                - dfd_ins_loss (FloatTensor): weighted instance loss of shape [1].

            ins_analysis_dict (Dict): Instance analysis dictionary containing following key:
                - dfd_ins_acc (FloatTensor): instance accuracy of shape [1].
        """

        # Initialize empty instance loss and analysis dictionaries
        tensor_kwargs = {'dtype': ins_feats.dtype, 'device': ins_feats.device}
        ins_loss_dict = {'dfd_ins_loss': torch.zeros(1, **tensor_kwargs)}
        ins_analysis_dict = {'dfd_ins_acc': torch.zeros(1, **tensor_kwargs)}

        # Get batch size, number of features in total and zip tuple
        batch_size = len(ins_feats)
        num_feats = len(feat_wh)
        zip_tuple = (ins_feats, tgt_boxes, tgt_ids, weights, tgt_map_ids)

        # Get feature areas and get unique feature widths and heights
        feat_areas = feat_wh[:, 0] * feat_wh[:, 1]
        feat_wh, map_numel = torch.unique_consecutive(feat_wh, return_counts=True, dim=0)

        # Get cumulative number of elements per map
        cum_map_numel = torch.tensor([0, *map_numel.tolist()]).to(map_numel)
        cum_map_numel = torch.cumsum(cum_map_numel[:-1], dim=0)

        # Iterate over every batch entry
        for ins_feats_i, tgt_boxes_i, tgt_ids_i, weights_i, tgt_map_ids_i in zip(*zip_tuple):

            # Get number of targets
            num_tgts = len(tgt_boxes_i)

            # Skip batch entry if there are no targets
            if num_tgts == 0:
                ins_loss_dict['dfd_ins_loss'] += 0.0 * ins_feats_i.sum()
                ins_analysis_dict['dfd_ins_acc'] += 100 / batch_size
                continue

            # Get ids of center feature for each target
            tgt_cts_i = tgt_boxes_i.to_format('cxcywh').boxes[:, :2]
            feat_wh_i = feat_wh[tgt_map_ids_i]
            ct_feat_ids_i = (tgt_cts_i // feat_wh_i).to(torch.int64)

            numel_row_i = (1/feat_wh_i[:, 0]).to(torch.int64)
            ct_feat_ids_i = ct_feat_ids_i[:, 0] + numel_row_i * ct_feat_ids_i[:, 1]
            ct_feat_ids_i = cum_map_numel[tgt_map_ids_i] + ct_feat_ids_i

            # Get instance logits
            ct_feats_i = ins_feats_i[ct_feat_ids_i]
            ins_logits_i = torch.mm(ct_feats_i, ins_feats_i.t()) + self.ins_bias
            ins_logits_i = ins_logits_i.view(-1, 1)

            # Get instance targets
            ins_targets_i = tgt_ids_i[ct_feat_ids_i, None] == tgt_ids_i[None, :]
            ins_targets_i = ins_targets_i.view(-1, 1).to(ins_logits_i.dtype)

            # Get unweighted instance losses
            ins_kwargs = {'alpha': self.ins_focal_alpha, 'gamma': self.ins_focal_gamma, 'reduction': 'none'}
            ins_losses = sigmoid_focal_loss(ins_logits_i, ins_targets_i, **ins_kwargs).squeeze(dim=1)

            # Get weights normalized per target
            ins_tgt_ids = tgt_ids_i[ct_feat_ids_i, None].expand(-1, num_feats).flatten()
            ins_tgt_ids = torch.where(ins_targets_i.squeeze(dim=1) == 1, ins_tgt_ids, -1)

            weights_i = weights_i[None, :].expand(num_tgts, -1).flatten()
            norm_weights = torch.zeros_like(weights_i)

            for ins_tgt_id in range(num_tgts):
                ins_tgt_mask = ins_tgt_ids == ins_tgt_id
                norm_weights[ins_tgt_mask] = weights_i[ins_tgt_mask] / weights_i[ins_tgt_mask].sum()

            non_ins_mask = ins_tgt_ids == -1
            feat_areas_i = feat_areas[None, :].expand(num_tgts, -1).flatten()
            norm_weights[non_ins_mask] = feat_areas_i[non_ins_mask] / feat_areas_i[non_ins_mask].sum()

            # Get weighted instance loss
            ins_losses = norm_weights * ins_losses
            ins_loss = self.ins_weight * ins_losses.sum()
            ins_loss_dict['dfd_ins_loss'] += ins_loss

            # Get instance accuracy
            with torch.no_grad():
                ins_preds_i = ins_logits_i >= 0
                ins_acc = torch.eq(ins_preds_i, ins_targets_i).sum() / len(ins_preds_i)
                ins_analysis_dict['dfd_ins_acc'] += 100 * ins_acc / batch_size

        return ins_loss_dict, ins_analysis_dict

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
                    - dfd_box_loss (FloatTensor): weighted bounding box loss of shape [1];
                    - dfd_ins_loss (FloatTensor): weighted instance loss of shape [1].

                analysis_dict (Dict): Analysis dictionary containing following keys:
                    - dfd_cls_acc (FloatTensor): classification accuracy of shape [1];
                    - dfd_obj_acc (FloatTensor): objectness accuracy of shape [1];
                    - dfd_box_acc (FloatTensor): bounding box accuracy of shape [1];
                    - dfd_ins_acc (FloatTensor): instance accuracy of shape [1].

            * If tgt_dict is None (i.e. during testing and possibly during validation):
                pred_dicts (List): List of prediction dictionaries with each dictionary containing following keys:
                    - labels (LongTensor): predicted class indices of shape [num_preds_total];
                    - boxes (Boxes): structure containing axis-aligned bounding boxes of size [num_preds_total];
                    - scores (FloatTensor): normalized prediction scores of shape [num_preds_total];
                    - batch_ids (LongTensor): batch indices of predictions of shape [num_preds_total].
        """

        # Get batch size
        batch_size = len(feat_maps[0])

        # Get feature centers
        _, feat_cts_maps = sine_pos_encodings(feat_maps, normalize=True)
        feat_cts = torch.cat([feat_cts_map.flatten(1).t() for feat_cts_map in feat_cts_maps], dim=0)

        # Get feature widths and heights
        map_numel = torch.tensor([feat_map.flatten(2).shape[-1] for feat_map in feat_maps]).to(feat_cts.device)
        feat_wh = torch.tensor([[1/s for s in feat_map.shape[:1:-1]] for feat_map in feat_maps]).to(feat_cts)
        feat_wh = torch.repeat_interleave(feat_wh, map_numel, dim=0)

        # Get classification, objectness and bounding box logits
        cls_logits = torch.cat([cls_map.flatten(2).permute(0, 2, 1) for cls_map in self.cls_head(feat_maps)], dim=1)
        obj_logits = torch.cat([obj_map.flatten(2).permute(0, 2, 1) for obj_map in self.obj_head(feat_maps)], dim=1)
        box_logits = torch.cat([box_map.flatten(2).permute(0, 2, 1) for box_map in self.box_head(feat_maps)], dim=1)

        # Get feature maps augmented with position encodings
        pos_maps = [self.pos_head(feat_cts_map[None, :]) for feat_cts_map in feat_cts_maps]
        pos_maps = [pos_map.expand(batch_size, -1, -1, -1) for pos_map in pos_maps]
        aug_maps = [torch.cat([feat_map, pos_map], dim=1) for feat_map, pos_map in zip(feat_maps, pos_maps)]

        # Get instance features
        ins_feats = torch.cat([ins_map.flatten(2).permute(0, 2, 1) for ins_map in self.ins_head(aug_maps)], dim=1)

        # Get loss and analysis dictionaries (trainval only)
        if tgt_dict is not None:

            # Get total number of features and feature indices
            num_feats = len(feat_cts)
            feat_ids = torch.arange(num_feats, device=feat_cts.device)

            # Get feature boxes and corresponding areas
            feat_boxes = torch.cat([feat_cts, feat_wh], dim=1)
            feat_boxes = Boxes(feat_boxes, format='cxcywh', normalized='img_with_padding')
            feat_areas = feat_boxes.area()

            # Initialize empty lists
            tgt_ids = []
            weights = []

            # Iterate over every batch entry
            for tgt_boxes_i in tgt_dict['boxes']:

                # Get number of targets
                num_tgts = len(tgt_boxes_i)

                # Get target indices and weights when there are targets
                if num_tgts > 0:

                    # Check which feature centers lie inside target boxes
                    inside_tgt_boxes_i = pts_inside_boxes(feat_cts, tgt_boxes_i)

                    # Get IoU and intersection matrices between feature boxes and target boxes
                    ious_i, inters_i = box_iou(feat_boxes, tgt_boxes_i, return_inters=True)

                    # Get target ids and weights corresponding to each feature
                    max_values, tgt_ids_i = torch.max(inside_tgt_boxes_i * ious_i, dim=1)
                    weights_i = inters_i[feat_ids, tgt_ids_i]

                    # Assign features with center outside every target box to background
                    background = max_values == 0
                    num_tgts = len(tgt_boxes_i)
                    tgt_ids_i[background] = num_tgts

                    bg_weights_i = torch.clamp(feat_areas - inters_i.sum(dim=1), min=0)
                    weights_i[background] = bg_weights_i[background]

                # Get target indices and weights when there are no targets
                else:
                    tgt_ids_i = torch.zeros_like(feat_ids)
                    weights_i = feat_areas

                # Add results to their corresponding lists
                tgt_ids.append(tgt_ids_i)
                weights.append(weights_i)

            # Initialize empty loss and analysis dictionaries
            loss_dict = {}
            analysis_dict = {}

            # Get classification loss and analysis
            tgt_labels = tgt_dict['labels']
            cls_loss_dict, cls_analysis_dict = self.cls_loss(cls_logits, tgt_labels, tgt_ids, weights)
            loss_dict.update(cls_loss_dict)
            analysis_dict.update(cls_analysis_dict)

            # Get objectness loss and analysis
            tgt_boxes = tgt_dict['boxes']
            obj_args = (obj_logits, tgt_boxes, tgt_ids, weights, feat_wh)
            obj_loss_dict, obj_analysis_dict, tgt_map_ids, obj_targets = self.obj_loss(*obj_args)
            loss_dict.update(obj_loss_dict)
            analysis_dict.update(obj_analysis_dict)

            # Get bounding box loss and analysis
            box_args = (box_logits, tgt_boxes, tgt_ids, weights, feat_cts, feat_wh, obj_targets)
            box_loss_dict, box_analysis_dict = self.box_loss(*box_args)
            loss_dict.update(box_loss_dict)
            analysis_dict.update(box_analysis_dict)

            # Get instance loss and analysis
            ins_args = (ins_feats, tgt_boxes, tgt_ids, weights, feat_wh, tgt_map_ids)
            ins_loss_dict, ins_analysis_dict = self.ins_loss(*ins_args)
            loss_dict.update(ins_loss_dict)
            analysis_dict.update(ins_analysis_dict)

            return loss_dict, analysis_dict

        # Get list of prediction dictionaries (validation/testing)
        if tgt_dict is None:

            # Get predicted classification labels
            cls_scores = cls_logits.softmax(dim=2)
            labels = cls_scores[:, :, :-1].argmax(dim=2)

            # Get objectness scores
            obj_scores = obj_logits.softmax(dim=2)
            obj_scores = obj_logits[:, :, 1]

            # Initialize list of prediciton dictionaries
            pred_dict = {k: [] for k in ['labels', 'boxes', 'scores', 'batch_ids']}
            pred_dicts = [deepcopy(pred_dict) for _ in range(4)]

            # Iterate over every batch entry
            for i in range(batch_size):

                # Get ids of highest object scores and get bounding boxes from box logits
                sort_ids = obj_scores[i].argsort(descending=True)
                boxes_i = apply_edge_dists(box_logits[i], feat_cts, scales=feat_wh, normalized='img_with_padding')

                # 1.1 Remove duplicate detections using NMS
                nms_candidate_ids = sort_ids[:self.inf_nms_candidates]
                nms_boxes = boxes_i[nms_candidate_ids].to_format('xyxy')
                nms_scores = obj_scores[i][nms_candidate_ids]
                nms_labels = labels[i][nms_candidate_ids]

                nms_ids = batched_nms(nms_boxes.boxes, nms_scores, nms_labels, iou_threshold=self.inf_nms_threshold)
                nms_ids = nms_ids[:self.inf_max_detections]

                nms_labels = nms_labels[nms_ids]
                nms_box_boxes = nms_boxes[nms_ids]
                nms_scores = nms_scores[nms_ids]

                pred_dicts[0]['labels'].append(nms_labels)
                pred_dicts[0]['boxes'].append(nms_box_boxes)
                pred_dicts[0]['scores'].append(nms_scores)
                pred_dicts[0]['batch_ids'].append(torch.full_like(nms_ids, i, dtype=torch.int64))

                # 1.2 Remove duplicate detections using instance head
                ins_candidate_ids = sort_ids[:self.inf_ins_candidates]
                ins_ins_feats = ins_feats[i][ins_candidate_ids]

                ins_dup_matrix = torch.mm(ins_ins_feats, ins_ins_feats.t()) + self.ins_bias
                ins_dup_matrix = ins_dup_matrix.sigmoid() >= self.inf_ins_threshold
                ins_dup_matrix = torch.tril(ins_dup_matrix, diagonal=-1)
                non_duplicates = ins_dup_matrix.sum(dim=1) == 0

                ins_ids = torch.arange(len(ins_candidate_ids)).to(ins_candidate_ids)
                ins_ids = ins_ids[non_duplicates]
                ins_ids = ins_ids[:self.inf_max_detections]

                ins_labels = labels[i][ins_candidate_ids][ins_ids]
                ins_box_boxes = boxes_i[ins_candidate_ids][ins_ids].to_format('xyxy')
                ins_scores = obj_scores[i][ins_candidate_ids][ins_ids]

                pred_dicts[1]['labels'].append(ins_labels)
                pred_dicts[1]['boxes'].append(ins_box_boxes)
                pred_dicts[1]['scores'].append(ins_scores)
                pred_dicts[1]['batch_ids'].append(torch.full_like(ins_ids, i, dtype=torch.int64))

                # 2. Use instance head to infer instance segmenations and corresponding bounding boxes
                bg_logits = cls_logits[i, :, -1:]
                ins_boxes_dict = {}

                nms_feat_cts = feat_cts[nms_candidate_ids][nms_ids]
                nms_feat_wh = feat_wh[nms_candidate_ids][nms_ids]
                nms_left_top = nms_feat_cts - nms_feat_wh/2
                nms_right_bottom = nms_feat_cts + nms_feat_wh/2
                ins_boxes_dict['nms'] = torch.cat([nms_left_top, nms_right_bottom], dim=1)

                ins_feat_cts = feat_cts[ins_candidate_ids][ins_ids]
                ins_feat_wh = feat_wh[ins_candidate_ids][ins_ids]
                ins_left_top = ins_feat_cts - ins_feat_wh/2
                ins_right_bottom = ins_feat_cts + ins_feat_wh/2
                ins_boxes_dict['ins'] = torch.cat([ins_left_top, ins_right_bottom], dim=1)

                nms_ins_feats = ins_feats[i][nms_candidate_ids][nms_ids]
                ins_ins_feats = ins_ins_feats[ins_ids]
                ins_feats_list = [nms_ins_feats, ins_ins_feats]

                for key, obj_ins_feats in zip(('nms', 'ins'), ins_feats_list):
                    ins_boxes = ins_boxes_dict[key]

                    ins_logits = torch.mm(ins_feats[i], obj_ins_feats.t()) + self.ins_bias
                    ins_logits = torch.cat([ins_logits, bg_logits], dim=1)

                    ins_scores = torch.softmax(ins_logits, dim=1)
                    det_ids = torch.argmax(ins_scores, dim=1)
                    num_dets = len(obj_ins_feats)

                    for det_id in range(num_dets):
                        det_mask = det_ids == det_id
                        det_feat_cts = feat_cts[det_mask]
                        det_feat_wh = feat_wh[det_mask]

                        if len(det_feat_cts) > 1:
                            left_top_ct, left_top_ids = torch.min(det_feat_cts, dim=0)
                            right_bottom_ct, right_bottom_ids = torch.max(det_feat_cts, dim=0)

                            left_top_wh = det_feat_wh[left_top_ids, torch.arange(2)]
                            right_bottom_wh = det_feat_wh[right_bottom_ids, torch.arange(2)]

                            ins_boxes[det_id, :2] = left_top_ct - left_top_wh/2
                            ins_boxes[det_id, 2:] = right_bottom_ct + right_bottom_wh/2

                    ins_boxes = Boxes(ins_boxes, format='xyxy', normalized='img_with_padding')
                    ins_boxes_dict[key] = ins_boxes

                pred_dicts[2]['labels'].append(nms_labels)
                pred_dicts[2]['boxes'].append(ins_boxes_dict['nms'])
                pred_dicts[2]['scores'].append(nms_scores)
                pred_dicts[2]['batch_ids'].append(torch.full_like(nms_ids, i, dtype=torch.int64))

                pred_dicts[3]['labels'].append(ins_labels)
                pred_dicts[3]['boxes'].append(ins_boxes_dict['ins'])
                pred_dicts[3]['scores'].append(ins_scores)
                pred_dicts[3]['batch_ids'].append(torch.full_like(ins_ids, i, dtype=torch.int64))

            # Concatenate different batch entry predictions
            for i, pred_dict in enumerate(pred_dicts):
                pred_dict.update({k: torch.cat(v, dim=0) for k, v in pred_dict.items() if k != 'boxes'})
                pred_dict['boxes'] = Boxes.cat(pred_dict['boxes'])
                pred_dicts[i] = pred_dict

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
