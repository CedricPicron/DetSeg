"""
State-Based Detector (SBD) head.
"""
from collections import OrderedDict
from copy import deepcopy

from detectron2.layers import batched_nms
from fvcore.nn import sigmoid_focal_loss, smooth_l1_loss
import torch
from torch import nn
import torch.nn.functional as F

from models.modules.mlp import OneStepMLP, TwoStepMLP
from structures.boxes import apply_box_deltas, Boxes, box_giou, box_iou, get_box_deltas


class SBD(nn.Module):
    """
    Class implementing the State-Based Detector (SBD) head.

    Attributes:
        dod (DOD): Dense object discovery (DOD) module selecting promising features corresponding to objects.
        osi (nn.ModuleList): List of object state initialization (OSI) modules computing the initial object states.
        obj_state_size (int): Integer containing the size of the object states.

        cls (nn.Sequential): Classification (CLS) module computing classification predictions from object states.
        box (nn.Sequential): Bounding box (BOX) module computing bounding box predictions from object states.

        match_mode (str): String containing the prediction-target matching mode.
        match_rel_thr (int): Integer containing the relative prediction-target matching threshold.

        with_bg (bool): Boolean indicating whether background label is added to the classification labels.
        bg_weight (float): Factor weighting the classification losses with background targets.
        cls_type (str): String containing the type of classification loss function.
        cls_alpha (float): Alpha value used by the classification sigmoid focal loss.
        cls_gamma (float): Gamma value used by the classification sigmoid focal loss.
        cls_weight (float): Factor weighting the classification loss.
        box_types (List): List with strings containing the types of bounding box loss functions.
        box_beta (float): Beta value used by the bounding box smooth L1 loss.
        box_weights (List): List with factors weighting the different bounding box losses.

        dup_removal (str): String containing the duplicate removal mechanism used during prediction.
        nms_candidates (int): Integer containing the maximum number of candidates retained before NMS.
        nms_thr (float): IoU threshold used during NMS to remove duplicate detections.
        max_dets (int): Integer containing the maximum number of detections during prediction.

        metadata (detectron2.data.Metadata): Metadata instance containing additional dataset information.
    """

    def __init__(self, dod, osi_dict, cls_dict, box_dict, match_dict, loss_dict, pred_dict, metadata):
        """
        Initializes the SBD module.

        Args:
            dod (DOD): Dense object discovery (DOD) module selecting promising features corresponding to objects.

            osi_dict (Dict): Object state initialization (OSI) network dictionary containing following keys:
                - type (str): string containing the type of OSI network;
                - layers (int): integer containing the number of OSI network layers;
                - in_size (int): input feature size of the OSI network;
                - hidden_size (int): hidden feature size of the OSI network;
                - out_size (int): output feature size of the OSI network;
                - norm (str): string containing the type of normalization of the OSI network;
                - act_fn (str): string containing the type of activation function of the OSI network;
                - skip (bool): boolean indicating whether layers of the OSI network contain skip connections.

            cls_dict (Dict): Classification (CLS) network dictionary containing following keys:
                - type (str): string containing the type of hidden CLS (HCLS) network;
                - layers (int): integer containing the number of HCLS network layers;
                - in_size (int): input feature size of the HCLS network;
                - hidden_size (int): hidden feature size of the HCLS network;
                - out_size (int): output feature size of the HCLS network;
                - norm (str): string containing the type of normalization of the HCLS network;
                - act_fn (str): string containing the type of activation function of the HCLS network;
                - skip (bool): boolean indicating whether layers of the HCLS network contain skip connections;
                - num_classes (int): integer containing the number of object classes (without background).

            box_dict (Dict): Bounding box (BOX) network dictionary containing following keys:
                - type (str): string containing the type of hidden BOX (HBOX) network;
                - layers (int): integer containing the number of HBOX network layers;
                - in_size (int): input feature size of the HBOX network;
                - hidden_size (int): hidden feature size of the HBOX network;
                - out_size (int): output feature size of the HBOX network;
                - norm (str): string containing the type of normalization of the HBOX network;
                - act_fn (str): string containing the type of activation function of the HBOX network;
                - skip (bool): boolean indicating whether layers of the HBOX network contain skip connections.

            match_dict (Dict): Matching dictionary containing following keys:
                - mode (str): string containing the prediction-target matching mode;
                - rel_thr (int): integer containing the relative prediction-target matching threshold.

            loss_dict (Dict): Loss dictionary containing following keys:
                - with_bg (bool): boolean indicating whether background label is added to the classification labels;
                - bg_weight (float): factor weighting the classification losses with background targets;
                - cls_type (str): string containing the type of classification loss function;
                - cls_alpha (float): alpha value used by the classification sigmoid focal loss;
                - cls_gamma (float): gamma value used by the classification sigmoid focal loss;
                - cls_weight (float): factor weighting the classification loss;
                - box_types (List): list with strings containing the types of bounding box loss functions;
                - box_beta (float): beta value used by the bounding box smooth L1 loss;
                - box_weights (List): list with factors weighting the different bounding box losses.

            pred_dict (Dict): Prediction dictionary containing following keys:
                - dup_removal (str): string containing the duplicate removal mechanism used during prediction;
                - nms_candidates (int): integer containing the maximum number of candidates retained before NMS;
                - nms_thr (float): IoU threshold used during NMS to remove duplicate detections;
                - max_dets (int): integer containing the maximum number of detections during prediction.

            metadata (detectron2.data.Metadata): Metadata instance containing additional dataset information.
        """

        # Initialization of default nn.Module
        super().__init__()

        # Set DOD attribute
        self.dod = dod

        # Initialization of object state initialization (OSI) networks
        osi = SBD.get_net(osi_dict)
        self.osi = nn.ModuleList([deepcopy(osi) for _ in range(dod.num_cell_anchors)])

        # Set attribute with size of object states
        self.obj_state_size = osi_dict['out_size']

        # Initialization of classification prediction (CLS) network
        num_classes = cls_dict.pop('num_classes')
        num_cls_labels = num_classes + 1 if loss_dict['with_bg'] else num_classes

        hcls = SBD.get_net(cls_dict)
        ocls = nn.Linear(cls_dict['out_size'], num_cls_labels)
        self.cls = nn.Sequential(OrderedDict([('hidden', hcls), ('out', ocls)]))

        # Initialization of bounding box prediction (BOX) network
        hbox = SBD.get_net(box_dict)
        obox = nn.Linear(cls_dict['out_size'], 4)
        self.box = nn.Sequential(OrderedDict([('hidden', hbox), ('out', obox)]))

        # Set matching attributes
        for k, v in match_dict.items():
            setattr(self, f'match_{k}', v)

        # Set loss attributes
        for k, v in loss_dict.items():
            setattr(self, k, v)

        # Set prediction attributes
        for k, v in pred_dict.items():
            setattr(self, k, v)

        # Set metadata attribute
        metadata.stuff_classes = metadata.thing_classes
        metadata.stuff_colors = metadata.thing_colors
        self.metadata = metadata

    @staticmethod
    def get_net(net_dict):
        """
        Get network from network dictionary.

        Args:
            net_dict (Dict): Network dictionary containing following keys:
                - type (str): string containing the type of network;
                - layers (int): integer containing the number of network layers;
                - in_size (int): input feature size of the network;
                - hidden_size (int): hidden feature size of the network;
                - out_size (int): output feature size of the network;
                - norm (str): string containing the type of normalization of the network;
                - act_fn (str): string containing the type of activation function of the etwork;
                - skip (bool): boolean indicating whether layers of the network contain skip connections.

        Returns:
            net (nn.Sequential): Module implementing the network specified by the given network dictionary.

        Raises:
            ValueError: Error when unsupported type of network is provided.
            ValueError: Error when the number of layers in non-positive.
        """

        if net_dict['type'] == 'one_step_mlp':
            net_args = (net_dict['in_size'], net_dict['out_size'])
            net_kwargs = {k: v for k, v in net_dict.items() if k in ('norm', 'act_fn', 'skip')}
            net_layer = OneStepMLP(*net_args, **net_kwargs)

        elif net_dict['type'] == 'two_step_mlp':
            net_args = (net_dict['in_size'], net_dict['hidden_size'], net_dict['out_size'])
            net_kwargs = {'norm1': net_dict['norm'], 'act_fn2': net_dict['act_fn'], 'skip': net_dict['skip']}
            net_layer = TwoStepMLP(*net_args, **net_kwargs)

        else:
            error_msg = f"The provided network type '{net_dict['type']}' is not supported."
            raise ValueError(error_msg)

        if net_dict['layers'] > 0:
            net = nn.Sequential(*[deepcopy(net_layer) for _ in range(net_dict['layers'])])
        else:
            error_msg = f"The number of network layers must be positive, but got {net_dict['layers']}."
            raise ValueError(error_msg)

        return net

    def get_loss(self, cls_preds, box_preds, pred_anchors, tgt_dict, pred_anchor_ids=None, tgt_sorted_ids=None):
        """
        Compute classification and bounding box losses from predictions.

        Args:
            cls_preds (List): List [batch_size] of classification predictions of shape [num_preds, num_cls_labels].
            box_preds (List): List [batch_size] of bounding box predictions of shape [num_preds, 4].
            pred_anchors (List): List [batch_size] of Boxes structures with prediction anchors of size [num_preds].

            tgt_dict (Dict): Target dictionary containing at least following keys:
                - labels (List): list of size [batch_size] with class indices of shape [num_targets];
                - boxes (List): list of size [batch_size] with Boxes structures of size [num_targets].

            pred_anchor_ids (List): List [batch_size] with anchor indices of shape [num_preds] (default=None).
            tgt_sorted_ids (List): List [batch_size] of sorted ids of shape [num_anchors, num_targets] (default=None).

        Returns:
            loss_dict (Dict): Loss dictionary containing following keys:
                - cls_loss (FloatTensor): tensor containing the weighted classification loss of shape [1];
                - box_loss (FloatTensor): tensor containing the weighted bounding box loss of shape [1].

            analysis_dict (Dict): Analysis dictionary containing following keys:
                - cls_acc (FloatTensor): tensor containing the classification accuracy (in percentage) of shape [1];
                - box_acc (FloatTensor): tensor containing the bounding box accuracy (in percentage) of shape [1].

            pos_pred_ids (List): List [batch_size] with indices of positive predictions of shape [num_pos_preds].
            tgt_found (List): List [batch_size] with masks of found targets of shape [num_targets].

        Raises:
            ValueError: Error when invalid prediction-target matching mode is provided.
            ValueError: Error when invalid classfication loss type is provided.
            ValueError: Error when the number of bounding box loss types and corresponding loss weights is different.
            ValueError: Error when invalid bounding box loss type is provided.
        """

        # Get batch size and number of classification labels
        batch_size = len(cls_preds)
        num_cls_labels = cls_preds[0].shape[1]

        # Initialize empty lists for prediction-target matching
        pred_ids = []
        tgt_ids = []
        pos_pred_ids = []
        tgt_found = []

        # Perform prediction-target matching
        with torch.no_grad():
            if self.match_mode == 'dod_rel':
                for i in range(batch_size):
                    pred_pos_masks = pred_anchor_ids[i][:, None, None] == tgt_sorted_ids[i][:self.match_rel_thr]
                    pred_pos_mask = pred_pos_masks.sum(dim=1)

                    pred_ids_i, tgt_ids_i = torch.nonzero(pred_pos_mask, as_tuple=True)
                    pos_pred_ids_i = torch.unique_consecutive(pred_ids_i)

                    num_tgts = pred_pos_mask.shape[1]
                    tgt_found_i = torch.zeros(num_tgts, dtype=torch.bool, device=tgt_ids_i.device)
                    tgt_found_i[tgt_ids_i] = True

                    pred_ids.append(pred_ids_i)
                    tgt_ids.append(tgt_ids_i)
                    pos_pred_ids.append(pos_pred_ids_i)
                    tgt_found.append(tgt_found_i)

            else:
                error_msg = f"Invalid prediction-target matching mode '{self.match_mode}'."
                raise ValueError(error_msg)

        # Initialize loss and analysis dicitonaries
        tensor_kwargs = {'dtype': cls_preds[0].dtype, 'device': cls_preds[0].device}
        loss_dict = {'cls_loss': torch.zeros(1, **tensor_kwargs), 'box_loss': torch.zeros(1, **tensor_kwargs)}
        analysis_dict = {'cls_acc': torch.zeros(1, **tensor_kwargs), 'box_acc': torch.zeros(1, **tensor_kwargs)}

        # Get weighted classification loss and classification accuracy
        for i in range(batch_size):

            # Get classification logits and corresponding targets
            cls_logits_i = cls_preds[i][pred_ids[i]]
            cls_targets_i = tgt_dict['labels'][i][tgt_ids[i]]

            # Add background targets if requested
            if self.with_bg:
                num_preds_i = len(cls_preds[i])
                bg_pred_mask_i = torch.ones(num_preds_i, dtype=torch.bool, device=tensor_kwargs['device'])
                bg_pred_mask_i[pos_pred_ids[i]] = False
                bg_pred_ids_i = torch.nonzero(bg_pred_mask_i, as_tuple=True)[0]

                bg_logits_i = cls_preds[i][bg_pred_ids_i]
                bg_targets_i = torch.full_like(bg_pred_ids_i, num_cls_labels-1)

                cls_logits_i = torch.cat([cls_logits_i, bg_logits_i], dim=0)
                cls_targets_i = torch.cat([cls_targets_i, bg_targets_i], dim=0)

            # Handle case where there are no targets
            if len(cls_targets_i) == 0:
                loss_dict['cls_loss'] += 0.0 * cls_preds[i].sum()
                analysis_dict['cls_acc'] += 100 / batch_size
                continue

            # Get classification losses
            if self.cls_type == 'sigmoid_focal':
                cls_targets_oh_i = F.one_hot(cls_targets_i, num_classes=num_cls_labels).to(cls_logits_i.dtype)
                cls_kwargs = {'alpha': self.cls_alpha, 'gamma': self.cls_gamma, 'reduction': 'none'}
                cls_losses = sigmoid_focal_loss(cls_logits_i, cls_targets_oh_i, **cls_kwargs).sum(dim=1)

            else:
                error_msg = f"Invalid classification loss type '{self.cls_type}'."
                raise ValueError(error_msg)

            # Weight losses of background targets if needed
            if self.with_bg:
                bg_tgt_mask = cls_targets_i == num_cls_labels-1
                cls_losses[bg_tgt_mask] = self.bg_weight * cls_losses[bg_tgt_mask]

            # Get weighted classification loss
            cls_loss = self.cls_weight * cls_losses.sum()
            loss_dict['cls_loss'] += cls_loss

            # Get classification accuracy
            with torch.no_grad():
                cls_labels_i = torch.argmax(cls_logits_i, dim=1)
                cls_acc = torch.eq(cls_labels_i, cls_targets_i).sum() / len(cls_labels_i)
                analysis_dict['cls_acc'] += 100 * cls_acc / batch_size

        # Get weighted bounding box loss and bounding box accuracy
        for i in range(batch_size):

            # Get box logits with corresponding anchors and target boxes
            box_logits_i = box_preds[i][pred_ids[i]]
            pred_anchors_i = pred_anchors[i][pred_ids[i]]
            tgt_boxes_i = tgt_dict['boxes'][i][tgt_ids[i]]

            # Handle case where there are no targets
            if len(tgt_boxes_i) == 0:
                loss_dict['box_loss'] += 0.0 * box_preds[i].sum()
                analysis_dict['box_acc'] += 100 / batch_size
                continue

            # Get bounding box targets and prediction boxes
            box_targets_i = get_box_deltas(pred_anchors_i, tgt_boxes_i)
            pred_boxes_i = apply_box_deltas(box_logits_i, pred_anchors_i)

            if len(self.box_types) != len(self.box_weights):
                error_msg = "The number of bounding box loss types and corresponding loss weights must be equal."
                raise ValueError(error_msg)

            # Get weighted bounding box loss
            for box_type, box_weight in zip(self.box_types, self.box_weights):
                if box_type == 'smooth_l1':
                    box_kwargs = {'beta': self.box_beta, 'reduction': 'sum'}
                    box_loss = smooth_l1_loss(box_logits_i, box_targets_i, **box_kwargs)
                    loss_dict['box_loss'] += box_weight * box_loss

                elif box_type == 'iou':
                    box_loss = len(pred_boxes_i) - box_iou(pred_boxes_i, tgt_boxes_i).diag().sum()
                    loss_dict['box_loss'] += box_weight * box_loss

                elif box_type == 'giou':
                    box_loss = len(pred_boxes_i) - box_giou(pred_boxes_i, tgt_boxes_i).diag().sum()
                    loss_dict['box_loss'] += box_weight * box_loss

                else:
                    error_msg = f"Invalid bounding box loss type '{box_type}'."
                    raise ValueError(error_msg)

            # Get bounding box accuracy
            with torch.no_grad():
                box_acc = box_iou(pred_boxes_i, tgt_boxes_i).diag().mean()
                analysis_dict['box_acc'] += 100 * box_acc / batch_size

        return loss_dict, analysis_dict, pos_pred_ids, tgt_found

    @torch.no_grad()
    def make_predictions(self, cls_preds, box_preds, pred_anchors):
        """
        Makes classified bounding box predictions.

        Args:
            cls_preds (List): List [batch_size] of classification predictions of shape [num_preds, num_cls_labels].
            box_preds (List): List [batch_size] of bounding box predictions of shape [num_preds, 4].
            pred_anchors (List): List [batch_size] of Boxes structures with prediction anchors of size [num_preds].

        Returns:
            pred_dict (Dict): Prediction dictionary containing following keys:
                - labels (LongTensor): predicted class indices of shape [num_preds_total];
                - boxes (Boxes): structure containing axis-aligned bounding boxes of size [num_preds_total];
                - scores (FloatTensor): normalized prediction scores of shape [num_preds_total];
                - batch_ids (LongTensor): batch indices of predictions of shape [num_preds_total].

        Raises:
            ValueError: Error when invalid duplicate removal mechanism is provided.
        """

        # Get batch size
        batch_size = len(cls_preds)

        # Initialize prediction dictionary
        pred_keys = ('labels', 'boxes', 'scores', 'batch_ids')
        pred_dict = {pred_key: [] for pred_key in pred_keys}

        # Get predictions for every batch entry
        for i in range(batch_size):

            # Get prediction labels and scores
            cls_preds_i = cls_preds[i][:, :-1] if self.with_bg else cls_preds[i]
            scores_i, labels_i = cls_preds_i.sigmoid().max(dim=1)

            # Get prediction boxes
            boxes_i = apply_box_deltas(box_preds[i], pred_anchors[i])

            # Get top prediction indices sorted by score
            top_pred_ids = torch.argsort(scores_i, dim=0, descending=True)

            # Perform duplicate removal
            if not self.dup_removal:
                pass

            elif self.dup_removal == 'nms':
                top_pred_ids = top_pred_ids[:self.nms_candidates]
                boxes_i = boxes_i[top_pred_ids].to_format('xyxy')
                scores_i = scores_i[top_pred_ids]
                labels_i = labels_i[top_pred_ids]
                top_pred_ids = batched_nms(boxes_i.boxes, scores_i, labels_i, iou_threshold=self.nms_thr)

            else:
                error_msg = f"Invalid duplicate removal mechanism '{self.dup_removal}'."
                raise ValueError(error_msg)

            # Keep maximum number of allowed predictions
            top_pred_ids = top_pred_ids[:self.max_dets]
            labels_i = labels_i[top_pred_ids]
            boxes_i = boxes_i[top_pred_ids]
            scores_i = scores_i[top_pred_ids]

            # Append predictions to their respective lists
            pred_dict['labels'].append(labels_i)
            pred_dict['boxes'].append(boxes_i)
            pred_dict['scores'].append(scores_i)
            pred_dict['batch_ids'].append(torch.full_like(labels_i, i))

        # Concatenate predictions of different batch entries
        pred_dict.update({k: torch.cat(v, dim=0) for k, v in pred_dict.items() if k != 'boxes'})
        pred_dict['boxes'] = Boxes.cat(pred_dict['boxes'])

        return pred_dict

    def forward(self, feat_maps, tgt_dict=None, visualize=False, **kwargs):
        """
        Forward method of the SBD head.

        Args:
            feat_maps (List): List of size [num_maps] with feature maps of shape [batch_size, feat_size, fH, fW].

            tgt_dict (Dict): Optional target dictionary used during trainval containing at least following keys:
                - labels (LongTensor): tensor of shape [num_targets_total] containing the class indices;
                - boxes (Boxes): structure containing axis-aligned bounding boxes of size [num_targets_total];
                - sizes (LongTensor): tensor of shape [batch_size+1] with the cumulative target sizes of batch entries.

            visualize (bool): Boolean indicating whether to compute dictionary with visualizations (default=False).
            kwargs (Dict): Dictionary of keyword arguments not used by this module, but passed to some sub-modules.

        Returns:
            * If SBD module in training mode (i.e. during training):
                loss_dict (Dict): Dictionary of different weighted loss terms used during training.
                analysis_dict (Dict): Dictionary of different analyses used for logging purposes only.

            * If SBD module not in training mode and tgt_dict is not None (i.e. during validation):
                pred_dicts (List): List of size [num_layers+1] with SBD prediction dictionaries.
                loss_dict (Dict): Dictionary of different weighted loss terms used during training.
                analysis_dict (Dict): Dictionary of different analyses used for logging purposes only.

            * If SBD module not in training mode and tgt_dict is None (i.e. during testing):
                pred_dicts (List): List of size [num_layers+1] with SBD prediction dictionaries.
                analysis_dict (Dict): Dictionary of different analyses used for logging purposes only.

        Raises:
            NotImplementedError: Error when visualizations are requested.
        """

        # Check whether visualizations are requested
        if visualize:
            raise NotImplementedError

        # Get batch size
        batch_size = len(feat_maps[0])

        # Apply DOD module and extract output
        dod_kwargs = {'tgt_dict': tgt_dict, 'stand_alone': False}
        dod_output = self.dod(feat_maps, **dod_kwargs, **kwargs)

        if tgt_dict is None:
            sel_ids, anchors, analysis_dict = dod_output
        elif self.dod.tgt_mode == 'ext_dynamic':
            logits, obj_probs, sel_ids, anchors, pos_masks, neg_masks, tgt_sorted_ids, analysis_dict = dod_output
        else:
            sel_ids, anchors, tgt_sorted_ids, dod_loss_dict, analysis_dict = dod_output

        # Get selected features
        feats = torch.cat([feat_map.flatten(2).permute(0, 2, 1) for feat_map in feat_maps], dim=1)
        feat_ids = [sel_ids_i // self.dod.num_cell_anchors for sel_ids_i in sel_ids]
        sel_feats = [feats[i, feat_ids[i], :] for i in range(batch_size)]

        # Get initial object states and corresponding anchors
        tensor_kwargs = {'dtype': feats.dtype, 'device': feats.device}
        obj_states = [torch.zeros(len(sel_ids[i]), self.obj_state_size, **tensor_kwargs) for i in range(batch_size)]
        osi_ids = [sel_ids_i % self.dod.num_cell_anchors for sel_ids_i in sel_ids]

        for i in range(batch_size):
            for j in range(self.dod.num_cell_anchors):
                osi_mask = osi_ids[i] == j
                obj_states[i][osi_mask] = self.osi[j](sel_feats[i][osi_mask])

        obj_anchors = [anchors[sel_ids_i] for sel_ids_i in sel_ids]
        num_states = sum(len(obj_states_i) for obj_states_i in obj_states)
        analysis_dict['num_states_0'] = num_states / batch_size

        # Get initial predictions
        cls_preds = [self.cls(obj_states_i) for obj_states_i in obj_states]
        box_preds = [self.box(obj_states_i) for obj_states_i in obj_states]

        # Get initial loss
        if tgt_dict is not None:

            # Get local target dictionary
            tgt_sizes = tgt_dict['sizes']
            tgt_labels = [tgt_dict['labels'][i0:i1] for i0, i1 in zip(tgt_sizes[:-1], tgt_sizes[1:])]
            tgt_boxes = [tgt_dict['boxes'][i0:i1] for i0, i1 in zip(tgt_sizes[:-1], tgt_sizes[1:])]
            local_tgt_dict = {'labels': tgt_labels, 'boxes': tgt_boxes}

            # Get initial classification and bounding box losses with corresponding analyses
            loss_kwargs = {}

            if self.match_mode == 'dod_rel':
                loss_kwargs = {'pred_anchor_ids': sel_ids, 'tgt_sorted_ids': tgt_sorted_ids}

            loss_output = self.get_loss(cls_preds, box_preds, obj_anchors, local_tgt_dict, **loss_kwargs)
            loss_dict, local_analysis_dict, pos_pred_ids, tgt_found = loss_output

            loss_dict = {f'{k}_0': v for k, v in loss_dict.items()}
            analysis_dict.update({f'{k}_0': v for k, v in local_analysis_dict.items()})

            # Get DOD losses if in DOD external dynamic target mode
            if self.dod.tgt_mode == 'ext_dynamic':

                # Get external dictionary
                pos_anchor_ids = [sel_ids[i][pos_pred_ids[i]] for i in range(batch_size)]
                ext_dict = {'logits': logits, 'obj_probs': obj_probs, 'pos_anchor_ids': pos_anchor_ids}
                ext_dict = {**ext_dict, 'tgt_found': tgt_found, 'pos_masks': pos_masks, 'neg_masks': neg_masks}
                ext_dict = {**ext_dict, 'tgt_sorted_ids': tgt_sorted_ids}

                # Get DOD loss and analysis dictionary
                dod_loss_dict, dod_analysis_dict = self.dod(feat_maps, ext_dict=ext_dict, **dod_kwargs, **kwargs)
                analysis_dict.update(dod_analysis_dict)

            # Add DOD losses to main loss dictionary
            loss_dict.update(dod_loss_dict)

        # Get initial prediction dictionary
        if not self.training:
            pred_dict = self.make_predictions(cls_preds, box_preds, obj_anchors)
            pred_dicts = [pred_dict]

        # Return desired dictionaries
        if self.training:
            return loss_dict, analysis_dict
        elif tgt_dict is not None:
            return pred_dicts, loss_dict, analysis_dict
        else:
            return pred_dicts, analysis_dict
