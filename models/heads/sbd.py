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

from models.functional.net import get_net
from models.modules.container import Sequential
from structures.boxes import apply_box_deltas, Boxes, box_giou, box_iou, get_box_deltas


class SBD(nn.Module):
    """
    Class implementing the State-Based Detector (SBD) head.

    Attributes:
        dod (DOD): Dense object discovery (DOD) module selecting promising features corresponding to objects.

        state_size (int): Integer containing the size of object states.
        state_type (str): String containing the type of object states.

        osi (nn.ModuleList): List of object state initialization (OSI) modules computing the initial object states.
        ae (Sequential): Optional anchor encoding (AE) module computing anchor encodings from normalized anchor boxes.
        se (Sequential): Optional scale encoding (SE) module computing scale encodings from normalized anchor sizes.
        cls (Sequential): Classification (CLS) module computing classification predictions from object states.
        box (Sequential): Bounding box (BOX) module computing bounding box predictions from object states.

        match_mode (str): String containing the prediction-target matching mode.
        match_static_mode (str): String containing the static prediction-target matching mode.
        match_abs_pos (float): Absolute positive threshold used static prediction-target matching.
        match_abs_neg (float): Absolute negative threshold used static prediction-target matching.
        match_rel_pos (int): Relative positive threshold used static prediction-target matching.
        match_rel_neg (int): Relative negative threshold used static prediction-target matching.

        ae_weight (float): Factor weighting the anchor encoding loss.
        apply_freq (str): String containing the frequency at which losses are computed and applied.
        freeze_inter (bool): Boolean indicating whether CLS and BOX modules are frozen for intermediate losses.
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

        up_ca_type (str): String containing the type of cross-attention used by the update layers.
        up_layers (nn.ModuleList): List of update layers computing new object states from current object states.
        up_iters (int): Integer containing the number of iterations over all update layers.

        metadata (detectron2.data.Metadata): Metadata instance containing additional dataset information.
    """

    def __init__(self, dod, state_dict, osi_dict, ae_dict, se_dict, cls_dict, box_dict, match_dict, loss_dict,
                 pred_dict, update_dict, ca_dict, sa_dict, ffn_dict, metadata):
        """
        Initializes the SBD module.

        Args:
            dod (DOD): Dense object discovery (DOD) module selecting promising features corresponding to objects.

            state_dict (Dict): State dictionary containing following keys:
                - size (int): integer containing the size of object states;
                - type (str): string containing the type of object states.

            osi_dict (Dict): Object state initialization (OSI) network dictionary containing following keys:
                - type (str): string containing the type of OSI network;
                - layers (int): integer containing the number of OSI network layers;
                - in_size (int): input feature size of the OSI network;
                - hidden_size (int): hidden feature size of the OSI network;
                - out_size (int): output feature size of the OSI network;
                - norm (str): string containing the type of normalization of the OSI network;
                - act_fn (str): string containing the type of activation function of the OSI network;
                - skip (bool): boolean indicating whether layers of the OSI network contain skip connections.

            ae_dict (Dict): Anchor encoding (AE) network dictionary containing following keys:
                - type (str): string containing the type of hidden AE (HAE) network;
                - layers (int): integer containing the number of HAE network layers;
                - in_size (int): input feature size of the HAE network;
                - hidden_size (int): hidden feature size of the HAE network;
                - out_size (int): output feature size of the HAE network;
                - norm (str): string containing the type of normalization of the HAE network;
                - act_fn (str): string containing the type of activation function of the HAE network;
                - skip (bool): boolean indicating whether layers of the HAE network contain skip connections.

            se_dict (Dict): Scale encoding (SE) dictionary containing following keys:
                - needed (bool): boolean indicating whether scale encodings are needed;
                - type (str): string containing the type of hidden SE (HSE) network;
                - layers (int): integer containing the number of HSE network layers;
                - in_size (int): input feature size of the HSE network;
                - hidden_size (int): hidden feature size of the HSE network;
                - out_size (int): output feature size of the HSE network;
                - norm (str): string containing the type of normalization of the HSE network;
                - act_fn (str): string containing the type of activation function of the HSE network;
                - skip (bool): boolean indicating whether layers of the HSE network contain skip connections.

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
                - static_mode (str): string containing the static prediction-target matching mode;
                - abs_pos (float): absolute positive threshold used static prediction-target matching;
                - abs_neg (float): absolute negative threshold used static prediction-target matching;
                - rel_pos (int): relative positive threshold used static prediction-target matching;
                - rel_neg (int): relative negative threshold used static prediction-target matching.

            loss_dict (Dict): Loss dictionary containing following keys:
                - ae_weight (float): factor weighting the anchor encoding loss;
                - apply_freq (str): string containing the frequency at which losses are computed and applied;
                - freeze_inter (bool): boolean indicating whether CLS and BOX are frozen for intermediate losses;
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

            update_dict (Dict): Update dictionary containing following keys:
                - types (List): list with strings containing the type of modules present in each update layer;
                - layers (int): integer containing the number of update layers;
                - iters (int): integer containing the number of iterations over all update layers.

            ca_dict: Cross-attention (CA) network dictionary containing following keys:
                - type (str): string containing the type of CA network;
                - layers (int): integer containing the number of CA network layers;
                - in_size (int): input feature size of the CA network;
                - sample_size (int): sample feature size of the CA network;
                - out_size (int): output feature size of the CA network;
                - norm (str): string containing the type of normalization of the CA network;
                - act_fn (str): string containing the type of activation function of the CA network;
                - skip (bool): boolean indicating whether layers of the CA network contain skip connections;
                - version (int): integer containing the version of the CA network;
                - num_heads (int): integer containing the number of attention heads of the CA network;
                - num_levels (int): integer containing the number of map levels for the CA network to sample from;
                - num_points (int): integer containing the number of points of the CA network;
                - rad_pts (int): integer containing the number of radial points of the CA network;
                - ang_pts (int): integer containing the number of angular points of the CA network;
                - lvl_pts (int): integer containing the number of level points of the CA network;
                - dup_pts (int): integer containing the number of duplicate points of the CA network;
                - qk_size (int): query and key feature size of the CA network;
                - val_size (int): value feature size of the CA network;
                - val_with_pos (bool): boolean indicating whether position info is added to CA value features;
                - step_size (float): size of the CA sample steps relative to the sample step normalization;
                - step_norm_xy (str): string containing the normalization type of CA sample steps in the XY-direction;
                - step_norm_z (float): value normalizing the CA sample steps in the Z-direction;
                - num_particles (int): integer containing the number of particles per CA head.

            sa_dict: Self-attention (SA) network dictionary containing following keys:
                - type (str): string containing the type of SA network;
                - layers (int): integer containing the number of SA network layers;
                - in_size (int): input feature size of the SA network;
                - out_size (int): output feature size of the SA network;
                - norm (str): string containing the type of normalization of the SA network;
                - act_fn (str): string containing the type of activation function of the SA network;
                - skip (bool): boolean indicating whether layers of the SA network contain skip connections;
                - num_heads (int): integer containing the number of attention heads of the SA network.

            ffn_dict (Dict): Feedforward network (FFN) dictionary containing following keys:
                - type (str): string containing the type of FFN network;
                - layers (int): integer containing the number of FFN network layers;
                - in_size (int): input feature size of the FFN network;
                - out_size (int): output feature size of the FFN network;
                - norm (str): string containing the type of normalization of the FFN network;
                - act_fn (str): string containing the type of activation function of the FFN network;
                - skip (bool): boolean indicating whether layers of the FFN network contain skip connections;
                - hidden_size (int): hidden feature size of the FFN network.

            metadata (detectron2.data.Metadata): Metadata instance containing additional dataset information.

        Raises:
            ValueError: Error when invalid module type is provided for update layer.
        """

        # Initialization of default nn.Module
        super().__init__()

        # Set DOD attribute
        self.dod = dod

        # Set state attributes
        for k, v in state_dict.items():
            setattr(self, f'state_{k}', v)

        # Initialization of object state initialization (OSI) networks
        osi = get_net(osi_dict)
        self.osi = nn.ModuleList([deepcopy(osi) for _ in range(dod.num_cell_anchors)])

        # Initialization of anchor encoding (AE) network if needed
        needs_ae = [t for t in update_dict['types'] if t in ['ca', 'sa']] and update_dict['layers'] > 0
        needs_ae = needs_ae or self.state_type == 'abs'

        if needs_ae:
            iae = nn.Linear(4, ae_dict['in_size'])
            hae = get_net(ae_dict)
            self.ae = Sequential(OrderedDict([('in', iae), ('hidden', hae)]))

        # Initialization of scale encoding (SE) network if needed
        if se_dict.pop('needed'):
            ise = nn.Linear(2, se_dict['in_size'])
            hse = get_net(se_dict)
            self.se = Sequential(OrderedDict([('in', ise), ('hidden', hse)]))

        # Initialization of classification prediction (CLS) network
        num_classes = cls_dict.pop('num_classes')
        num_cls_labels = num_classes + 1

        hcls = get_net(cls_dict)
        ocls = nn.Linear(cls_dict['out_size'], num_cls_labels)
        self.cls = Sequential(OrderedDict([('hidden', hcls), ('out', ocls)]))

        # Initialization of bounding box prediction (BOX) network
        hbox = get_net(box_dict)
        obox = nn.Linear(cls_dict['out_size'], 4)
        self.box = Sequential(OrderedDict([('hidden', hbox), ('out', obox)]))

        # Set matching attributes
        for k, v in match_dict.items():
            setattr(self, f'match_{k}', v)

        # Set loss attributes
        for k, v in loss_dict.items():
            setattr(self, k, v)

        # Set prediction attributes
        for k, v in pred_dict.items():
            setattr(self, k, v)

        # Initialization of update layers and attributes
        module_dict = OrderedDict()
        self.up_ca_type = ''

        if update_dict['layers'] > 0:
            for module_type in update_dict['types']:
                if not module_type:
                    update_dict['layers'] = 0
                    break

                elif module_type == 'ca':
                    module_dict['ca'] = get_net(ca_dict)
                    self.up_ca_type = ca_dict['type']

                elif module_type == 'sa':
                    module_dict['sa'] = get_net(sa_dict)

                elif module_type == 'ffn':
                    module_dict['ffn'] = get_net(ffn_dict)

                else:
                    error_msg = f"Invalid module type '{module_type}' for update layer."
                    raise ValueError(error_msg)

        up_layer = Sequential(module_dict)
        self.up_layers = nn.ModuleList([deepcopy(up_layer) for _ in range(update_dict['layers'])])
        self.up_iters = update_dict['iters']

        if self.up_ca_type == 'particle_attn':
            ca_id = update_dict['types'].index('ca')
            last_pa_layer = self.up_layers[-1][ca_id][0].pa
            last_pa_layer.no_sample_locations_update()

        # Set metadata attribute
        metadata.stuff_classes = metadata.thing_classes
        metadata.stuff_colors = metadata.thing_colors
        self.metadata = metadata

    @torch.no_grad()
    def perform_matching(self, pred_anchors=None, tgt_boxes=None, pred_anchor_ids=None, tgt_sorted_ids=None):
        """
        Perform prediction-target matching.

        Args:
            pred_anchors (List): List [batch_size] of prediction anchors of size [num_preds] (default=None).
            tgt_boxes (List): List [batch_size] of target boxes of size [num_targets] (default=None).
            pred_anchor_ids (List): List [batch_size] with anchor indices of shape [num_preds] (default=None).
            tgt_sorted_ids (List): List [batch_size] of sorted ids of shape [sort_thr, num_targets] (default=None).

        Returns:
            pos_pred_ids (List): List [batch_size] with indices of positive predictions of shape [num_pos_preds].
            pos_tgt_ids (List): List [batch_size] with indices of positive targets of shape [num_pos_preds].
            neg_pred_ids (List): List [batch_size] with indices of negative predictions of shape [num_neg_preds].

        Raises:
            ValueError: Error when the 'pred_anchors' input is missing when needed.
            ValueError: Error when the 'tgt_boxes' input is missing when needed.
            ValueError: Error when the 'pred_anchor_ids' input is missing when needed.
            ValueError: Error when the 'tgt_sorted_ids' input is missing when needed.
            ValueError: Error when invalid static prediction-target matching mode is provided.
            ValueError: Error when invalid prediction-target matching mode is provided.
        """

        # Initialize empty lists for prediction-target matching
        pos_pred_ids = []
        pos_tgt_ids = []
        neg_pred_ids = []

        # Perform prediction-target matching
        if self.match_mode == 'static':
            if 'abs' in self.match_static_mode:
                if pred_anchors is None:
                    mode = self.match_static_mode
                    error_msg = f"The 'pred_anchors' input must be provided when in '{mode}' static match mode."
                    raise ValueError(error_msg)

                if tgt_boxes is None:
                    mode = self.match_static_mode
                    error_msg = f"The 'tgt_boxes' input must be provided when in '{mode}' static match mode."
                    raise ValueError(error_msg)

            if 'rel' in self.match_static_mode:
                if pred_anchor_ids is None:
                    mode = self.match_static_mode
                    error_msg = f"The 'pred_anchor_ids' input must be provided when in '{mode}' static match mode."
                    raise ValueError(error_msg)

                if tgt_sorted_ids is None:
                    mode = self.match_static_mode
                    error_msg = f"The 'tgt_sorted_ids' input must be provided when in '{mode}' static match mode."
                    raise ValueError(error_msg)

            for i in range(len(pred_anchor_ids)):
                if 'abs' in self.match_static_mode:
                    sim_matrix = box_iou(pred_anchors[i], tgt_boxes[i])
                    abs_pos_mask = sim_matrix >= self.match_abs_pos
                    abs_neg_mask = sim_matrix < self.match_abs_neg

                if 'rel' in self.match_static_mode:
                    pos_masks = pred_anchor_ids[i][:, None, None] == tgt_sorted_ids[i][:self.match_rel_pos]
                    non_neg_masks = pred_anchor_ids[i][:, None, None] == tgt_sorted_ids[i][:self.match_rel_neg]

                    rel_pos_mask = pos_masks.any(dim=1)
                    rel_neg_mask = ~non_neg_masks.any(dim=1)

                if self.match_static_mode == 'abs':
                    pos_mask = abs_pos_mask
                    neg_mask = abs_neg_mask

                elif self.match_static_mode == 'abs_and_rel':
                    pos_mask = abs_pos_mask & rel_pos_mask
                    neg_mask = abs_neg_mask | rel_neg_mask

                elif self.match_static_mode == 'abs_or_rel':
                    pos_mask = abs_pos_mask | rel_pos_mask
                    neg_mask = abs_neg_mask & rel_neg_mask

                elif self.match_static_mode == 'rel':
                    pos_mask = rel_pos_mask
                    neg_mask = rel_neg_mask

                else:
                    error_msg = f"Invalid static prediction-target matching mode '{self.match_static_mode}'."
                    raise ValueError(error_msg)

                pos_pred_ids_i, pos_tgt_ids_i = torch.nonzero(pos_mask, as_tuple=True)
                neg_pred_ids_i = neg_mask.all(dim=1).nonzero(as_tuple=True)[0]

                pos_pred_ids.append(pos_pred_ids_i)
                pos_tgt_ids.append(pos_tgt_ids_i)
                neg_pred_ids.append(neg_pred_ids_i)

        else:
            error_msg = f"Invalid prediction-target matching mode '{self.match_mode}'."
            raise ValueError(error_msg)

        return pos_pred_ids, pos_tgt_ids, neg_pred_ids

    def get_cls_loss(self, cls_preds, tgt_dict, pos_pred_ids, pos_tgt_ids, neg_pred_ids):
        """
        Compute classification loss from predictions.

        Args:
            cls_preds (List): List [batch_size] of classification predictions of shape [num_preds, num_cls_labels].

            tgt_dict (Dict): Target dictionary containing at least following key:
                - labels (List): list of size [batch_size] with class indices of shape [num_targets].

            pos_pred_ids (List): List [batch_size] with indices of positive predictions of shape [num_pos_preds].
            pos_tgt_ids (List): List [batch_size] with indices of positive targets of shape [num_pos_preds].
            neg_pred_ids (List): List [batch_size] with indices of negative predictions of shape [num_neg_preds].

        Returns:
            cls_loss (FloatTensor): Tensor containing the classification loss of shape [1].
            cls_acc (FloatTensor): Tensor containing the classification accuracy (in percentage) of shape [1].

        Raises:
            ValueError: Error when invalid classfication loss type is provided.
        """

        # Get batch size and number of classification labels
        batch_size = len(cls_preds)
        num_cls_labels = cls_preds[0].shape[1]

        # Initialize classification loss and accuracy
        tensor_kwargs = {'dtype': cls_preds[0].dtype, 'device': cls_preds[0].device}
        cls_loss = torch.zeros(1, **tensor_kwargs)
        cls_acc = torch.zeros(1, **tensor_kwargs)

        # Iterate over every batch entry
        for i in range(batch_size):

            # Get classification predictions and corresponding targets
            cls_preds_i = cls_preds[i][pos_pred_ids[i]]
            cls_targets_i = tgt_dict['labels'][i][pos_tgt_ids[i]]

            # Add predictions with background targets
            bg_preds_i = cls_preds[i][neg_pred_ids[i]]
            bg_targets_i = torch.full_like(neg_pred_ids[i], num_cls_labels-1)

            cls_preds_i = torch.cat([cls_preds_i, bg_preds_i], dim=0)
            cls_targets_i = torch.cat([cls_targets_i, bg_targets_i], dim=0)

            # Handle case where there are no targets
            if len(cls_targets_i) == 0:
                cls_loss += 0.0 * cls_preds[i].sum()
                cls_acc += 100 / batch_size
                continue

            # Get classification losses
            if self.cls_type == 'sigmoid_focal':
                cls_targets_oh_i = F.one_hot(cls_targets_i, num_classes=num_cls_labels).to(cls_preds_i.dtype)
                cls_kwargs = {'alpha': self.cls_alpha, 'gamma': self.cls_gamma, 'reduction': 'none'}
                cls_losses = sigmoid_focal_loss(cls_preds_i, cls_targets_oh_i, **cls_kwargs).sum(dim=1)

            else:
                error_msg = f"Invalid classification loss type '{self.cls_type}'."
                raise ValueError(error_msg)

            # Weight losses of background targets if needed
            if self.bg_weight != 1.0:
                bg_tgt_mask = cls_targets_i == num_cls_labels-1
                cls_losses[bg_tgt_mask] = self.bg_weight * cls_losses[bg_tgt_mask]

            # Get classification loss
            cls_loss_i = self.cls_weight * cls_losses.sum()
            cls_loss += cls_loss_i

            # Get classification accuracy
            with torch.no_grad():
                cls_labels_i = torch.argmax(cls_preds_i, dim=1)
                cls_acc_i = torch.eq(cls_labels_i, cls_targets_i).sum() / len(cls_labels_i)
                cls_acc += 100 * cls_acc_i / batch_size

        return cls_loss, cls_acc

    def get_box_loss(self, box_preds, tgt_dict, pred_ids, tgt_ids, pred_anchors=None):
        """
        Computes bounding box losses from predictions.

        Args:
            box_preds (List): List [batch_size] of bounding box predictions of shape [num_preds, 4].

            tgt_dict (Dict): Target dictionary containing at least following key:
                - boxes (List): list of size [batch_size] with Boxes structures of size [num_targets].

            pred_ids (List): List [batch_size] with indices of predictions of shape [num_preds].
            tgt_ids (List): List [batch_size] with indices of targets matching with predictions of shape [num_preds].
            pred_anchors (List): List [batch_size] of prediction anchors of size [num_preds] (default=None).

        Returns:
            box_loss (FloatTensor): Tensor containing the bounding box loss of shape [1].
            box_acc (FloatTensor): Tensor containing the bounding box accuracy (in percentage) of shape [1].

        Raises:
            ValueError: Error when no prediction anchors are provided when using relative object states.
            ValueError: Error when invalid object state type is provided.
            ValueError: Error when the number of bounding box loss types and corresponding loss weights is different.
            ValueError: Error when invalid bounding box loss type is provided.
        """

        # Get batch size
        batch_size = len(box_preds)

        # Initialize bounding box loss and accuracy
        tensor_kwargs = {'dtype': box_preds[0].dtype, 'device': box_preds[0].device}
        box_loss = torch.zeros(1, **tensor_kwargs)
        box_acc = torch.zeros(1, **tensor_kwargs)

        # Iteratre over every batch entry
        for i in range(batch_size):

            # Get box predictions and corresponding targets
            box_preds_i = box_preds[i][pred_ids[i]]
            tgt_boxes_i = tgt_dict['boxes'][i][tgt_ids[i]]

            # Handle case where there are no targets
            if len(tgt_boxes_i) == 0:
                box_loss += 0.0 * box_preds[i].sum()
                box_acc += 100 / batch_size
                continue

            # Prepare predictions and targets
            if self.state_type == 'abs':
                box_targets_i = tgt_boxes_i.to_format('cxcywh').boxes
                pred_boxes_i = Boxes(box_preds_i, format='cxcywh', normalized='img_with_padding')

                well_defined = pred_boxes_i.well_defined()
                pred_boxes_i = pred_boxes_i[well_defined]
                tgt_boxes_i = tgt_boxes_i[well_defined]

            elif self.state_type in ['rel_static', 'rel_dynamic']:
                if pred_anchors is not None:
                    pred_anchors_i = pred_anchors[i][pred_ids[i]]
                    box_targets_i = get_box_deltas(pred_anchors_i, tgt_boxes_i)
                    pred_boxes_i = apply_box_deltas(box_preds_i, pred_anchors_i)

                else:
                    error_msg = "A list with prediction anchors must be provided when using relative object states."
                    raise ValueError(error_msg)

            else:
                error_msg = f"Invalid object state type '{self.state_type}'."
                raise ValueError(error_msg)

            # Get bounding box loss
            if len(self.box_types) != len(self.box_weights):
                error_msg = "The number of bounding box loss types and corresponding loss weights must be equal."
                raise ValueError(error_msg)

            for box_type, box_weight in zip(self.box_types, self.box_weights):
                if box_type == 'smooth_l1':
                    box_kwargs = {'beta': self.box_beta, 'reduction': 'sum'}
                    box_loss_i = smooth_l1_loss(box_preds_i, box_targets_i, **box_kwargs)
                    box_loss += box_weight * box_loss_i

                elif box_type == 'iou':
                    if len(pred_boxes_i) > 0:
                        box_loss_i = len(pred_boxes_i) - box_iou(pred_boxes_i, tgt_boxes_i).diag().sum()
                        box_loss += box_weight * box_loss_i
                    else:
                        box_loss += 0.0 * box_preds[i].sum()

                elif box_type == 'giou':
                    if len(pred_boxes_i) > 0:
                        box_loss_i = len(pred_boxes_i) - box_giou(pred_boxes_i, tgt_boxes_i).diag().sum()
                        box_loss += box_weight * box_loss_i
                    else:
                        box_loss += 0.0 * box_preds[i].sum()

                else:
                    error_msg = f"Invalid bounding box loss type '{box_type}'."
                    raise ValueError(error_msg)

            # Get bounding box accuracy
            with torch.no_grad():
                if len(pred_boxes_i) > 0:
                    box_acc_i = box_iou(pred_boxes_i, tgt_boxes_i).diag().mean()
                    box_acc += 100 * box_acc_i / batch_size

        return box_loss, box_acc

    def get_loss(self, cls_preds, box_preds, tgt_dict, pred_anchors=None, pred_anchor_ids=None, tgt_sorted_ids=None):
        """
        Compute classification and bounding box losses from predictions.

        Args:
            cls_preds (List): List [batch_size] of classification predictions of shape [num_preds, num_cls_labels].
            box_preds (List): List [batch_size] of bounding box predictions of shape [num_preds, 4].

            tgt_dict (Dict): Target dictionary containing at least following keys:
                - labels (List): list of size [batch_size] with class indices of shape [num_targets];
                - boxes (List): list of size [batch_size] with Boxes structures of size [num_targets].

            pred_anchors (List): List [batch_size] of prediction anchors of size [num_preds] (default=None).
            pred_anchor_ids (List): List [batch_size] with anchor indices of shape [num_preds] (default=None).
            tgt_sorted_ids (List): List [batch_size] of sorted ids of shape [sort_thr, num_targets] (default=None).

        Returns:
            loss_dict (Dict): Loss dictionary containing following keys:
                - cls_loss (FloatTensor): tensor containing the classification loss of shape [1];
                - box_loss (FloatTensor): tensor containing the bounding box loss of shape [1].

            analysis_dict (Dict): Analysis dictionary containing following keys:
                - cls_acc (FloatTensor): tensor containing the classification accuracy (in percentage) of shape [1];
                - box_acc (FloatTensor): tensor containing the bounding box accuracy (in percentage) of shape [1].

            pos_pred_ids (List): List [batch_size] with indices of positive predictions of shape [num_pos_preds].
        """

        # Perform prediction-target matching
        match_args = (pred_anchors, tgt_dict['boxes'], pred_anchor_ids, tgt_sorted_ids)
        pos_pred_ids, pos_tgt_ids, neg_pred_ids = self.perform_matching(*match_args)

        # Get classification loss and accuracy
        cls_loss, cls_acc = self.get_cls_loss(cls_preds, tgt_dict, pos_pred_ids, pos_tgt_ids, neg_pred_ids)

        # Get bounding box loss and accuracy
        box_loss, box_acc = self.get_box_loss(box_preds, tgt_dict, pos_pred_ids, pos_tgt_ids, pred_anchors)

        # Get loss and analysis dictionaries
        loss_dict = {'cls_loss': cls_loss, 'box_loss': box_loss}
        analysis_dict = {'cls_acc': cls_acc, 'box_acc': box_acc}

        return loss_dict, analysis_dict, pos_pred_ids, pos_tgt_ids

    @torch.no_grad()
    def make_predictions(self, cls_preds, box_preds, pred_anchors=None, return_obj_ids=False):
        """
        Makes classified bounding box predictions.

        Args:
            cls_preds (List): List [batch_size] of classification predictions of shape [num_preds, num_cls_labels].
            box_preds (List): List [batch_size] of bounding box predictions of shape [num_preds, 4].
            pred_anchors (List): List [batch_size] of prediction anchors of size [num_preds] (default=None).

        Returns:
            pred_dict (Dict): Prediction dictionary containing following keys:
                - labels (LongTensor): predicted class indices of shape [num_preds_total];
                - boxes (Boxes): structure containing axis-aligned bounding boxes of size [num_preds_total];
                - scores (FloatTensor): normalized prediction scores of shape [num_preds_total];
                - batch_ids (LongTensor): batch indices of predictions of shape [num_preds_total].

            If 'return_obj_ids' is True, the prediction dictionary additionally contains following key:
                - obj_ids (List): list [batch_size] containing the object indices of predicitons of shape [num_preds].

        Raises:
            ValueError: Error when no prediction anchors are provided when using relative object states.
            ValueError: Error when invalid object state type is provided.
            ValueError: Error when invalid duplicate removal mechanism is provided.
        """

        # Get batch size
        batch_size = len(cls_preds)

        # Initialize prediction dictionary
        pred_keys = ('labels', 'boxes', 'scores', 'batch_ids')
        pred_dict = {pred_key: [] for pred_key in pred_keys}

        if return_obj_ids:
            pred_dict['obj_ids'] = []

        # Get predictions for every batch entry
        for i in range(batch_size):

            # Get prediction labels and scores
            cls_preds_i = cls_preds[i][:, :-1]
            scores_i, labels_i = cls_preds_i.sigmoid().max(dim=1)

            # Get prediction boxes
            if self.state_type == 'abs':
                boxes_i = Boxes(box_preds[i], format='cxcywh', normalized='img_with_padding')

            elif self.state_type in ['rel_static', 'rel_dynamic']:
                if pred_anchors is not None:
                    boxes_i = apply_box_deltas(box_preds[i], pred_anchors[i])
                else:
                    error_msg = "A list with prediction anchors must be provided when using relative object states."
                    raise ValueError(error_msg)

            else:
                error_msg = f"Invalid object state type '{self.state_type}'."
                raise ValueError(error_msg)

            # Only keep entries with well-defined boxes
            well_defined = boxes_i.well_defined()
            boxes_i = boxes_i[well_defined]
            scores_i = scores_i[well_defined]
            labels_i = labels_i[well_defined]

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
                non_dup_ids = batched_nms(boxes_i.boxes, scores_i, labels_i, iou_threshold=self.nms_thr)

            else:
                error_msg = f"Invalid duplicate removal mechanism '{self.dup_removal}'."
                raise ValueError(error_msg)

            # Keep maximum number of allowed predictions
            non_dup_ids = non_dup_ids[:self.max_dets]
            labels_i = labels_i[non_dup_ids]
            boxes_i = boxes_i[non_dup_ids]
            scores_i = scores_i[non_dup_ids]

            # Append predictions to their respective lists
            pred_dict['labels'].append(labels_i)
            pred_dict['boxes'].append(boxes_i)
            pred_dict['scores'].append(scores_i)
            pred_dict['batch_ids'].append(torch.full_like(labels_i, i))

            if return_obj_ids:
                num_preds_i = len(cls_preds_i)
                obj_ids_i = torch.arange(num_preds_i, device=cls_preds_i.device)
                obj_ids_i = obj_ids_i[well_defined][top_pred_ids][non_dup_ids]
                pred_dict['obj_ids'].append(obj_ids_i)

        # Concatenate predictions of different batch entries
        pred_dict.update({k: torch.cat(v, dim=0) for k, v in pred_dict.items() if k not in ['boxes', 'obj_ids']})
        pred_dict['boxes'] = Boxes.cat(pred_dict['boxes'])

        return pred_dict

    def forward(self, feat_maps, tgt_dict=None, images=None, stand_alone=True, visualize=False, **kwargs):
        """
        Forward method of the SBD head.

        Args:
            feat_maps (List): List of size [num_maps] with feature maps of shape [batch_size, feat_size, fH, fW].

            tgt_dict (Dict): Optional target dictionary used during trainval containing at least following keys:
                - labels (LongTensor): tensor of shape [num_targets_total] containing the class indices;
                - boxes (Boxes): structure containing axis-aligned bounding boxes of size [num_targets_total];
                - sizes (LongTensor): tensor of shape [batch_size+1] with the cumulative target sizes of batch entries.

            images (Images): Images structure containing the batched images (default=None).
            stand_alone (bool): Boolean indicating whether the SBD module operates as stand-alone (default=True).
            visualize (bool): Boolean indicating whether to compute dictionary with visualizations (default=False).
            kwargs (Dict): Dictionary of keyword arguments not used by this module, but passed to some sub-modules.

        Returns:
            * If SBD module in training mode (i.e. during training):
                loss_dict (Dict): Dictionary of different weighted loss terms used during training.
                analysis_dict (Dict): Dictionary of different analyses used for logging purposes only.

                If SBD module is not stand-alone:
                    pred_dict (Dict): SBD prediction dictionary containing the final SBD predictions.
                    obj_states (List): List [batch_size] with state features corresponding to each object prediction.
                    obj_anchors (List): List [batch_size] with anchors corresponding to each object prediction.
                    feats (FloatTensor): Concatenated input features of shape [batch_size, num_feats, feat_size].
                    feat_ids (List): List [batch_size] with feature indices corresponding to each object prediction.

            * If SBD module not in training mode and tgt_dict is not None (i.e. during validation):
                pred_dicts (List): List with SBD prediction dictionaries.
                loss_dict (Dict): Dictionary of different weighted loss terms used during training.
                analysis_dict (Dict): Dictionary of different analyses used for logging purposes only.

                If SBD module is not stand-alone:
                    obj_states (List): List [batch_size] with state features corresponding to each object prediction.
                    obj_anchors (List): List [batch_size] with anchors corresponding to each object prediction.
                    feats (FloatTensor): Concatenated input features of shape [batch_size, num_feats, feat_size].
                    feat_ids (List): List [batch_size] with feature indices corresponding to each object prediction.

            * If SBD module not in training mode and tgt_dict is None (i.e. during testing):
                pred_dicts (List): List with SBD prediction dictionaries.
                analysis_dict (Dict): Dictionary of different analyses used for logging purposes only.

                If SBD module is not stand-alone:
                    obj_states (List): List [batch_size] with state features corresponding to each object prediction.
                    obj_anchors (List): List [batch_size] with anchors corresponding to each object prediction.
                    feats (FloatTensor): Concatenated input features of shape [batch_size, num_feats, feat_size].
                    feat_ids (List): List [batch_size] with feature indices corresponding to each object prediction.

        Raises:
            NotImplementedError: Error when visualizations are requested.
            ValueError: Error when no Images structure is provided.
            ValueError: Error when invalid loss application frequency is provided.
            ValueError: Error when the number of update iterations is smaller than one.
        """

        # Check whether visualizations are requested
        if visualize:
            raise NotImplementedError

        # Get batch size
        batch_size = len(feat_maps[0])

        # Initialize empty loss dictionary
        if tgt_dict is not None:
            loss_dict = {}

        # Initialize empty list for prediction dictionaries
        if not self.training:
            pred_dicts = []

        # Apply DOD module and extract output
        dod_kwargs = {'tgt_dict': tgt_dict, 'stand_alone': False}
        dod_output = self.dod(feat_maps, **dod_kwargs, **kwargs)

        if tgt_dict is None:
            sel_ids, anchors, analysis_dict = dod_output
        elif self.dod.tgt_mode == 'ext_dynamic':
            logits, obj_probs, sel_ids, anchors, pos_masks, neg_masks, tgt_sorted_ids, analysis_dict = dod_output
        else:
            sel_ids, anchors, tgt_sorted_ids, dod_loss_dict, analysis_dict = dod_output
            loss_dict.update(dod_loss_dict)

        # Get selected features
        feats = torch.cat([feat_map.flatten(2).permute(0, 2, 1) for feat_map in feat_maps], dim=1)
        feat_ids = [sel_ids_i // self.dod.num_cell_anchors for sel_ids_i in sel_ids]
        sel_feats = [feats[i, feat_ids[i], :] for i in range(batch_size)]

        # Get initial relative object states
        tensor_kwargs = {'dtype': feats.dtype, 'device': feats.device}
        obj_states = [torch.zeros(len(sel_ids[i]), self.state_size, **tensor_kwargs) for i in range(batch_size)]
        osi_ids = [sel_ids_i % self.dod.num_cell_anchors for sel_ids_i in sel_ids]

        for i in range(batch_size):
            for j in range(self.dod.num_cell_anchors):
                osi_mask = osi_ids[i] == j
                obj_states[i][osi_mask] = self.osi[j](sel_feats[i][osi_mask])

        num_states = sum(len(obj_states_i) for obj_states_i in obj_states)
        analysis_dict['num_states_0'] = torch.tensor([num_states], device=feats.device) / batch_size

        # Get anchors and corresponding anchor encodings if needed
        obj_anchors = [anchors[sel_ids_i] for sel_ids_i in sel_ids]

        if hasattr(self, 'ae'):
            if images is None:
                error_msg = "An Images structure containing the batched images must be provided."
                raise ValueError(error_msg)

            norm_anchors = [obj_anchors[i].clone().normalize(images[i]).to_format('cxcywh') for i in range(batch_size)]
            anchor_encs = [self.ae(norm_anchors[i].boxes) for i in range(batch_size)]

            if hasattr(self, 'se'):
                scale_encs = [self.se(norm_anchors[i].boxes[:, 2:]) for i in range(batch_size)]

        # Get absolute object states if desired
        if self.state_type == 'abs':
            if hasattr(self, 'se'):
                obj_states = [obj_states[i] * scale_encs[i] for i in range(batch_size)]
            obj_states = [obj_states[i] + anchor_encs[i].detach() for i in range(batch_size)]

        # Get boolean indicating whether initial losses/prediction dictionary should be computed
        if self.apply_freq == 'last':
            compute_loss_preds = len(self.up_layers) == 0
        elif self.apply_freq in ['iters', 'layers']:
            compute_loss_preds = True
        else:
            error_msg = f"Invalid type of loss application frequency '{self.apply_freq}'."
            raise ValueError(error_msg)

        # Get initial predictions
        if self.training and self.freeze_inter and len(self.up_layers) > 0:
            cls_module = deepcopy(self.cls).requires_grad_(False)
            box_module = deepcopy(self.box).requires_grad_(False)
        else:
            cls_module = self.cls
            box_module = self.box

        cls_preds = [cls_module(obj_states[i]) for i in range(batch_size)]
        box_preds = [box_module(obj_states[i]) for i in range(batch_size)]

        # Get initial loss if desired
        if tgt_dict is not None:

            # Get anchor encoding loss if desired
            if self.state_type == 'abs':
                ae_box_preds = [self.box(anchor_encs[i]) for i in range(batch_size)]
                ae_tgt_dict = {'boxes': norm_anchors}

                ae_pred_ids = [torch.arange(len(sel_ids[i]), device=feats.device) for i in range(batch_size)]
                ae_tgt_ids = [ae_pred_ids[i].clone() for i in range(batch_size)]

                ae_loss, ae_acc = self.get_box_loss(ae_box_preds, ae_tgt_dict, ae_pred_ids, ae_tgt_ids)
                loss_dict['ae_loss'] = self.ae_weight * ae_loss
                analysis_dict['ae_acc'] = ae_acc

            # Get SBD target dictionary of desired format
            tgt_boxes = tgt_dict['boxes']
            tgt_sizes = tgt_dict['sizes']

            if self.state_type == 'abs':
                tgt_boxes = tgt_boxes.normalize(images)

            tgt_labels = [tgt_dict['labels'][i0:i1] for i0, i1 in zip(tgt_sizes[:-1], tgt_sizes[1:])]
            tgt_boxes = [tgt_boxes[i0:i1] for i0, i1 in zip(tgt_sizes[:-1], tgt_sizes[1:])]
            sbd_tgt_dict = {'labels': tgt_labels, 'boxes': tgt_boxes}

            # Get static loss keyword arguments
            loss_kwargs = {'pred_anchor_ids': sel_ids, 'tgt_sorted_ids': tgt_sorted_ids}

            # Get initial classification and bounding box losses with corresponding analyses if desired
            if compute_loss_preds:
                loss_kwargs['pred_anchors'] = obj_anchors
                loss_output = self.get_loss(cls_preds, box_preds, sbd_tgt_dict, **loss_kwargs)
                local_loss_dict, local_analysis_dict, pos_pred_ids, pos_tgt_ids = loss_output

                loss_dict.update({f'{k}_0': v for k, v in local_loss_dict.items()})
                analysis_dict.update({f'{k}_0': v for k, v in local_analysis_dict.items()})

        # Get initial prediction dictionary if desired
        if not self.training and compute_loss_preds:
            return_obj_ids = len(self.up_layers) == 0 and not stand_alone
            pred_dict = self.make_predictions(cls_preds, box_preds, obj_anchors, return_obj_ids=return_obj_ids)
            pred_dicts.append(pred_dict)

        # Get static keyword arguments for update layers
        up_kwargs = [{} for _ in range(batch_size)]

        if self.state_type == 'rel_static' and hasattr(self, 'ae'):
            for i in range(batch_size):
                up_kwargs[i]['add_encs'] = anchor_encs[i]
                up_kwargs[i]['mul_encs'] = scale_encs[i] if hasattr(self, 'se') else None

        if self.up_ca_type in ('deformable_attn', 'particle_attn'):
            sample_map_shapes = torch.tensor([feat_map.shape[-2:] for feat_map in feat_maps], device=feats.device)
            sample_map_start_ids = sample_map_shapes.prod(dim=1).cumsum(dim=0)[:-1]
            sample_map_start_ids = torch.cat([sample_map_shapes.new_zeros((1,)), sample_map_start_ids], dim=0)

            for i in range(batch_size):
                if self.state_type == 'rel_static':
                    up_kwargs[i]['sample_priors'] = norm_anchors[i].boxes

                up_kwargs[i]['sample_feats'] = feats[i]
                up_kwargs[i]['sample_map_shapes'] = sample_map_shapes
                up_kwargs[i]['sample_map_start_ids'] = sample_map_start_ids

                level_mask = (feat_ids[i][:, None] - sample_map_start_ids) >= 0
                level_ids = level_mask.sum(dim=1) - 1
                up_kwargs[i]['level_ids'] = level_ids

                if self.up_ca_type == 'particle_attn':
                    up_kwargs[i]['storage_dict'] = {}

        # Check whether the number of update iteration is greater than zero
        if self.up_iters < 1:
            error_msg = f"The number of update iterations should be greater than zero (got {self.up_iters})."
            raise ValueError(error_msg)

        # Update object states and get losses and prediction dictionaries if desired
        for i in range(1, self.up_iters+1):
            for j, up_layer in enumerate(self.up_layers, 1):

                # Update anchors if desired
                if self.state_type == 'rel_dynamic':
                    obj_anchors = [apply_box_deltas(box_preds[i], obj_anchors[i]) for i in range(batch_size)]
                    norm_anchors = [obj_anchors[i].clone().normalize(images[i]) for i in range(batch_size)]
                    norm_anchors = [norm_anchors[i].to_format('cxcywh') for i in range(batch_size)]
                    anchor_encs = [self.ae(norm_anchors[i].boxes) for i in range(batch_size)]

                    if hasattr(self, 'se'):
                        scale_encs = [self.se(norm_anchors[i].boxes[:, 2:]) for i in range(batch_size)]

                # Get dynamic keyword arguments for update layers
                if self.state_type == 'abs' and self.up_ca_type in ('deformable_attn', 'particle_attn'):
                    for i in range(batch_size):
                        up_kwargs[i]['sample_priors'] = box_preds[i]

                elif self.state_type == 'rel_dynamic':
                    for i in range(batch_size):
                        up_kwargs[i]['add_encs'] = anchor_encs[i]
                        up_kwargs[i]['mul_encs'] = scale_encs[i] if hasattr(self, 'se') else None

                        if self.up_ca_type in ('deformable_attn', 'particle_attn'):
                            up_kwargs[i]['sample_priors'] = norm_anchors[i].boxes

                # Update object states
                obj_states = [up_layer(obj_states[i], **up_kwargs[i]) for i in range(batch_size)]

                # Get boolean indicating whether losses/prediction dictionary should be computed
                last_iter = i == self.up_iters
                last_layer = j == len(self.up_layers)

                compute_loss_preds = (last_iter and last_layer) or (last_layer and self.apply_freq != 'last')
                compute_loss_preds = compute_loss_preds or (self.apply_freq == 'layers')

                # Get predictions
                if self.training and self.freeze_inter and last_iter and last_layer:
                    cls_module = self.cls
                    box_module = self.box

                cls_preds = [cls_module(obj_states[i]) for i in range(batch_size)]
                box_preds = [box_module(obj_states[i]) for i in range(batch_size)]

                # Get loss if desired
                if tgt_dict is not None and compute_loss_preds:
                    loss_kwargs['pred_anchors'] = obj_anchors
                    loss_output = self.get_loss(cls_preds, box_preds, sbd_tgt_dict, **loss_kwargs)
                    local_loss_dict, local_analysis_dict, pos_pred_ids, pos_tgt_ids = loss_output

                    loss_dict.update({f'{k}_{i}_{j}': v for k, v in local_loss_dict.items()})
                    analysis_dict.update({f'{k}_{i}_{j}': v for k, v in local_analysis_dict.items()})

                # Get prediction dictionary if desired
                if not self.training and compute_loss_preds:
                    return_obj_ids = last_iter and last_layer and not stand_alone
                    pred_dict = self.make_predictions(cls_preds, box_preds, obj_anchors, return_obj_ids=return_obj_ids)
                    pred_dicts.append(pred_dict)

        # Get DOD losses if in trainval and in DOD external dynamic target mode
        if tgt_dict is not None and self.dod.tgt_mode == 'ext_dynamic':

            # Get external dictionary
            pos_pred_ids = [torch.unique_consecutive(pos_pred_ids[i]) for i in range(batch_size)]
            pos_anchor_ids = [sel_ids[i][pos_pred_ids[i]] for i in range(batch_size)]
            tgt_found = []

            for i in range(batch_size):
                num_tgts = tgt_sorted_ids[i].shape[1]
                tgt_found_i = torch.zeros(num_tgts, dtype=torch.bool, device=logits.device)
                tgt_found_i[pos_tgt_ids[i]] = True
                tgt_found.append(tgt_found_i)

            ext_dict = {'logits': logits, 'obj_probs': obj_probs, 'pos_anchor_ids': pos_anchor_ids}
            ext_dict = {**ext_dict, 'tgt_found': tgt_found, 'pos_masks': pos_masks, 'neg_masks': neg_masks}
            ext_dict = {**ext_dict, 'tgt_sorted_ids': tgt_sorted_ids}

            # Get DOD loss and analysis dictionary
            dod_loss_dict, dod_analysis_dict = self.dod(feat_maps, ext_dict=ext_dict, **dod_kwargs, **kwargs)
            loss_dict.update(dod_loss_dict)
            analysis_dict.update(dod_analysis_dict)

        # Return desired dictionaries if SBD module is stand-alone
        if stand_alone:
            if self.training:
                return loss_dict, analysis_dict
            elif tgt_dict is not None:
                return pred_dicts, loss_dict, analysis_dict
            else:
                return pred_dicts, analysis_dict

        # Return additional items if SBD module is not stand-alone
        if self.training:
            pred_dict = self.make_predictions(cls_preds, box_preds, obj_anchors, return_obj_ids=True)

        obj_ids = pred_dict.pop('obj_ids')
        obj_states = [obj_states[i][obj_ids[i]] for i in range(batch_size)]
        obj_anchors = [obj_anchors[i][obj_ids[i]] for i in range(batch_size)]
        feat_ids = [feat_ids[i][obj_ids[i]] for i in range(batch_size)]

        if self.training:
            return loss_dict, analysis_dict, pred_dict, obj_states, obj_anchors, feats, feat_ids
        elif tgt_dict is not None:
            return pred_dicts, loss_dict, analysis_dict, obj_states, obj_anchors, feats, feat_ids
        else:
            return pred_dicts, analysis_dict, obj_states, obj_anchors, feats, feat_ids