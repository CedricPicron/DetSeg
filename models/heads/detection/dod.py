"""
Dense Object Discovery (DOD) head.
"""
from collections import OrderedDict
from copy import deepcopy
import math

from fvcore.nn import sigmoid_focal_loss
import torch
from torch import nn

from models.modules.convolution import BottleneckConv, ProjConv
from structures.boxes import Boxes, box_iou, get_anchors


class DOD(nn.Module):
    """
    Class implementing the Dense Object Discovery (DOD) head.

    Attributes:
        net (nn.Sequential): DOD network computing the logits.
        num_cell_anchors (int): Integer containing the number of cell anchors (i.e. the number of anchors per feature).

        anchor_dict (Dict): Dictionary with items used for anchor generation containing following keys:
            - map_ids (List): list [num_maps] containing the map ids (i.e. downsampling exponents) of each map;
            - num_sizes (int): integer containing the number of different anchor sizes per aspect ratio;
            - scale_factor (float): factor scaling the anchors w.r.t. non-overlapping tiling anchors;
            - aspect_ratios (List): list [num_aspect_ratios] containing the different anchor aspect ratios.

        sel_mode (str): String containing the anchor selection mode.
        sel_abs_thr (float): Absolute threshold determining the selected anchors.
        sel_rel_thr (int): Relative threshold determining the selected anchors.

        tgt_metric (str): String containing the anchor-target matching metric.
        tgt_sort_thr (int): Threshold containing the amount of sorted anchor indices to return per target.
        tgt_decision (str): String containing the target decision maker type.
        tgt_abs_pos (float): Absolute threshold used during positive target decision making.
        tgt_abs_neg (float): Absolute threshold used during negative target decision making.
        tgt_rel_pos (int): Relative threshold used during positive target decision making.
        tgt_rel_neg (int): Relative threshold used during negative target decision making.
        tgt_mode (str): String containing the target mode.

        loss_type (str): String containing the type of loss.
        focal_alpha (float): Alpha value of the sigmoid focal loss.
        focal_gamma (float): Gamma value of the sigmoid focal loss.
        pos_weight (float): Factor weighting the loss terms with positive targets.
        neg_weight (float): Factor weighting the loss terms with negative targets.

        pred_num_pos (int): Integer containing the number of positive anchors per target during prediction.
        pred_max_dets (int): Integer containing the maximum number of detections during prediction.

        metadata (detectron2.data.Metadata): Metadata instance containing additional dataset information.
    """

    def __init__(self, in_feat_size, net_dict, anchor_dict, sel_dict, tgt_dict, loss_dict, pred_dict, metadata):
        """
        Initializes the DOD module.

        Args:
            in_feat_size (int): Integer containing the feature size of the input feature pyramid.

            net_dict (Dict): Network dictionary containing following keys:
                - feat_size (int): hidden feature size of the DOD network;
                - norm (str): string specifying the type of normalization used within the DOD network;
                - kernel_size (int): kernel size used by hidden layers of the DOD network;
                - bottle_size (int): bottleneck size used by bottleneck layers of the DOD network;
                - hidden_layers (int): number of hidden layers of the DOD network;
                - rel_preds (bool): whether or not DOD network should make relative predictions;
                - prior_prob (float): prior object probability determining the initial output layer bias value(s).

            anchor_dict (Dict): Dictionary with items used for anchor generation containing following keys:
                - map_ids (List): list [num_maps] containing the map ids (i.e. downsampling exponents) of each map;
                - num_sizes (int): integer containing the number of different anchor sizes per aspect ratio;
                - scale_factor (float): factor scaling the anchors w.r.t. non-overlapping tiling anchors;
                - aspect_ratios (List): list [num_aspect_ratios] containing the different anchor aspect ratios.

            sel_dict (Dict): Anchor selection dictionary containing following keys:
                - mode (str): string containing the anchor selection mode;
                - abs_thr (float): absolute threshold determining the selected anchors;
                - rel_thr (int): relative threshold determining the selected anchors;
                - eval (str): string containing the selected anchors evalution mode.

            tgt_dict (Dict): Dictionary with items used during target computation containing following keys:
                - metric (str): string containing the anchor-target matching metric;
                - sort_thr (int): threshold containing the amount of sorted anchor indices to return per target;
                - decision (str): string containing the target decision maker type;
                - abs_pos (float): absolute threshold used during positive target decision making;
                - abs_neg (float): absolute threshold used during negative target decision making;
                - rel_pos (int): relative threshold used during positive target decision making;
                - rel_neg (int): relative threshold used during negative target decision making;
                - mode (str): string containing the target mode.

            loss_dict (Dict): Loss dictionary containing following keys:
                - type (str): string containing the type of loss;
                - focal_alpha (float): alpha value of the sigmoid focal loss;
                - focal_gamma (float): gamma value of the sigmoid focal loss;
                - pos_weight (float): factor weighting the loss terms with positive targets;
                - neg_weight (float): factor weighting the loss terms with negative targets.

            pred_dict (Dict): Dictionary with items used during prediction containing following keys:
                - num_pos (int): integer containing the number of positive anchors per target during prediction;
                - max_dets (int): integer containing the maximum number of detections during prediction.

            metadata (detectron2.data.Metadata): Metadata instance containing additional dataset information.
        """

        # Initialization of default nn.Module
        super().__init__()

        # Initialization of the DOD network
        feat_size = net_dict['feat_size']
        norm = net_dict['norm']
        in_layer = ProjConv(in_feat_size, feat_size, norm=norm, skip=False)

        if net_dict['kernel_size'] == 1:
            hidden_layer = ProjConv(feat_size, norm=norm, skip=True)
        else:
            bottle_size = net_dict['bottle_size']
            bottleneck_kwargs = {'kernel_size': net_dict['kernel_size'], 'norm': norm, 'skip': True}
            hidden_layer = BottleneckConv(feat_size, bottle_size, **bottleneck_kwargs)

        num_hidden_layers = net_dict['hidden_layers']
        hidden_layers = nn.Sequential(*[deepcopy(hidden_layer) for _ in range(num_hidden_layers)])

        num_cell_anchors = anchor_dict['num_sizes'] * len(anchor_dict['aspect_ratios'])
        out_feat_size = 2*num_cell_anchors if net_dict['rel_preds'] else num_cell_anchors
        out_layer = ProjConv(feat_size, out_feat_size, norm=norm, skip=False)

        obj_prior_prob = net_dict['prior_prob']
        out_bias_value = -(math.log((1 - obj_prior_prob) / obj_prior_prob))

        if net_dict['rel_preds']:
            with torch.no_grad():
                out_bias = torch.tensor([out_bias_value, -out_bias_value]).repeat(num_cell_anchors)
                out_layer.conv.bias.copy_(out_bias)
        else:
            torch.nn.init.constant_(out_layer.conv.bias, out_bias_value)

        net_dict = OrderedDict([('in', in_layer), ('hidden', hidden_layers), ('out', out_layer)])
        self.net = nn.Sequential(net_dict)

        # Set anchor-related attributes
        self.num_cell_anchors = num_cell_anchors
        self.anchor_dict = anchor_dict

        # Set anchor selection attributes
        for k, v in sel_dict.items():
            setattr(self, f'sel_{k}', v)

        # Set target-related attributes
        for k, v in tgt_dict.items():
            setattr(self, f'tgt_{k}', v)

        # Set loss-related attributes
        self.loss_type = loss_dict['type']
        self.focal_alpha = loss_dict['focal_alpha']
        self.focal_gamma = loss_dict['focal_gamma']
        self.pos_weight = loss_dict['pos_weight']
        self.neg_weight = loss_dict['neg_weight']

        # Set prediction-related attributes
        self.pred_num_pos = pred_dict['num_pos']
        self.pred_max_dets = pred_dict['max_dets']

        # Set metadata attribute
        metadata.stuff_classes = metadata.thing_classes
        metadata.stuff_colors = metadata.thing_colors
        self.metadata = metadata

    @torch.no_grad()
    def get_static_tgt_masks(self, feat_maps, anchors, tgt_boxes, return_ids=True):
        """
        Get positive and negative static target masks.

        Args:
            feat_maps (List): List of size [num_maps] with feature maps of shape [batch_size, feat_size, fH, fW].
            anchors (Boxes): Structure containing axis-aligned anchor boxes of size [num_anchors].
            tgt_boxes (Boxes): Structure containing axis-aligned target boxes of size [num_targets_total].
            return_ids (bool): Boolean indicating whether sorted anchor indices should be returned (default=True).

        Returns:
            pos_masks (List): List [batch_size] of positive static target masks of shape [num_anchors, num_targets].
            neg_masks (List): List [batch_size] of negative static target masks of shape [num_anchors, num_targets].

            If 'return_ids' is True:
                tgt_sorted_ids (List): List [batch_size] of sorted anchor indices of shape [sort_thr, num_targets].

        Raises:
            ValueError: Error when unknown anchor-target metric is present in 'tgt_metric' attribute.
            ValueError: Error when unknown target decision maker type is present in 'tgt_decision' attribute.
        """

        # Get batch size and number of target boxes
        batch_size = len(feat_maps[0])
        num_tgts = len(tgt_boxes)

        # Return if there are no target boxes
        if num_tgts == 0:
            num_anchors = len(anchors)
            device = feat_maps[0].device

            pos_masks = [torch.zeros(num_anchors, 0, dtype=torch.bool, device=device) for _ in range(batch_size)]
            neg_masks = [torch.ones(num_anchors, 0, dtype=torch.bool, device=device) for _ in range(batch_size)]

            if not return_ids:
                return pos_masks, neg_masks

            tensor_kwargs = {'dtype': torch.int64, 'device': device}
            tgt_sorted_ids = [torch.zeros(self.tgt_sort_thr, 0, **tensor_kwargs) for _ in range(batch_size)]

            return pos_masks, neg_masks, tgt_sorted_ids

        # Get anchor-target similarity matrix and target sorted anchor indices
        if self.tgt_metric == 'iou':
            sim_matrix = box_iou(anchors, tgt_boxes)
            tgt_sorted_ids = torch.topk(sim_matrix, self.tgt_sort_thr, dim=0, sorted=True).indices

        else:
            error_msg = f"Unknown anchor-target metric '{self.tgt_metric}'."
            raise ValueError(error_msg)

        # Get positive and negative static target masks
        if 'abs' in self.tgt_decision:
            abs_pos_mask = sim_matrix >= self.tgt_abs_pos
            abs_neg_mask = sim_matrix < self.tgt_abs_neg

        if 'rel' in self.tgt_decision:
            pos_ids = tgt_sorted_ids[:self.tgt_rel_pos, :]
            rel_pos_mask = torch.zeros_like(sim_matrix, dtype=torch.bool)
            rel_pos_mask[pos_ids, torch.arange(num_tgts)] = True

            non_neg_ids = tgt_sorted_ids[:self.tgt_rel_neg, :]
            rel_neg_mask = torch.ones_like(sim_matrix, dtype=torch.bool)
            rel_neg_mask[non_neg_ids, torch.arange(num_tgts)] = False

        if self.tgt_decision == 'abs':
            pos_mask = abs_pos_mask
            neg_mask = abs_neg_mask

        elif self.tgt_decision == 'abs_and_rel':
            pos_mask = abs_pos_mask & rel_pos_mask
            neg_mask = abs_neg_mask & rel_neg_mask

        elif self.tgt_decision == 'abs_or_rel':
            pos_mask = abs_pos_mask | rel_pos_mask
            neg_mask = abs_neg_mask | rel_neg_mask

        elif self.tgt_decision == 'rel':
            pos_mask = rel_pos_mask
            neg_mask = rel_neg_mask

        else:
            error_msg = f"Unknown target decision maker type '{self.tgt_decision}'."
            raise ValueError(error_msg)

        tgts_per_img = tgt_boxes.boxes_per_img.tolist()
        pos_masks = torch.split(pos_mask, tgts_per_img, dim=1)
        neg_masks = torch.split(neg_mask, tgts_per_img, dim=1)

        # Return masks with sorted anchor indices if requested
        if return_ids:
            tgt_sorted_ids = torch.split(tgt_sorted_ids, tgts_per_img, dim=1)
            return pos_masks, neg_masks, tgt_sorted_ids

        return pos_masks, neg_masks

    @torch.no_grad()
    def get_ap(self, obj_probs, tgt_sorted_ids):
        """
        Get average precision (AP) based on DOD object probabilities and sorted anchor indices.

        Args:
            obj_probs (FloatTensor): Tensor containing object probabilities of shape [batch_size, num_anchors].
            tgt_sorted_ids (List): List [batch_size] of sorted anchor indices of shape [sort_thr, num_targets].

        Returns:
            ap (FloatTensor): Tensor containing the average precision (AP) of shape [1].
        """

        # Get batch size and number of anchors
        batch_size, num_anchors = obj_probs.shape

        # Initialize average precision tensor
        ap = torch.zeros(1).to(obj_probs)

        # Get average precision for every batch entry
        for i in range(batch_size):

            # Get number of targets
            num_tgts = tgt_sorted_ids[i].shape[1]

            # Handle case where there are no targets
            if num_tgts == 0:
                ap += 100/batch_size
                continue

            # Get positives mask sorted according to object probabilities
            pos_ids = tgt_sorted_ids[i][:self.pred_num_pos, :]
            pos_mask = torch.zeros(num_anchors, num_tgts, dtype=torch.bool, device=ap.device)
            pos_mask[pos_ids, torch.arange(num_tgts)] = True

            pred_sorted_ids = torch.argsort(obj_probs[i], dim=0, descending=True)[:self.pred_max_dets]
            pos_mask = pos_mask[pred_sorted_ids, :]

            # Get precisions
            positives = pos_mask.sum(dim=1) > 0
            precisions = positives.cumsum(dim=0) / torch.arange(1, self.pred_max_dets+1).to(positives.device)
            precisions = precisions.flip([0]).cummax(dim=0)[0].flip([0])

            # Get recalls
            recalls = pos_mask.cummax(dim=0)[0].sum(dim=1) / num_tgts

            # Get average precision
            recalls, counts = torch.unique_consecutive(recalls, return_counts=True)
            cum_counts = torch.tensor([0, *counts.cumsum(dim=0).tolist()]).to(counts)
            precisions = [precisions[i0:i1].max()[None] for i0, i1 in zip(cum_counts[:-1], cum_counts[1:])]
            precisions = torch.cat(precisions, dim=0)

            if recalls[0] == 0:
                precisions = precisions[1:]
            else:
                recalls = torch.tensor([0, *recalls.tolist()]).to(recalls)

            areas = precisions * (recalls[1:] - recalls[:-1])
            ap += 100 * areas.sum() / batch_size

        return ap

    @torch.no_grad()
    def make_predictions(self, obj_probs, tgt_sorted_ids, tgt_dict):
        """
        Makes predictions based on DOD object probabilities, sorted anchor indices and corresponding targets.

        Args:
            obj_probs (FloatTensor): Tensor containing object probabilities of shape [batch_size, num_anchors].
            tgt_sorted_ids (List): List [batch_size] of sorted anchor indices of shape [sort_thr, num_targets].

            tgt_dict (Dict): Target dictionary containing at least following keys:
                - labels (LongTensor): tensor of shape [num_targets_total] containing the class indices;
                - boxes (Boxes): structure containing axis-aligned bounding boxes of size [num_targets_total];
                - sizes (LongTensor): tensor of shape [batch_size+1] with the cumulative target sizes of batch entries.

        Returns:
            pred_dicts (List): List of size [3] with the DOD prediction dictionaries containing following keys:
                - labels (LongTensor): predicted class indices of shape [num_preds_total];
                - boxes (Boxes): structure containing axis-aligned bounding boxes of size [num_preds_total];
                - scores (FloatTensor): normalized prediction scores of shape [num_preds_total];
                - batch_ids (LongTensor): batch indices of predictions of shape [num_preds_total].
        """

        # Get batch size and number of anchors
        batch_size, num_anchors = obj_probs.shape

        # Group targets per batch entry
        tgt_sizes = tgt_dict['sizes']
        tgt_labels = [tgt_dict['labels'][i0:i1] for i0, i1 in zip(tgt_sizes[:-1], tgt_sizes[1:])]
        tgt_boxes = [tgt_dict['boxes'][i0:i1] for i0, i1 in zip(tgt_sizes[:-1], tgt_sizes[1:])]

        # Get device, default label and default bounding box
        device = obj_probs.device
        def_label = torch.tensor([0], dtype=torch.int64, device=device)
        def_box_kwargs = {'format': tgt_boxes[0].format, 'normalized': tgt_boxes[0].normalized}
        def_box = Boxes(torch.tensor([[0.01, 0.01, 0.02, 0.02]], device=device), **def_box_kwargs)

        # Initialize prediction dictionaries
        pred_keys = ('labels', 'boxes', 'scores', 'batch_ids')
        pred_dict = {pred_key: [] for pred_key in pred_keys}
        pred_dicts = [deepcopy(pred_dict) for _ in range(3)]

        # Get predictions for every batch entry
        for i in range(batch_size):

            # Get target labels and boxes with default detection appended
            tgt_labels_i = torch.cat([tgt_labels[i], def_label], dim=0)
            tgt_boxes_i = Boxes.cat([tgt_boxes[i], def_box], same_image=True)

            # Get positives mask sorted according to object probabilities
            num_tgts = tgt_sorted_ids[i].shape[1]
            pos_ids = tgt_sorted_ids[i][:self.pred_num_pos, :]
            pos_mask = torch.zeros(num_anchors, num_tgts, dtype=torch.bool, device=device)
            pos_mask[pos_ids, torch.arange(num_tgts)] = True

            pred_sorted_ids = torch.argsort(obj_probs[i], dim=0, descending=True)[:self.pred_max_dets]
            pos_mask = pos_mask[pred_sorted_ids, :]

            # Get target ids
            def_values = torch.full((self.pred_max_dets, 1), 0.5, device=device)
            pos_matrix = torch.cat([pos_mask.to(def_values.dtype), def_values], dim=1)
            tgt_ids = torch.argmax(pos_matrix, dim=1)

            # Get scores for different prediction dictionaries
            scores0 = obj_probs[i, pred_sorted_ids]
            scores1 = torch.rand(self.pred_max_dets)

            duplicate_mask = torch.zeros(self.pred_max_dets, dtype=torch.bool, device=device)
            num_tgts = len(tgt_labels_i) - 1

            for tgt_id in range(num_tgts):
                duplicate_ids = torch.arange(self.pred_max_dets, device=device)[tgt_ids == tgt_id][1:]
                duplicate_mask[duplicate_ids] = True

            scores2 = torch.clone(scores0)
            scores2[duplicate_mask] = scores2[duplicate_mask] * scores2[-1]

            # Add predictions to prediction dictionaries
            pred_dicts[0]['labels'].append(tgt_labels_i[tgt_ids])
            pred_dicts[0]['boxes'].append(tgt_boxes_i[tgt_ids])
            pred_dicts[0]['scores'].append(scores0)
            pred_dicts[0]['batch_ids'].append(torch.full_like(tgt_ids, i, dtype=torch.int64))

            pred_dicts[1]['labels'].append(tgt_labels_i[tgt_ids])
            pred_dicts[1]['boxes'].append(tgt_boxes_i[tgt_ids])
            pred_dicts[1]['scores'].append(scores1)
            pred_dicts[1]['batch_ids'].append(torch.full_like(tgt_ids, i, dtype=torch.int64))

            pred_dicts[2]['labels'].append(tgt_labels_i[tgt_ids])
            pred_dicts[2]['boxes'].append(tgt_boxes_i[tgt_ids])
            pred_dicts[2]['scores'].append(scores2)
            pred_dicts[2]['batch_ids'].append(torch.full_like(tgt_ids, i, dtype=torch.int64))

        # Concatenate predictions of different batch entries
        for pred_dict in pred_dicts:
            pred_dict.update({k: torch.cat(v, dim=0) for k, v in pred_dict.items() if k != 'boxes'})
            pred_dict['boxes'] = Boxes.cat(pred_dict['boxes'])

        return pred_dicts

    def forward(self, feat_maps, tgt_dict=None, stand_alone=True, ext_dict=None, visualize=False, **kwargs):
        """
        Forward method of the DOD module.

        Args:
            feat_maps (List): List of size [num_maps] with feature maps of shape [batch_size, feat_size, fH, fW].

            tgt_dict (Dict): Optional target dictionary used during trainval containing at least following keys:
                - labels (LongTensor): tensor of shape [num_targets_total] containing the class indices;
                - boxes (Boxes): structure containing axis-aligned bounding boxes of size [num_targets_total];
                - sizes (LongTensor): tensor of shape [batch_size+1] with the cumulative target sizes of batch entries.

            stand_alone (bool): Boolean indicating whether the DOD module operates as stand-alone (default=True).

            ext_dict (Dict): Optional dictionary required when in 'ext_dynamic' target mode containing following keys:
                - logits (FloatTensor): tensor containing DOD logits of shape [batch_size, num_anchors, {1, 2}];
                - obj_probs (FloatTensor): tensor containing object probabilities of shape [batch_size, num_anchors];
                - pos_anchor_ids (List): list [batch_size] with indices of positive anchors of shape [num_pos_anchors];
                - tgt_found (List): list [batch_size] with masks of found targets of shape [num_targets];
                - pos_masks (List): list [batch_size] of positive target masks of shape [num_anchors, num_targets];
                - neg_masks (List): list [batch_size] of negative target masks of shape [num_anchors, num_targets];
                - tgt_sorted_ids (List): list [batch_size] of sorted anchor indices [sort_thr, num_targets].

            visualize (bool): Boolean indicating whether to compute dictionary with visualizations (default=False).
            kwargs (Dict): Dictionary of keyword arguments not used by this head module.

        Returns:
            * If tgt_dict is None:
                * If DOD module is not stand-alone:
                    sel_ids (List): List [batch_size] with indices of selected anchors of shape [num_sel_anchors].
                    anchors (Boxes): Structure containing axis-aligned anchor boxes of size [num_anchors].
                    analysis_dict (Dict): Dictionary of different analyses used for logging purposes only.

                * If DOD module is stand-alone:
                    pred_dicts (List): List of size [3] with empty dictionaries.
                    analysis_dict (Dict):  Dictionary of different analyses used for logging purposes only.

            * If tgt_dict is not None:
                * If ext_dict is None and in 'ext_dynamic' target mode:
                    logits (FloatTensor): Tensor containing DOD logits of shape [batch_size, num_anchors, {1, 2}].
                    obj_probs (FloatTensor): Tensor containing object probabilities of shape [batch_size, num_anchors].
                    sel_ids (List): List [batch_size] with indices of selected anchors of shape [num_sel_anchors].
                    anchors (Boxes): Structure containing axis-aligned anchor boxes of size [num_anchors].
                    pos_masks (List): List [batch_size] of positive target masks of shape [num_anchors, num_targets].
                    neg_masks (List): List [batch_size] of negative target masks of shape [num_anchors, num_targets].
                    tgt_sorted_ids (List): List [batch_size] of sorted indices of shape [sort_thr, num_targets].
                    analysis_dict (Dict): Dictionary of different analyses used for logging purposes only.

                * If in remaining cases:
                    * If DOD module is stand-alone and not in training mode:
                        pred_dicts (List): List of size [3] with DOD prediction dictionaries.

                    * If DOD module is not stand-alone and not in 'ext_dynamic' target mode:
                        sel_ids (List): List [batch_size] with indices of selected anchors of shape [num_sel_anchors].
                        anchors (Boxes): Structure containing axis-aligned anchor boxes of size [num_anchors].
                        tgt_sorted_ids (List): List [batch_size] of sorted indices of shape [sort_thr, num_targets].

                    loss_dict (Dict): Dictionary of different weighted loss terms used during training.
                    analysis_dict (Dict): Dictionary of different analyses used for logging purposes only.

        Raises:
            NotImplementedError: Error when visualizations are requested.
            RuntimeError: Error when logits have size different than 1 or 2.
            ValueError: Error when invalid anchor selection mode is provided.
            RuntimeError: Error when stand-alone DOD module has 'ext_dynamic' target mode.
            RuntimeError: Error when external dictionary is provided while not in 'ext_dynamic' target mode.
            RuntimeError: Error when no target dictionary is provided along with the external dictionary.
            ValueError: Error when one of the required keys of the external dictionary is missing.
            ValueError: Error when invalid target mode is provided.
            ValueError: Error when unknown loss type is present in 'loss_type' attribute.
        """

        # Check whether visualizations are requested
        if visualize:
            raise NotImplementedError

        # Get batch size and initialize empty analysis dictionary
        batch_size = len(feat_maps[0])
        analysis_dict = {}

        # Skip the following when external dictionary is provided
        if ext_dict is None:

            # Get anchors
            anchors = get_anchors(feat_maps, **self.anchor_dict)

            # Get logits
            logits = torch.cat([logit_map.flatten(2).permute(0, 2, 1) for logit_map in self.net(feat_maps)], dim=1)
            logits = logits.view(batch_size, len(anchors), -1)

            # Get object probabilities
            with torch.no_grad():
                if logits.shape[-1] == 1:
                    obj_probs = torch.sigmoid(logits)[:, :, 0]
                elif logits.shape[-1] == 2:
                    obj_probs = torch.softmax(logits, dim=2)[:, :, 0]
                else:
                    error_msg = f"Logits should have size 1 or 2, but got size {logits.shape[-1]}."
                    raise RuntimeError(error_msg)

            # Get and report number of positive and negative predictions
            pos_preds = obj_probs >= 0.5
            analysis_dict['pos_preds'] = torch.sum(pos_preds)[None] / batch_size
            analysis_dict['neg_preds'] = torch.sum(~pos_preds)[None] / batch_size

            # Get indices of selected anchors
            if self.sel_mode == 'abs_or_rel':
                ids = torch.arange(len(anchors), device=obj_probs.device)
                abs_sel_ids = [ids[obj_probs_i >= self.sel_abs_thr] for obj_probs_i in obj_probs]
                rel_sel_ids = torch.topk(obj_probs, self.sel_rel_thr, dim=1, sorted=False).indices
                sel_ids = [torch.cat([abs_sel_ids[i], rel_sel_ids[i]]).unique() for i in range(batch_size)]

            elif self.sel_mode == 'rel':
                sel_ids = torch.topk(obj_probs, self.sel_rel_thr, dim=1, sorted=False).indices
                sel_ids = [*sel_ids]

            else:
                error_msg = f"Invalid anchor selection mode '{self.sel_mode}'."
                raise ValueError(error_msg)

            # Return when no target dictionary is provided
            if tgt_dict is None:
                analysis_dict = {f'dod_{k}': v for k, v in analysis_dict.items()}

                if not stand_alone:
                    return sel_ids, anchors, analysis_dict

                pred_dicts = [{}, {}, {}]
                return pred_dicts, analysis_dict

            # Get static target masks
            tgt_masks_output = self.get_static_tgt_masks(feat_maps, anchors, tgt_dict['boxes'], return_ids=True)
            pos_masks, neg_masks, tgt_sorted_ids = tgt_masks_output

            # Return if in external dynamic target mode
            if self.tgt_mode == 'ext_dynamic':
                if not stand_alone:
                    analysis_dict = {f'dod_{k}': v for k, v in analysis_dict.items()}
                    return logits, obj_probs, sel_ids, anchors, pos_masks, neg_masks, tgt_sorted_ids, analysis_dict
                else:
                    error_msg = "Stand-alone DOD modules do not support the 'ext_dynamic' target mode."
                    raise RuntimeError(error_msg)

        # Perform some checks when an external dictionary is provided
        if ext_dict is not None:

            if self.tgt_mode != 'ext_dynamic':
                error_msg = "An external dictionary should only be provided when in 'ext_dynamic' target mode."
                raise RuntimeError(error_msg)

            if tgt_dict is None:
                error_msg = "A target dictionary must be provided along with the external dictionary."
                raise RuntimeError(error_msg)

            required_keys = ('logits', 'obj_probs', 'pos_anchor_ids', 'tgt_found', 'pos_masks', 'neg_masks')
            required_keys = (*required_keys, 'tgt_sorted_ids')

            for required_key in required_keys:
                if required_key not in ext_dict:
                    error_msg = f"The key '{required_key}' is missing from the external dictionary."
                    raise ValueError(error_msg)

            logits = ext_dict['logits']
            obj_probs = ext_dict['obj_probs']
            pos_anchor_ids = ext_dict['pos_anchor_ids']
            tgt_found = ext_dict['tgt_found']
            pos_masks = ext_dict['pos_masks']
            neg_masks = ext_dict['neg_masks']
            tgt_sorted_ids = ext_dict['tgt_sorted_ids']

        # Get final target masks and mask with found targets
        if self.tgt_mode == 'static':
            pos_tgts = torch.stack([torch.sum(pos_mask, dim=1) > 0 for pos_mask in pos_masks], dim=0)
            tgt_found = [pos_masks[i][sel_ids[i]].sum(dim=0) > 0 for i in range(batch_size)]

        elif self.tgt_mode == 'int_dynamic':
            num_anchors = obj_probs.shape[1]
            pos_tgts = torch.zeros_like(obj_probs, dtype=torch.bool)
            tgt_found = []

            for i in range(batch_size):
                sel_pos_mask = pos_masks[i][sel_ids[i]]
                tgt_sums = torch.cummax(sel_pos_mask, dim=0)[0].sum(dim=0)

                tgt_found_i = tgt_sums > 0
                tgt_found.append(tgt_found_i)

                pos_tgts[i] = pos_masks[i][:, ~tgt_found_i].sum(dim=1) > 0
                pos_anchor_ids_i = (num_anchors - tgt_sums)[tgt_found_i]
                pos_tgts[i, pos_anchor_ids_i] = True

        elif self.tgt_mode == 'ext_dynamic':
            pos_tgts = torch.zeros_like(obj_probs, dtype=torch.bool)

            for i in range(batch_size):
                pos_tgts[i] = pos_masks[i][:, ~tgt_found[i]].sum(dim=1) > 0
                pos_tgts[i, pos_anchor_ids[i]] = True

        else:
            error_msg = f"Invalid target mode '{self.tgt_mode}'."
            raise ValueError(error_msg)

        neg_tgts = torch.stack([torch.sum(neg_mask, dim=1) > 0 for neg_mask in neg_masks], dim=0)
        analysis_dict['pos_tgts'] = torch.sum(pos_tgts)[None] / batch_size
        analysis_dict['neg_tgts'] = torch.sum(neg_tgts)[None] / batch_size

        tgt_found = torch.cat(tgt_found, dim=0)
        num_tgts = len(tgt_found)

        if num_tgts > 0:
            analysis_dict['tgt_found'] = 100 * torch.sum(tgt_found)[None] / num_tgts
        else:
            analysis_dict['tgt_found'] = 100 * torch.ones(1).to(tgt_found)

        # Get weighted positive and negative losses
        targets = torch.full_like(obj_probs, fill_value=-1, dtype=torch.int64)
        targets[neg_tgts] = 0
        targets[pos_tgts] = 1

        loss_feats = targets != -1
        logits = logits[loss_feats]
        targets = targets[loss_feats]

        if self.loss_type == 'sigmoid_focal':
            if logits.shape[1] == 1:
                focal_targets = targets[:, None].to(logits.dtype)
            elif logits.shape[1] == 2:
                focal_targets = torch.stack([targets == 1, targets == 0], dim=1).to(logits.dtype)
            else:
                error_msg = f"Logits should have size 1 or 2, but got size {logits.shape[-1]}."
                raise RuntimeError(error_msg)

            focal_kwargs = {'alpha': self.focal_alpha, 'gamma': self.focal_gamma, 'reduction': 'none'}
            losses = sigmoid_focal_loss(logits, focal_targets, **focal_kwargs).sum(dim=1)

        else:
            error_msg = f"Unknown loss type '{self.loss_type}' during DOD loss computation."
            raise ValueError(error_msg)

        loss_dict = {}
        loss_dict['pos_loss'] = self.pos_weight * losses[targets == 1].sum(dim=0, keepdim=True)
        loss_dict['neg_loss'] = self.neg_weight * losses[targets == 0].sum(dim=0, keepdim=True)

        # Get average precision
        analysis_dict['ap'] = self.get_ap(obj_probs, tgt_sorted_ids)

        # Add identifier to keys of loss and analysis dictionaries
        loss_dict = {f'dod_{k}': v for k, v in loss_dict.items()}
        analysis_dict = {f'dod_{k}': v for k, v in analysis_dict.items()}

        # Get prediction dictionaries if desired and return
        if stand_alone and not self.training:
            pred_dicts = self.make_predictions(obj_probs, tgt_sorted_ids, tgt_dict)
            return pred_dicts, loss_dict, analysis_dict

        if not stand_alone and self.tgt_mode != 'ext_dynamic':
            return sel_ids, anchors, tgt_sorted_ids, loss_dict, analysis_dict

        return loss_dict, analysis_dict
