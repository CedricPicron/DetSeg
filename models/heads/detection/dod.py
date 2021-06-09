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
from structures.boxes import Boxes, box_iou, get_feat_boxes


class DOD(nn.Module):
    """
    Class implementing the Dense Object Discovery (DOD) head.

    Attributes:
        net (nn.Sequential): DOD network computing the logits.

        sel_mode (str): String containing the feature selection mode.
        sel_abs_thr (float): Absolute threshold determining the selected features.
        sel_rel_thr (int): Relative threshold determining the selected features.

        tgt_metric (str): String containing the feature-target matching metric.
        tgt_decision (str): String containing the target decision maker type.
        abs_pos_tgt (float): Absolute threshold used during positive target decision making.
        abs_neg_tgt (float): Absolute threshold used during negative target decision making.
        rel_pos_tgt (int): Relative threshold used during positive target decision making.
        rel_neg_tgt (int): Relative threshold used during negative target decision making.
        tgt_mode (str): String containing the target mode.

        loss_type (str): String containing the type of loss.
        focal_alpha (float): Alpha value of the sigmoid focal loss.
        focal_gamma (float): Gamma value of the sigmoid focal loss.
        pos_weight (float): Factor weighting the loss terms with positive targets.
        neg_weight (float): Factor weighting the loss terms with negative targets.

        pred_num_pos (int): Integer containing the number of positive features per target during prediction.
        pred_max_dets (int): Integer containing the maximum number of detections during prediction.

        metadata (detectron2.data.Metadata): Metadata instance containing additional dataset information.
    """

    def __init__(self, in_feat_size, net_dict, sel_dict, tgt_dict, loss_dict, pred_dict, metadata):
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

            sel_dict (Dict): Feature selection dictionary containing following keys:
                - mode (str): string containing the feature selection mode;
                - abs_thr (float): absolute threshold determining the selected features;
                - rel_thr (int): relative threshold determining the selected features;
                - eval (str): string containing the selected features evalution mode.

            tgt_dict (Dict): Dictionary with items used during target computation containing following keys:
                - metric (str): string containing the feature-target matching metric;
                - decision (str): string containing the target decision maker type;
                - abs_pos_tgt (float): absolute threshold used during positive target decision making;
                - abs_neg_tgt (float): absolute threshold used during negative target decision making;
                - rel_pos_tgt (int): relative threshold used during positive target decision making;
                - rel_neg_tgt (int): relative threshold used during negative target decision making;
                - mode (str): string containing the target mode.

            loss_dict (Dict): Loss dictionary containing following keys:
                - type (str): string containing the type of loss;
                - focal_alpha (float): alpha value of the sigmoid focal loss;
                - focal_gamma (float): gamma value of the sigmoid focal loss;
                - pos_weight (float): factor weighting the loss terms with positive targets;
                - neg_weight (float): factor weighting the loss terms with negative targets.

            pred_dict (Dict): Dictionary with items used during prediction containing following keys:
                - num_pos (int): integer containing the number of positive features per target during prediction;
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

        out_feat_size = 2 if net_dict['rel_preds'] else 1
        out_layer = ProjConv(feat_size, out_feat_size, norm=norm, skip=False)

        obj_prior_prob = net_dict['prior_prob']
        obj_bias_value = -(math.log((1 - obj_prior_prob) / obj_prior_prob))
        torch.nn.init.constant_(out_layer.conv.bias[0], obj_bias_value)

        if out_feat_size == 2:
            torch.nn.init.constant_(out_layer.conv.bias[1], -obj_bias_value)

        net_dict = OrderedDict([('in', in_layer), ('hidden', hidden_layers), ('out', out_layer)])
        self.net = nn.Sequential(net_dict)

        # Set feature selection attributes
        self.sel_mode = sel_dict['mode']
        self.sel_abs_thr = sel_dict['abs_thr']
        self.sel_rel_thr = sel_dict['rel_thr']

        # Set target-related attributes
        self.tgt_metric = tgt_dict['metric']
        self.tgt_decision = tgt_dict['decision']
        self.abs_pos_tgt = tgt_dict['abs_pos_tgt']
        self.abs_neg_tgt = tgt_dict['abs_neg_tgt']
        self.rel_pos_tgt = tgt_dict['rel_pos_tgt']
        self.rel_neg_tgt = tgt_dict['rel_neg_tgt']
        self.tgt_mode = tgt_dict['mode']

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
    def get_static_tgt_masks(self, feat_maps, tgt_boxes, return_ids=True):
        """
        Get positive and negative static target masks.

        Args:
            feat_maps (List): List of size [num_maps] with feature maps of shape [batch_size, feat_size, fH, fW].
            tgt_boxes (List): List of size [batch_size] with normalized Boxes structure of size [num_targets].
            return_ids (bool): Boolean indicating whether sorted feature indices should be returned (default=True).

        Returns:
            pos_masks (List): List [batch_size] of positive static target masks of shape [num_feats, num_targets].
            neg_masks (List): List [batch_size] of negative static target masks of shape [num_feats, num_targets].

            If 'return_ids' is True:
                tgt_sorted_ids (List): List [batch_size] of sorted feature indices of shape [num_feats, num_targets].

        Raises:
            ValueError: Error when unknown feature-target metric is present in 'tgt_metric' attribute.
            ValueError: Error when unknown target decision maker type is present in 'tgt_decision' attribute.
        """

        # Get batch size and feature boxes
        batch_size = len(tgt_boxes)
        feat_boxes = get_feat_boxes(feat_maps)

        # Concatenate target boxes and get number of targets
        tgt_boxes = Boxes.cat(tgt_boxes)
        num_tgts = len(tgt_boxes)

        # Return if there are no target boxes
        if num_tgts == 0:
            num_feats = len(feat_boxes)
            tensor_kwargs = {'dtype': torch.bool, 'device': feat_maps[0].device}
            pos_masks = [torch.zeros(num_feats, 0, **tensor_kwargs) for _ in range(batch_size)]

            return pos_masks

        # Get feature-target similarity matrix and target sorted feature indices
        if self.tgt_metric == 'iou':
            sim_matrix = box_iou(feat_boxes, tgt_boxes)
            tgt_sorted_ids = torch.argsort(sim_matrix, dim=0, descending=True)

        else:
            error_msg = f"Unknown feature-target metric '{self.tgt_metric}'."
            raise ValueError(error_msg)

        # Get positive and negative static target masks
        if self.tgt_decision == 'abs':
            pos_mask = sim_matrix >= self.abs_pos_tgt
            neg_mask = sim_matrix < self.abs_neg_tgt

        elif self.tgt_decision == 'rel':
            pos_ids = tgt_sorted_ids[:self.rel_pos_tgt, :]
            pos_mask = torch.zeros_like(sim_matrix, dtype=torch.bool)
            pos_mask[pos_ids, torch.arange(num_tgts)] = True

            non_neg_ids = tgt_sorted_ids[:self.rel_neg_tgt, :]
            neg_mask = torch.ones_like(sim_matrix, dtype=torch.bool)
            neg_mask[non_neg_ids, torch.arange(num_tgts)] = False

        else:
            error_msg = f"Unknown target decision maker type '{self.tgt_decision}'."
            raise ValueError(error_msg)

        tgts_per_img = tgt_boxes.boxes_per_img.tolist()
        pos_masks = torch.split(pos_mask, tgts_per_img, dim=1)
        neg_masks = torch.split(neg_mask, tgts_per_img, dim=1)

        # Return masks with sorted feature indices if requested
        if return_ids:
            tgt_sorted_ids = torch.split(tgt_sorted_ids, tgts_per_img, dim=1)
            return pos_masks, neg_masks, tgt_sorted_ids

        return pos_masks, neg_masks

    @torch.no_grad()
    def get_ap(self, obj_probs, tgt_sorted_ids):
        """
        Get average precision (AP) based on DOD object probabilities and sorted feature indices.

        Args:
            obj_probs (FloatTensor): Tensor containing object probabilities of shape [batch_size, num_feats].
            tgt_sorted_ids (List): List [batch_size] of sorted feature indices of shape [num_feats, num_targets].

        Returns:
            ap (FloatTensor): Tensor containing the average precision (AP) of shape [1].
        """

        # Get batch size and initialize average precision tensor
        batch_size = len(obj_probs)
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
            pos_mask = torch.zeros_like(tgt_sorted_ids[i], dtype=torch.bool)
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
        Makes predictions based on DOD object probabilities, sorted feature indices and corresponding targets.

        Args:
            obj_probs (FloatTensor): Tensor containing object probabilities of shape [batch_size, num_feats].
            tgt_sorted_ids (List): List [batch_size] of sorted feature indices of shape [num_feats, num_targets].

            tgt_dict (Dict): Target dictionary containing at least following keys:
                - labels (List): list of size [batch_size] with class indices of shape [num_targets];
                - boxes (List): list of size [batch_size] with normalized Boxes structure of size [num_targets].

        Returns:
            pred_dicts (List): List of size [3] with the DOD prediction dictionaries containing following keys:
                - labels (LongTensor): predicted class indices of shape [num_preds_total];
                - boxes (Boxes): structure containing axis-aligned bounding boxes of size [num_preds_total];
                - scores (FloatTensor): normalized prediction scores of shape [num_preds_total];
                - batch_ids (LongTensor): batch indices of predictions of shape [num_preds_total].
        """

        # Get batch size
        batch_size = len(obj_probs)

        # Get device, default label and default bounding box
        device = obj_probs.device
        def_label = torch.tensor([0], dtype=torch.int64, device=device)
        def_box_kwargs = {'format': 'cxcywh', 'normalized': 'img_with_padding'}
        def_box = Boxes(torch.tensor([[0.01, 0.01, 0.01, 0.01]], device=device), **def_box_kwargs)

        # Initialize prediction dictionaries
        pred_keys = ('labels', 'boxes', 'scores', 'batch_ids')
        pred_dict = {pred_key: [] for pred_key in pred_keys}
        pred_dicts = [deepcopy(pred_dict) for _ in range(3)]

        # Get predictions for every batch entry
        for i in range(batch_size):

            # Get target labels and boxes with default detection appended
            tgt_labels = torch.cat([tgt_dict['labels'][i], def_label], dim=0)
            tgt_boxes = Boxes.cat([tgt_dict['boxes'][i].to_format('cxcywh'), def_box], same_image=True)

            # Get positives mask sorted according to object probabilities
            num_tgts = tgt_sorted_ids[i].shape[1]
            pos_ids = tgt_sorted_ids[i][:self.pred_num_pos, :]
            pos_mask = torch.zeros_like(tgt_sorted_ids[i], dtype=torch.bool)
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
            num_tgts = len(tgt_labels) - 1

            for tgt_id in range(num_tgts):
                duplicate_ids = torch.arange(self.pred_max_dets, device=device)[tgt_ids == tgt_id][1:]
                duplicate_mask[duplicate_ids] = True

            scores2 = torch.clone(scores0)
            scores2[duplicate_mask] = scores2[duplicate_mask] * scores2[-1]

            # Add predictions to prediction dictionaries
            pred_dicts[0]['labels'].append(tgt_labels[tgt_ids])
            pred_dicts[0]['boxes'].append(tgt_boxes[tgt_ids])
            pred_dicts[0]['scores'].append(scores0)
            pred_dicts[0]['batch_ids'].append(torch.full_like(tgt_ids, i, dtype=torch.int64))

            pred_dicts[1]['labels'].append(tgt_labels[tgt_ids])
            pred_dicts[1]['boxes'].append(tgt_boxes[tgt_ids])
            pred_dicts[1]['scores'].append(scores1)
            pred_dicts[1]['batch_ids'].append(torch.full_like(tgt_ids, i, dtype=torch.int64))

            pred_dicts[2]['labels'].append(tgt_labels[tgt_ids])
            pred_dicts[2]['boxes'].append(tgt_boxes[tgt_ids])
            pred_dicts[2]['scores'].append(scores2)
            pred_dicts[2]['batch_ids'].append(torch.full_like(tgt_ids, i, dtype=torch.int64))

        # Concatenate predictions of different batch entries
        for pred_dict in pred_dicts:
            pred_dict.update({k: torch.cat(v, dim=0) for k, v in pred_dict.items() if k != 'boxes'})
            pred_dict['boxes'] = Boxes.cat(pred_dict['boxes'])

        return pred_dicts

    def forward(self, feat_maps, tgt_dict=None, images=None, stand_alone=True, ext_dict=None, visualize=False,
                **kwargs):
        """
        Forward method of the DOD module.

        Args:
            feat_maps (List): List of size [num_maps] with feature maps of shape [batch_size, feat_size, fH, fW].

            tgt_dict (Dict): Optional target dictionary used during trainval containing at least following keys:
                - labels (LongTensor): tensor of shape [num_targets_total] containing the class indices;
                - boxes (Boxes): structure containing axis-aligned bounding boxes of size [num_targets_total];
                - sizes (LongTensor): tensor of shape [batch_size+1] with the cumulative target sizes of batch entries.

            images (Images): Images structure containing the batched images (default=None).
            stand_alone (bool): Boolean indicating whether the DOD module operates as stand-alone (default=True).

            ext_dict (Dict): Optional dictionary required when in 'ext_dynamic' target mode containing following keys:
                - logits (FloatTensor): tensor containing DOD logits of shape [batch_size, num_feats, {1, 2}];
                - obj_probs (FloatTensor): tensor containing object probabilities of shape [batch_size, num_feats];
                - pos_feat_ids (List): list [batch_size] with indices of positive features of shape [num_pos_feats];
                - tgt_found (List): list [batch_size] with masks of found targets of shape [num_targets];
                - pos_masks (List): list [batch_size] of positive target masks of shape [num_feats, num_targets];
                - neg_masks (List): list [batch_size] of negative target masks of shape [num_feats, num_targets];
                - tgt_sorted_ids (List): list [batch_size] of sorted feature indices of shape [num_feats, num_targets].

            visualize (bool): Boolean indicating whether to compute dictionary with visualizations (default=False).
            kwargs (Dict): Dictionary of keyword arguments not used by this head module.

        Returns:
            * If tgt_dict is None:
                * If DOD module is not stand-alone:
                    sel_ids (List): List [batch_size] with indices of selected features of shape [num_sel_feats].
                    analysis_dict (Dict): Dictionary of different analyses used for logging purposes only.

                * If DOD module is stand-alone:
                    pred_dicts (List): List of size [3] with empty dictionaries.
                    analysis_dict (Dict):  Dictionary of different analyses used for logging purposes only.

            * If tgt_dict is not None:
                * If ext_dict is None and in 'ext_dynamic' target mode:
                    logits (FloatTensor): Tensor containing DOD logits of shape [batch_size, num_feats, {1, 2}].
                    obj_probs (FloatTensor): Tensor containing object probabilities of shape [batch_size, num_feats].
                    sel_ids (List): List [batch_size] with indices of selected features of shape [num_sel_feats].
                    pos_masks (List): List [batch_size] of positive target masks of shape [num_feats, num_targets].
                    neg_masks (List): List [batch_size] of negative target masks of shape [num_feats, num_targets].
                    tgt_sorted_ids (List): List [batch_size] of sorted indices of shape [num_feats, num_targets].

                * If in remaining cases:
                    * If DOD module is stand-alone and not in training mode:
                        pred_dicts (List): List of size [3] with DOD prediction dictionaries.

                    * If DOD module is not stand-alone and not in 'ext_dynamic' target mode:
                        sel_ids (List): List [batch_size] with indices of selected features of shape [num_sel_feats].
                        pos_masks (List): List [batch_size] of positive target masks of shape [num_feats, num_targets].

                    loss_dict (Dict): Dictionary of different weighted loss terms used during training.
                    analysis_dict (Dict): Dictionary of different analyses used for logging purposes only.

        Raises:
            NotImplementedError: Error when visualizations are requested.
            RuntimeError: Error when logits have size different than 1 or 2.
            ValueError: Error when invalid feature selection mode is provided.
            ValueError: Error when target dictionary is provided without corresponding Images structure.
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

            # Get logits
            logits = torch.cat([logit_map.flatten(2).permute(0, 2, 1) for logit_map in self.net(feat_maps)], dim=1)

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

            # Get indices of selected features
            if self.sel_mode == 'abs':
                vals, ids = torch.sort(obj_probs, dim=1, descending=True)
                sel_ids = [ids_i[vals_i >= self.sel_abs_thr] for vals_i, ids_i in zip(vals, ids)]

            elif self.sel_mode == 'rel':
                sel_ids = torch.argsort(obj_probs, dim=1, descending=True)[:, :self.sel_rel_thr]
                sel_ids = [*sel_ids]

            else:
                error_msg = f"Invalid feature selection mode '{self.sel_mode}'."
                raise ValueError(error_msg)

            # Return when no target dictionary is provided
            if tgt_dict is None:
                analysis_dict = {f'dod_{k}': v for k, v in analysis_dict.items()}

                if not stand_alone:
                    return sel_ids, analysis_dict

                pred_dicts = [{}, {}, {}]
                return pred_dicts, analysis_dict

            # Get target boxes in desired format
            if images is not None:
                sizes = tgt_dict['sizes']
                tgt_boxes = tgt_dict['boxes'].normalize(images, with_padding=True)
                tgt_boxes = [tgt_boxes[i0:i1] for i0, i1 in zip(sizes[:-1], sizes[1:])]

            else:
                error_msg = "A corresponding Images structure must be provided along with the target dictionary."
                raise ValueError(error_msg)

            # Get static target masks
            pos_masks, neg_masks, tgt_sorted_ids = self.get_static_tgt_masks(feat_maps, tgt_boxes, return_ids=True)

            # Return if in external dynamic target mode
            if self.tgt_mode == 'ext_dynamic':
                if not stand_alone:
                    return logits, obj_probs, sel_ids, pos_masks, neg_masks, tgt_sorted_ids
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

            required_keys = ('logits', 'obj_probs', 'pos_feat_ids', 'tgt_found', 'pos_masks', 'neg_masks')
            required_keys = (*required_keys, 'tgt_sorted_ids')

            for required_key in required_keys:
                if required_key not in ext_dict:
                    error_msg = f"The key '{required_key}' is missing from the external dictionary."
                    raise ValueError(error_msg)

            logits = ext_dict['logits']
            obj_probs = ext_dict['obj_probs']
            pos_feat_ids = ext_dict['pos_feat_ids']
            tgt_found = ext_dict['tgt_found']
            pos_masks = ext_dict['pos_masks']
            neg_masks = ext_dict['neg_masks']
            tgt_sorted_ids = ext_dict['tgt_sorted_ids']

        # Get final target masks and mask with found targets
        if self.tgt_mode == 'static':
            pos_tgts = torch.stack([torch.sum(pos_mask, dim=1) > 0 for pos_mask in pos_masks], dim=0)
            tgt_found = [pos_masks[i][sel_ids[i]].sum(dim=0) > 0 for i in range(batch_size)]

        elif self.tgt_mode == 'int_dynamic':
            num_feats = obj_probs.shape[1]
            pos_tgts = torch.zeros_like(obj_probs, dtype=torch.bool)
            tgt_found = []

            for i in range(batch_size):
                sel_pos_mask = pos_masks[i][sel_ids[i]]
                tgt_sums = torch.cummax(sel_pos_mask, dim=0)[0].sum(dim=0)

                tgt_found_i = tgt_sums > 0
                tgt_found.append(tgt_found_i)

                pos_tgts[i] = pos_masks[i][:, ~tgt_found_i].sum(dim=1) > 0
                pos_feat_ids_i = (num_feats - tgt_sums)[tgt_found_i]
                pos_tgts[i, pos_feat_ids_i] = True

        elif self.tgt_mode == 'ext_dynamic':
            pos_tgts = torch.zeros_like(obj_probs, dtype=torch.bool)

            for i in range(batch_size):
                pos_tgts[i] = pos_masks[i][:, ~tgt_found[i]].sum(dim=1) > 0
                pos_tgts[i, pos_feat_ids[i]] = True

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
        targets = torch.full_like(pos_preds, fill_value=-1, dtype=torch.int64)
        targets[pos_tgts] = 1
        targets[neg_tgts] = 0

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
            local_tgt_dict = {}
            local_tgt_dict['labels'] = [tgt_dict['labels'][i0:i1] for i0, i1 in zip(sizes[:-1], sizes[1:])]
            local_tgt_dict['boxes'] = tgt_boxes

            pred_dicts = self.make_predictions(obj_probs, tgt_sorted_ids, local_tgt_dict)
            return pred_dicts, loss_dict, analysis_dict

        if not stand_alone and self.tgt_mode != 'ext_dynamic':
            return sel_ids, pos_masks, loss_dict, analysis_dict

        return loss_dict, analysis_dict
