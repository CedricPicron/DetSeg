"""
Collection of selector modules.
"""
from collections import OrderedDict
import math

import torch
from torch import nn

from models.build import build_model, MODELS
from models.modules.container import Sequential
from models.modules.convolution import ProjConv
from structures.boxes import get_anchors


@MODELS.register_module()
class AnchorSelector(nn.Module):
    """
    Class implementing the AnchorSelector module.

    The AnchorSelector module selects features from a given set of feature maps, with each feature map having the same
    feature size. The AnchorSelector module is trained to select the features for which their corresponding anchors
    overlap the most with the prodived ground-truth target boxes.

    Attributes:
        anchor_attrs (Dict): Dictionary containing following anchor-related attributes:
            - map_ids (List): list [num_maps] containing the map ids (i.e. downsampling exponents) of each map;
            - num_sizes (int): integer containing the number of different anchor sizes per aspect ratio;
            - scale_factor (float): factor scaling the anchors w.r.t. non-overlapping tiling anchors;
            - aspect_ratios (List): list [num_aspect_ratios] containing the different anchor aspect ratios.

        logits (Sequential): Module computing the selection logits for each feature-anchor combination.

        sel_attrs (Dict): Dictionary containing following selection-related attributes:
            - type (str): string containing the type of selection procedure;
            - abs_thr (float): absolute threshold used during selection;
            - rel_thr (int): relative threshold used during selection.

        post (Sequential): Post-processing module operating on selected features.
        match (nn.Module): Module performing matching between anchors and target boxes.
        loss (nn.Module): Module computing the weighted selection loss.
    """

    def __init__(self, anchor_cfg, pre_logits_cfg, sel_cfg, post_cfg, match_cfg, loss_cfg, init_prob=0.01):
        """
        Initializes the AnchorSelector module.

        Args:
            anchor_cfg (Dict): Configuration dictionary specifying the anchors.
            pre_logits_cfg (Dict): Configuration dictionary specifying the pre-logits module.
            sel_cfg (Dict): Configuration dictionary specifying the selection procedure.
            post_cfg (Dict): Configuration dictionary specifying the post-processing module.
            match_cfg (Dict): Configuration dictionary specifying the matcher module.
            loss_cfg (Dict): Configuration dictionary specifying the loss module.
            init_prob (float): Probability determining initial bias value of last logits sub-module (default=0.01).
        """

        # Set anchor attributes
        self.anchor_attrs = anchor_cfg

        # Build logits module
        pre_logits = build_model(pre_logits_cfg)

        in_feat_size = pre_logits_cfg['out_channels']
        num_cell_anchors = anchor_cfg['num_sizes'] * len(anchor_cfg['aspect_ratios'])
        proj_logits = ProjConv(in_feat_size, num_cell_anchors)

        init_logits_bias = -(math.log((1 - init_prob) / init_prob))
        torch.nn.init.constant_(proj_logits.conv.bias, init_logits_bias)

        seq_dict = OrderedDict([('pre', pre_logits), ('proj', proj_logits)])
        self.logits = Sequential(seq_dict)

        # Set selection attributes
        self.sel_attrs = sel_cfg

        # Build post-processing module
        self.post = build_model(post_cfg)
        self.post = Sequential(self.post) if not isinstance(self.post, Sequential) else self.post

        # Build matcher module
        self.match = build_model(match_cfg)

        # Build loss module
        self.loss = build_model(loss_cfg)

    def perform_selection(self, feat_maps):
        """
        Method performing the selection procedure.

        Args:
            feat_maps (List): List of size [num_maps] with feature maps of shape [batch_size, feat_size, fH, fW].

        Returns:
            sel_logits (FloatTensor): Selection logits of shape [batch_size, num_anchors].
            sel_ids (LongTensor): Indices of selected feature-anchor combinations of shape [num_selecs].
            sel_feats (FloatTensor): Features of selected feature-anchor combinations of shape [num_selecs].

        Raises:
            ValueError: Error when an invalid type of selection procedure is provided.
        """

        # Get selection logits
        logit_maps = self.logits(feat_maps)
        sel_logits = torch.cat([logit_map.permute(0, 2, 3, 1).flatten(1) for logit_map in logit_maps], dim=1)

        # Get selected indices
        sel_probs = torch.sigmoid(sel_logits.detach())

        batch_size, num_anchors = sel_probs.size()
        device = sel_probs.device
        sel_type = self.sel_attrs['type']

        if 'abs' in sel_type:
            abs_sel_ids = torch.arange(batch_size * num_anchors, device=device)
            abs_sel_ids = abs_sel_ids[sel_probs.flatten() >= self.sel_attrs['abs_thr']]

        elif 'rel' in sel_type:
            rel_sel_ids = torch.topk(sel_probs, self.sel_attrs['rel_thr'], dim=1, sorted=False).indices
            rel_sel_ids = num_anchors * torch.arange(batch_size, device)[:, None] * rel_sel_ids
            rel_sel_ids = rel_sel_ids.flatten()

        if sel_type == 'abs':
            sel_ids = abs_sel_ids

        elif sel_type == 'abs_and_rel':
            sel_ids, counts = torch.cat([abs_sel_ids, rel_sel_ids], dim=0).unique(return_counts=True)
            sel_ids = sel_ids[counts == 2]

        elif sel_type == 'abs_or_rel':
            sel_ids = torch.cat([abs_sel_ids, rel_sel_ids], dim=0).unique()

        elif sel_type == 'rel':
            sel_ids = rel_sel_ids

        else:
            error_msg = f"Invalid type of selection procedure (got '{sel_type}')."
            raise ValueError(error_msg)

        # Get selected features
        feats = torch.cat([feat_map.flatten(2).permute(0, 2, 1) for feat_map in feat_maps], dim=1)
        feats = feats.flatten(0, 1)

        num_cell_anchors = logit_maps[0].size(dim=1)
        feat_ids = sel_ids // num_cell_anchors
        sel_feats = feats[feat_ids]

        return sel_logits, sel_ids, sel_feats

    def forward(self, feat_maps, tgt_dict=None, **kwargs):
        """
        Forward method of the AnchorSelector module.

        Args:
            feat_maps (List): List of size [num_maps] with feature maps of shape [batch_size, feat_size, fH, fW].

            tgt_dict (Dict): Optional target dictionary used during trainval containing at least following key:
                - boxes (Boxes): structure containing axis-aligned bounding boxes of size [num_targets_total].

            kwargs (Dict): Dictionary of unused keyword arguments.

        Returns:
            out_dict (Dict): Output dictionary containing following keys:
                - anchors (Boxes): structure with axis-aligned anchor boxes of size [num_anchors];
                - sel_ids (LongTensor): indices of selected feature-anchor combinations of shape [num_selecs];
                - batch_ids (LongTensor): batch indices of selected features of shape [num_selecs];
                - sel_feats (FloatTensor): selected features after post-processing of shape [num_selecs, feat_size];
                - sel_boxes (Boxes): structure with boxes corresponding to selected features of size [num_selecs];
                - loss_dict (Dict): dictionary with the selection loss (trainval only);
                - analysis_dict (Dict): dictionary with different selection analyses used for logging purposes only.
        """

        # Initialize empty analysis and output dictionary
        analysis_dict = {}
        out_dict = {}

        # Get anchors
        anchors = get_anchors(feat_maps, **self.anchor_attrs)
        out_dict['anchors'] = anchors

        # Perform selection
        sel_logits, sel_ids, sel_feats = self.perform_selection(feat_maps)
        out_dict['sel_ids'] = sel_ids

        # Get batch and cell anchor indices
        batch_ids = sel_ids // len(anchors)
        out_dict['batch_ids'] = batch_ids

        num_cell_anchors = self.anchor_attrs['num_sizes'] * len(self.anchor_attrs['aspect_ratios'])
        cell_anchor_ids = sel_ids % num_cell_anchors

        # Perform post-processing on selected features
        sel_feats = self.post(sel_feats, module_ids=cell_anchor_ids)
        out_dict['sel_feats'] = sel_feats

        # Get boxes corresponding to selected features
        batch_size = len(sel_ids) / len(anchors)
        sel_boxes = anchors[sel_ids // batch_size]
        out_dict['sel_boxes'] = sel_boxes

        # Get selection loss (trainval only)
        if tgt_dict is not None:

            # Perform matching
            match_labels, matched_qry_ids, matched_tgt_ids = self.match(anchors, tgt_dict['boxes'])
            match_labels = None

            # Get selection loss
            loss_mask = match_labels != 1
            pred_logits = sel_logits.flatten()[loss_mask]
            tgt_labels = match_labels[loss_mask]

            sel_loss = self.loss(pred_logits, tgt_labels)
            loss_dict = {'sel_loss': sel_loss}
            out_dict['loss_dict'] = loss_dict

            # Get percentage of targets found
            tgt_found = None
            analysis_dict['tgt_found'] = tgt_found

        # Add analysis dictionary to output dictionary
        out_dict['analysis_dict'] = analysis_dict

        return out_dict
