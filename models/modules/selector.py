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


@MODELS.register_module()
class AnchorSelector(nn.Module):
    """
    Class implementing the AnchorSelector module.

    The AnchorSelector module selects features from a given set of feature maps, with each feature map having the same
    feature size. The AnchorSelector module is trained to select the features for which their corresponding anchors
    overlap the most with the prodived ground-truth target boxes.

    Attributes:
        feat_map_ids (Tuple): Tuple containing the indices of feature maps used by this module (or None).
        anchor (nn.Module): Module computing the anchor boxes.
        logits (Sequential): Module computing the selection logits for each feature-anchor combination.

        sel_attrs (Dict): Dictionary containing following selection-related attributes:
            - mode (str): string containing the selection mode;
            - abs_thr (float): absolute threshold used during selection;
            - rel_thr (int): relative threshold used during selection.

        post (Sequential): Post-processing module operating on selected features.
        box_encoder (nn.Module): Module computing the box encodings of selected boxes.
        matcher (nn.Module): Module performing matching between anchors and target boxes.
        loss (nn.Module): Module computing the weighted selection loss.
    """

    def __init__(self, anchor_cfg, pre_logits_cfg, sel_attrs, post_cfg, matcher_cfg, loss_cfg, feat_map_ids=None,
                 init_prob=0.01, box_encoder_cfg=None):
        """
        Initializes the AnchorSelector module.

        Args:
            anchor_cfg (Dict): Configuration dictionary specifying the anchor module.
            pre_logits_cfg (Dict): Configuration dictionary specifying the pre-logits module.
            sel_attrs (Dict): Attribute dictionary specifying the selection procedure.
            post_cfg (Dict): Configuration dictionary specifying the post-processing module.
            matcher_cfg (Dict): Configuration dictionary specifying the matcher module.
            loss_cfg (Dict): Configuration dictionary specifying the loss module.
            feat_map_ids (Tuple): Tuple containing the indices of feature maps used by this module (default=None).
            init_prob (float): Probability determining initial bias value of last logits sub-module (default=0.01).
            box_encoder_cfg (Dict): Configuration dictionary specifying the box encoder module (default=None).
        """

        # Initialization of default nn.Module
        super().__init__()

        # Set attribute with feature map indices
        self.feat_map_ids = feat_map_ids

        # Build anchor module
        self.anchor = build_model(anchor_cfg)

        # Build logits module
        pre_logits = build_model(pre_logits_cfg)

        if isinstance(pre_logits_cfg, list):
            in_feat_size = pre_logits_cfg[-1]['out_channels']
        else:
            in_feat_size = pre_logits_cfg['out_channels']

        num_cell_anchors = self.anchor.num_cell_anchors
        proj_logits = ProjConv(in_feat_size, num_cell_anchors, skip=False)

        init_logits_bias = -(math.log((1 - init_prob) / init_prob))
        torch.nn.init.constant_(proj_logits.conv.bias, init_logits_bias)

        seq_dict = OrderedDict([('pre', pre_logits), ('proj', proj_logits)])
        self.logits = Sequential(seq_dict)

        # Set selection-related attributes
        self.sel_attrs = sel_attrs

        # Build post-processing module
        self.post = build_model(post_cfg, sequential=True)

        # Build box encoder module if needed
        if box_encoder_cfg is not None:
            self.box_encoder = build_model(box_encoder_cfg)

        # Build matcher module
        self.matcher = build_model(matcher_cfg)

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
            ValueError: Error when an invalid selection mode is provided.
        """

        # Get selection logits
        logit_maps = self.logits(feat_maps)
        sel_logits = torch.cat([logit_map.permute(0, 2, 3, 1).flatten(1) for logit_map in logit_maps], dim=1)

        # Get selected indices
        sel_probs = torch.sigmoid(sel_logits.detach())

        batch_size, num_anchors = sel_probs.size()
        device = sel_probs.device
        sel_mode = self.sel_attrs['mode']

        if 'abs' in sel_mode:
            abs_sel_ids = torch.arange(batch_size * num_anchors, device=device)
            abs_sel_ids = abs_sel_ids[sel_probs.flatten() >= self.sel_attrs['abs_thr']]

        elif 'rel' in sel_mode:
            rel_sel_ids = torch.topk(sel_probs, self.sel_attrs['rel_thr'], dim=1, sorted=False).indices
            rel_sel_ids = rel_sel_ids + num_anchors * torch.arange(batch_size, device=device)[:, None]
            rel_sel_ids = rel_sel_ids.flatten()

        if sel_mode == 'abs':
            sel_ids = abs_sel_ids

        elif sel_mode == 'abs_and_rel':
            sel_ids, counts = torch.cat([abs_sel_ids, rel_sel_ids], dim=0).unique(return_counts=True)
            sel_ids = sel_ids[counts == 2]

        elif sel_mode == 'abs_or_rel':
            sel_ids = torch.cat([abs_sel_ids, rel_sel_ids], dim=0).unique()

        elif sel_mode == 'rel':
            sel_ids = rel_sel_ids

        else:
            error_msg = f"Invalid selection mode (got '{sel_mode}')."
            raise ValueError(error_msg)

        return sel_logits, sel_ids

    def get_selection_loss(self, sel_logits, storage_dict, tgt_dict, loss_dict, analysis_dict=None, **kwargs):
        """
        Method computing the selection loss.

        The method also computes the percentage of targets found, if an analysis dictionary is provided.

        Args:
            sel_logits (FloatTensor): Selection logits of shape [batch_size, num_anchors].

            storage_dict (Dict): Storage dictionary containing at least following key:
                - sel_ids (LongTensor): indices of selected feature-anchor combinations of shape [num_selecs].

            tgt_dict (Dict): Target dictionary used containing at least following key:
                - boxes (Boxes): structure containing axis-aligned bounding boxes of size [num_targets_total].

            loss_dict (Dict): Dictionary containing different weighted loss terms.
            analysis_dict (Dict): Dictionary containing different analyses (default=None).
            kwargs (Dict): Dictionary of keyword arguments passed to some underlying modules.

        Returns:
            storage_dict (Dict): Storage dictionary possibly containing following additional key:
                - sel_top_ids (LongTensor): top indices of selections per target of shape [top_limit, num_targets].

            loss_dict (Dict): Loss dictionary containing following additional key:
                - sel_loss (FloatTensor): tensor containing the selection loss of shape [].

            analysis_dict (Dict): Analysis dictionary containing following additional key (if not None):
                - sel_tgt_found (FloatTensor): tensor containing the percentage of targets found of shape [].
        """

        # Retrieve indices of selected feature-anchor combinations and get device
        sel_ids = storage_dict['sel_ids']
        device = sel_ids.device

        # Perform matching
        self.matcher(storage_dict=storage_dict, tgt_dict=tgt_dict, analysis_dict=analysis_dict, **kwargs)

        # Get top indices of selections per target if needed
        if 'top_qry_ids' in storage_dict:
            top_old_ids = storage_dict['top_qry_ids']

            num_selecs = len(sel_ids)
            max_old_id = sel_ids.amax().item() if num_selecs > 0 else 0
            max_old_id = max(max_old_id, top_old_ids.amax().item()) if top_old_ids.numel() > 0 else max_old_id

            new_ids = torch.full(size=[max_old_id+1], fill_value=-1, dtype=torch.int64, device=device)
            new_ids[sel_ids] = torch.arange(num_selecs, device=device)

            top_new_ids = new_ids[top_old_ids]
            storage_dict['sel_top_ids'] = top_new_ids

        # Get selection loss
        match_labels = storage_dict['match_labels']

        loss_mask = match_labels != -1
        pred_logits = sel_logits.flatten()[loss_mask]
        tgt_labels = match_labels[loss_mask]

        sel_loss = self.loss(pred_logits, tgt_labels)
        loss_dict['sel_loss'] = sel_loss

        # Get percentage of targets found
        if analysis_dict is not None:
            matched_qry_ids = storage_dict['matched_qry_ids']
            matched_tgt_ids = storage_dict['matched_tgt_ids']

            cat_ids = torch.cat([matched_qry_ids, sel_ids], dim=0)
            inv_ids, counts = cat_ids.unique(return_inverse=True, return_counts=True)[1:]

            num_matches = len(matched_qry_ids)
            inv_ids = inv_ids[:num_matches]

            tgt_found_mask = counts[inv_ids] == 2
            tgt_found_ids = matched_tgt_ids[tgt_found_mask].unique()

            num_tgts_found = len(tgt_found_ids)
            num_tgts = len(tgt_dict['boxes'])

            tgt_found = num_tgts_found / num_tgts if num_tgts > 0 else 1.0
            tgt_found = torch.tensor(tgt_found, device=device)
            analysis_dict['sel_tgt_found'] = 100 * tgt_found

        return storage_dict, loss_dict, analysis_dict

    def forward(self, storage_dict, tgt_dict=None, **kwargs):
        """
        Forward method of the AnchorSelector module.

        Args:
            storage_dict (Dict): Storage dictionary containing at least following keys:
                - feat_maps (List): list [num_maps] with maps of shape [batch_size, feat_size, fH, fW];
                - images (Images): images structure of size [batch_size] containing the batched images.

            tgt_dict (Dict): Dictionary containing the ground-truth targets during trainval (default=None).
            kwargs (Dict): Dictionary of keyword arguments passed to some underlying methods.

        Returns:
            storage_dict (Dict): Storage dictionary possibly containing following additional keys:
                - anchors (Boxes): structure with axis-aligned anchor boxes of size [num_anchors];
                - sel_ids (LongTensor): indices of selected feature-anchor combinations of shape [num_selecs];
                - cum_feats_batch (LongTensor): cumulative number of selected features per batch entry [batch_size+1];
                - feat_ids (LongTensor): indices of selected features of shape [num_selecs];
                - sel_feats (FloatTensor): selected features after post-processing of shape [num_selecs, feat_size];
                - sel_boxes (Boxes): structure with boxes corresponding to selected features of size [num_selecs];
                - sel_box_encs (FloatTensor): optional encodings of selected boxes of shape [num_selecs, feat_size].
        """

        # Retrieve feature maps from storage dictionary
        feat_maps = storage_dict['feat_maps']

        # Select desired feature maps if needed
        if self.feat_map_ids is not None:
            feat_maps = feat_maps[self.feat_map_ids]

        # Get anchors
        anchors = self.anchor(feat_maps)
        storage_dict['anchors'] = anchors

        # Perform selection
        sel_logits, sel_ids = self.perform_selection(feat_maps)
        storage_dict['sel_ids'] = sel_ids

        # Get cumulative number of selected features per batch entry
        batch_size = sel_logits.size(dim=0)
        batch_ids = torch.div(sel_ids, len(anchors), rounding_mode='floor')

        cum_feats_batch = torch.stack([(batch_ids == i).sum() for i in range(batch_size)]).cumsum(dim=0)
        cum_feats_batch = torch.cat([cum_feats_batch.new_zeros([1]), cum_feats_batch], dim=0)
        storage_dict['cum_feats_batch'] = cum_feats_batch

        # Get indices of selected features
        num_cell_anchors = self.anchor.num_cell_anchors
        feat_ids = torch.div(sel_ids, num_cell_anchors, rounding_mode='floor')
        storage_dict['feat_ids'] = feat_ids

        # Get selected features
        feats = torch.cat([feat_map.flatten(2).permute(0, 2, 1) for feat_map in feat_maps], dim=1)
        feats = feats.flatten(0, 1)
        sel_feats = feats[feat_ids]

        # Perform post-processing on selected features
        cell_anchor_ids = sel_ids % num_cell_anchors
        sel_feats = self.post(sel_feats, module_ids=cell_anchor_ids)
        storage_dict['sel_feats'] = sel_feats

        # Get boxes corresponding to selected features
        anchor_ids = sel_ids % len(anchors)
        sel_boxes = anchors[anchor_ids]
        sel_boxes.batch_ids = batch_ids
        storage_dict['sel_boxes'] = sel_boxes

        # Get encodings of selected boxes if needed
        if hasattr(self, 'box_encoder'):
            images = storage_dict['images']
            norm_boxes = sel_boxes.clone().normalize(images).to_format('cxcywh')

            box_encs = self.box_encoder(norm_boxes.boxes)
            storage_dict['sel_box_encs'] = box_encs

        # Get selection loss during trainval
        if tgt_dict is not None:
            self.get_selection_loss(sel_logits, storage_dict=storage_dict, tgt_dict=tgt_dict, **kwargs)

        return storage_dict
