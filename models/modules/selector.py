"""
Collection of selector modules.
"""
import math

import torch
from torch import nn

from models.build import build_model, MODELS
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
        pre (nn.Module): Pre-processing module operating on input set of feature maps.

        anchor_attrs (Dict): Dictionary containing following anchor-related attributes:
            map_ids (List): list [num_maps] containing the map ids (i.e. downsampling exponents) of each map;
            num_sizes (int): integer containing the number of different anchor sizes per aspect ratio;
            scale_factor (float): factor scaling the anchors w.r.t. non-overlapping tiling anchors;
            aspect_ratios (List): list [num_aspect_ratios] containing the different anchor aspect ratios.

        logits (ProjConv): Module computing the selection logits for each feature-anchor combination.

        sel_attrs (Dict): Dictionary containing following selection-related attributes:
            - type (str): string containing the type of selection procedure;
            - abs_thr (float): absolute threshold used during selection;
            - rel_thr (int): relative threshold used during selection.

        post (nn.Module): Post-processing module operating on selected features.

        match_attrs (Dict): Dictionary containing following matching-related attributes:
            - metric (str): string containing the matching metric;
            - sort_thr (int): threshold containing the amount of sorted feature-anchor indices to return per target;
            - type (str): string containing the type of matching procedure;
            - abs_pos (float): absolute threshold used during matching of positives;
            - abs_neg (float): absolute threshold used during matching of negatives;
            - rel_pos (int): relative threshold used during matching of positives;
            - rel_neg (int): relative threshold used during matching of negatives.

        loss (nn.Module): Module computing the weighted selection loss.
    """

    def __init__(self, pre_cfg, anchor_cfg, sel_cfg, post_cfg, match_cfg, loss_cfg, init_prob=0.01):
        """
        Initializes the AnchorSelector module.

        Args:
            pre_cfg (Dict): Configuration dictionary specifying the pre-processing module.
            anchor_cfg (Dict): Configuration dictionary specifying the anchors.
            sel_cfg (Dict): Configuration dictionary specifying the selection procedure.
            post_cfg (Dict): Configuration dictionary specifying the post-processing module.
            match_cfg (Dict): Configuration dictionary specifying the matching procedure.
            loss_cfg (Dict): Configuration dictionary specifying the loss module.
            init_prob (float): Probability determining initial bias value of logits module (default=0.01).
        """

        # Build pre-processing module
        self.pre = build_model(pre_cfg)

        # Set anchor attributes
        self.anchor_attrs = anchor_cfg

        # Initialize logits module
        in_feat_size = pre_cfg['out_channels']
        num_cell_anchors = anchor_cfg['num_sizes'] * len(anchor_cfg['aspect_ratios'])
        self.logits = ProjConv(in_feat_size, num_cell_anchors)

        init_logits_bias = -(math.log((1 - init_prob) / init_prob))
        torch.nn.init.constant_(self.logits.conv.bias, init_logits_bias)

        # Set selection attributes
        self.sel_attrs = sel_cfg

        # Build post-processing module
        self.post = build_model(post_cfg)

        # Set matching attributes
        self.match_attrs = match_cfg

        # Build loss module
        self.loss = build_model(loss_cfg)

    def forward(self, feat_maps, tgt_dict=None, **kwargs):
        """
        Forward method of the AnchorSelector module.

        Args:
            feat_maps (List): List of size [num_maps] with feature maps of shape [batch_size, feat_size, fH, fW].

            tgt_dict (Dict): Optional target dictionary used during trainval containing at least following keys:
                - labels (LongTensor): tensor of shape [num_targets_total] containing the class indices;
                - boxes (Boxes): structure containing axis-aligned bounding boxes of size [num_targets_total];
                - sizes (LongTensor): tensor of shape [batch_size+1] with the cumulative target sizes of batch entries.

            kwargs (Dict): Dictionary of unused keyword arguments.

        Returns:
            out_dict (Dict): Output dictionary containing following keys:
                - batch_ids (LongTensor): batch indices of selected features of shape [num_sel_feats];
                - feat_ids (LongTensor): feature indices of selected features of shape [num_sel_feats];
                - cell_anchor_ids (LongTensor): cell anchor indices of selected features of shape [num_sel_feats];
                - sel_feats (FloatTensor): selected features after post-processing of shape [num_sel_feats, feat_size];
                - loss_dict (Dict): dictionary with the weighted selection loss (trainval only);
                - analysis_dict (Dict): dictionary with different selection analyses used for logging purposes only.
        """

        # Initialize empty analysis and output dictionary
        analysis_dict = {}
        out_dict = {}

        # Perform pre-processing on feature maps
        feat_maps = self.pre(feat_maps)

        # Get anchors
        anchors = get_anchors(feat_maps, **self.anchor_attrs)
        out_dict['anchors'] = anchors

        # Get selection logits
        logit_maps = self.logits(feat_maps)
        sel_logits = torch.cat([logit_map.flatten(2) for logit_map in logit_maps], dim=2)
        sel_logits = sel_logits.permute(0, 2, 1)

        # Get selection probabilities
        sel_probs = torch.sigmoid(sel_logits.detach())

        # Perform selection
        sel_ids, sel_feats = self.perform_selection(sel_probs)

        # Get cell anchor and batch indices
        num_feats, num_cell_anchors = sel_logits.size()[1:]
        cell_anchor_ids = sel_ids % num_cell_anchors

        batch_ids = sel_ids // (num_feats * num_cell_anchors)
        out_dict['batch_ids'] = batch_ids

        # Perform post-processing on selected features
        sel_feats = self.post(sel_feats, module_ids=cell_anchor_ids)
        out_dict['sel_feats'] = sel_feats

        # Get weighted selection loss (trainval only)
        if tgt_dict is not None:

            # Perform matching
            pos_masks, neg_masks, tgt_sorted_ids = self.perform_matching(feat_maps, anchors, tgt_dict['boxes'])

            # Compute weighted selection loss
            pass

        # Add analysis dictionary to output dictionary
        out_dict['analysis_dict'] = analysis_dict

        return out_dict
