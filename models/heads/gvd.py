"""
General Vision Decoder (GVD) head.
"""

import torch
from torch import nn
from torch.nn.parameter import Parameter

from models.build import build_model, MODELS


@MODELS.register_module()
class GVD(nn.Module):
    """
    Class implementing the General Vision Decoder (GVD) head.

    Attributes:
        group_init_type (str): String containing the type of group initialization.

        If group_init_type is 'learned':
            group_init_feats (Parameter): Parameter with group initialization features [num_groups, group_feat_size].

        If group_init_type is 'selected':
            group_init_sel (nn.Module): Module obtaining group initialization features by selecting from input maps.
    """

    def __init__(self, group_init_cfg):
        """
        Initializes the GVD head.

        Args:
            group_init_cfg (Dict): Configuration dictionary specifying the group initialization.

        Raises:
            ValueError: Error when an invalid type of group initialization is provided.
        """

        # Initialization of default nn.Module
        super().__init__()

        # Set attributes related to group initialization
        self.group_init_type = group_init_cfg['type']

        if self.group_init_type == 'learned':
            num_groups = group_init_cfg['num_groups']
            group_feat_size = group_init_cfg['feat_size']
            param_data = torch.randn(num_groups, group_feat_size)
            self.group_init_feats = Parameter(param_data, requires_grad=True)

        elif self.group_init_type == 'selected':
            sel_cfg = group_init_cfg['sel_cfg']
            self.group_init_sel = build_model(sel_cfg)

        else:
            error_msg = f"Invalid type of group initialization (got '{self.group_init_type}')."
            raise ValueError(error_msg)

    def group_initialization(self, feat_maps=None, tgt_dict=None, loss_dict=None, analysis_dict=None,
                             storage_dict=None, **kwargs):
        """
        Method performing group initialization, i.e. obtaining the initial group features.

        Key-value pairs might be added to the given loss, analysis and storage dictionaries.

        Args:
            feat_maps (List): List [num_maps] with maps of shape [batch_size, feat_size, fH, fW] (default=None).
            tgt_dict (Dict): Dictionary with ground-truth targets used during trainval (default=None).
            loss_dict (Dict): Dictionary with different weighted loss terms used during training (default=None).
            analysis_dict (Dict): Dictionary with different analyses used for logging purposes only (default=None).
            storage_dict (Dict): Dictionary storing all kinds of key-value pairs of interest (default=None).
            kwargs (Dict): Dictionary of additional keyword arguments passed to underlying modules.

        Returns:
            group_init_feats (FloatTensor): Group initialization features of shape [num_groups, group_feat_size].
            group_batch_ids (LongTensor): Batch indices of group features of shape [num_groups].

        Raises:
            ValueError: Error when an invalid type of group initialization is provided.
        """

        # Perform group initialization
        if self.group_init_type == 'learned':
            batch_size = feat_maps[0].size(dim=0)
            group_init_feats = self.group_init_feats.expand(batch_size, -1)

            device = self.group_init_feats.device
            num_groups = self.group_init_feats.size(dim=0)
            group_batch_ids = torch.arange(batch_size, device=device).repeat_interleave(num_groups)

        elif self.group_init_type == 'selected':
            sel_out_dict = self.group_init_sel(feat_maps=feat_maps, tgt_dict=tgt_dict, **kwargs)

            group_init_feats = sel_out_dict.pop('sel_feats')
            group_batch_ids = sel_out_dict.pop('batch_ids')

            sel_loss_dict = sel_out_dict.pop('loss_dict', {})
            sel_analysis_dict = sel_out_dict.pop('analysis_dict', {})

            sel_loss_dict = {f'sel_{k}': v for k, v in sel_loss_dict.items()}
            sel_analysis_dict = {f'sel_{k}': v for k, v in sel_analysis_dict.items()}

            loss_dict.update(sel_loss_dict) if loss_dict is not None else None
            analysis_dict.update(sel_analysis_dict) if analysis_dict is not None else None
            storage_dict.update(sel_out_dict) if storage_dict is not None else None

        else:
            error_msg = f"Invalid type of group initialization (got '{self.group_init_type}')."
            raise ValueError(error_msg)

        return group_init_feats, group_batch_ids

    def forward(self, feat_maps, tgt_dict=None, visualize=False, **kwargs):
        """
        Forward method of the GVD head.

        Args:
            feat_maps (List): List of size [num_maps] with feature maps of shape [batch_size, feat_size, fH, fW].

            tgt_dict (Dict): Optional target dictionary used during trainval containing following keys:
                - labels (LongTensor): tensor of shape [num_targets_total] containing the class indices;
                - boxes (Boxes): structure containing axis-aligned bounding boxes of size [num_targets_total];
                - sizes (LongTensor): tensor of shape [batch_size+1] with the cumulative target sizes of batch entries;
                - masks (ByteTensor): padded segmentation masks of shape [num_targets_total, max_iH, max_iW].

            visualize (bool): Boolean indicating whether to compute dictionary with visualizations (default=False).
            kwargs (Dict): Dictionary of additional keyword arguments passed to some underlying modules and methods.

        Raises:
            ValueError: Error when visualizations are requested.
            ValueError: Error when an invalid type of group initialization is provided.
        """

        # Check inputs
        if visualize:
            error_msg = "The GVD head currently does not provide visualizations."
            raise ValueError(error_msg)

        # Initialize empty loss, analysis and storage dictionaries
        loss_dict = {} if tgt_dict is not None else None
        analysis_dict = {}
        storage_dict = {}

        # Get initial group features
        group_init_kwargs = {'feat_maps': feat_maps, 'tgt_dict': tgt_dict, 'loss_dict': loss_dict}
        group_init_kwargs = {**group_init_kwargs, 'analysis_dict': analysis_dict, 'storage_dict': storage_dict}
        group_feats, group_batch_ids = self.perform_group_initialization(**group_init_kwargs, **kwargs)

        return group_feats, group_batch_ids, loss_dict, analysis_dict
