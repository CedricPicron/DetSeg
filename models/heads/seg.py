"""
Collection of segmentation heads.
"""

from detectron2.projects.point_rend.point_features import get_uncertain_point_coords_with_randomness
import torch
from torch import nn
import torch.nn.functional as F

from models.build import build_model, MODELS
from models.functional.loss import sigmoid_dice_loss


@MODELS.register_module()
class BaseSegHead(nn.Module):
    """
    Class implementing the BaseSegHead module.

    Attributes:
        qry (nn.Module): Module computing the query features.
        key (nn.Module): Module computing the key feature map.
        matcher (nn.Module): Optional matcher module determining the target segmentation maps.

        sample_attrs (Dict): Dictionary specifying the sample procedure, possibly containing following keys:
            - type (str): string containing the type of sample procedure (mandatory);
            - num_points (int): number of points to sample during PointRend sampling;
            - oversample_ratio (float): value of oversample ratio used during PointRend sampling;
            - importance_sample_ratio (float): ratio of importance sampling during PointRend sampling.

        loss (nn.Module): Module computing the segmentation loss.
    """

    def __init__(self, qry_cfg, key_cfg, sample_attrs, loss_cfg, matcher_cfg=None):
        """
        Initializes the BaseSegHead module.

        Args:
            qry_cfg (Dict): Configuration dictionary specifying the query module.
            key_cfg (Dict): Configuration dictionary specifying the key module.
            sample_attrs (Dict): Attribute dictionary specifying the sample procedure during loss computation.
            loss_cfg (Dict): Configuration dictionary specifying the loss module.
            matcher_cfg (Dict): Configuration dictionary specifying the matcher module (default=None).
        """

        # Initialization of default nn.Module
        super().__init__()

        # Build query module
        self.qry = build_model(qry_cfg)

        # Build key module
        self.key = build_model(key_cfg)

        # Build matcher module if needed
        if matcher_cfg is not None:
            self.matcher = build_model(matcher_cfg)

        # Build loss module
        self.loss = build_model(loss_cfg)

        # Set remaining attributes
        self.sample_attrs = sample_attrs

    def forward_pred(self, in_feats, storage_dict, cum_feats_batch=None, **kwargs):
        """
        Forward prediction method of the BaseSegHead module.

        Args:
            in_feats (FloatTensor): Input features of shape [num_feats, in_feat_size].

            storage_dict (Dict): Storage dictionary containing at least following key:
                - feat_maps (List): list of size [num_maps] with feature maps of shape [batch_size, feat_size, fH, fW].

            cum_feats_batch (LongTensor): Cumulative number of features per batch entry [batch_size+1] (default=None).
            kwargs (Dict): Dictionary of keyword arguments not used by this module.

        Returns:
            storage_dict (Dict): Storage dictionary containing following additional key:
                - seg_logits (FloatTensor): map with segmentation logits of shape [num_feats, fH, fW].
        """

        # Get number of features and device
        num_feats = len(in_feats)
        device = in_feats.device

        # Get cumulative number of features per batch entry if missing
        if cum_feats_batch is None:
            cum_feats_batch = torch.tensor([0, num_feats], device=device)

        # Get query features
        qry_feats = self.qry(in_feats)

        # Get key feature map
        in_feat_map = storage_dict['feat_maps'][0]
        key_feat_map = self.key(in_feat_map)

        # Get segmentation logits
        batch_size, _, fH, fW = key_feat_map.size()
        seg_logits_list = []

        for i in range(batch_size):
            i0 = cum_feats_batch[i].item()
            i1 = cum_feats_batch[i+1].item()

            conv_input = key_feat_map[i:i+1]
            conv_weight = qry_feats[i0:i1, :, None, None]

            seg_logits_i = F.conv2d(conv_input, conv_weight)[0]
            seg_logits_list.append(seg_logits_i)

        seg_logits = torch.cat(seg_logits_list, dim=0)
        storage_dict['seg_logits'] = seg_logits

        return storage_dict

    @staticmethod
    def get_uncertainties(logits):
        """
        Function computing uncertainties from the given input logits.

        Args:
            logits (FloatTensor): Tensor containing the input logits of shape [*].

        Returns:
            uncertainties (FloatTensor): Tensor containing the output uncertainties of shape [*].
        """

        # Get uncertainties
        uncertainties = -logits.abs()

        return uncertainties

    @staticmethod
    def point_sample(in_maps, sample_pts, **kwargs):
        """
        Function sampling the given input maps at the given sample points.

        Args:
            in_maps (FloatTensor): Input maps to sample from of shape [num_maps, H, W].
            sample_pts (FloatTensor): Normalized sample points within [0, 1] of shape [num_maps, num_samples, 2].
            kwargs (Dict): Dictionary of keyword arguments passed to underlying 'grid_sample' function.

        Returns:
            out_samples (FloatTensor): Output samples of shape [num_maps, num_samples].
        """

        # Get output samples
        in_maps = in_maps.unsqueeze(dim=1)

        sample_pts = 2*sample_pts - 1
        sample_pts = sample_pts.unsqueeze(dim=2)

        out_samples = F.grid_sample(in_maps, sample_pts, **kwargs)
        out_samples = out_samples[:, 0, :, 0]

        return out_samples

    def forward_loss(self, storage_dict, tgt_dict, loss_dict, analysis_dict=None, id=None, **kwargs):
        """
        Forward loss method of the BaseSegHead module.

        Args:
            storage_dict (Dict): Storage dictionary containing following keys (after matching):
                - seg_logits (FloatTensor): map with segmentation logits of shape [num_feats, fH, fW];
                - matched_qry_ids (LongTensor): indices of matched queries of shape [num_pos_queries];
                - matched_tgt_ids (LongTensor): indices of corresponding matched targets of shape [num_pos_queries].

            tgt_dict (Dict): Target dictionary containing at least following key:
                - masks (BoolTensor): segmentation masks of shape [num_targets, iH, iW].

            loss_dict (Dict): Dictionary containing different weighted loss terms.
            analysis_dict (Dict): Dictionary containing different analyses (default=None).
            id (int): Integer containing the head id (default=None).
            kwargs (Dict): Dictionary of keyword arguments passed to some underlying modules.

        Returns:
            loss_dict (Dict): Loss dictionary containing following additional key:
                - seg_loss (FloatTensor): segmentation loss of shape [].

            analysis_dict (Dict): Analysis dictionary containing following additional key (if not None):
                - seg_acc (FloatTensor): segmentation accuracy of shape [].

        Raises:
            ValueError: Error when an invalid type of sample procedure is provided.
        """

        # Perform matching if matcher is available
        if hasattr(self, 'matcher'):
            self.matcher(storage_dict=storage_dict, tgt_dict=tgt_dict, analysis_dict=analysis_dict, **kwargs)

        # Retrieve segmentation logits and matching results
        seg_logits = storage_dict['seg_logits']
        matched_qry_ids = storage_dict['matched_qry_ids']
        matched_tgt_ids = storage_dict['matched_tgt_ids']

        # Get device
        device = seg_logits.device

        # Handle case where there are no positive matches
        if len(matched_qry_ids) == 0:

            # Get segmentation loss
            seg_loss = 0.0 * seg_logits.sum()
            key_name = f'seg_loss_{id}' if id is not None else 'seg_loss'
            loss_dict[key_name] = seg_loss

            # Get segmentation accuracy if needed
            if analysis_dict is not None:
                seg_acc = 1.0 if len(tgt_dict['masks']) == 0 else 0.0
                seg_acc = torch.tensor(seg_acc, dtype=seg_loss.dtype, device=device)

                key_name = f'seg_acc_{id}' if id is not None else 'seg_acc'
                analysis_dict[key_name] = 100 * seg_acc

            return loss_dict, analysis_dict

        # Get matched segmentation logits with corresponding targets
        seg_logits = seg_logits[matched_qry_ids]
        seg_targets = tgt_dict['masks'][matched_tgt_ids].float()

        # Get sample points
        with torch.no_grad():
            sample_type = self.sample_attrs['type']

            if sample_type == 'dense':
                fH, fW = seg_logits.size()[1:]

                sample_pts_x = torch.linspace(0.5/fW, 1-0.5/fW, steps=fW, device=device)
                sample_pts_y = torch.linspace(0.5/fH, 1-0.5/fH, steps=fH, device=device)

                sample_pts = torch.meshgrid(sample_pts_x, sample_pts_y, indexing='xy')
                sample_pts = torch.stack(sample_pts, dim=2).flatten(0, 1)

                num_matches = len(matched_qry_ids)
                sample_pts = sample_pts[None, :, :].expand(num_matches, -1, -1)

            elif sample_type == 'point_rend':
                point_rend_keys = ('num_points', 'oversample_ratio', 'importance_sample_ratio')
                point_rend_kwargs = {k: v for k, v in self.sample_attrs.items() if k in point_rend_keys}

                point_rend_kwargs['coarse_logits'] = seg_logits.unsqueeze(dim=1)
                point_rend_kwargs['uncertainty_func'] = lambda logits: self.get_uncertainties(logits)
                sample_pts = get_uncertain_point_coords_with_randomness(**point_rend_kwargs)

            else:
                error_msg = f"Invalid type of sample procedure (got '{sample_type}')."
                raise ValueError(error_msg)

        # Get sampled segmenatation logits and corresponding targets
        seg_logits = self.point_sample(seg_logits, sample_pts, align_corners=False)
        seg_targets = self.point_sample(seg_targets, sample_pts, align_corners=False)

        # Get segmentation loss
        seg_loss = self.loss(seg_logits, seg_targets)
        key_name = f'seg_loss_{id}' if id is not None else 'seg_loss'
        loss_dict[key_name] = seg_loss

        # Get segmentation accuracy if needed
        if analysis_dict is not None:
            seg_acc = 1 - sigmoid_dice_loss(seg_logits, seg_targets, reduction='mean')
            key_name = f'seg_acc_{id}' if id is not None else 'seg_acc'
            analysis_dict[key_name] = 100 * seg_acc

        return loss_dict, analysis_dict

    def forward(self, mode, **kwargs):
        """
        Forward method of the BaseSegHead module.

        Args:
            mode (str): String containing the forward mode chosen from ['pred', 'loss'].
            kwargs (Dict): Dictionary of keyword arguments passed to the underlying forward method.

        Raises:
            ValueError: Error when an invalid forward mode is provided.
        """

        # Choose underlying forward method
        if mode == 'pred':
            self.forward_pred(**kwargs)

        elif mode == 'loss':
            self.forward_loss(**kwargs)

        else:
            error_msg = f"Invalid forward mode (got '{mode}')."
            raise ValueError(error_msg)
