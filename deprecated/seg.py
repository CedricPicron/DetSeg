"""
Collection of segmentation heads.
"""

import torch
from torch import nn
import torch.nn.functional as F

from models.build import build_model, MODELS


@MODELS.register_module()
class SegBndHead(nn.Module):
    """
    Class implementing the SegBndHead module.

    Attributes:
        map_id (int): Integer containing the feature map index.
        logits (nn.Module): Module computing the segmentation boundary logits.
        get_mask (bool): Boolean indicating whether to get the segmentation boundary mask.
        mask_thr (float): Unnormalized threshold used to obtain the segmentation boundary mask.
        mask_ext (int): Integer containing the segmentation boundary mask extension size.
        loss (nn.Module): Module computing the segmentation boundary loss.
        apply_ids (List): List with integers determining when the head should be applied.
    """

    def __init__(self, logits_cfg, loss_cfg, map_id=0, get_mask=True, mask_thr=0.0, mask_ext=0, apply_ids=None,
                 **kwargs):
        """
        Initializes the SegBndHead module.

        Args:
            logits_cfg (Dict): Configuration dictionary specifying the logits module.
            loss_cfg (Dict): Configuration dictionary specifying the loss module.
            map_id (int): Integer containing the feature map index (default=0).
            get_mask (bool): Boolean indicating whether to get the segmentation boundary mask (default=True).
            mask_thr (float): Unnormalized threshold used to obtain the segmentation boundary mask (default=0.0).
            mask_ext (int): Integer containing the segmentation boundary mask extension size (default=0).
            apply_ids (List): List with integers determining when the head should be applied (default=None).
            kwargs (Dict): Dictionary of unused keyword arguments.
        """

        # Initialization of default nn.Module
        super().__init__()

        # Build logits module
        self.logits = build_model(logits_cfg)

        # Build loss module
        self.loss = build_model(loss_cfg)

        # Set remaining attributes
        self.map_id = map_id
        self.get_mask = get_mask
        self.mask_thr = mask_thr
        self.mask_ext = mask_ext
        self.apply_ids = apply_ids

    def forward_pred(self, storage_dict, **kwargs):
        """
        Forward prediction method of the SegBndHead module.

        Args:
            storage_dict (Dict): Storage dictionary containing at least following key:
                - feat_maps (List): list of size [num_maps] with feature maps of shape [batch_size, feat_size, fH, fW].

            kwargs (Dict): Dictionary of unused keyword arguments.

        Returns:
            storage_dict (Dict): Storage dictionary (possibly) containing following additional keys:
                - seg_bnd_logits (FloatTensor): segmentation boundary logits of shape [batch_size, 1, fH, fW];
                - seg_bnd_mask (BoolTensor): segmentation boundary mask of shape [batch_size, 1, fH, fW].
        """

        # Get feature map
        feat_map = storage_dict['feat_maps'][self.map_id]

        # Get segmentation boundary logits
        seg_bnd_logits = self.logits(feat_map)
        storage_dict['seg_bnd_logits'] = seg_bnd_logits

        # Get segmentation boundary mask if needed
        if self.get_mask:
            seg_bnd_mask = seg_bnd_logits > self.mask_thr

            if self.mask_ext > 0:
                kernel_size = 2*self.mask_ext + 1
                kernel = feat_map.new_ones([1, 1, kernel_size, kernel_size])

                seg_bnd_mask = seg_bnd_mask.float()
                seg_bnd_mask = F.conv2d(seg_bnd_mask, kernel, padding=self.mask_ext)
                seg_bnd_mask = seg_bnd_mask > 0

            storage_dict['seg_bnd_mask'] = seg_bnd_mask

        return storage_dict

    def forward_loss(self, storage_dict, tgt_dict, loss_dict, analysis_dict=None, id=None, **kwargs):
        """
        Forward loss method of the SegBndHead module.

        Args:
            storage_dict (Dict): Storage dictionary containing at least following key:
                - seg_bnd_logits (FloatTensor): segmentation boundary logits of shape [batch_size, 1, fH, fW].

            tgt_dict (Dict): Target dictionary containing at least following keys:
                - sizes (LongTensor): tensor of shape [batch_size+1] with the cumulative target sizes of batch entries;
                - masks (BoolTensor): target segmentation masks of shape [num_targets, iH, iW].

            loss_dict (Dict): Dictionary containing different weighted loss terms.
            analysis_dict (Dict): Dictionary containing different analyses (default=None).
            id (int or str): Integer or string containing the head id (default=None).
            kwargs (Dict): Dictionary of unused keyword arguments.

        Returns:
            loss_dict (Dict): Loss dictionary containing following additional key:
                - seg_bnd_loss (FloatTensor): segmentation boundary loss of shape [].

            analysis_dict (Dict): Analysis dictionary containing following additional key (if not None):
                - seg_bnd_acc (FloatTensor): segmentation boundary accuracy of shape [].
        """

        # Get segmentation boundary logits
        seg_bnd_logits = storage_dict['seg_bnd_logits']

        # Get segmentation boundary targets
        tgt_sizes = tgt_dict['sizes']
        tgt_masks = tgt_dict['masks']

        fH, fW = seg_bnd_logits.size()[2:]
        tgt_masks = tgt_masks[:, None, :, :].float()
        tgt_masks = F.interpolate(tgt_masks, size=(fH, fW), mode='bilinear', align_corners=False)
        tgt_masks = tgt_masks > 0.5

        kernel = seg_bnd_logits.new_ones([1, 1, 3, 3])
        tgt_bnd_masks = F.conv2d(tgt_masks.float(), kernel, padding=1)
        tgt_bnd_masks = (tgt_bnd_masks > 0) & (tgt_bnd_masks < 9)

        batch_size = len(tgt_sizes) - 1
        seg_bnd_targets_list = []

        for i in range(batch_size):
            i0 = tgt_sizes[i]
            i1 = tgt_sizes[i+1]

            seg_bnd_targets_i = tgt_bnd_masks[i0:i1].any(dim=0)
            seg_bnd_targets_list.append(seg_bnd_targets_i)

        seg_bnd_targets = torch.stack(seg_bnd_targets_list, dim=0)

        # Get segmentation boundary loss
        seg_bnd_loss = self.loss(seg_bnd_logits, seg_bnd_targets)

        key_name = f'seg_bnd_loss_{id}' if id is not None else 'seg_bnd_loss'
        loss_dict[key_name] = seg_bnd_loss

        # Get segmentation boundary accuary if needed
        if analysis_dict is not None:
            seg_bnd_preds = seg_bnd_logits > self.mask_thr
            seg_bnd_acc = (seg_bnd_preds == seg_bnd_targets).sum() / seg_bnd_preds.numel()

            key_name = f'seg_bnd_acc_{id}' if id is not None else 'seg_bnd_acc'
            analysis_dict[key_name] = 100 * seg_bnd_acc

        return loss_dict, analysis_dict

    def forward(self, mode, **kwargs):
        """
        Forward method of the SegBndHead module.

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
