"""
Collection of MMDetection RoI (Region of Interest) heads.
"""
from inspect import signature

from mmcv.ops import point_sample
from mmdet.structures.mask import BitmapMasks
from mmdet.structures.mask.mask_target import mask_target_single
from mmdet.models.roi_heads import StandardRoIHead as MMDetStandardRoIHead
from mmdet.models.roi_heads.mask_heads.fcn_mask_head import _do_paste_mask
from mmdet.models.roi_heads.point_rend_roi_head import PointRendRoIHead as MMDetPointRendRoIHead
from mmengine.config import Config
from mmengine.structures import InstanceData
import torch
import torch.nn.functional as F

from models.build import build_model, MODELS
from models.heads.seg.base import BaseSegHead
from models.modules.refine_mask import generate_block_target


@MODELS.register_module()
class StandardRoIHead(BaseSegHead, MMDetStandardRoIHead):
    """
    Class implementing the StandardRoIHead module.

    The module is based on the StandardRoIHead module from MMDetection.

    Attributes:
        pos_enc (nn.Module): Optional module adding position encodings to the RoI features.
        qry (nn.Module): Optional module updating the query features.
        fuse_qry (nn.Module): Optional module fusing the query features with the RoI features.
    """

    def __init__(self, pos_enc_cfg=None, qry_cfg=None, fuse_qry_cfg=None, **kwargs):
        """
        Initializes the StandardRoIHead module.

        Args:
            pos_enc_cfg (Dict): Configuration dictionary specifying the position encoder module (default=None).
            qry_cfg (Dict): Configuration dictionary specifying the query module (default=None).
            fuse_qry_cfg (Dict): Configuration dictionary specifying the fuse query module (default=None).
            kwargs (Dict): Dictionary of keyword arguments passed to some underlying methods.
        """

        # Initialize module using BaseRoIHead __init__ method
        BaseSegHead.__init__(self, **kwargs)

        # Additional initialization using MMDetStandardRoIHead __init__ method
        init_kwargs = {k: kwargs[k] for k in signature(MMDetStandardRoIHead.__init__).parameters if k in kwargs}
        MMDetStandardRoIHead.__init__(self, **init_kwargs)

        # Build position encoder, query and fuse query modules
        self.pos_enc = build_model(pos_enc_cfg) if pos_enc_cfg is not None else None
        self.qry = build_model(qry_cfg) if qry_cfg is not None else None
        self.fuse_qry = build_model(fuse_qry_cfg) if fuse_qry_cfg is not None else None

    def get_mask_logits(self, roi_feats, pred_labels, **kwargs):
        """
        Method computing the segmentation mask logits at RoI resolution.

        Args:
            roi_feats (FloatTensor): RoI features of shape [num_preds, feat_size, in_rH, in_rW].
            pred_labels (LongTensor): Class indices of predictions of shape [num_preds].
            kwargs (Dict): Dictionary of unused keyword arguments.

        Returns:
            mask_logits (FloatTensor): Segmentation mask logits of shape [num_preds, 1, out_rH, out_rW].
        """

        # Get mask logits at RoI resolution
        mask_logits = self.mask_head(roi_feats)
        mask_logits = mask_logits[range(len(mask_logits)), pred_labels]
        mask_logits = mask_logits[:, None]

        return mask_logits

    def get_mask_scores(self, batch_id, pred_qry_ids, pred_labels, pred_boxes, storage_dict, **kwargs):
        """
        Method computing the segmentation mask scores at image resolution.

        Args:
            batch_id (int): Integer containing the batch index.
            pred_qry_ids (LongTensor): Query indices of predictions of shape [num_preds].
            pred_labels (LongTensor): Class indices of predictions of shape [num_preds].
            pred_boxes (Boxes): Predicted 2D bounding boxes of size [num_preds].

            storage_dict (Dict): Storage dictionary containing at least following keys:
                - qry_feats (FloatTensor): query features of shape [num_qrys, qry_feat_size];
                - feat_maps (List): list of size [num_maps] with feature maps of shape [batch_size, feat_size, fH, fW];
                - images (Images): Images structure containing the batched images of size [batch_size].

            kwargs (Dict): Dictionary of unused keyword arguments.

        Returns:
            mask_scores (FloatTensor): Segmentation mask scores of shape [num_preds, iH, iW].
        """

        # Get RoI features
        feat_maps = storage_dict['feat_maps']
        roi_feat_maps = feat_maps[:self.mask_roi_extractor.num_inputs]

        batch_ids = torch.full_like(pred_labels, batch_id)
        roi_boxes = torch.cat([batch_ids[:, None], pred_boxes], dim=1)
        roi_feats = self.mask_roi_extractor(roi_feat_maps, roi_boxes)

        # Add position encodings if needed
        if self.pos_enc is not None:
            rH, rW = roi_feats.size()[2:]
            device = roi_feats.device

            pts_x = torch.linspace(0.5/rW, 1-0.5/rW, steps=rW, device=device)
            pts_y = torch.linspace(0.5/rH, 1-0.5/rH, steps=rH, device=device)

            norm_xy = torch.meshgrid(pts_x, pts_y, indexing='xy')
            norm_xy = torch.stack(norm_xy, dim=2).flatten(0, 1)
            roi_feats = roi_feats + self.pos_enc(norm_xy).t().view(1, -1, rH, rW)

        # Fuse query features if needed
        if self.fuse_qry is not None:
            pred_qry_feats = storage_dict['qry_feats'][pred_qry_ids]
            pred_qry_feats = self.qry(pred_qry_feats) if self.qry is not None else pred_qry_feats
            pred_qry_feats = pred_qry_feats[:, :, None, None].expand_as(roi_feats)

            fuse_qry_feats = torch.cat([pred_qry_feats, roi_feats], dim=1)
            roi_feats = roi_feats + self.fuse_qry(fuse_qry_feats)

        # Get mask logits at RoI resolution
        in_kwargs = {'roi_feats': roi_feats, 'roi_boxes': roi_boxes, 'pred_labels': pred_labels}
        in_kwargs = {**in_kwargs, 'storage_dict': storage_dict}
        mask_logits = self.get_mask_logits(**in_kwargs)

        # Get mask scores at image resolution
        iW, iH = storage_dict['images'].size()
        mask_scores = mask_logits.sigmoid()
        mask_scores = _do_paste_mask(mask_scores, pred_boxes, iH, iW, skip_empty=False)[0]

        return mask_scores

    def forward_pred(self, storage_dict, images_dict=None, **kwargs):
        """
        Forward prediction method of the StandardRoIHead module.

        Args:
            storage_dict (Dict): Dictionary storing various items of interest.
            images_dict (Dict): Dictionary with annotated images of predictions/targets (default=None).
            kwargs (Dict): Dictionary of keyword arguments passed to some underlying methods.

        Returns:
            storage_dict (Dict): Dictionary with (possibly) additional stored items of interest.
            images_dict (Dict): Dictionary with (possibly) additional images annotated with 2D boxes/segmentations.

        Raises:
            NotImplementedError: Error when the StandardRoIHead module contains a bounding box head.
        """

        # Get box-related predictions
        if self.with_bbox:
            raise NotImplementedError

        # Get mask-related predictions
        if self.with_mask:

            # Get segmentation predictions if needed
            if self.get_segs and not self.training:
                self.compute_segs(storage_dict=storage_dict, **kwargs)

            # Draw predicted and target segmentations if needed
            if images_dict is not None:
                self.draw_segs(storage_dict=storage_dict, images_dict=images_dict, **kwargs)

        return storage_dict, images_dict

    def compute_losses(self, storage_dict, tgt_dict, loss_dict, analysis_dict=None, id=None, **kwargs):
        """
        Method computing the losses and collecting analysis metrics.

        Args:
            storage_dict (Dict): Storage dictionary containing at least following keys (after matching):
                - qry_feats (FloatTensor): query features of shape [num_qrys, qry_feat_size];
                - batch_ids (LongTensor): batch indices of queries of shape [num_qrys];
                - feat_maps (List): list of size [num_maps] with feature maps of shape [batch_size, feat_size, fH, fW];
                - images (Images): Images structure containing the batched images of size [batch_size];
                - pred_boxes (Boxes): predicted 2D bounding boxes of size [num_qrys];
                - matched_qry_ids (LongTensor): indices of matched queries of shape [num_pos_qrys];
                - matched_tgt_ids (LongTensor): indices of corresponding matched targets of shape [num_pos_qrys].

            tgt_dict (Dict): Target dictionary containing at least following keys:
                - labels (LongTensor): target class indices of shape [num_targets];
                - masks (BoolTensor): target segmentation masks of shape [num_targets, iH, iW].

            loss_dict (Dict): Dictionary containing different weighted loss terms.
            analysis_dict (Dict): Dictionary containing different analysis metrics (default=None).
            id (int or str): Integer or string containing the head id (default=None).
            kwargs (Dict): Dictionary of keyword arguments passed to some underlying modules.

        Returns:
            loss_dict (Dict): Loss dictionary containing following additional key:
                - mask_loss (FloatTensor): mask loss of shape [].

            analysis_dict (Dict): Analysis dictionary containing following additional key (if not None):
                - mask_acc (FloatTensor): mask accuracy of shape [].
        """

        # Retrieve desired items from storage dictionary
        batch_ids = storage_dict['batch_ids']
        feat_maps = storage_dict['feat_maps']
        images = storage_dict['images']
        pred_boxes = storage_dict['pred_boxes']
        matched_qry_ids = storage_dict['matched_qry_ids']
        matched_tgt_ids = storage_dict['matched_tgt_ids']

        # Get device and number of positive matches
        device = matched_qry_ids.device
        num_pos_matches = len(matched_qry_ids)

        # Handle case where there are no positive matches
        if num_pos_matches == 0:

            # Get mask loss
            mask_loss = sum(0.0 * feat_map.flatten()[0] for feat_map in feat_maps)
            mask_loss += sum(0.0 * p.flatten()[0] for p in self.parameters())

            key_name = f'mask_loss_{id}' if id is not None else 'mask_loss'
            loss_dict[key_name] = mask_loss

            # Get mask accuracy if needed
            if analysis_dict is not None:
                mask_acc = 1.0 if len(tgt_dict['masks']) == 0 else 0.0
                mask_acc = torch.tensor(mask_acc, dtype=mask_loss.dtype, device=device)

                key_name = f'mask_acc_{id}' if id is not None else 'mask_acc'
                analysis_dict[key_name] = 100 * mask_acc

            return loss_dict, analysis_dict

        # Get batch indices
        batch_ids = batch_ids[matched_qry_ids]

        # Get RoI boxes
        roi_boxes = pred_boxes[matched_qry_ids].to_format('xyxy')
        roi_boxes = roi_boxes.to_img_scale(images).boxes.detach()
        roi_boxes = torch.cat([batch_ids[:, None], roi_boxes], dim=1)

        # Get mask logits
        roi_feat_maps = feat_maps[:self.mask_roi_extractor.num_inputs]
        roi_feats = self.mask_roi_extractor(roi_feat_maps, roi_boxes)

        if self.pos_enc is not None:
            rH, rW = roi_feats.size()[2:]
            pts_x = torch.linspace(0.5/rW, 1-0.5/rW, steps=rW, device=device)
            pts_y = torch.linspace(0.5/rH, 1-0.5/rH, steps=rH, device=device)

            norm_xy = torch.meshgrid(pts_x, pts_y, indexing='xy')
            norm_xy = torch.stack(norm_xy, dim=2).flatten(0, 1)
            roi_feats = roi_feats + self.pos_enc(norm_xy).t().view(1, -1, rH, rW)

        if self.fuse_qry is not None:
            qry_feats = storage_dict['qry_feats']

            qry_feats = qry_feats[matched_qry_ids]
            qry_feats = self.qry(qry_feats) if self.qry is not None else qry_feats
            qry_feats = qry_feats[:, :, None, None].expand_as(roi_feats)

            fuse_qry_feats = torch.cat([qry_feats, roi_feats], dim=1)
            roi_feats = roi_feats + self.fuse_qry(fuse_qry_feats)

        mask_logits = self.mask_head(roi_feats)

        tgt_labels = tgt_dict['labels'][matched_tgt_ids]
        mask_logits = mask_logits[range(num_pos_matches), tgt_labels]

        # Get mask targets
        tgt_masks = tgt_dict['masks'].cpu().numpy()
        tgt_masks = BitmapMasks(tgt_masks, *tgt_masks.shape[-2:])

        mask_size = tuple(mask_logits.size()[-2:])
        mask_tgt_cfg = Config({'mask_size': mask_size})
        mask_targets = mask_target_single(roi_boxes[:, 1:], matched_tgt_ids, tgt_masks, mask_tgt_cfg)

        # Get mask loss
        mask_loss = self.mask_head.loss_mask(mask_logits, mask_targets)
        mask_loss = mask_loss / (mask_size[0] * mask_size[1])

        key_name = f'mask_loss_{id}' if id is not None else 'mask_loss'
        loss_dict[key_name] = mask_loss

        # Get mask accuracy if needed
        if analysis_dict is not None:
            mask_preds = mask_logits > 0
            mask_targets = mask_targets.bool()
            mask_acc = (mask_preds == mask_targets).sum() / (mask_preds).numel()

            key_name = f'mask_acc_{id}' if id is not None else 'mask_acc'
            analysis_dict[key_name] = 100 * mask_acc

        return loss_dict, analysis_dict


@MODELS.register_module()
class PointRendRoIHead(StandardRoIHead, MMDetPointRendRoIHead):
    """
    Class implementing the PointRendRoIHead module.

    The module is based on the PointRendRoIHead module from MMDetection.

    Attributes:
        point_head (nn.Module): Module implementing the point head.

        train_cfg (Config): Dictionary containing following training attributes:
            - num_points (int): number of points to sample during training;
            - oversample_ratio (float): value of oversample ratio used during training point sampling;
            - importance_sample_ratio (float): ratio of importance sampling used during training point sampling.

        test_cfg (Config): Dictionary containing following test attributes:
            - subdivision_steps (int): number of subdivision steps used during inference;
            - subdivision_num_points (int): number subdivision points per subdivision step during inference;
            - scale_factor (float): scale factor used during inference upsampling.
    """

    def __init__(self, point_head_cfg, train_attrs, test_attrs, **kwargs):
        """
        Initializes the PointRendRoIHead module.

        Args:
            point_head_cfg (Dict): Configuration dictionary specifying the point head module.

            train_attrs (Dict): Dictionary containing following training attributes:
                - num_points (int): number of points to sample during training;
                - oversample_ratio (float): value of oversample ratio used during training point sampling;
                - importance_sample_ratio (float): ratio of importance sampling used during training point sampling.

            test_attrs (Dict): Dictionary containing following test attributes:
                - subdivision_steps (int): number of subdivision steps used during inference;
                - subdivision_num_points (int): number subdivision points per subdivision step during inference;
                - scale_factor (float): scale factor used during inference upsampling.

            kwargs (Dict): Dictionary of keyword arguments passed to the parent __init__ method.
        """

        # Initialize module using parent __init__ method
        super().__init__(**kwargs)

        # Build point head module
        self.point_head = build_model(point_head_cfg)

        # Set training and test attributes
        self.train_cfg = Config(train_attrs)
        self.test_cfg = Config(test_attrs)

    def get_mask_logits(self, roi_feats, roi_boxes, pred_labels, storage_dict, **kwargs):
        """
        Method computing the segmentation mask logits at RoI resolution.

        Args:
            roi_feats (FloatTensor): RoI features of shape [num_preds, feat_size, in_rH, in_rW].
            roi_boxes (FloatTensor): RoI boxes with batch indices of shape [num_preds, 5].
            pred_labels (LongTensor): Class indices of predictions of shape [num_preds].

            storage_dict (Dict): Storage dictionary containing at least following key:
                - feat_maps (List): list of size [num_maps] with feature maps of shape [batch_size, feat_size, fH, fW].

            kwargs (Dict): Dictionary of unused keyword arguments.

        Returns:
            mask_logits (FloatTensor): Segmentation mask logits of shape [num_preds, 1, out_rH, out_rW].
        """

        # Get mask logits at RoI resolution
        mask_logits = self.mask_head(roi_feats)

        feat_maps = storage_dict['feat_maps']
        mask_logits = self._mask_point_forward_test(feat_maps, roi_boxes, pred_labels, mask_logits)

        mask_logits = mask_logits[range(len(mask_logits)), pred_labels]
        mask_logits = mask_logits[:, None]

        return mask_logits

    def compute_losses(self, storage_dict, tgt_dict, loss_dict, analysis_dict=None, id=None, **kwargs):
        """
        Method computing the losses and collecting analysis metrics.

        Args:
            storage_dict (Dict): Storage dictionary containing at least following keys (after matching):
                - qry_feats (FloatTensor): query features of shape [num_qrys, qry_feat_size];
                - batch_ids (LongTensor): batch indices of queries of shape [num_qrys];
                - feat_maps (List): list of size [num_maps] with feature maps of shape [batch_size, feat_size, fH, fW];
                - images (Images): Images structure containing the batched images of size [batch_size];
                - pred_boxes (Boxes): predicted 2D bounding boxes of size [num_qrys];
                - matched_qry_ids (LongTensor): indices of matched queries of shape [num_pos_qrys];
                - matched_tgt_ids (LongTensor): indices of corresponding matched targets of shape [num_pos_qrys].

            tgt_dict (Dict): Target dictionary containing at least following keys:
                - labels (LongTensor): target class indices of shape [num_targets];
                - masks (BoolTensor): target segmentation masks of shape [num_targets, iH, iW].

            loss_dict (Dict): Dictionary containing different weighted loss terms.
            analysis_dict (Dict): Dictionary containing different analysis metrics (default=None).
            id (int or str): Integer or string containing the head id (default=None).
            kwargs (Dict): Dictionary of keyword arguments passed to some underlying modules.

        Returns:
            loss_dict (Dict): Loss dictionary containing following additional keys:
                - mask_loss (FloatTensor): mask loss of shape [];
                - point_loss (FloatTensor): point loss of shape [].

            analysis_dict (Dict): Analysis dictionary containing following additional keys (if not None):
                - mask_acc (FloatTensor): mask accuracy of shape [];
                - point_acc (FloatTensor): point accuracy of shape [].
        """

        # Retrieve desired items from storage dictionary
        batch_ids = storage_dict['batch_ids']
        feat_maps = storage_dict['feat_maps']
        images = storage_dict['images']
        pred_boxes = storage_dict['pred_boxes']
        matched_qry_ids = storage_dict['matched_qry_ids']
        matched_tgt_ids = storage_dict['matched_tgt_ids']

        # Get device and number of positive matches
        device = matched_qry_ids.device
        num_pos_matches = len(matched_qry_ids)

        # Handle case where there are no positive matches
        if num_pos_matches == 0:

            # Get mask loss
            mask_loss = sum(0.0 * feat_map.flatten()[0] for feat_map in feat_maps)
            mask_loss += sum(0.0 * p.flatten()[0] for p in self.parameters())

            key_name = f'mask_loss_{id}' if id is not None else 'mask_loss'
            loss_dict[key_name] = mask_loss

            # Get point loss
            key_name = f'point_loss_{id}' if id is not None else 'point_loss'
            loss_dict[key_name] = mask_loss

            # Get mask and point accuracies if needed
            if analysis_dict is not None:

                # Get mask accuracy
                mask_acc = 1.0 if len(tgt_dict['masks']) == 0 else 0.0
                mask_acc = torch.tensor(mask_acc, dtype=mask_loss.dtype, device=device)

                key_name = f'mask_acc_{id}' if id is not None else 'mask_acc'
                analysis_dict[key_name] = 100 * mask_acc

                # Get point accuracy
                point_acc = mask_acc.clone()

                key_name = f'point_acc_{id}' if id is not None else 'point_acc'
                analysis_dict[key_name] = 100 * point_acc

            return loss_dict, analysis_dict

        # Get batch indices
        batch_ids = batch_ids[matched_qry_ids]

        # Get RoI boxes
        roi_boxes = pred_boxes[matched_qry_ids].to_format('xyxy')
        roi_boxes = roi_boxes.to_img_scale(images).boxes.detach()
        roi_boxes = torch.cat([batch_ids[:, None], roi_boxes], dim=1)

        # Get mask logits
        roi_feat_maps = feat_maps[:self.mask_roi_extractor.num_inputs]
        roi_feats = self.mask_roi_extractor(roi_feat_maps, roi_boxes)

        if self.pos_enc is not None:
            rH, rW = roi_feats.size()[2:]
            pts_x = torch.linspace(0.5/rW, 1-0.5/rW, steps=rW, device=device)
            pts_y = torch.linspace(0.5/rH, 1-0.5/rH, steps=rH, device=device)

            norm_xy = torch.meshgrid(pts_x, pts_y, indexing='xy')
            norm_xy = torch.stack(norm_xy, dim=2).flatten(0, 1)
            roi_feats = roi_feats + self.pos_enc(norm_xy).t().view(1, -1, rH, rW)

        if self.fuse_qry is not None:
            qry_feats = storage_dict['qry_feats']

            qry_feats = qry_feats[matched_qry_ids]
            qry_feats = self.qry(qry_feats) if self.qry is not None else qry_feats
            qry_feats = qry_feats[:, :, None, None].expand_as(roi_feats)

            fuse_qry_feats = torch.cat([qry_feats, roi_feats], dim=1)
            roi_feats = roi_feats + self.fuse_qry(fuse_qry_feats)

        mask_logits = self.mask_head(roi_feats)

        tgt_labels = tgt_dict['labels'][matched_tgt_ids]
        cls_mask_logits = mask_logits[range(num_pos_matches), tgt_labels]

        # Get mask targets
        tgt_masks = tgt_dict['masks'].cpu().numpy()
        tgt_masks = BitmapMasks(tgt_masks, *tgt_masks.shape[-2:])

        mask_size = tuple(mask_logits.size()[-2:])
        mask_tgt_cfg = Config({'mask_size': mask_size})
        mask_targets = mask_target_single(roi_boxes[:, 1:], matched_tgt_ids, tgt_masks, mask_tgt_cfg)

        # Get mask loss
        mask_loss = self.mask_head.loss_mask(cls_mask_logits, mask_targets)
        mask_loss = mask_loss / (mask_size[0] * mask_size[1])

        key_name = f'mask_loss_{id}' if id is not None else 'mask_loss'
        loss_dict[key_name] = mask_loss

        # Get point logits
        tgt_labels = tgt_dict['labels'][matched_tgt_ids]
        roi_pts = self.point_head.get_roi_rel_points_train(mask_logits, tgt_labels, self.train_cfg)

        feat_maps = storage_dict['feat_maps']
        fine_feats = self._get_fine_grained_point_feats(feat_maps, roi_boxes, roi_pts)

        coarse_feats = point_sample(mask_logits, roi_pts)
        point_logits = self.point_head(fine_feats, coarse_feats)
        point_logits = point_logits[range(num_pos_matches), tgt_labels]

        # Get point targets
        gt_instances = InstanceData(masks=tgt_masks)
        target_single_args = (roi_boxes, roi_pts, matched_tgt_ids, gt_instances, self.train_cfg)
        point_targets = self.point_head._get_targets_single(*target_single_args)

        # Get point loss
        point_loss = self.point_head.loss_point(point_logits, point_targets)
        point_loss = point_loss / self.train_cfg['num_points']

        key_name = f'point_loss_{id}' if id is not None else 'point_loss'
        loss_dict[key_name] = point_loss

        # Get mask and point accuracies if needed
        if analysis_dict is not None:

            # Get mask accuracy
            mask_preds = cls_mask_logits > 0
            mask_targets = mask_targets.bool()
            mask_acc = (mask_preds == mask_targets).sum() / (mask_preds).numel()

            key_name = f'mask_acc_{id}' if id is not None else 'mask_acc'
            analysis_dict[key_name] = 100 * mask_acc

            # Get point accuracy
            point_preds = point_logits > 0
            point_targets = point_targets.bool()
            point_acc = (point_preds == point_targets).sum() / (point_preds).numel()

            key_name = f'point_acc_{id}' if id is not None else 'point_acc'
            analysis_dict[key_name] = 100 * point_acc

        return loss_dict, analysis_dict


@MODELS.register_module()
class RefineMaskRoIHead(StandardRoIHead):
    """
    Class implementing the RefineMaskRoIHead module.

    The module is based on the SimpleRefineRoIHead module from https://github.com/zhanggang001/RefineMask.
    """

    def get_mask_logits(self, roi_feats, roi_boxes, pred_labels, storage_dict, **kwargs):
        """
        Method computing the segmentation mask logits at RoI resolution.

        Args:
            roi_feats (FloatTensor): RoI features of shape [num_preds, feat_size, in_rH, in_rW].
            roi_boxes (FloatTensor): RoI boxes with batch indices of shape [num_preds, 5].
            pred_labels (LongTensor): Class indices of predictions of shape [num_preds].

            storage_dict (Dict): Storage dictionary containing at least following key:
                - feat_maps (List): list of size [num_maps] with feature maps of shape [batch_size, feat_size, fH, fW].

            kwargs (Dict): Dictionary of unused keyword arguments.

        Returns:
            mask_logits (FloatTensor): Segmentation mask logits of shape [num_preds, 1, out_rH, out_rW].
        """

        # Get mask logits at RoI resolution
        feat_maps = storage_dict['feat_maps']
        mask_logits_list = self.mask_head(roi_feats, feat_maps[0], roi_boxes, pred_labels)

        num_stages = len(self.mask_head.stage_sup_size)
        itp_kwargs = {'mode': 'bilinear', 'align_corners': True}

        for i in range(1, num_stages-1):
            mask_logits_i = mask_logits_list[i]
            pred_masks_i = mask_logits_i.squeeze(dim=1).sigmoid() >= 0.5
            next_shape = mask_logits_list[i+1].shape[-2:]

            non_boundary_mask = generate_block_target(pred_masks_i, boundary_width=1) != 1
            non_boundary_mask = non_boundary_mask.unsqueeze(dim=1).float()
            non_boundary_mask = F.interpolate(non_boundary_mask, next_shape, **itp_kwargs) >= 0.5

            mask_logits_i = F.interpolate(mask_logits_i, next_shape, **itp_kwargs)
            mask_logits_list[i+1][non_boundary_mask] = mask_logits_i[non_boundary_mask]

        mask_logits = mask_logits_list[-1]

        return mask_logits

    def compute_losses(self, storage_dict, tgt_dict, loss_dict, analysis_dict=None, id=None, **kwargs):
        """
        Method computing the losses and collecting analysis metrics.

        Args:
            storage_dict (Dict): Storage dictionary containing at least following keys (after matching):
                - qry_feats (FloatTensor): query features of shape [num_qrys, qry_feat_size];
                - batch_ids (LongTensor): batch indices of queries of shape [num_qrys];
                - feat_maps (List): list of size [num_maps] with feature maps of shape [batch_size, feat_size, fH, fW];
                - images (Images): Images structure containing the batched images of size [batch_size];
                - pred_boxes (Boxes): predicted 2D bounding boxes of size [num_qrys];
                - matched_qry_ids (LongTensor): indices of matched queries of shape [num_pos_qrys];
                - matched_tgt_ids (LongTensor): indices of corresponding matched targets of shape [num_pos_qrys].

            tgt_dict (Dict): Target dictionary containing at least following keys:
                - labels (LongTensor): target class indices of shape [num_targets];
                - masks (BoolTensor): target segmentation masks of shape [num_targets, iH, iW].

            loss_dict (Dict): Dictionary containing different weighted loss terms.
            analysis_dict (Dict): Dictionary containing different analysis metrics (default=None).
            id (int or str): Integer or string containing the head id (default=None).
            kwargs (Dict): Dictionary of keyword arguments passed to some underlying modules.

        Returns:
            loss_dict (Dict): Loss dictionary containing following additional key:
                - mask_loss (FloatTensor): mask loss of shape [].

            analysis_dict (Dict): Analysis dictionary containing following additional key (if not None):
                - mask_acc (FloatTensor): mask accuracy of shape [].
        """

        # Retrieve desired items from storage dictionary
        batch_ids = storage_dict['batch_ids']
        feat_maps = storage_dict['feat_maps']
        images = storage_dict['images']
        pred_boxes = storage_dict['pred_boxes']
        matched_qry_ids = storage_dict['matched_qry_ids']
        matched_tgt_ids = storage_dict['matched_tgt_ids']

        # Get device and number of positive matches
        device = matched_qry_ids.device
        num_pos_matches = len(matched_qry_ids)

        # Handle case where there are no positive matches
        if num_pos_matches == 0:

            # Get mask loss
            mask_loss = sum(0.0 * feat_map.flatten()[0] for feat_map in feat_maps)
            mask_loss += sum(0.0 * p.flatten()[0] for p in self.parameters())

            key_name = f'mask_loss_{id}' if id is not None else 'mask_loss'
            loss_dict[key_name] = mask_loss

            # Get mask accuracy if needed
            if analysis_dict is not None:
                mask_acc = 1.0 if len(tgt_dict['masks']) == 0 else 0.0
                mask_acc = torch.tensor(mask_acc, dtype=mask_loss.dtype, device=device)

                key_name = f'mask_acc_{id}' if id is not None else 'mask_acc'
                analysis_dict[key_name] = 100 * mask_acc

            return loss_dict, analysis_dict

        # Get batch indices
        batch_ids = batch_ids[matched_qry_ids]

        # Get RoI boxes
        roi_boxes = pred_boxes[matched_qry_ids].to_format('xyxy')
        roi_boxes = roi_boxes.to_img_scale(images).boxes.detach()
        roi_boxes = torch.cat([batch_ids[:, None], roi_boxes], dim=1)

        # Get mask logits
        roi_feat_maps = feat_maps[:self.mask_roi_extractor.num_inputs]
        roi_feats = self.mask_roi_extractor(roi_feat_maps, roi_boxes)

        if self.pos_enc is not None:
            rH, rW = roi_feats.size()[2:]
            pts_x = torch.linspace(0.5/rW, 1-0.5/rW, steps=rW, device=device)
            pts_y = torch.linspace(0.5/rH, 1-0.5/rH, steps=rH, device=device)

            norm_xy = torch.meshgrid(pts_x, pts_y, indexing='xy')
            norm_xy = torch.stack(norm_xy, dim=2).flatten(0, 1)
            roi_feats = roi_feats + self.pos_enc(norm_xy).t().view(1, -1, rH, rW)

        if self.fuse_qry is not None:
            qry_feats = storage_dict['qry_feats']

            qry_feats = qry_feats[matched_qry_ids]
            qry_feats = self.qry(qry_feats) if self.qry is not None else qry_feats
            qry_feats = qry_feats[:, :, None, None].expand_as(roi_feats)

            fuse_qry_feats = torch.cat([qry_feats, roi_feats], dim=1)
            roi_feats = roi_feats + self.fuse_qry(fuse_qry_feats)

        tgt_labels = tgt_dict['labels'][matched_tgt_ids]
        mask_logits = self.mask_head(roi_feats, feat_maps[0], roi_boxes, tgt_labels)

        # Get mask targets
        tgt_masks = tgt_dict['masks'].cpu().numpy()
        tgt_masks = BitmapMasks(tgt_masks, *tgt_masks.shape[-2:])
        mask_targets = self.mask_head.get_targets([roi_boxes[:, 1:]], [matched_tgt_ids], [tgt_masks])

        # Get mask loss
        mask_loss = self.mask_head.loss(mask_logits, mask_targets)['loss_instance']
        mask_loss = num_pos_matches * mask_loss

        key_name = f'mask_loss_{id}' if id is not None else 'mask_loss'
        loss_dict[key_name] = mask_loss

        # Get mask accuracy if needed
        if analysis_dict is not None:
            mask_logits = torch.cat([mask_logits_i.flatten() for mask_logits_i in mask_logits], dim=0)
            mask_targets = torch.cat([mask_targets_i.flatten() for mask_targets_i in mask_targets], dim=0)

            mask_preds = mask_logits > 0
            mask_targets = mask_targets.bool()
            mask_acc = (mask_preds == mask_targets).sum() / (mask_preds).numel()

            key_name = f'mask_acc_{id}' if id is not None else 'mask_acc'
            analysis_dict[key_name] = 100 * mask_acc

        return loss_dict, analysis_dict
