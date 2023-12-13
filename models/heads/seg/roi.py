"""
Collection of RoI (Region of Interest) heads.
"""

from detectron2.layers import batched_nms
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

        get_segs (bool): Boolean indicating whether to get segmentation predictions.
        seg_type (str): String containing the type of segmentation task.

        dup_attrs (Dict): Optional dictionary specifying the duplicate removal mechanism, possibly containing:
            - type (str): string containing the type of duplicate removal mechanism (mandatory);
            - nms_candidates (int): integer containing the maximum number of candidate detections retained before NMS;
            - nms_thr (float): value of IoU threshold used during NMS to remove duplicate detections.

        max_segs (int): Optional integer with the maximum number of returned segmentation predictions.

        pan_post_attrs (Dict): Dictionary specifying the panoptic post-processing mechanism possibly containing:
            - score_thr (float): value containing the instance score threshold (or None);
            - nms_thr (float): value containing the IoU threshold used during mask IoU (or None);
            - pan_mask_thr (float): value containing the normalized panoptic segmentation mask threshold;
            - ins_pan_thr (float): value containing the IoU threshold between instance and panoptic masks;
            - area_thr (int): integer containing the mask area threshold (or None).

        metadata (detectron2.data.Metadata): Metadata instance containing additional dataset information.
        matcher (nn.Module): Optional matcher module matching predictions with targets.
        apply_ids (List): List with integers determining when the head should be applied.
    """

    def __init__(self, metadata, pos_enc_cfg=None, qry_cfg=None, fuse_qry_cfg=None, get_segs=True, seg_type='instance',
                 dup_attrs=None, max_segs=None, pan_post_attrs=None, matcher_cfg=None, apply_ids=None, **kwargs):
        """
        Initializes the StandardRoIHead module.

        Args:
            metadata (detectron2.data.Metadata): Metadata instance containing additional dataset information.
            pos_enc_cfg (Dict): Configuration dictionary specifying the position encoder module (default=None).
            qry_cfg (Dict): Configuration dictionary specifying the query module (default=None).
            fuse_qry_cfg (Dict): Configuration dictionary specifying the fuse query module (default=None).
            get_segs (bool): Boolean indicating whether to get segmentation predictions (default=True).
            seg_type (str): String containing the type of segmentation task (default='instance').
            dup_attrs (Dict): Attribute dictionary specifying the duplicate removal mechanism (default=None).
            max_segs (int): Integer with the maximum number of returned segmentation predictions (default=None).
            pan_post_attrs (Dict): Attribute dictionary specifying the panoptic post-processing (default=None).
            matcher_cfg (Dict): Configuration dictionary specifying the matcher module (default=None).
            apply_ids (List): List with integers determining when the head should be applied (default=None).
            kwargs (Dict): Dictionary of keyword arguments passed to the parent __init__ method.
        """

        # Initialize module using parent __init__ method
        MMDetStandardRoIHead.__init__(self, **kwargs)

        # Build position encoder, query and fuse query modules
        self.pos_enc = build_model(pos_enc_cfg) if pos_enc_cfg is not None else None
        self.qry = build_model(qry_cfg) if qry_cfg is not None else None
        self.fuse_qry = build_model(fuse_qry_cfg) if fuse_qry_cfg is not None else None

        # Build matcher module if needed
        self.matcher = build_model(matcher_cfg) if matcher_cfg is not None else None

        # Set additional attributes
        self.get_segs = get_segs
        self.dup_attrs = dup_attrs
        self.max_segs = max_segs
        self.seg_type = seg_type
        self.pan_post_attrs = pan_post_attrs if pan_post_attrs is not None else dict()
        self.metadata = metadata
        self.apply_ids = apply_ids

    @torch.no_grad()
    def compute_segs(self, storage_dict, pred_dicts, **kwargs):
        """
        Method computing the segmentation predictions.

        Args:
            storage_dict (Dict): Storage dictionary containing at least following keys:
                - qry_feats (FloatTensor): query features of shape [num_qrys, qry_feat_size];
                - batch_ids (LongTensor): batch indices of queries of shape [num_qrys];
                - feat_maps (List): list of size [num_maps] with feature maps of shape [batch_size, feat_size, fH, fW];
                - images (Images): Images structure containing the batched images of size [batch_size];
                - cls_logits (FloatTensor): classification logits of shape [num_qrys, num_labels];
                - pred_boxes (Boxes): predicted 2D bounding boxes of size [num_qrys].

            pred_dicts (List): List of size [num_pred_dicts] collecting various prediction dictionaries.
            kwargs (Dict): Dictionary of unused keyword arguments.

        Returns:
            pred_dicts (List): List with prediction dictionaries containing following additional entry:
                pred_dict (Dict): Prediction dictionary containing following keys:
                    - labels (LongTensor): predicted class indices of shape [num_preds];
                    - masks (BoolTensor): predicted segmentation masks of shape [num_preds, iH, iW];
                    - scores (FloatTensor): normalized prediction scores of shape [num_preds];
                    - batch_ids (LongTensor): batch indices of predictions of shape [num_preds].

        Raises:
            ValueError: Error when an invalid type of duplicate removal mechanism is provided.
            ValueError: Error when an invalid type of segmentation task if provided.
        """

        # Retrieve desired items from storage dictionary
        batch_ids = storage_dict['batch_ids']
        feat_maps = storage_dict['feat_maps']
        images = storage_dict['images']
        cls_logits = storage_dict['cls_logits']
        pred_boxes = storage_dict['pred_boxes']

        # Get RoI feat maps
        roi_feat_maps = feat_maps[:self.mask_roi_extractor.num_inputs]

        # Get image width and height with padding
        iW, iH = images.size()

        # Get number of queries and number of classes
        num_qrys = cls_logits.size(dim=0)
        num_classes = cls_logits.size(dim=1) - 1

        # Get batch size and device
        batch_size = len(images)
        device = cls_logits.device

        # Get query and batch indices
        qry_ids = torch.arange(num_qrys, device=device)

        if self.seg_type == 'instance':
            qry_ids = qry_ids[:, None].expand(-1, num_classes).reshape(-1)
            batch_ids = batch_ids[:, None].expand(-1, num_classes).reshape(-1)

        # Get prediction labels and scores
        if self.seg_type == 'instance':
            pred_labels = torch.arange(num_classes, device=device)[None, :].expand(num_qrys, -1).reshape(-1)
            pred_scores = cls_logits[:, :-1].sigmoid().view(-1)

        else:
            pred_scores, pred_labels = cls_logits[:, :-1].sigmoid().max(dim=1)

        # Get thing indices if needed
        if self.seg_type == 'panoptic':
            thing_ids = tuple(self.metadata.thing_dataset_id_to_contiguous_id.values())
            thing_ids = torch.as_tensor(thing_ids, device=device)

        # Initialize prediction dictionary
        pred_keys = ('labels', 'masks', 'scores', 'batch_ids')
        pred_dict = {pred_key: [] for pred_key in pred_keys}

        # Iterate over every batch entry
        for i in range(batch_size):

            # Get predictions corresponding to batch entry
            batch_mask = batch_ids == i

            qry_ids_i = qry_ids[batch_mask]
            pred_labels_i = pred_labels[batch_mask]
            pred_scores_i = pred_scores[batch_mask]

            # Remove duplicate predictions if needed
            if self.dup_attrs is not None:
                dup_removal_type = self.dup_attrs['type']

                if dup_removal_type == 'nms':
                    num_candidates = self.dup_attrs['nms_candidates']
                    num_preds_i = len(pred_scores_i)
                    candidate_ids = pred_scores_i.topk(min(num_candidates, num_preds_i))[1]

                    qry_ids_i = qry_ids_i[candidate_ids]
                    pred_labels_i = pred_labels_i[candidate_ids]
                    pred_scores_i = pred_scores_i[candidate_ids]

                    pred_boxes_i = pred_boxes[qry_ids_i].to_format('xyxy')
                    pred_boxes_i = pred_boxes_i.to_img_scale(images).boxes

                    iou_thr = self.dup_attrs['nms_thr']
                    non_dup_ids = batched_nms(pred_boxes_i, pred_scores_i, pred_labels_i, iou_thr)

                    qry_ids_i = qry_ids_i[non_dup_ids]
                    pred_labels_i = pred_labels_i[non_dup_ids]
                    pred_boxes_i = pred_boxes_i[non_dup_ids]
                    pred_scores_i = pred_scores_i[non_dup_ids]

                else:
                    error_msg = f"Invalid type of duplicate removal mechanism (got '{dup_removal_type}')."
                    raise ValueError(error_msg)

            # Only keep top predictions if needed
            if self.max_segs is not None:
                if len(pred_scores_i) > self.max_segs:
                    top_pred_ids = pred_scores_i.topk(self.max_segs)[1]

                    qry_ids_i = qry_ids_i[top_pred_ids]
                    pred_labels_i = pred_labels_i[top_pred_ids]
                    pred_boxes_i = pred_boxes_i[top_pred_ids]
                    pred_scores_i = pred_scores_i[top_pred_ids]

            # Get mask scores and instance segmentation masks
            batch_ids_i = torch.full_like(pred_labels_i, i)
            roi_boxes_i = torch.cat([batch_ids_i[:, None], pred_boxes_i], dim=1)
            roi_feats_i = self.mask_roi_extractor(roi_feat_maps, roi_boxes_i)

            if self.pos_enc is not None:
                rH, rW = roi_feats_i.size()[2:]
                pts_x = torch.linspace(0.5/rW, 1-0.5/rW, steps=rW, device=device)
                pts_y = torch.linspace(0.5/rH, 1-0.5/rH, steps=rH, device=device)

                norm_xy = torch.meshgrid(pts_x, pts_y, indexing='xy')
                norm_xy = torch.stack(norm_xy, dim=2).flatten(0, 1)
                roi_feats_i = roi_feats_i + self.pos_enc(norm_xy).t().view(1, -1, rH, rW)

            if self.fuse_qry is not None:
                qry_feats = storage_dict['qry_feats']

                qry_feats_i = qry_feats[qry_ids_i]
                qry_feats_i = self.qry(qry_feats_i) if self.qry is not None else qry_feats_i
                qry_feats_i = qry_feats_i[:, :, None, None].expand_as(roi_feats_i)

                fuse_qry_feats_i = torch.cat([qry_feats_i, roi_feats_i], dim=1)
                roi_feats_i = roi_feats_i + self.fuse_qry(fuse_qry_feats_i)

            mask_logits_i = self.mask_head(roi_feats_i)
            mask_logits_i = mask_logits_i[range(len(mask_logits_i)), pred_labels_i]
            mask_logits_i = mask_logits_i[:, None]

            mask_scores_i = mask_logits_i.sigmoid()
            mask_scores_i = _do_paste_mask(mask_scores_i, pred_boxes_i, iH, iW, skip_empty=False)[0]
            ins_seg_masks = mask_scores_i > 0.5

            # Update prediction scores based on mask scores if needed
            if self.seg_type == 'instance':
                pred_scores_i = pred_scores_i * (ins_seg_masks * mask_scores_i).flatten(1).sum(dim=1)
                pred_scores_i = pred_scores_i / (ins_seg_masks.flatten(1).sum(dim=1) + 1e-6)

            # Perform panoptic post-processing if needed
            if self.seg_type == 'panoptic':

                # Filter based on score if needed
                score_thr = self.pan_post_attrs.get('score_thr', None)

                if score_thr is not None:
                    keep_mask = pred_scores_i > score_thr

                    pred_labels_i = pred_labels_i[keep_mask]
                    mask_scores_i = mask_scores_i[keep_mask]
                    pred_scores_i = pred_scores_i[keep_mask]
                    ins_seg_masks = ins_seg_masks[keep_mask]

                # Filter based on mask NMS if needed
                nms_thr = self.pan_post_attrs.get('nms_thr', None)

                if nms_thr is not None:
                    pred_scores_i, sort_ids = pred_scores_i.sort(descending=True)

                    pred_labels_i = pred_labels_i[sort_ids]
                    mask_scores_i = mask_scores_i[sort_ids]
                    ins_seg_masks = ins_seg_masks[sort_ids]

                    num_preds = len(ins_seg_masks)
                    flat_masks = ins_seg_masks.flatten(1)
                    inter = torch.zeros(num_preds, num_preds, dtype=torch.float, device=device)

                    for j in range(1, num_preds):
                        inter[j, :j] = (flat_masks[j, None, :] * flat_masks[None, :j, :]).sum(dim=2)

                    areas = flat_masks.sum(dim=1)
                    union = areas[:, None] + areas[None, :] - inter

                    ious = inter / union
                    keep_mask = ious.amax(dim=1) < nms_thr

                    pred_labels_i = pred_labels_i[keep_mask]
                    mask_scores_i = mask_scores_i[keep_mask]
                    pred_scores_i = pred_scores_i[keep_mask]
                    ins_seg_masks = ins_seg_masks[keep_mask]

                # Get panoptic segmentation masks
                rel_mask_scores = pred_scores_i[:, None, None] * mask_scores_i
                num_preds = len(rel_mask_scores)

                if num_preds > 0:
                    pan_seg_mask = rel_mask_scores.argmax(dim=0)
                else:
                    pan_seg_mask = ins_seg_masks.new_zeros([*rel_mask_scores.size()[1:]])

                pred_ids = torch.arange(num_preds, device=device)
                pan_seg_masks = pan_seg_mask[None, :, :] == pred_ids[:, None, None]

                # Apply panoptic mask threshold if needed
                pan_mask_thr = self.pan_post_attrs.get('pan_mask_thr', None)

                if pan_mask_thr is not None:
                    pan_seg_masks &= mask_scores_i > pan_mask_thr

                # Filter based on instance-panoptic IoU if needed
                ins_pan_thr = self.pan_post_attrs.get('ins_pan_thr', None)

                if ins_pan_thr is not None:
                    ins_flat_masks = ins_seg_masks.flatten(1)
                    pan_flat_masks = pan_seg_masks.flatten(1)

                    ins_areas = ins_flat_masks.sum(dim=1)
                    pan_areas = pan_flat_masks.sum(dim=1)

                    inter = (ins_flat_masks * pan_flat_masks).sum(dim=1)
                    union = ins_areas + pan_areas - inter

                    ious = inter / union
                    keep_mask = ious > ins_pan_thr

                    pred_labels_i = pred_labels_i[keep_mask]
                    pred_scores_i = pred_scores_i[keep_mask]
                    pan_seg_masks = pan_seg_masks[keep_mask]

                # Filter based on mask area if needed
                area_thr = self.pan_post_attrs.get('area_thr', None)

                if area_thr is not None:
                    areas = pan_seg_masks.flatten(1).sum(dim=1)
                    keep_mask = areas > area_thr

                    pred_labels_i = pred_labels_i[keep_mask]
                    pred_scores_i = pred_scores_i[keep_mask]
                    pan_seg_masks = pan_seg_masks[keep_mask]

                # Merge stuff predictions
                thing_mask = (pred_labels_i[:, None] == thing_ids[None, :]).any(dim=1)
                stuff_mask = ~thing_mask

                stuff_labels = pred_labels_i[stuff_mask]
                stuff_labels, stuff_ids, stuff_counts = stuff_labels.unique(return_inverse=True, return_counts=True)
                pred_labels_i = torch.cat([pred_labels_i[thing_mask], stuff_labels], dim=0)

                stuff_scores = torch.zeros_like(stuff_labels, dtype=torch.float)
                stuff_scores.scatter_add_(dim=0, index=stuff_ids, src=pred_scores_i[stuff_mask])
                stuff_scores = stuff_scores / stuff_counts
                pred_scores_i = torch.cat([pred_scores_i[thing_mask], stuff_scores], dim=0)

                stuff_ids = stuff_ids[:, None, None].expand(-1, iH, iW)
                unmerged_stuff_masks = pan_seg_masks[stuff_mask]

                num_stuff_preds = len(stuff_labels)
                stuff_seg_masks = pan_seg_masks.new_zeros([num_stuff_preds, iH, iW])
                stuff_seg_masks.scatter_add_(dim=0, index=stuff_ids, src=unmerged_stuff_masks)

                thing_seg_masks = pan_seg_masks[thing_mask]
                pan_seg_masks = torch.cat([thing_seg_masks, stuff_seg_masks], dim=0)

            # Add predictions to prediction dictionary
            pred_dict['labels'].append(pred_labels_i)
            pred_dict['scores'].append(pred_scores_i)
            pred_dict['batch_ids'].append(torch.full_like(pred_labels_i, i))

            if self.seg_type == 'instance':
                pred_dict['masks'].append(ins_seg_masks)

            elif self.seg_type == 'panoptic':
                pred_dict['masks'].append(pan_seg_masks)

            else:
                error_msg = f"Invalid type of segmentation task (got '{self.seg_type}')."
                raise ValueError(error_msg)

        # Concatenate predictions of different batch entries
        pred_dict.update({k: torch.cat(v, dim=0) for k, v in pred_dict.items()})

        # Add prediction dictionary to list of prediction dictionaries
        pred_dicts.append(pred_dict)

        return pred_dicts

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

    def forward_loss(self, storage_dict, tgt_dict, loss_dict, analysis_dict=None, id=None, **kwargs):
        """
        Forward loss method of the StandardRoIHead module.

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
            analysis_dict (Dict): Dictionary containing different analyses (default=None).
            id (int or str): Integer or string containing the head id (default=None).
            kwargs (Dict): Dictionary of keyword arguments passed to some underlying modules.

        Returns:
            loss_dict (Dict): Loss dictionary containing following additional key:
                - mask_loss (FloatTensor): mask loss of shape [].

            analysis_dict (Dict): Analysis dictionary containing following additional key (if not None):
                - mask_acc (FloatTensor): mask accuracy of shape [].
        """

        # Perform matching if matcher is available
        if self.matcher is not None:
            self.matcher(storage_dict, tgt_dict=tgt_dict, analysis_dict=analysis_dict, **kwargs)

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

    @torch.no_grad()
    def compute_segs(self, storage_dict, pred_dicts, **kwargs):
        """
        Method computing the segmentation predictions.

        Args:
            storage_dict (Dict): Storage dictionary containing at least following keys:
                - qry_feats (FloatTensor): query features of shape [num_qrys, qry_feat_size];
                - batch_ids (LongTensor): batch indices of queries of shape [num_qrys];
                - feat_maps (List): list of size [num_maps] with feature maps of shape [batch_size, feat_size, fH, fW];
                - images (Images): Images structure containing the batched images of size [batch_size];
                - cls_logits (FloatTensor): classification logits of shape [num_qrys, num_labels];
                - pred_boxes (Boxes): predicted 2D bounding boxes of size [num_qrys].

            pred_dicts (List): List of size [num_pred_dicts] collecting various prediction dictionaries.
            kwargs (Dict): Dictionary of unused keyword arguments.

        Returns:
            pred_dicts (List): List with prediction dictionaries containing following additional entry:
                pred_dict (Dict): Prediction dictionary containing following keys:
                    - labels (LongTensor): predicted class indices of shape [num_preds];
                    - masks (BoolTensor): predicted segmentation masks of shape [num_preds, iH, iW];
                    - scores (FloatTensor): normalized prediction scores of shape [num_preds];
                    - batch_ids (LongTensor): batch indices of predictions of shape [num_preds].

        Raises:
            ValueError: Error when an invalid type of duplicate removal mechanism is provided.
            ValueError: Error when an invalid type of segmentation task if provided.
        """

        # Retrieve desired items from storage dictionary
        batch_ids = storage_dict['batch_ids']
        feat_maps = storage_dict['feat_maps']
        images = storage_dict['images']
        cls_logits = storage_dict['cls_logits']
        pred_boxes = storage_dict['pred_boxes']

        # Get RoI feat maps
        roi_feat_maps = feat_maps[:self.mask_roi_extractor.num_inputs]

        # Get image width and height with padding
        iW, iH = images.size()

        # Get number of queries and number of classes
        num_qrys = cls_logits.size(dim=0)
        num_classes = cls_logits.size(dim=1) - 1

        # Get batch size and device
        batch_size = len(images)
        device = cls_logits.device

        # Get query and batch indices
        qry_ids = torch.arange(num_qrys, device=device)

        if self.seg_type == 'instance':
            qry_ids = qry_ids[:, None].expand(-1, num_classes).reshape(-1)
            batch_ids = batch_ids[:, None].expand(-1, num_classes).reshape(-1)

        # Get prediction labels and scores
        if self.seg_type == 'instance':
            pred_labels = torch.arange(num_classes, device=device)[None, :].expand(num_qrys, -1).reshape(-1)
            pred_scores = cls_logits[:, :-1].sigmoid().view(-1)

        else:
            pred_scores, pred_labels = cls_logits[:, :-1].sigmoid().max(dim=1)

        # Get thing indices if needed
        if self.seg_type == 'panoptic':
            thing_ids = tuple(self.metadata.thing_dataset_id_to_contiguous_id.values())
            thing_ids = torch.as_tensor(thing_ids, device=device)

        # Initialize prediction dictionary
        pred_keys = ('labels', 'masks', 'scores', 'batch_ids')
        pred_dict = {pred_key: [] for pred_key in pred_keys}

        # Iterate over every batch entry
        for i in range(batch_size):

            # Get predictions corresponding to batch entry
            batch_mask = batch_ids == i

            qry_ids_i = qry_ids[batch_mask]
            pred_labels_i = pred_labels[batch_mask]
            pred_scores_i = pred_scores[batch_mask]

            # Remove duplicate predictions if needed
            if self.dup_attrs is not None:
                dup_removal_type = self.dup_attrs['type']

                if dup_removal_type == 'nms':
                    num_candidates = self.dup_attrs['nms_candidates']
                    num_preds_i = len(pred_scores_i)
                    candidate_ids = pred_scores_i.topk(min(num_candidates, num_preds_i))[1]

                    qry_ids_i = qry_ids_i[candidate_ids]
                    pred_labels_i = pred_labels_i[candidate_ids]
                    pred_scores_i = pred_scores_i[candidate_ids]

                    pred_boxes_i = pred_boxes[qry_ids_i].to_format('xyxy')
                    pred_boxes_i = pred_boxes_i.to_img_scale(images).boxes

                    iou_thr = self.dup_attrs['nms_thr']
                    non_dup_ids = batched_nms(pred_boxes_i, pred_scores_i, pred_labels_i, iou_thr)

                    qry_ids_i = qry_ids_i[non_dup_ids]
                    pred_labels_i = pred_labels_i[non_dup_ids]
                    pred_boxes_i = pred_boxes_i[non_dup_ids]
                    pred_scores_i = pred_scores_i[non_dup_ids]

                else:
                    error_msg = f"Invalid type of duplicate removal mechanism (got '{dup_removal_type}')."
                    raise ValueError(error_msg)

            # Only keep top predictions if needed
            if self.max_segs is not None:
                if len(pred_scores_i) > self.max_segs:
                    top_pred_ids = pred_scores_i.topk(self.max_segs)[1]

                    qry_ids_i = qry_ids_i[top_pred_ids]
                    pred_labels_i = pred_labels_i[top_pred_ids]
                    pred_boxes_i = pred_boxes_i[top_pred_ids]
                    pred_scores_i = pred_scores_i[top_pred_ids]

            # Get mask scores and instance segmentation masks
            batch_ids_i = torch.full_like(pred_labels_i, i)
            roi_boxes_i = torch.cat([batch_ids_i[:, None], pred_boxes_i], dim=1)
            roi_feats_i = self.mask_roi_extractor(roi_feat_maps, roi_boxes_i)

            if self.pos_enc is not None:
                rH, rW = roi_feats_i.size()[2:]
                pts_x = torch.linspace(0.5/rW, 1-0.5/rW, steps=rW, device=device)
                pts_y = torch.linspace(0.5/rH, 1-0.5/rH, steps=rH, device=device)

                norm_xy = torch.meshgrid(pts_x, pts_y, indexing='xy')
                norm_xy = torch.stack(norm_xy, dim=2).flatten(0, 1)
                roi_feats_i = roi_feats_i + self.pos_enc(norm_xy).t().view(1, -1, rH, rW)

            if self.fuse_qry is not None:
                qry_feats = storage_dict['qry_feats']

                qry_feats_i = qry_feats[qry_ids_i]
                qry_feats_i = self.qry(qry_feats_i) if self.qry is not None else qry_feats_i
                qry_feats_i = qry_feats_i[:, :, None, None].expand_as(roi_feats_i)

                fuse_qry_feats_i = torch.cat([qry_feats_i, roi_feats_i], dim=1)
                roi_feats_i = roi_feats_i + self.fuse_qry(fuse_qry_feats_i)

            mask_logits_i = self.mask_head(roi_feats_i)
            mask_logits_i = self._mask_point_forward_test(feat_maps, roi_boxes_i, pred_labels_i, mask_logits_i)

            mask_logits_i = mask_logits_i[range(len(mask_logits_i)), pred_labels_i]
            mask_logits_i = mask_logits_i[:, None]

            mask_scores_i = mask_logits_i.sigmoid()
            mask_scores_i = _do_paste_mask(mask_scores_i, pred_boxes_i, iH, iW, skip_empty=False)[0]
            ins_seg_masks = mask_scores_i > 0.5

            # Update prediction scores based on mask scores if needed
            if self.seg_type == 'instance':
                pred_scores_i = pred_scores_i * (ins_seg_masks * mask_scores_i).flatten(1).sum(dim=1)
                pred_scores_i = pred_scores_i / (ins_seg_masks.flatten(1).sum(dim=1) + 1e-6)

            # Perform panoptic post-processing if needed
            if self.seg_type == 'panoptic':

                # Filter based on score if needed
                score_thr = self.pan_post_attrs.get('score_thr', None)

                if score_thr is not None:
                    keep_mask = pred_scores_i > score_thr

                    pred_labels_i = pred_labels_i[keep_mask]
                    mask_scores_i = mask_scores_i[keep_mask]
                    pred_scores_i = pred_scores_i[keep_mask]
                    ins_seg_masks = ins_seg_masks[keep_mask]

                # Filter based on mask NMS if needed
                nms_thr = self.pan_post_attrs.get('nms_thr', None)

                if nms_thr is not None:
                    pred_scores_i, sort_ids = pred_scores_i.sort(descending=True)

                    pred_labels_i = pred_labels_i[sort_ids]
                    mask_scores_i = mask_scores_i[sort_ids]
                    ins_seg_masks = ins_seg_masks[sort_ids]

                    num_preds = len(ins_seg_masks)
                    flat_masks = ins_seg_masks.flatten(1)
                    inter = torch.zeros(num_preds, num_preds, dtype=torch.float, device=device)

                    for j in range(1, num_preds):
                        inter[j, :j] = (flat_masks[j, None, :] * flat_masks[None, :j, :]).sum(dim=2)

                    areas = flat_masks.sum(dim=1)
                    union = areas[:, None] + areas[None, :] - inter

                    ious = inter / union
                    keep_mask = ious.amax(dim=1) < nms_thr

                    pred_labels_i = pred_labels_i[keep_mask]
                    mask_scores_i = mask_scores_i[keep_mask]
                    pred_scores_i = pred_scores_i[keep_mask]
                    ins_seg_masks = ins_seg_masks[keep_mask]

                # Get panoptic segmentation masks
                rel_mask_scores = pred_scores_i[:, None, None] * mask_scores_i
                num_preds = len(rel_mask_scores)

                if num_preds > 0:
                    pan_seg_mask = rel_mask_scores.argmax(dim=0)
                else:
                    pan_seg_mask = ins_seg_masks.new_zeros([*rel_mask_scores.size()[1:]])

                pred_ids = torch.arange(num_preds, device=device)
                pan_seg_masks = pan_seg_mask[None, :, :] == pred_ids[:, None, None]

                # Apply panoptic mask threshold if needed
                pan_mask_thr = self.pan_post_attrs.get('pan_mask_thr', None)

                if pan_mask_thr is not None:
                    pan_seg_masks &= mask_scores_i > pan_mask_thr

                # Filter based on instance-panoptic IoU if needed
                ins_pan_thr = self.pan_post_attrs.get('ins_pan_thr', None)

                if ins_pan_thr is not None:
                    ins_flat_masks = ins_seg_masks.flatten(1)
                    pan_flat_masks = pan_seg_masks.flatten(1)

                    ins_areas = ins_flat_masks.sum(dim=1)
                    pan_areas = pan_flat_masks.sum(dim=1)

                    inter = (ins_flat_masks * pan_flat_masks).sum(dim=1)
                    union = ins_areas + pan_areas - inter

                    ious = inter / union
                    keep_mask = ious > ins_pan_thr

                    pred_labels_i = pred_labels_i[keep_mask]
                    pred_scores_i = pred_scores_i[keep_mask]
                    pan_seg_masks = pan_seg_masks[keep_mask]

                # Filter based on mask area if needed
                area_thr = self.pan_post_attrs.get('area_thr', None)

                if area_thr is not None:
                    areas = pan_seg_masks.flatten(1).sum(dim=1)
                    keep_mask = areas > area_thr

                    pred_labels_i = pred_labels_i[keep_mask]
                    pred_scores_i = pred_scores_i[keep_mask]
                    pan_seg_masks = pan_seg_masks[keep_mask]

                # Merge stuff predictions
                thing_mask = (pred_labels_i[:, None] == thing_ids[None, :]).any(dim=1)
                stuff_mask = ~thing_mask

                stuff_labels = pred_labels_i[stuff_mask]
                stuff_labels, stuff_ids, stuff_counts = stuff_labels.unique(return_inverse=True, return_counts=True)
                pred_labels_i = torch.cat([pred_labels_i[thing_mask], stuff_labels], dim=0)

                stuff_scores = torch.zeros_like(stuff_labels, dtype=torch.float)
                stuff_scores.scatter_add_(dim=0, index=stuff_ids, src=pred_scores_i[stuff_mask])
                stuff_scores = stuff_scores / stuff_counts
                pred_scores_i = torch.cat([pred_scores_i[thing_mask], stuff_scores], dim=0)

                stuff_ids = stuff_ids[:, None, None].expand(-1, iH, iW)
                unmerged_stuff_masks = pan_seg_masks[stuff_mask]

                num_stuff_preds = len(stuff_labels)
                stuff_seg_masks = pan_seg_masks.new_zeros([num_stuff_preds, iH, iW])
                stuff_seg_masks.scatter_add_(dim=0, index=stuff_ids, src=unmerged_stuff_masks)

                thing_seg_masks = pan_seg_masks[thing_mask]
                pan_seg_masks = torch.cat([thing_seg_masks, stuff_seg_masks], dim=0)

            # Add predictions to prediction dictionary
            pred_dict['labels'].append(pred_labels_i)
            pred_dict['scores'].append(pred_scores_i)
            pred_dict['batch_ids'].append(torch.full_like(pred_labels_i, i))

            if self.seg_type == 'instance':
                pred_dict['masks'].append(ins_seg_masks)

            elif self.seg_type == 'panoptic':
                pred_dict['masks'].append(pan_seg_masks)

            else:
                error_msg = f"Invalid type of segmentation task (got '{self.seg_type}')."
                raise ValueError(error_msg)

        # Concatenate predictions of different batch entries
        pred_dict.update({k: torch.cat(v, dim=0) for k, v in pred_dict.items()})

        # Add prediction dictionary to list of prediction dictionaries
        pred_dicts.append(pred_dict)

        return pred_dicts

    def forward_loss(self, storage_dict, tgt_dict, loss_dict, analysis_dict=None, id=None, **kwargs):
        """
        Forward loss method of the PointRendRoIHead module.

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
            analysis_dict (Dict): Dictionary containing different analyses (default=None).
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

        # Perform matching if matcher is available
        if self.matcher is not None:
            self.matcher(storage_dict, tgt_dict=tgt_dict, analysis_dict=analysis_dict, **kwargs)

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

    @torch.no_grad()
    def compute_segs(self, storage_dict, pred_dicts, **kwargs):
        """
        Method computing the segmentation predictions.

        Args:
            storage_dict (Dict): Storage dictionary containing at least following keys:
                - qry_feats (FloatTensor): query features of shape [num_qrys, qry_feat_size];
                - batch_ids (LongTensor): batch indices of queries of shape [num_qrys];
                - feat_maps (List): list of size [num_maps] with feature maps of shape [batch_size, feat_size, fH, fW];
                - images (Images): Images structure containing the batched images of size [batch_size];
                - cls_logits (FloatTensor): classification logits of shape [num_qrys, num_labels];
                - pred_boxes (Boxes): predicted 2D bounding boxes of size [num_qrys].

            pred_dicts (List): List of size [num_pred_dicts] collecting various prediction dictionaries.
            kwargs (Dict): Dictionary of unused keyword arguments.

        Returns:
            pred_dicts (List): List with prediction dictionaries containing following additional entry:
                pred_dict (Dict): Prediction dictionary containing following keys:
                    - labels (LongTensor): predicted class indices of shape [num_preds];
                    - masks (BoolTensor): predicted segmentation masks of shape [num_preds, iH, iW];
                    - scores (FloatTensor): normalized prediction scores of shape [num_preds];
                    - batch_ids (LongTensor): batch indices of predictions of shape [num_preds].

        Raises:
            ValueError: Error when an invalid type of duplicate removal mechanism is provided.
            ValueError: Error when an invalid type of segmentation task if provided.
        """

        # Retrieve desired items from storage dictionary
        batch_ids = storage_dict['batch_ids']
        feat_maps = storage_dict['feat_maps']
        images = storage_dict['images']
        cls_logits = storage_dict['cls_logits']
        pred_boxes = storage_dict['pred_boxes']

        # Get RoI feat maps
        roi_feat_maps = feat_maps[:self.mask_roi_extractor.num_inputs]

        # Get image width and height with padding
        iW, iH = images.size()

        # Get number of queries and number of classes
        num_qrys = cls_logits.size(dim=0)
        num_classes = cls_logits.size(dim=1) - 1

        # Get batch size and device
        batch_size = len(images)
        device = cls_logits.device

        # Get query and batch indices
        qry_ids = torch.arange(num_qrys, device=device)

        if self.seg_type == 'instance':
            qry_ids = qry_ids[:, None].expand(-1, num_classes).reshape(-1)
            batch_ids = batch_ids[:, None].expand(-1, num_classes).reshape(-1)

        # Get prediction labels and scores
        if self.seg_type == 'instance':
            pred_labels = torch.arange(num_classes, device=device)[None, :].expand(num_qrys, -1).reshape(-1)
            pred_scores = cls_logits[:, :-1].sigmoid().view(-1)

        else:
            pred_scores, pred_labels = cls_logits[:, :-1].sigmoid().max(dim=1)

        # Get number of stages and interpolation keyword arguments
        num_stages = len(self.mask_head.stage_sup_size)
        itp_kwargs = {'mode': 'bilinear', 'align_corners': True}

        # Get thing indices if needed
        if self.seg_type == 'panoptic':
            thing_ids = tuple(self.metadata.thing_dataset_id_to_contiguous_id.values())
            thing_ids = torch.as_tensor(thing_ids, device=device)

        # Initialize prediction dictionary
        pred_keys = ('labels', 'masks', 'scores', 'batch_ids')
        pred_dict = {pred_key: [] for pred_key in pred_keys}

        # Iterate over every batch entry
        for i in range(batch_size):

            # Get predictions corresponding to batch entry
            batch_mask = batch_ids == i

            qry_ids_i = qry_ids[batch_mask]
            pred_labels_i = pred_labels[batch_mask]
            pred_scores_i = pred_scores[batch_mask]

            # Remove duplicate predictions if needed
            if self.dup_attrs is not None:
                dup_removal_type = self.dup_attrs['type']

                if dup_removal_type == 'nms':
                    num_candidates = self.dup_attrs['nms_candidates']
                    num_preds_i = len(pred_scores_i)
                    candidate_ids = pred_scores_i.topk(min(num_candidates, num_preds_i))[1]

                    qry_ids_i = qry_ids_i[candidate_ids]
                    pred_labels_i = pred_labels_i[candidate_ids]
                    pred_scores_i = pred_scores_i[candidate_ids]

                    pred_boxes_i = pred_boxes[qry_ids_i].to_format('xyxy')
                    pred_boxes_i = pred_boxes_i.to_img_scale(images).boxes

                    iou_thr = self.dup_attrs['nms_thr']
                    non_dup_ids = batched_nms(pred_boxes_i, pred_scores_i, pred_labels_i, iou_thr)

                    qry_ids_i = qry_ids_i[non_dup_ids]
                    pred_labels_i = pred_labels_i[non_dup_ids]
                    pred_boxes_i = pred_boxes_i[non_dup_ids]
                    pred_scores_i = pred_scores_i[non_dup_ids]

                else:
                    error_msg = f"Invalid type of duplicate removal mechanism (got '{dup_removal_type}')."
                    raise ValueError(error_msg)

            # Only keep top predictions if needed
            if self.max_segs is not None:
                if len(pred_scores_i) > self.max_segs:
                    top_pred_ids = pred_scores_i.topk(self.max_segs)[1]

                    qry_ids_i = qry_ids_i[top_pred_ids]
                    pred_labels_i = pred_labels_i[top_pred_ids]
                    pred_boxes_i = pred_boxes_i[top_pred_ids]
                    pred_scores_i = pred_scores_i[top_pred_ids]

            # Get mask scores and instance segmentation masks
            batch_ids_i = torch.full_like(pred_labels_i, i)
            roi_boxes_i = torch.cat([batch_ids_i[:, None], pred_boxes_i], dim=1)
            roi_feats_i = self.mask_roi_extractor(roi_feat_maps, roi_boxes_i)

            if self.pos_enc is not None:
                rH, rW = roi_feats_i.size()[2:]
                pts_x = torch.linspace(0.5/rW, 1-0.5/rW, steps=rW, device=device)
                pts_y = torch.linspace(0.5/rH, 1-0.5/rH, steps=rH, device=device)

                norm_xy = torch.meshgrid(pts_x, pts_y, indexing='xy')
                norm_xy = torch.stack(norm_xy, dim=2).flatten(0, 1)
                roi_feats_i = roi_feats_i + self.pos_enc(norm_xy).t().view(1, -1, rH, rW)

            if self.fuse_qry is not None:
                qry_feats = storage_dict['qry_feats']

                qry_feats_i = qry_feats[qry_ids_i]
                qry_feats_i = self.qry(qry_feats_i) if self.qry is not None else qry_feats_i
                qry_feats_i = qry_feats_i[:, :, None, None].expand_as(roi_feats_i)

                fuse_qry_feats_i = torch.cat([qry_feats_i, roi_feats_i], dim=1)
                roi_feats_i = roi_feats_i + self.fuse_qry(fuse_qry_feats_i)

            mask_logits_i = self.mask_head(roi_feats_i, feat_maps[0], roi_boxes_i, pred_labels_i)

            for j in range(1, num_stages-1):
                mask_logits_ij = mask_logits_i[j]
                pred_masks_ij = mask_logits_ij.squeeze(dim=1).sigmoid() >= 0.5
                next_shape = mask_logits_i[j+1].shape[-2:]

                non_boundary_mask = generate_block_target(pred_masks_ij, boundary_width=1) != 1
                non_boundary_mask = non_boundary_mask.unsqueeze(dim=1).float()
                non_boundary_mask = F.interpolate(non_boundary_mask, next_shape, **itp_kwargs) >= 0.5

                mask_logits_ij_up = F.interpolate(mask_logits_ij, next_shape, **itp_kwargs)
                mask_logits_i[j+1][non_boundary_mask] = mask_logits_ij_up[non_boundary_mask]

            mask_scores_i = mask_logits_i[-1].sigmoid()
            mask_scores_i = _do_paste_mask(mask_scores_i, pred_boxes_i, iH, iW, skip_empty=False)[0]
            ins_seg_masks = mask_scores_i > 0.5

            # Update prediction scores based on mask scores if needed
            if self.seg_type == 'instance':
                pred_scores_i = pred_scores_i * (ins_seg_masks * mask_scores_i).flatten(1).sum(dim=1)
                pred_scores_i = pred_scores_i / (ins_seg_masks.flatten(1).sum(dim=1) + 1e-6)

            # Perform panoptic post-processing if needed
            if self.seg_type == 'panoptic':

                # Filter based on score if needed
                score_thr = self.pan_post_attrs.get('score_thr', None)

                if score_thr is not None:
                    keep_mask = pred_scores_i > score_thr

                    pred_labels_i = pred_labels_i[keep_mask]
                    mask_scores_i = mask_scores_i[keep_mask]
                    pred_scores_i = pred_scores_i[keep_mask]
                    ins_seg_masks = ins_seg_masks[keep_mask]

                # Filter based on mask NMS if needed
                nms_thr = self.pan_post_attrs.get('nms_thr', None)

                if nms_thr is not None:
                    pred_scores_i, sort_ids = pred_scores_i.sort(descending=True)

                    pred_labels_i = pred_labels_i[sort_ids]
                    mask_scores_i = mask_scores_i[sort_ids]
                    ins_seg_masks = ins_seg_masks[sort_ids]

                    num_preds = len(ins_seg_masks)
                    flat_masks = ins_seg_masks.flatten(1)
                    inter = torch.zeros(num_preds, num_preds, dtype=torch.float, device=device)

                    for j in range(1, num_preds):
                        inter[j, :j] = (flat_masks[j, None, :] * flat_masks[None, :j, :]).sum(dim=2)

                    areas = flat_masks.sum(dim=1)
                    union = areas[:, None] + areas[None, :] - inter

                    ious = inter / union
                    keep_mask = ious.amax(dim=1) < nms_thr

                    pred_labels_i = pred_labels_i[keep_mask]
                    mask_scores_i = mask_scores_i[keep_mask]
                    pred_scores_i = pred_scores_i[keep_mask]
                    ins_seg_masks = ins_seg_masks[keep_mask]

                # Get panoptic segmentation masks
                rel_mask_scores = pred_scores_i[:, None, None] * mask_scores_i
                num_preds = len(rel_mask_scores)

                if num_preds > 0:
                    pan_seg_mask = rel_mask_scores.argmax(dim=0)
                else:
                    pan_seg_mask = ins_seg_masks.new_zeros([*rel_mask_scores.size()[1:]])

                pred_ids = torch.arange(num_preds, device=device)
                pan_seg_masks = pan_seg_mask[None, :, :] == pred_ids[:, None, None]

                # Apply panoptic mask threshold if needed
                pan_mask_thr = self.pan_post_attrs.get('pan_mask_thr', None)

                if pan_mask_thr is not None:
                    pan_seg_masks &= mask_scores_i > pan_mask_thr

                # Filter based on instance-panoptic IoU if needed
                ins_pan_thr = self.pan_post_attrs.get('ins_pan_thr', None)

                if ins_pan_thr is not None:
                    ins_flat_masks = ins_seg_masks.flatten(1)
                    pan_flat_masks = pan_seg_masks.flatten(1)

                    ins_areas = ins_flat_masks.sum(dim=1)
                    pan_areas = pan_flat_masks.sum(dim=1)

                    inter = (ins_flat_masks * pan_flat_masks).sum(dim=1)
                    union = ins_areas + pan_areas - inter

                    ious = inter / union
                    keep_mask = ious > ins_pan_thr

                    pred_labels_i = pred_labels_i[keep_mask]
                    pred_scores_i = pred_scores_i[keep_mask]
                    pan_seg_masks = pan_seg_masks[keep_mask]

                # Filter based on mask area if needed
                area_thr = self.pan_post_attrs.get('area_thr', None)

                if area_thr is not None:
                    areas = pan_seg_masks.flatten(1).sum(dim=1)
                    keep_mask = areas > area_thr

                    pred_labels_i = pred_labels_i[keep_mask]
                    pred_scores_i = pred_scores_i[keep_mask]
                    pan_seg_masks = pan_seg_masks[keep_mask]

                # Merge stuff predictions
                thing_mask = (pred_labels_i[:, None] == thing_ids[None, :]).any(dim=1)
                stuff_mask = ~thing_mask

                stuff_labels = pred_labels_i[stuff_mask]
                stuff_labels, stuff_ids, stuff_counts = stuff_labels.unique(return_inverse=True, return_counts=True)
                pred_labels_i = torch.cat([pred_labels_i[thing_mask], stuff_labels], dim=0)

                stuff_scores = torch.zeros_like(stuff_labels, dtype=torch.float)
                stuff_scores.scatter_add_(dim=0, index=stuff_ids, src=pred_scores_i[stuff_mask])
                stuff_scores = stuff_scores / stuff_counts
                pred_scores_i = torch.cat([pred_scores_i[thing_mask], stuff_scores], dim=0)

                stuff_ids = stuff_ids[:, None, None].expand(-1, iH, iW)
                unmerged_stuff_masks = pan_seg_masks[stuff_mask]

                num_stuff_preds = len(stuff_labels)
                stuff_seg_masks = pan_seg_masks.new_zeros([num_stuff_preds, iH, iW])
                stuff_seg_masks.scatter_add_(dim=0, index=stuff_ids, src=unmerged_stuff_masks)

                thing_seg_masks = pan_seg_masks[thing_mask]
                pan_seg_masks = torch.cat([thing_seg_masks, stuff_seg_masks], dim=0)

            # Add predictions to prediction dictionary
            pred_dict['labels'].append(pred_labels_i)
            pred_dict['scores'].append(pred_scores_i)
            pred_dict['batch_ids'].append(torch.full_like(pred_labels_i, i))

            if self.seg_type == 'instance':
                pred_dict['masks'].append(ins_seg_masks)

            elif self.seg_type == 'panoptic':
                pred_dict['masks'].append(pan_seg_masks)

            else:
                error_msg = f"Invalid type of segmentation task (got '{self.seg_type}')."
                raise ValueError(error_msg)

        # Concatenate predictions of different batch entries
        pred_dict.update({k: torch.cat(v, dim=0) for k, v in pred_dict.items()})

        # Add prediction dictionary to list of prediction dictionaries
        pred_dicts.append(pred_dict)

        return pred_dicts

    def forward_loss(self, storage_dict, tgt_dict, loss_dict, analysis_dict=None, id=None, **kwargs):
        """
        Forward loss method of the RefineMaskRoIHead module.

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
            analysis_dict (Dict): Dictionary containing different analyses (default=None).
            id (int or str): Integer or string containing the head id (default=None).
            kwargs (Dict): Dictionary of keyword arguments passed to some underlying modules.

        Returns:
            loss_dict (Dict): Loss dictionary containing following additional key:
                - mask_loss (FloatTensor): mask loss of shape [].

            analysis_dict (Dict): Analysis dictionary containing following additional key (if not None):
                - mask_acc (FloatTensor): mask accuracy of shape [].
        """

        # Perform matching if matcher is available
        if self.matcher is not None:
            self.matcher(storage_dict, tgt_dict=tgt_dict, analysis_dict=analysis_dict, **kwargs)

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
