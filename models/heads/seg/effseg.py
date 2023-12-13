"""
Efficient Segmentation (EffSeg) head.
"""

from detectron2.layers import batched_nms
from mmdet.models.roi_heads.mask_heads.fcn_mask_head import _do_paste_mask
import torch
from torch import nn
import torch.nn.functional as F

from models.build import build_model, MODELS
from models.heads.seg.base import BaseSegHead


@MODELS.register_module()
class EffSegHead(BaseSegHead):
    """
    Class implementing the EffSegHead module.

    Attributes:
        key_2d (nn.Module): Optional module updating the 2D key feature maps.
        roi_ext (nn.Module): Module extracting RoI features from feature maps based on RoI boxes.

        pos_enc (nn.Module): Optional module adding position features to RoI features.
        qry (nn.Module): Optional module updating the query features.
        fuse_qry (nn.Module): Optional module fusing query features with RoI features.
        roi_ins (nn.Module): Optional module updating the RoI features.

        seg (nn.ModuleList): List [seg_iters] of modules computing segmentation logits from core features.
        ref (nn.ModuleList): List [seg_iters] of modules computing refinement logits from core features.

        fuse_td (nn.ModuleList): List [seg_iters-1] of modules fusing top-down features with core features.
        fuse_key (nn.ModuleList): List [seg_iters-1] of modules fusing key features with core features.
        trans (nn.ModuleList): List [seg_iters-1] of modules transitioning core and auxiliary features to new space.
        proc (nn.ModuleList): List [seg_iters-1] of modules processing the core features.

        key_min_id (int): Integer containing the downsampling index of the highest resolution key feature map.
        key_max_id (int): Integer containing the downsampling index of the lowest resolution key feature map.
        seg_iters (int): Integer containing the number of segmentation iterations.
        refines_per_iter (int): Integer containing the number of refinements per segmentation iteration.
        get_segs (bool): Boolean indicating whether to get segmentation predictions.

        dup_attrs (Dict): Optional dictionary specifying the duplicate removal mechanism, possibly containing:
            - type (str): string containing the type of duplicate removal mechanism (mandatory);
            - nms_candidates (int): integer containing the maximum number of candidate detections retained before NMS;
            - nms_thr (float): value of IoU threshold used during NMS to remove duplicate detections.

        max_segs (int): Optional integer with the maximum number of returned segmentation predictions.
        mask_thr (float): Value containing the normalized (instance) segmentation mask threshold.

        pan_post_attrs (Dict): Dictionary specifying the panoptic post-processing mechanism possibly containing:
            - score_thr (float): value containing the instance score threshold (or None);
            - nms_thr (float): value containing the IoU threshold used during mask IoU (or None);
            - pan_mask_thr (float): value containing the normalized panoptic segmentation mask threshold;
            - ins_pan_thr (float): value containing the IoU threshold between instance and panoptic masks;
            - area_thr (int): integer containing the mask area threshold (or None).

        matcher (nn.Module): Optional matcher module determining the target segmentation maps.
        roi_sizes (Tuple): Tuple of size [seg_iters] containing the RoI sizes.
        tgt_roi_ext (nn.ModuleList): List [seg_iters] of modules extracting the RoI-based target segmentation masks.
        seg_loss (nn.Module): Module computing the segmentation loss.
        seg_loss_weights (Tuple): Tuple of size [seg_iters] containing the segmentation loss weights.
        ref_loss (nn.Module): Module computing the refinement loss.
        ref_loss_weights (Tuple): Tuple of size [seg_iters] containing the refinement loss weights.
        apply_ids (List): List with integers determining when the head should be applied.
    """

    def __init__(self, roi_ext_cfg, seg_cfg, ref_cfg, fuse_td_cfg, fuse_key_cfg, trans_cfg, proc_cfg, key_min_id,
                 key_max_id, seg_iters, refines_per_iter, mask_thr, metadata, seg_loss_cfg, seg_loss_weights,
                 ref_loss_cfg, ref_loss_weights, key_2d_cfg=None, pos_enc_cfg=None, qry_cfg=None, fuse_qry_cfg=None,
                 roi_ins_cfg=None, get_segs=True, seg_type='instance', dup_attrs=None, max_segs=None,
                 pan_post_attrs=None, matcher_cfg=None, roi_sizes=None, apply_ids=None, **kwargs):
        """
        Initializes the EffSegHead module.

        Args:
            roi_ext_cfg: Configuration dictionary specifying the RoI-extractor module.
            seg_cfg (Dict): Configuration dictionary specifying the segmentation module.
            ref_cfg (Dict): Configuration dictionary specifying the refinement module.
            fuse_td_cfg (Dict): Configuration dictionary specifying the fuse top-down module.
            fuse_key_cfg (Dict): Configuration dictionary specifying the fuse key module.
            trans_cfg (Dict): Configuration dictionary specifying the transition module.
            proc_cfg (Dict): Configuration dictionary specifying the processing module.
            key_min_id (int): Integer containing the downsampling index of the highest resolution key feature map.
            key_max_id (int): Integer containing the downsampling index of the lowest resolution key feature map.
            seg_iters (int): Integer containing the number of segmentation iterations.
            refines_per_iter (int): Integer containing the number of refinements per segmentation iteration.
            mask_thr (float): Value containing the mask threshold used to determine the segmentation masks.
            metadata (detectron2.data.Metadata): Metadata instance containing additional dataset information.
            seg_loss_cfg (Dict): Configuration dictionary specifying the segmentation loss module.
            seg_loss_weights (Tuple): Tuple of size [seg_iters] containing the segmentation loss weights.
            ref_loss_cfg (Dict): Configuration dictionary specifying the refinement loss module.
            ref_loss_weights (Tuple): Tuple of size [seg_iters] containing the refinement loss weights.
            key_2d_cfg (Dict): Configuration dictionary specifying the key 2D module (default=None).
            pos_enc_cfg (Dict): Configuration dictionary specifying the position encoder module (default=None).
            qry_cfg (Dict): Configuration dictionary specifying the query module (default=None).
            fuse_qry_cfg (Dict): Configuration dictionary specifying the fuse query module (default=None).
            roi_ins_cfg (Dict): Configuration dictionary specifying the RoI-instance module (default=None).
            get_segs (bool): Boolean indicating whether to get segmentation predictions (default=True).
            seg_type (str): String containing the type of segmentation task (default='instance').
            dup_attrs (Dict): Attribute dictionary specifying the duplicate removal mechanism (default=None).
            max_segs (int): Integer with the maximum number of returned segmentation predictions (default=None).
            pan_post_attrs (Dict): Attribute dictionary specifying the panoptic post-processing (default=None).
            matcher_cfg (Dict): Configuration dictionary specifying the matcher module (default=None).
            roi_sizes (Tuple): Tuple of size [seg_iters] containing the RoI sizes (default=None).
            apply_ids (List): List with integers determining when the head should be applied (default=None).
            kwargs (Dict): Dictionary of unused keyword arguments.
        """

        # Initialization of BaseSegHead module
        super().__init__(seg_type, metadata)

        # Build various modules used to obtain segmentation and refinement logits from inputs
        self.key_2d = build_model(key_2d_cfg) if key_2d_cfg is not None else None
        self.roi_ext = build_model(roi_ext_cfg)

        self.pos_enc = build_model(pos_enc_cfg) if pos_enc_cfg is not None else None
        self.qry = build_model(qry_cfg) if qry_cfg is not None else None
        self.fuse_qry = build_model(fuse_qry_cfg) if fuse_qry_cfg is not None else None
        self.roi_ins = build_model(roi_ins_cfg) if roi_ins_cfg is not None else None

        self.seg = nn.ModuleList([build_model(cfg_i) for cfg_i in seg_cfg])
        self.ref = nn.ModuleList([build_model(cfg_i) for cfg_i in ref_cfg])

        self.fuse_td = nn.ModuleList([build_model(cfg_i) for cfg_i in fuse_td_cfg])
        self.fuse_key = nn.ModuleList([build_model(cfg_i) for cfg_i in fuse_key_cfg])
        self.trans = nn.ModuleList([build_model(cfg_i) for cfg_i in trans_cfg])
        self.proc = nn.ModuleList([build_model(cfg_i) for cfg_i in proc_cfg])

        # Build matcher module if needed
        self.matcher = build_model(matcher_cfg) if matcher_cfg is not None else None

        # Set attribute containing the RoI sizes
        if roi_sizes is None:
            self.roi_sizes = tuple(2**i * roi_ext_cfg['roi_layer']['output_size'] for i in range(seg_iters))
        else:
            self.roi_sizes = roi_sizes

        # Build target RoI extractor
        tgt_roi_ext_cfg = dict(type='mmdet.SingleRoIExtractor')
        tgt_roi_ext_cfg['roi_layer'] = dict(type='RoIAlign', sampling_ratio=0)
        tgt_roi_ext_cfg['out_channels'] = 1
        tgt_roi_ext_cfg['featmap_strides'] = [1]
        self.tgt_roi_ext = nn.ModuleList()

        for roi_size in self.roi_sizes:
            tgt_roi_ext_cfg['roi_layer']['output_size'] = roi_size
            self.tgt_roi_ext.append(build_model(tgt_roi_ext_cfg))

        # Build segmentation and refinement loss modules
        self.seg_loss = build_model(seg_loss_cfg)
        self.ref_loss = build_model(ref_loss_cfg)

        # Set remaining attributes
        self.key_min_id = key_min_id
        self.key_max_id = key_max_id
        self.seg_iters = seg_iters
        self.refines_per_iter = refines_per_iter
        self.get_segs = get_segs
        self.dup_attrs = dup_attrs
        self.max_segs = max_segs
        self.mask_thr = mask_thr
        self.pan_post_attrs = pan_post_attrs if pan_post_attrs is not None else dict()
        self.seg_loss_weights = seg_loss_weights
        self.ref_loss_weights = ref_loss_weights
        self.apply_ids = apply_ids

    @torch.no_grad()
    def compute_segs(self, storage_dict, pred_dicts, **kwargs):
        """
        Method computing the segmentation predictions.

        Args:
            storage_dict (Dict): Storage dictionary containing at least following keys:
                - batch_ids (LongTensor): batch indices of queries of shape [num_qrys];
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
            ValueError: Error when an invalid type of duplicate removal or rescoring mechanism is provided.
            ValueError: Error when an invalid type of segmentation task if provided.
        """

        # Retrieve various items from storage dictionary
        batch_ids = storage_dict['batch_ids']
        images = storage_dict['images']
        cls_logits = storage_dict['cls_logits']
        pred_boxes = storage_dict['pred_boxes']

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

        # Get prediction boxes in desired format
        pred_boxes = pred_boxes.to_format('xyxy').to_img_scale(images).boxes

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
                    num_preds = len(pred_scores_i)
                    candidate_ids = pred_scores_i.topk(min(num_candidates, num_preds))[1]

                    qry_ids_i = qry_ids_i[candidate_ids]
                    pred_labels_i = pred_labels_i[candidate_ids]
                    pred_scores_i = pred_scores_i[candidate_ids]

                    pred_boxes_i = pred_boxes[qry_ids_i]
                    iou_thr = self.dup_attrs['nms_thr']
                    non_dup_ids = batched_nms(pred_boxes_i, pred_scores_i, pred_labels_i, iou_thr)

                    qry_ids_i = qry_ids_i[non_dup_ids]
                    pred_labels_i = pred_labels_i[non_dup_ids]
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
                    pred_scores_i = pred_scores_i[top_pred_ids]

            # Get query indices for which to compute segmentations
            qry_ids_i, non_unique_ids = qry_ids_i.unique(sorted=True, return_inverse=True)

            # Get segmentation and refinement predictions for desired queries
            self.get_preds(storage_dict, seg_qry_ids=qry_ids_i, **kwargs)

            # Retrieve various items related to segmentation predictions from storage dictionary
            roi_ids_list = storage_dict['roi_ids_list']
            pos_ids_list = storage_dict['pos_ids_list']
            seg_logits_list = storage_dict['seg_logits_list']

            # Get mask scores and instance segmentation masks
            num_rois = len(qry_ids_i)
            rH = rW = self.roi_sizes[0]
            mask_logits = torch.zeros(num_rois, 1, rH, rW, device=device)

            for j in range(self.seg_iters):
                roi_ids = roi_ids_list[j]
                pos_ids = pos_ids_list[j]

                seg_logits = seg_logits_list[j]
                mask_logits[roi_ids, 0, pos_ids[:, 1], pos_ids[:, 0]] = seg_logits

                if j < self.seg_iters-1:
                    rH = rW = self.roi_sizes[j+1]
                    mask_logits = F.interpolate(mask_logits, (rH, rW), mode='bilinear', align_corners=False)

            mask_scores_i = mask_logits.sigmoid()
            pred_boxes_i = pred_boxes[qry_ids_i]

            mask_scores_i = _do_paste_mask(mask_scores_i, pred_boxes_i, iH, iW, skip_empty=False)[0]
            mask_scores_i = mask_scores_i[non_unique_ids]
            ins_seg_masks = mask_scores_i > self.mask_thr

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

    def get_preds(self, storage_dict, seg_qry_ids, **kwargs):
        """
        Method computing the segmentation and refinement logits for the desired queries.

        Args:
            storage_dict (Dict): Storage dictionary containing at least following keys:
                - qry_feats (FloatTensor): query features of shape [num_qrys, qry_feat_size];
                - batch_ids (LongTensor): batch indices of queries of shape [num_qrys];
                - feat_maps (List): list of size [num_maps] with feature maps of shape [batch_size, feat_size, fH, fW];
                - images (Images): Images structure containing the batched images of size [batch_size];
                - feat_ids (LongTensor): indices of selected features resulting in query features of shape [num_qrys];
                - pred_boxes (Boxes): predicted 2D bounding boxes obtained from query features of size [num_qrys].

            seg_qry_ids (LongTensor): Query indices for which to compute segmentations of shape [num_segs].
            kwargs (Dict): Dictionary of keyword arguments not used by this module.

        Returns:
            storage_dict (Dict): Storage dictionary containing following additional keys:
                - roi_ids_list (List): list [seg_iters] with RoI indices of predictions [num_preds_i];
                - pos_ids_list (List): list [seg_iters] with RoI-based position ids in (X, Y) format [num_preds_i, 2];
                - seg_logits_list (List): list [seg_iters] with segmentation logits of shape [num_preds_i];
                - ref_logits_list (List): list [seg_iters] with refinement logits of shape [num_preds_i].
        """

        # Retrieve various items from storage dictionary
        qry_feats = storage_dict['qry_feats']
        batch_ids = storage_dict['batch_ids']
        key_feat_maps = storage_dict['feat_maps']
        images = storage_dict['images']
        qry_boxes = storage_dict['pred_boxes'].clone()

        # Get batch size and device
        batch_size = len(images)
        device = qry_feats.device

        # Select for which queries to compute segmentations
        qry_feats = qry_feats[seg_qry_ids]
        qry_boxes = qry_boxes[seg_qry_ids]
        batch_ids = batch_ids[seg_qry_ids]

        # Extract RoI features
        roi_boxes = qry_boxes.to_format('xyxy').to_img_scale(images).boxes.detach()
        roi_boxes = torch.cat([batch_ids[:, None], roi_boxes], dim=1)

        key_feat_maps = self.key_2d(key_feat_maps) if self.key_2d is not None else key_feat_maps
        roi_feat_maps = key_feat_maps[:self.roi_ext.num_inputs]
        roi_feats = self.roi_ext(roi_feat_maps, roi_boxes)

        # Add position encodings if needed
        if self.pos_enc is not None:
            rH, rW = roi_feats.size()[2:]
            pts_x = torch.linspace(0.5/rW, 1-0.5/rW, steps=rW, device=device)
            pts_y = torch.linspace(0.5/rH, 1-0.5/rH, steps=rH, device=device)

            norm_xy = torch.meshgrid(pts_x, pts_y, indexing='xy')
            norm_xy = torch.stack(norm_xy, dim=2).flatten(0, 1)
            roi_feats = roi_feats + self.pos_enc(norm_xy).t().view(1, -1, rH, rW)

        # Fuse query features if needed
        if self.fuse_qry is not None:
            qry_feats = self.qry(qry_feats) if self.qry is not None else qry_feats
            qry_feats = qry_feats[:, :, None, None].expand_as(roi_feats)

            fuse_qry_feats = torch.cat([qry_feats, roi_feats], dim=1)
            roi_feats = roi_feats + self.fuse_qry(fuse_qry_feats)

        # Update RoI features if needed
        if self.roi_ins is not None:
            roi_ins_kwargs = {'semantic_feat': key_feat_maps[0], 'rois': roi_boxes}
            roi_feats = self.roi_ins(roi_feats, **roi_ins_kwargs)

        # Get map indices from which RoI features were extracted
        map_ids = self.roi_ext.map_roi_levels(roi_boxes, self.roi_ext.num_inputs)
        max_map_id = map_ids.max().item()

        num_rois, feat_size, rH, rW = roi_feats.size()
        map_ids = map_ids[:, None].expand(-1, rH*rW).flatten()

        # Get RoI and batch indices
        roi_ids = torch.arange(num_rois, device=device)[:, None].expand(-1, rH*rW).flatten()
        batch_ids = batch_ids[:, None].expand(-1, rH*rW).flatten()

        # Get RoI-based position indices in (X, Y) format
        x_ids = torch.arange(rW, device=device)
        y_ids = torch.arange(rH, device=device)

        pos_ids = torch.meshgrid(x_ids, y_ids, indexing='xy')
        pos_ids = torch.stack(pos_ids, dim=2)
        pos_ids = pos_ids[None, :, :, :].expand(num_rois, -1, -1, -1).flatten(0, 2)

        # Get core and auxiliary features
        core_feats = roi_feats.permute(0, 2, 3, 1).flatten(0, 2)
        aux_feats = roi_feats.new_empty([0, feat_size])

        # Get number of core features
        num_core_feats = len(core_feats)

        # Get index map
        id_map = torch.arange(num_rois*rH*rW, device=device).view(num_rois, rH, rW)

        # Get position offsets
        pos_offs = torch.tensor([[0, 0], [1, 0], [0, 1], [1, 1]], device=device)

        # Get key feature size
        key_feat_size = key_feat_maps[0].size(dim=1)

        # Get boxes used to find normalized segmentation locations
        seg_boxes = qry_boxes.to_format('xywh').normalize(images).boxes

        # Store desired items in lists
        roi_ids_list = [roi_ids]
        pos_ids_list = [pos_ids]

        # Initialize empty lists
        seg_logits_list = []
        ref_logits_list = []

        # Perform segmentation iterations
        for i in range(self.seg_iters):

            # Get segmentation and refinement logits
            seg_logits = self.seg[i](core_feats)
            ref_logits = self.ref[i](core_feats)

            seg_logits_list.append(seg_logits)
            ref_logits_list.append(ref_logits)

            # Refine if needed
            if i < self.seg_iters-1:

                # Get refine mask
                if num_core_feats > self.refines_per_iter:
                    refine_ids = torch.topk(ref_logits, self.refines_per_iter, sorted=False)[1]
                    refine_mask = torch.zeros(num_core_feats, dtype=torch.bool, device=device)
                    refine_mask[refine_ids] = True

                else:
                    refine_mask = torch.ones(num_core_feats, dtype=torch.bool, device=device)

                # Update id map
                new_core_ids = torch.empty(num_core_feats, dtype=torch.int64, device=device)

                num_refines = min(num_core_feats, self.refines_per_iter)
                num_non_refines = num_core_feats - num_refines
                num_core_feats = 4 * num_refines
                num_aux_feats = len(aux_feats)

                mid = num_core_feats + num_aux_feats
                end = mid + num_non_refines

                new_core_ids[refine_mask] = torch.arange(0, num_core_feats, step=4, device=device)
                new_aux_ids = torch.arange(num_core_feats, mid, device=device)
                new_core_ids[~refine_mask] = torch.arange(mid, end, device=device)

                new_ids = torch.cat([new_core_ids, new_aux_ids], dim=0)
                id_map_0 = new_ids[id_map]

                off_mask = id_map_0 < num_core_feats
                id_map_1 = torch.where(off_mask, id_map_0 + 1, id_map_0)
                id_map_2 = torch.where(off_mask, id_map_0 + 2, id_map_0)
                id_map_3 = torch.where(off_mask, id_map_0 + 3, id_map_0)

                id_map_01 = torch.stack([id_map_0, id_map_1], dim=3).flatten(2)
                id_map_23 = torch.stack([id_map_2, id_map_3], dim=3).flatten(2)
                id_map = torch.stack([id_map_01, id_map_23], dim=2).flatten(1, 2)

                # Update core and auxiliary features
                aux_feats = torch.cat([aux_feats, core_feats[~refine_mask]], dim=0)
                core_feats = core_feats[refine_mask]

                # Update map indices
                map_ids = torch.clamp(map_ids-1, min=0)
                map_ids = map_ids[refine_mask].repeat_interleave(4, dim=0)
                max_map_id = max(max_map_id-1, 0)

                # Update RoI indices
                roi_ids = roi_ids[refine_mask].repeat_interleave(4, dim=0)
                roi_ids_list.append(roi_ids)

                # Update batch indices
                batch_ids = batch_ids[refine_mask].repeat_interleave(4, dim=0)

                # Update position indices
                pos_ids = 2*pos_ids[refine_mask, None, :] + pos_offs
                pos_ids = pos_ids.flatten(0, 1)
                pos_ids_list.append(pos_ids)

                # Fuse top-down features
                fuse_td_feats = self.fuse_td[i](core_feats)
                core_feats = core_feats.repeat_interleave(4, dim=0) + fuse_td_feats

                # Fuse key features
                key_feats = torch.empty(num_core_feats, key_feat_size, device=device)

                pos_wh = 2**(-i-1) * torch.tensor([1/rW, 1/rH], device=device)
                pos_xy = 0.5*pos_wh + pos_ids * pos_wh

                seg_boxes_i = seg_boxes[roi_ids]
                seg_xy = seg_boxes_i[:, :2] + pos_xy * seg_boxes_i[:, 2:]

                for j in range(batch_size):
                    mask_j = batch_ids == j

                    for k in range(max_map_id+1):
                        mask_jk = mask_j & (map_ids == k)

                        if mask_jk.sum().item() > 0:
                            sample_grid = 2 * seg_xy[mask_jk][None, None, :, :] - 1
                            sample_key_feats = F.grid_sample(key_feat_maps[k][j:j+1], sample_grid, align_corners=False)
                            key_feats[mask_jk] = sample_key_feats[0, :, 0, :].t()

                fuse_key_feats = torch.cat([core_feats, key_feats], dim=1)
                core_feats += self.fuse_key[i](fuse_key_feats)

                # Transition core and auxiliary features
                core_feats = self.trans[i](core_feats)
                aux_feats = self.trans[i](aux_feats)

                # Update core features
                id_kwargs = {'id_map': id_map, 'roi_ids': roi_ids, 'pos_ids': pos_ids}
                core_feats = self.proc[i](core_feats, aux_feats=aux_feats, **id_kwargs)

        # Store desired items in storage dictionary
        storage_dict['roi_ids_list'] = roi_ids_list
        storage_dict['pos_ids_list'] = pos_ids_list
        storage_dict['seg_logits_list'] = seg_logits_list
        storage_dict['ref_logits_list'] = ref_logits_list

        return storage_dict

    def forward_pred(self, storage_dict, images_dict=None, **kwargs):
        """
        Forward prediction method of the EffSegHead module.

        Args:
            storage_dict (Dict): Dictionary storing various items of interest.
            images_dict (Dict): Dictionary with annotated images of predictions/targets (default=None).
            kwargs (Dict): Dictionary of keyword arguments passed to some underlying methods.

        Returns:
            storage_dict (Dict): Dictionary with (possibly) additional stored items of interest.
            images_dict (Dict): Dictionary (possibly) containing additional images annotated with segmentations.
        """

        # Get segmentation predictions if needed
        if self.get_segs and not self.training:
            self.compute_segs(storage_dict=storage_dict, **kwargs)

        # Draw predicted and target segmentations if needed
        if images_dict is not None:
            self.draw_segs(storage_dict=storage_dict, images_dict=images_dict, **kwargs)

        return storage_dict, images_dict

    def forward_loss(self, storage_dict, tgt_dict, loss_dict, analysis_dict, id=None, **kwargs):
        """
        Forward loss method of the EffSegHead module.

        Args:
            storage_dict (Dict): Storage dictionary containing following keys (after matching):
                - qry_feats (FloatTensor): query features of shape [num_qrys, qry_feat_size];
                - images (Images): Images structure containing the batched images of size [batch_size];
                - pred_boxes (Boxes): predicted 2D bounding boxes obtained from query features of size [num_qrys];
                - matched_qry_ids (LongTensor): indices of matched queries of shape [num_pos_qrys];
                - matched_tgt_ids (LongTensor): indices of corresponding matched targets of shape [num_pos_qrys].

            tgt_dict (Dict): Target dictionary containing at least following key:
                - masks (BoolTensor): target segmentation masks of shape [num_targets, iH, iW].

            loss_dict (Dict): Dictionary containing different weighted loss terms.
            analysis_dict (Dict): Dictionary containing different analyses.
            id (int or str): Integer or string containing the head id (default=None).
            kwargs (Dict): Dictionary of keyword arguments passed to some underlying modules.

        Returns:
            loss_dict (Dict): Loss dictionary containing following additional keys:
                - seg_loss (FloatTensor): segmentation loss over all iterations of shape [];
                - ref_loss (FloatTensor): refinement loss over all iterations of shape [].

            analysis_dict (Dict): Analysis dictionary containing following additional keys (if not None):
                - seg_loss_{i} (FloatTensor): segmentation loss of iteration {i} of shape [];
                - seg_acc_{i} (FloatTensor): segmentation accuracy of iteration {i} of shape [];
                - ref_loss_{i} (FloatTensor): refinement loss of iteration {i} of shape [];
                - ref_acc_{i} (FloatTensor): refinement accuracy of iteration {i} of shape [].

        Raises:
            ValueError: Error when a single query is matched with multiple targets.
        """

        # Retrieve query features and get device
        qry_feats = storage_dict['qry_feats']
        device = qry_feats.device

        # Perform matching if matcher is available
        if self.matcher is not None:
            self.matcher(storage_dict, tgt_dict=tgt_dict, analysis_dict=analysis_dict, **kwargs)

        # Retrieve matched query and target indices from storage dictionary
        matched_qry_ids = storage_dict['matched_qry_ids']
        matched_tgt_ids = storage_dict['matched_tgt_ids']

        # Get number of positive matches
        num_matches = len(matched_qry_ids)

        # Handle case where there are no positive matches
        if num_matches == 0:

            # Get segmentation loss
            seg_loss = 0.0 * qry_feats.sum() + sum(0.0 * p.flatten()[0] for p in self.parameters())

            for i in range(self.seg_iters):
                key_name = f'seg_loss_{id}_{i}' if id is not None else f'seg_loss_{i}'
                analysis_dict[key_name] = seg_loss.detach()

            key_name = f'seg_loss_{id}' if id is not None else 'seg_loss'
            loss_dict[key_name] = seg_loss

            # Get segmentation accuracies
            with torch.no_grad():
                seg_acc = 1.0 if len(tgt_dict['masks']) == 0 else 0.0
                seg_acc = torch.tensor(seg_acc, dtype=seg_loss.dtype, device=device)

                for i in range(self.seg_iters):
                    key_name = f'seg_acc_{id}_{i}' if id is not None else f'seg_acc_{i}'
                    analysis_dict[key_name] = 100 * seg_acc

            # Get refinement loss
            ref_loss = 0.0 * qry_feats.sum() + sum(0.0 * p.flatten()[0] for p in self.parameters())

            for i in range(self.seg_iters):
                key_name = f'ref_loss_{id}_{i}' if id is not None else f'ref_loss_{i}'
                analysis_dict[key_name] = ref_loss.detach()

            key_name = f'ref_loss_{id}' if id is not None else 'ref_loss'
            loss_dict[key_name] = ref_loss

            # Get refinement accuracies
            with torch.no_grad():
                ref_acc = 1.0 if len(tgt_dict['masks']) == 0 else 0.0
                ref_acc = torch.tensor(ref_acc, dtype=ref_loss.dtype, device=device)

                for i in range(self.seg_iters):
                    key_name = f'ref_acc_{id}_{i}' if id is not None else f'ref_acc_{i}'
                    analysis_dict[key_name] = 100 * ref_acc

            return loss_dict, analysis_dict

        # Check that no query is matched with multiple targets
        counts = matched_qry_ids.unique(sorted=False, return_counts=True)[1]

        if torch.any(counts > 1):
            error_msg = "The EffSegHead does not support a single query to be matched with multiple targets."
            raise ValueError(error_msg)

        # Get segmentation and refinement predictions for desired queries
        self.get_preds(storage_dict, seg_qry_ids=matched_qry_ids, **kwargs)

        # Retrieve various items related to segmentation predictions from storage dictionary
        roi_ids_list = storage_dict['roi_ids_list']
        pos_ids_list = storage_dict['pos_ids_list']
        seg_logits_list = storage_dict['seg_logits_list']
        ref_logits_list = storage_dict['ref_logits_list']

        # Get initial target maps
        images = storage_dict['images']
        pred_boxes = storage_dict['pred_boxes'].clone()
        tgt_maps = tgt_dict['masks'][:, None, :, :].float()

        roi_boxes = pred_boxes[matched_qry_ids]
        roi_boxes = roi_boxes.to_format('xyxy').to_img_scale(images).boxes.detach()
        roi_boxes = torch.cat([matched_tgt_ids[:, None], roi_boxes], dim=1)

        # Initialize segmentation and refinement loss
        seg_loss = sum(0.0 * p.flatten()[0] for p in self.parameters())
        ref_loss = sum(0.0 * p.flatten()[0] for p in self.parameters())

        # Get segmentation and refinement losses and accuracies
        for i in range(self.seg_iters):

            # Get target maps
            tgt_maps_i = self.tgt_roi_ext[i]([tgt_maps], roi_boxes)

            # Get target values and segmentation mask
            roi_ids = roi_ids_list[i]
            pos_ids = pos_ids_list[i]
            tgt_vals = tgt_maps_i[roi_ids, 0, pos_ids[:, 1], pos_ids[:, 0]]

            # Get segmentation loss
            seg_logits = seg_logits_list[i]
            seg_targets = (tgt_vals > 0.5).float()

            if len(seg_logits) > 0:
                seg_loss_i = self.seg_loss(seg_logits, seg_targets)
                seg_loss_i *= self.seg_loss_weights[i] * num_matches
                seg_loss += seg_loss_i

            else:
                seg_loss_i = torch.tensor(0.0, device=device)

            key_name = f'seg_loss_{id}_{i}' if id is not None else f'seg_loss_{i}'
            analysis_dict[key_name] = seg_loss_i.detach()

            # Get segmentation accuracy
            with torch.no_grad():
                seg_preds = seg_logits > 0
                seg_targets = seg_targets.bool()

                if len(seg_preds) > 0:
                    seg_acc_i = (seg_preds == seg_targets).sum() / len(seg_preds)
                else:
                    seg_acc_i = torch.tensor(1.0, dtype=torch.float, device=device)

                key_name = f'seg_acc_{id}_{i}' if id is not None else f'seg_acc_{i}'
                analysis_dict[key_name] = 100 * seg_acc_i

            # Get refinement loss
            ref_logits = ref_logits_list[i]
            ref_targets = ((tgt_vals > 0) & (tgt_vals < 1)).float()

            if len(ref_logits) > 0:
                ref_loss_i = self.ref_loss(ref_logits, ref_targets)
                ref_loss_i *= self.ref_loss_weights[i] * num_matches
                ref_loss += ref_loss_i

            else:
                ref_loss_i = torch.tensor(0.0, device=device)

            key_name = f'ref_loss_{id}_{i}' if id is not None else f'ref_loss_{i}'
            analysis_dict[key_name] = ref_loss_i.detach()

            # Get refinement accuracy
            with torch.no_grad():
                ref_preds = ref_logits > 0
                ref_targets = ref_targets.bool()

                if len(ref_preds) > 0:
                    ref_acc_i = (ref_preds == ref_targets).sum() / len(ref_preds)
                else:
                    ref_acc_i = torch.tensor(1.0, dtype=torch.float, device=device)

                key_name = f'ref_acc_{id}_{i}' if id is not None else f'ref_acc_{i}'
                analysis_dict[key_name] = 100 * ref_acc_i

        # Add segmentation and refinement losses to loss dictionary
        key_name = f'seg_loss_{id}' if id is not None else 'seg_loss'
        loss_dict[key_name] = seg_loss

        key_name = f'ref_loss_{id}' if id is not None else 'ref_loss'
        loss_dict[key_name] = ref_loss

        return loss_dict, analysis_dict
