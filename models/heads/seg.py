"""
Collection of segmentation heads.
"""

from detectron2.layers import batched_nms
from detectron2.structures.instances import Instances
from detectron2.utils.visualizer import ColorMode, Visualizer
from mmdet.models.roi_heads.mask_heads.fcn_mask_head import _do_paste_mask
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.transforms.functional as T

from models.build import build_model, MODELS
from structures.boxes import mask_to_box


@MODELS.register_module()
class BaseSegHead(nn.Module):
    """
    Class implementing the BaseSegHead module.

    Attributes:
        process_all_qrys (bool): Boolean indicating whether to process all queries.
        qry (nn.Module): Optional module updating the query features.
        key (nn.Module): Optional module updating the key features.
        qry_key (nn.Module): Optional module computing the mask logits based on the query and key features.
        mask_update (bool): Boolean indicating whether to update a previously predicted mask.
        mask_type (str): String containing the type of predicted segmentation mask.
        key_map_id (int): Integer containing the map index from which to extract key features.
        update_mask_key (str): String with key to retrieve update mask from the storage dictionary.
        roi_ext (nn.Module): Optional module containing the RoI-extractor.
        get_bnd_mask (bool): Boolean indicating whether to get the segmentation boundary mask.
        get_unc_mask (bool): Boolean indicating whether to get the segmentation uncertainty mask.
        unc_thr (int): Integer containing the uncertainty threshold per query.
        get_segs (bool): Boolean indicating whether to get segmentation predictions.

        score_attrs (Dict): Dictionary specifying the scoring mechanism possibly containing following keys:
            - cls_power (float): value containing the classification score power;
            - box_power (float): value containing the box score power;
            - mask_power (float): value containing the mask score power.

        dup_attrs (Dict): Dictionary specifying the duplicate removal or rescoring mechanism possibly containing:
            - type (str): string containing the type of duplicate removal or rescoring mechanism;
            - needs_masks (boolean): boolean indicating whether duplicate mechanism needs segmentation masks;
            - nms_candidates (int): integer containing the maximum number of candidate detections retained before NMS;
            - nms_thr (float): value of IoU threshold used during NMS to remove duplicate detections.

        max_segs (int): Optional integer with the maximum number of returned segmentation predictions.
        mask_thr (float): Value containing the normalized segmentation mask threshold.
        pred_mask_type (str): String containing the type of predicted segmentation mask.
        metadata (detectron2.data.Metadata): Metadata instance containing additional dataset information.
        matcher (nn.Module): Optional matcher module determining the target segmentation maps.
        tgt_roi_ext (nn.Module): Optional module containing the target RoI-extractor.
        loss_sample (nn.Module): Optional module sampling mask loss points.
        loss_updated_only (bool): Boolean indicating whether to apply loss in updated points only.
        loss (nn.Module): Module computing the segmentation mask loss.
        apply_ids (List): List with integers determining when the head should be applied.
    """

    def __init__(self, mask_type, metadata, loss_cfg, process_all_qrys=False, qry_cfg=None, key_cfg=None,
                 qry_key_cfg=None, mask_update=False, key_map_id=0, update_mask_key=None, roi_ext_cfg=None,
                 get_bnd_mask=False, get_unc_mask=False, unc_thr=100, get_segs=True, score_attrs=None, dup_attrs=None,
                 max_segs=None, mask_thr=0.5, pred_mask_type='instance', matcher_cfg=None, loss_sample_cfg=None,
                 loss_updated_only=True, apply_ids=None, **kwargs):
        """
        Initializes the BaseSegHead module.

        Args:
            mask_type (str): String containing the type of predicted segmentation mask.
            metadata (detectron2.data.Metadata): Metadata instance containing additional dataset information.
            loss_cfg (Dict): Configuration dictionary specifying the loss module.
            process_all_qrys (bool): Boolean indicating whether to process all queries (default=False).
            qry_cfg (Dict): Configuration dictionary specifying the query module (default=None).
            key_cfg (Dict): Configuration dictionary specifying the key module (default=None).
            qry_key_cfg (Dict): Configuration dictionary specifying the query-key module (default=None).
            mask_update (bool): Boolean indicating whether to update a previously predicted mask (default=False).
            key_map_id (int): Integer containing the map index from which to extract key features (default=0).
            update_mask_key (str): String with key to retrieve update mask from the storage dictionary (default=None).
            roi_ext_cfg (Dict): Configuration dictionary specifying the RoI-extractor module (default=None).
            get_bnd_mask (bool): Boolean indicating whether to get the segmentation boundary mask (default=False).
            get_unc_mask (bool): Boolean indicating whether to get the segmentation uncertainty mask (default=False).
            unc_thr (int): Integer containing the uncertainty threshold per query (default=100).
            get_segs (bool): Boolean indicating whether to get segmentation predictions (default=True).
            score_attrs (Dict): Attribute dictionary specifying the scoring mechanism (default=None).
            dup_attrs (Dict): Dictionary specifying the duplicate removal or rescoring mechanism (default=None).
            max_segs (int): Integer with the maximum number of returned segmentation predictions (default=None).
            mask_thr (float): Value containing the normalized segmentation mask threshold (default=0.5).
            pred_mask_type (str): String containing the type of predicted segmentation mask (default='instance').
            matcher_cfg (Dict): Configuration dictionary specifying the matcher module (default=None).
            loss_sample_cfg (Dict): Configuration dictionary specifying the loss sample module (default=None).
            loss_updated_only (bool): Boolean indicating whether to apply loss in updated points only (default=True).
            apply_ids (List): List with integers determining when the head should be applied (default=None).
            kwargs (Dict): Dictionary of unused keyword arguments.
        """

        # Initialization of default nn.Module
        super().__init__()

        # Build query, key and query-key modules if needed
        self.qry = build_model(qry_cfg, sequential=True) if qry_cfg is not None else None
        self.key = build_model(key_cfg, sequential=True) if key_cfg is not None else None
        self.qry_key = build_model(qry_key_cfg) if qry_key_cfg is not None else None

        # Build RoI-extractor module if needed
        self.roi_ext = build_model(roi_ext_cfg) if roi_ext_cfg is not None else None

        # Build target RoI-extractor module if needed
        if roi_ext_cfg is not None:
            tgt_roi_ext_cfg = roi_ext_cfg.copy()
            tgt_roi_ext_cfg.update({'out_channels': 1, 'featmap_strides': [1]})
            self.tgt_roi_ext = build_model(tgt_roi_ext_cfg)

        # Build matcher module if needed
        self.matcher = build_model(matcher_cfg) if matcher_cfg is not None else None

        # Build loss sample module if needed
        self.loss_sample = build_model(loss_sample_cfg) if loss_sample_cfg is not None else None

        # Build loss module
        self.loss = build_model(loss_cfg)

        # Set remaining attributes
        self.process_all_qrys = process_all_qrys
        self.mask_update = mask_update
        self.mask_type = mask_type
        self.key_map_id = key_map_id
        self.update_mask_key = update_mask_key
        self.get_bnd_mask = get_bnd_mask
        self.get_unc_mask = get_unc_mask
        self.unc_thr = unc_thr
        self.get_segs = get_segs
        self.score_attrs = score_attrs if score_attrs is not None else dict()
        self.dup_attrs = dup_attrs if dup_attrs is not None else dict()
        self.max_segs = max_segs
        self.mask_thr = mask_thr
        self.pred_mask_type = pred_mask_type
        self.metadata = metadata
        self.loss_updated_only = loss_updated_only
        self.apply_ids = apply_ids

    def get_mask_logits(self, qry_feats, seg_qry_ids, storage_dict):
        """
        Method computing the mask logits for the desired queries.

        Args:
            qry_feats (FloatTensor): Query features of shape [num_qrys, qry_feat_size].
            seg_qry_ids (LongTensor): Query indices for which to compute segmentation logits of shape [num_segs].

            storage_dict (Dict): Storage dictionary (possibly) containing following keys:
                - feat_maps (List): list of size [num_maps] with feature maps of shape [batch_size, feat_size, fH, fW];
                - images (Images): images structure containing the batched images of size [batch_size];
                - batch_ids (LongTensor): batch indices of queries of shape [num_qrys];
                - pred_boxes (Boxes): predicted 2D bounding boxes of size [num_qrys];
                - {self.update_mask_key} (BoolTensor): mask with locations to update of shape [num_qrys, 1, mH, mW];
                - seg_mask_logits (FloatTensor): segmentation mask logits of shape [num_qrys, mH, mW].

        Returns:
            mask_logits (FloatTensor): Segmentation mask logits of shape [num_qrys, mH, mW].

        Raises:
            ValueError: Error when an invalid mask type is provided.
        """

        # Get device
        device = qry_feats.device

        # Update query features
        qry_feats = qry_feats[seg_qry_ids]

        if self.qry is not None:
            storage_dict['qry_boxes'] = storage_dict['pred_boxes'][seg_qry_ids]
            qry_feats = self.qry(qry_feats, storage_dict=storage_dict)

        # Get key feature maps
        key_feat_maps = storage_dict['feat_maps']

        if self.key is not None:
            iW, iH = storage_dict['images'].size(mode='with_padding')
            key_feat_maps = self.key(key_feat_maps, base_map_size=(iH, iW), storage_dict=storage_dict)

        # Get batch size and batch indices
        batch_size = len(key_feat_maps[0])
        batch_ids = storage_dict['batch_ids'][seg_qry_ids]

        # Get mask logits
        if self.mask_update:

            if self.mask_type == 'image':
                key_feat_map = key_feat_maps[self.key_map_id]

                update_mask = storage_dict[self.update_mask_key][seg_qry_ids]
                mask_logits = storage_dict['seg_mask_logits'][seg_qry_ids]

                for i in range(batch_size):
                    batch_mask = batch_ids == i

                    update_mask_i = update_mask[batch_mask, 0]
                    qry_ids, y_ids, x_ids = update_mask_i.nonzero(as_tuple=True)

                    qry_feats_i = qry_feats[batch_mask][qry_ids]
                    key_feats_i = key_feat_map[i, :, y_ids, x_ids]

                    if self.qry_key is not None:
                        feats_list = [qry_feats_i, key_feats_i.t()]
                        update_logits = self.qry_key(feats_list, feats_list=feats_list).squeeze(dim=1)

                    else:
                        update_logits = torch.einsum('kc,ck->k', qry_feats_i, key_feats_i)

                    mask_logits_i = mask_logits[batch_mask]
                    mask_logits_i[qry_ids, y_ids, x_ids] = update_logits
                    mask_logits[batch_mask] = mask_logits_i

            elif self.mask_type == 'roi':
                images = storage_dict['images']
                pred_boxes = storage_dict['pred_boxes'][seg_qry_ids]

                roi_boxes = pred_boxes.to_format('xyxy').to_img_scale(images).boxes.detach()
                roi_boxes = torch.cat([batch_ids[:, None], roi_boxes], dim=1)
                roi_key_feat_map = self.roi_ext([key_feat_maps[self.key_map_id]], roi_boxes)

                update_mask = storage_dict[self.update_mask_key][seg_qry_ids]
                qry_ids, y_ids, x_ids = update_mask.squeeze(dim=1).nonzero(as_tuple=True)

                qry_feats = qry_feats[qry_ids]
                key_feats = roi_key_feat_map[qry_ids, :, y_ids, x_ids]

                if self.qry_key is not None:
                    feats_list = [qry_feats, key_feats]
                    update_logits = self.qry_key(feats_list, feats_list=feats_list).squeeze(dim=1)

                else:
                    update_logits = torch.einsum('kc,kc->k', qry_feats, key_feats)

                mask_logits = storage_dict['seg_mask_logits'][seg_qry_ids]
                mask_logits[qry_ids, y_ids, x_ids] = update_logits

            else:
                error_msg = f"Invalid mask type in BaseSegHead (got '{self.mask_type}')."
                raise ValueError(error_msg)

        elif self.mask_type == 'image':
            key_feat_map = key_feat_maps[self.key_map_id]

            num_qrys = len(qry_feats)
            kH, kW = key_feat_map.size()[2:]
            mask_logits = torch.empty(num_qrys, kH, kW, dtype=torch.float, device=device)

            for i in range(batch_size):
                batch_mask = batch_ids == i

                qry_feats_i = qry_feats[batch_mask]
                key_feat_map_i = key_feat_map[i]

                if self.qry_key is not None:
                    qry_feats_i = qry_feats_i[:, None, :].expand(-1, kH*kW, -1).flatten(0, 1)
                    key_feats_i = key_feat_map_i.permute(1, 2, 0)
                    key_feats_i = key_feats_i[None, :, :, :].expand(num_qrys, -1, -1, -1).flatten(0, 2)

                    feats_list = [qry_feats_i, key_feats_i]
                    mask_logits_i = self.qry_key(feats_list, feats_list=feats_list)
                    mask_logits_i = mask_logits_i.view(num_qrys, kH, kW)

                else:
                    mask_logits_i = torch.einsum('qc,chw->qhw', qry_feats_i, key_feat_map_i)

                mask_logits[batch_mask] = mask_logits_i

        elif self.mask_type == 'roi':
            images = storage_dict['images']
            pred_boxes = storage_dict['pred_boxes'][seg_qry_ids]

            roi_boxes = pred_boxes.to_format('xyxy').to_img_scale(images).boxes.detach()
            roi_boxes = torch.cat([batch_ids[:, None], roi_boxes], dim=1)
            roi_key_feat_map = self.roi_ext([key_feat_maps[self.key_map_id]], roi_boxes)

            if self.qry_key is not None:
                num_qrys, _, kH, kW = roi_key_feat_map.size()

                qry_feats = qry_feats[:, None, :].expand(-1, kH*kW, -1).flatten(0, 1)
                key_feats = roi_key_feat_map.permute(0, 2, 3, 1).flatten(0, 2)

                feats_list = [qry_feats, key_feats]
                mask_logits = self.qry_key(feats_list, feats_list=feats_list)
                mask_logits = mask_logits.view(num_qrys, kH, kW)

            else:
                mask_logits = torch.einsum('qc,qchw->qhw', qry_feats, roi_key_feat_map)

        else:
            error_msg = f"Invalid mask type in BaseSegHead (got '{self.mask_type}')."
            raise ValueError(error_msg)

        return mask_logits

    @torch.no_grad()
    def compute_segs(self, qry_feats, storage_dict, pred_dicts, **kwargs):
        """
        Method computing the segmentation predictions.

        Args:
            qry_feats (FloatTensor): Query features of shape [num_qrys, qry_feat_size].

            storage_dict (Dict): Storage dictionary containing at least following keys:
                - images (Images): images structure containing the batched images of size [batch_size];
                - cls_logits (FloatTensor): classification logits of shape [num_qrys, num_labels];
                - batch_ids (LongTensor): batch indices of queries of shape [num_qrys];
                - pred_boxes (Boxes): predicted 2D bounding boxes of size [num_qrys];
                - box_scores (FloatTensor): unnormalized 2D bounding box scores of shape [num_qrys];
                - seg_mask_logits (FloatTensor): segmentation mask logits of shape [num_qrys, mH, mW].

            pred_dicts (List): List of size [num_pred_dicts] collecting various prediction dictionaries.
            kwargs (Dict): Dictionary of unused keyword arguments.

        Returns:
            pred_dicts (List): List with prediction dictionaries containing following additional entry:
                pred_dict (Dict): Prediction dictionary containing following keys:
                    - labels (LongTensor): predicted class indices of shape [num_preds];
                    - mask_scores (FloatTensor): predicted segmentation mask scores of shape [num_preds, iH, iW];
                    - scores (FloatTensor): normalized prediction scores of shape [num_preds];
                    - batch_ids (LongTensor): batch indices of predictions of shape [num_preds].

        Raises:
            ValueError: Error when an invalid type of duplicate removal or rescoring mechanism is provided.
            ValueError: Error when an invalid mask type is provided.
        """

        # Retrieve desired items from storage dictionary
        images = storage_dict['images']
        cls_logits = storage_dict['cls_logits']
        batch_ids = storage_dict['batch_ids']
        pred_boxes = storage_dict.get('pred_boxes', None)
        box_scores = storage_dict.get('box_scores', None)
        mask_logits = storage_dict.get('seg_mask_logits', None)

        # Get batch size and device
        batch_size = len(images)
        device = cls_logits.device

        # Get image width and height with padding
        iW, iH = images.size()

        # Get number of queries and number of classes
        num_qrys, num_labels = cls_logits.size()
        num_classes = num_labels - 1

        # Get query and batch indices
        qry_ids = torch.arange(num_qrys, device=device)[:, None].expand(-1, num_classes).reshape(-1)
        batch_ids = batch_ids[:, None].expand(-1, num_classes).reshape(-1)

        # Get prediction labels and scores
        pred_labels = torch.arange(num_classes, device=device)[None, :].expand(num_qrys, -1).reshape(-1)
        pred_scores = cls_logits[:, :-1].sigmoid().view(-1)

        # Update prediction scores with box scores if needed
        if box_scores is not None:
            box_scores = box_scores.sigmoid()
            box_scores = box_scores[:, None].expand(-1, num_classes).flatten()

            cls_power = self.score_attrs.get('cls_power', 1.0)
            box_power = self.score_attrs.get('box_power', 1.0)
            pred_scores = (pred_scores**cls_power) * (box_scores**box_power)

        # Initialize prediction dictionary
        pred_keys = ('labels', 'mask_scores', 'scores', 'batch_ids')
        pred_dict = {pred_key: [] for pred_key in pred_keys}
        pred_dict['mask_thr'] = self.mask_thr

        # Iterate over every batch entry
        for i in range(batch_size):

            # Get predictions corresponding to batch entry
            batch_mask = batch_ids == i

            qry_ids_i = qry_ids[batch_mask]
            pred_labels_i = pred_labels[batch_mask]
            pred_scores_i = pred_scores[batch_mask]

            # Remove duplicate predictions if needed
            dup_type = self.dup_attrs.get('type', None)
            dup_needs_masks = self.dup_attrs.get('needs_masks', False)

            if dup_type is not None and not dup_needs_masks:
                if dup_type == 'box_nms':
                    num_preds_i = len(pred_scores_i)

                    num_candidates = self.dup_attrs.get('nms_candidates', 1000)
                    candidate_ids = pred_scores_i.topk(min(num_candidates, num_preds_i))[1]

                    qry_ids_i = qry_ids_i[candidate_ids]
                    pred_labels_i = pred_labels_i[candidate_ids]
                    pred_scores_i = pred_scores_i[candidate_ids]

                    pred_boxes_i = pred_boxes[qry_ids_i].to_format('xyxy')
                    iou_thr = self.dup_attrs.get('nms_thr', 0.65)
                    non_dup_ids = batched_nms(pred_boxes_i.boxes, pred_scores_i, pred_labels_i, iou_thr)

                    qry_ids_i = qry_ids_i[non_dup_ids]
                    pred_labels_i = pred_labels_i[non_dup_ids]
                    pred_scores_i = pred_scores_i[non_dup_ids]

                else:
                    error_msg = f"Invalid type of duplicate removal or rescoring mechanism (got '{dup_type}')."
                    raise ValueError(error_msg)

            # Only keep top predictions if needed
            if self.max_segs is not None and not dup_needs_masks:
                if len(pred_scores_i) > self.max_segs:
                    top_pred_ids = pred_scores_i.topk(self.max_segs)[1]

                    qry_ids_i = qry_ids_i[top_pred_ids]
                    pred_labels_i = pred_labels_i[top_pred_ids]
                    pred_scores_i = pred_scores_i[top_pred_ids]

            # Get mask scores and prediction masks
            qry_ids_i, non_unique_ids = qry_ids_i.unique(sorted=False, return_inverse=True)

            if self.process_all_qrys:
                mask_logits_i = mask_logits[qry_ids_i]
            else:
                mask_logits_i = self.get_mask_logits(qry_feats, qry_ids_i, storage_dict)

            mask_scores_i = mask_logits_i.sigmoid()

            if self.mask_type == 'image':
                mask_scores_i = mask_scores_i.unsqueeze(dim=1)
                mask_scores_i = F.interpolate(mask_scores_i, size=(iH, iW), mode='bilinear', align_corners=False)
                mask_scores_i = mask_scores_i.squeeze(dim=1)

            elif self.mask_type == 'roi':
                pred_boxes_i = pred_boxes[qry_ids_i]
                pred_boxes_i = pred_boxes_i.to_format('xyxy').to_img_scale(images).boxes

                mask_scores_i = mask_scores_i.unsqueeze(dim=1)
                mask_scores_i = _do_paste_mask(mask_scores_i, pred_boxes_i, iH, iW, skip_empty=False)[0]

            else:
                error_msg = f"Invalid mask type in BaseSegHead (got '{self.mask_type}')."
                raise ValueError(error_msg)

            mask_scores_i = mask_scores_i[non_unique_ids]
            pred_masks_i = mask_scores_i > self.mask_thr

            # Update prediction scores based on mask scores
            mask_areas = pred_masks_i.flatten(1).sum(dim=1).clamp_(min=1)
            avg_mask_scores = (pred_masks_i * mask_scores_i).flatten(1).sum(dim=1) / mask_areas

            mask_power = self.score_attrs.get('mask_power', 1.0)
            pred_scores_i = pred_scores_i * (avg_mask_scores ** mask_power)

            # Remove duplicate predictions if needed
            if dup_type is not None and dup_needs_masks:
                if dup_type == 'box_nms':
                    num_preds_i = len(pred_scores_i)

                    num_candidates = self.dup_attrs.get('nms_candidates', 1000)
                    candidate_ids = pred_scores_i.topk(min(num_candidates, num_preds_i))[1]

                    pred_labels_i = pred_labels_i[candidate_ids]
                    mask_scores_i = mask_scores_i[candidate_ids]
                    pred_masks_i = pred_masks_i[candidate_ids]
                    pred_scores_i = pred_scores_i[candidate_ids]

                    pred_boxes_i = mask_to_box(pred_masks_i).to_format('xyxy')
                    iou_thr = self.dup_attrs.get('nms_thr', 0.65)
                    non_dup_ids = batched_nms(pred_boxes_i.boxes, pred_scores_i, pred_labels_i, iou_thr)

                    pred_labels_i = pred_labels_i[non_dup_ids]
                    mask_scores_i = mask_scores_i[non_dup_ids]
                    pred_scores_i = pred_scores_i[non_dup_ids]

                else:
                    error_msg = f"Invalid type of duplicate removal or rescoring mechanism (got '{dup_type}')."
                    raise ValueError(error_msg)

            # Only keep top predictions if needed
            if self.max_segs is not None and dup_needs_masks:
                if len(pred_scores_i) > self.max_segs:
                    top_pred_ids = pred_scores_i.topk(self.max_segs)[1]

                    pred_labels_i = pred_labels_i[top_pred_ids]
                    mask_scores_i = mask_scores_i[top_pred_ids]
                    pred_scores_i = pred_scores_i[top_pred_ids]

            # Reweight mask scores if needed
            if self.pred_mask_type == 'panoptic':
                mask_scores_i = pred_scores_i[:, None, None] * mask_scores_i

            # Add predictions to prediction dictionary
            pred_dict['labels'].append(pred_labels_i)
            pred_dict['mask_scores'].append(mask_scores_i)
            pred_dict['scores'].append(pred_scores_i)
            pred_dict['batch_ids'].append(torch.full_like(pred_labels_i, i))

        # Concatenate predictions of different batch entries
        pred_dict.update({k: torch.cat(v, dim=0) for k, v in pred_dict.items() if k != 'mask_thr'})

        # Add prediction dictionary to list of prediction dictionaries
        pred_dicts.append(pred_dict)

        return pred_dicts

    @torch.no_grad()
    def draw_segs(self, storage_dict, images_dict, pred_dicts, tgt_dict=None, vis_score_thr=0.4, id=None, **kwargs):
        """
        Draws predicted and target segmentations on the corresponding images.

        Segmentations must have a score of at least the score threshold to be drawn. Target segmentations get a default
        100% score.

        Args:
            storage_dict (Dict): Storage dictionary containing at least following key:
                - images (Images): images structure containing the batched images of size [batch_size].

            images_dict (Dict): Dictionary with annotated images of predictions/targets.

            pred_dicts (List): List with prediction dictionaries containing as last entry:
                pred_dict (Dict): Prediction dictionary containing following keys:
                    - labels (LongTensor): predicted class indices of shape [num_preds];
                    - mask_scores (FloatTensor): predicted segmentation mask scores of shape [num_preds, iH, iW];
                    - scores (FloatTensor): normalized prediction scores of shape [num_preds];
                    - batch_ids (LongTensor): batch indices of predictions of shape [num_preds].

            tgt_dict (Dict): Optional target dictionary containing at least following keys when given:
                - labels (LongTensor): target class indices of shape [num_targets];
                - masks (BoolTensor): target segmentation masks of shape [num_targets, iH, iW];
                - sizes (LongTensor): cumulative number of targets per batch entry of size [batch_size+1].

            vis_score_thr (float): Threshold indicating the minimum score for an instance to be drawn (default=0.4).
            id (int or str): Integer or string containing the head id (default=None).
            kwargs (Dict): Dictionary of unused keyword arguments.

        Returns:
            images_dict (Dict): Dictionary containing additional images annotated with segmentations.

        Raises:
            ValueError: Error when an invalid prediction mask type is provided.
        """

        # Retrieve images from storage dictionary and get batch size
        images = storage_dict['images']
        batch_size = len(images)

        # Initialize list of draw dictionaries and list of dictionary names
        draw_dicts = []
        dict_names = []

        # Get prediction draw dictionary and dictionary name
        pred_dict = pred_dicts[-1]
        draw_dict = {}

        if self.pred_mask_type == 'instance':
            pred_scores = pred_dict['scores']
            sufficient_score = pred_scores >= vis_score_thr

            draw_dict['labels'] = pred_dict['labels'][sufficient_score]
            draw_dict['masks'] = pred_dict['mask_scores'][sufficient_score] > 0.5
            draw_dict['scores'] = pred_scores[sufficient_score]

            pred_batch_ids = pred_dict['batch_ids'][sufficient_score]
            pred_sizes = [0] + [(pred_batch_ids == i).sum() for i in range(batch_size)]
            pred_sizes = torch.tensor(pred_sizes, device=pred_scores.device).cumsum(dim=0)
            draw_dict['sizes'] = pred_sizes

        elif self.pred_mask_type == 'panoptic':
            thing_ids = tuple(self.metadata.thing_dataset_id_to_contiguous_id.values())

            pred_labels = pred_dict['labels']
            mask_scores = pred_dict['mask_scores']
            pred_batch_ids = pred_dict['batch_ids']

            pred_sizes = [0] + [(pred_batch_ids == i).sum() for i in range(batch_size)]
            pred_sizes = torch.tensor(pred_sizes, device=pred_labels.device).cumsum(dim=0)
            draw_dict['sizes'] = pred_sizes

            mask_list = []
            segments = []

            for i0, i1 in zip(pred_sizes[:-1], pred_sizes[1:]):
                mask_i = mask_scores[i0:i1].argmax(dim=0)
                mask_list.append(mask_i)

                for j in range(i1-i0):
                    cat_id = pred_labels[i0+j]
                    is_thing = cat_id in thing_ids

                    segment_ij = {'id': j, 'category_id': cat_id, 'isthing': is_thing}
                    segments.append(segment_ij)

            draw_dict['masks'] = torch.stack(mask_list, dim=0)
            draw_dict['segments'] = segments

        else:
            error_msg = f"Invalid prediction mask type in BaseSegHead (got '{self.pred_mask_type}')."
            raise ValueError(error_msg)

        draw_dicts.append(draw_dict)
        dict_name = f"seg_pred_{id}" if id is not None else "seg_pred"
        dict_names.append(dict_name)

        # Get target draw dictionary and dictionary name if needed
        if tgt_dict is not None and not any('seg_tgt' in key for key in images_dict.keys()):
            draw_dict = {}

            if self.pred_mask_type == 'instance':
                draw_dict['labels'] = tgt_dict['labels']
                draw_dict['masks'] = tgt_dict['masks']
                draw_dict['scores'] = torch.ones_like(tgt_dict['labels'], dtype=torch.float)
                draw_dict['sizes'] = tgt_dict['sizes']

            elif self.pred_mask_type == 'panoptic':
                tgt_labels = tgt_dict['labels']
                tgt_masks = tgt_dict['masks']
                tgt_sizes = tgt_dict['sizes']

                mask_list = []
                segments = []

                for i0, i1 in zip(tgt_sizes[:-1], tgt_sizes[1:]):
                    mask_i = tgt_masks[i0:i1].int().argmax(dim=0)
                    mask_list.append(mask_i)

                    for j in range(i1-i0):
                        cat_id = tgt_labels[i0+j]
                        is_thing = cat_id in thing_ids

                        segment_ij = {'id': j, 'category_id': cat_id, 'isthing': is_thing}
                        segments.append(segment_ij)

                draw_dict['masks'] = torch.stack(mask_list, dim=0)
                draw_dict['segments'] = segments
                draw_dict['sizes'] = tgt_sizes

            else:
                error_msg = f"Invalid prediction mask type in BaseSegHead (got '{self.pred_mask_type}')."
                raise ValueError(error_msg)

            draw_dicts.append(draw_dict)
            dict_names.append('seg_tgt')

        # Get image sizes without padding in (width, height) format
        img_sizes = images.size(mode='without_padding')

        # Get image sizes with padding in (height, width) format
        img_sizes_pad = images.size(mode='with_padding')
        img_sizes_pad = (img_sizes_pad[1], img_sizes_pad[0])

        # Draw 2D object detections on images and add them to images dictionary
        for dict_name, draw_dict in zip(dict_names, draw_dicts):
            sizes = draw_dict['sizes']

            for i, i0, i1 in zip(range(batch_size), sizes[:-1], sizes[1:]):
                img_size = img_sizes[i]
                img_size = (img_size[1], img_size[0])

                image = images.images[i].clone()
                image = T.crop(image, 0, 0, *img_size)
                image = image.permute(1, 2, 0) * 255
                image = image.to(torch.uint8).cpu().numpy()

                if self.pred_mask_type == 'instance':
                    visualizer = Visualizer(image, metadata=self.metadata, instance_mode=ColorMode.IMAGE)

                    if i1 > i0:
                        labels_i = draw_dict['labels'][i0:i1].cpu().numpy()
                        scores_i = draw_dict['scores'][i0:i1].cpu().numpy()

                        masks_i = draw_dict['masks'][i0:i1]
                        masks_i = T.resize(masks_i, img_sizes_pad)
                        masks_i = T.crop(masks_i, 0, 0, *img_size)
                        masks_i = masks_i.cpu().numpy()

                        instances = Instances(img_size, pred_classes=labels_i, pred_masks=masks_i, scores=scores_i)
                        visualizer.draw_instance_predictions(instances)

                elif self.pred_mask_type == 'panoptic':
                    mask_i = draw_dict['masks'][i].cpu()
                    segments_i = draw_dict['segments'][i0:i1]

                    visualizer = Visualizer(image, metadata=self.metadata, instance_mode=ColorMode.SEGMENTATION)
                    visualizer.draw_panoptic_seg(mask_i, segments_i)

                else:
                    error_msg = f"Invalid prediction mask type in BaseSegHead (got '{self.pred_mask_type}')."
                    raise ValueError(error_msg)

                annotated_image = visualizer.output.get_image()
                images_dict[f'{dict_name}_{i}'] = annotated_image

        return images_dict

    def forward_pred(self, qry_feats, storage_dict, images_dict=None, **kwargs):
        """
        Forward prediction method of the BaseSegHead module.

        Args:
            qry_feats (FloatTensor): Query features of shape [num_qrys, qry_feat_size].

            storage_dict (Dict): Storage dictionary containing at least following keys:
                - images (Images): images structure containing the batched images of size [batch_size];
                - feat_maps (List): list of size [num_maps] with feature maps of shape [batch_size, feat_size, fH, fW];
                - cum_feats_batch (LongTensor): cumulative number of queries per batch entry [batch_size+1];
                - pred_boxes (Boxes): predicted 2D bounding boxes of size [num_qrys].

            images_dict (Dict): Dictionary with annotated images of predictions/targets (default=None).
            kwargs (Dict): Dictionary of keyword arguments passed to underlying methods.

        Returns:
            storage_dict (Dict): Storage dictionary (possibly) containing following additional keys:
                - seg_mask_logits (FloatTensor): segmentation mask logits of shape [num_qrys, mH, mW];
                - seg_qry_bnd_mask (BoolTensor): segmentation boundary mask of shape [num_qrys, 1, mH, mW];
                - seg_batch_bnd_mask (BoolTensor): segmentation boundary mask of shape [batch_size, 1, fH, fW];
                - seg_qry_unc_mask (BoolTensor): segmentation uncertainty mask of shape [num_qrys, 1, mH, mW];
                - seg_batch_unc_mask (BoolTensor): segmentation uncertainty mask of shape [batch_size, 1, fH, fW].

            images_dict (Dict): Dictionary containing additional images annotated with segmentations (if given).
        """

        # Compute mask logits for all queries if needed
        if self.process_all_qrys:
            seg_qry_ids = torch.arange(len(qry_feats), device=qry_feats.device)
            mask_logits = self.get_mask_logits(qry_feats, seg_qry_ids, storage_dict)
            storage_dict['seg_mask_logits'] = mask_logits

        # Compute segmentation boundary mask if needed
        if self.get_bnd_mask:
            seg_mask_logits = storage_dict['seg_mask_logits'].unsqueeze(dim=1)
            seg_masks = seg_mask_logits.sigmoid() > self.mask_thr

            kernel = seg_mask_logits.new_ones([1, 1, 3, 3])
            bnd_masks = F.conv2d(seg_masks.float(), kernel, padding=1)

            bnd_masks = (bnd_masks > 0) & (bnd_masks < 9)
            storage_dict['seg_qry_bnd_mask'] = bnd_masks

            if self.mask_type == 'roi':
                key_feat_map = storage_dict['feat_maps'][self.key_map_id]
                fH, fW = key_feat_map.size()[2:]

                images = storage_dict['images']
                boxes = storage_dict['pred_boxes'].clone()

                scale = torch.tensor([[fW, fH, fW, fH]], device=key_feat_map.device)
                boxes = scale * boxes.to_format('xyxy').normalize(images).boxes

                bnd_masks = _do_paste_mask(bnd_masks, boxes, fH, fW, skip_empty=False)[0]
                bnd_masks = bnd_masks > 0.5

            cum_feats_batch = storage_dict['cum_feats_batch']
            batch_size = len(cum_feats_batch) - 1
            bnd_mask_list = []

            for i in range(batch_size):
                i0 = cum_feats_batch[i]
                i1 = cum_feats_batch[i+1]

                bnd_mask_i = bnd_masks[i0:i1].any(dim=0)
                bnd_mask_list.append(bnd_mask_i)

            bnd_mask = torch.stack(bnd_mask_list, dim=0)
            storage_dict['seg_batch_bnd_mask'] = bnd_mask

        # Compute segmentation uncertainty mask if needed
        if self.get_unc_mask:
            seg_mask_logits = storage_dict['seg_mask_logits']
            num_qrys, mH, mW = seg_mask_logits.size()

            unc_maps = -seg_mask_logits.abs().flatten(1)
            unc_ids = unc_maps.topk(self.unc_thr, dim=1, sorted=False)[1]
            qry_ids = torch.arange(num_qrys, device=unc_maps.device)[:, None].expand(-1, self.unc_thr)

            unc_masks = torch.zeros(num_qrys, mH*mW, device=unc_maps.device, dtype=torch.bool)
            unc_masks[qry_ids, unc_ids] = True

            unc_masks = unc_masks.view(num_qrys, 1, mH, mW)
            storage_dict['seg_qry_unc_mask'] = unc_masks

            if self.mask_type == 'roi':
                key_feat_map = storage_dict['feat_maps'][self.key_map_id]
                fH, fW = key_feat_map.size()[2:]

                images = storage_dict['images']
                boxes = storage_dict['pred_boxes'].clone()

                scale = torch.tensor([[fW, fH, fW, fH]], device=key_feat_map.device)
                boxes = scale * boxes.to_format('xyxy').normalize(images).boxes

                unc_masks = _do_paste_mask(unc_masks, boxes, fH, fW, skip_empty=False)[0]
                unc_masks = unc_masks > 0.5

            cum_feats_batch = storage_dict['cum_feats_batch']
            batch_size = len(cum_feats_batch) - 1
            unc_mask_list = []

            for i in range(batch_size):
                i0 = cum_feats_batch[i]
                i1 = cum_feats_batch[i+1]

                unc_mask_i = unc_masks[i0:i1].any(dim=0)
                unc_mask_list.append(unc_mask_i)

            unc_mask = torch.stack(unc_mask_list, dim=0)
            storage_dict['seg_batch_unc_mask'] = unc_mask

        # Get segmentation predictions if needed
        if self.get_segs and not self.training:
            self.compute_segs(qry_feats, storage_dict=storage_dict, **kwargs)

        # Draw predicted and target segmentations if needed
        if self.get_segs and images_dict is not None:
            self.draw_segs(storage_dict=storage_dict, images_dict=images_dict, **kwargs)

        return storage_dict, images_dict

    def forward_loss(self, qry_feats, storage_dict, tgt_dict, loss_dict, analysis_dict=None, id=None, **kwargs):
        """
        Forward loss method of the BaseSegHead module.

        Args:
            qry_feats (FloatTensor): Query features of shape [num_qrys, qry_feat_size].

            storage_dict (Dict): Storage dictionary (possibly) containing following keys (after matching):
                - {self.update_mask_key} (BoolTensor): mask with locations to update of shape [num_qrys, 1, mH, mW];
                - seg_mask_logits (FloatTensor): segmentation mask logits of shape [num_qrys, mH, mW];
                - matched_qry_ids (LongTensor): indices of matched queries of shape [num_pos_qrys];
                - matched_tgt_ids (LongTensor): indices of corresponding matched targets of shape [num_pos_qrys].

            tgt_dict (Dict): Target dictionary containing at least following key:
                - masks (BoolTensor): target segmentation masks of shape [num_targets, iH, iW].

            loss_dict (Dict): Dictionary containing different weighted loss terms.
            analysis_dict (Dict): Dictionary containing different analyses (default=None).
            id (int or str): Integer or string containing the head id (default=None).
            kwargs (Dict): Dictionary of keyword arguments passed to some underlying modules.

        Returns:
            loss_dict (Dict): Loss dictionary containing following additional key:
                - mask_loss (FloatTensor): segmentation mask loss of shape [].

            analysis_dict (Dict): Analysis dictionary containing following additional key (if not None):
                - mask_acc (FloatTensor): segmentation mask accuracy of shape [].

        Raises:
            ValueError: Error when an invalid mask type is provided.
        """

        # Get device
        device = qry_feats.device

        # Perform matching if matcher is available
        if self.matcher is not None:
            self.matcher(storage_dict=storage_dict, tgt_dict=tgt_dict, analysis_dict=analysis_dict, **kwargs)

        # Retrieve matching results
        matched_qry_ids = storage_dict['matched_qry_ids']
        matched_tgt_ids = storage_dict['matched_tgt_ids']

        # Handle case where there are no positive matches
        if len(matched_qry_ids) == 0:

            # Get mask loss
            mask_loss = 0.0 * qry_feats.sum() + sum(0.0 * p.flatten()[0] for p in self.parameters())
            key_name = f'mask_loss_{id}' if id is not None else 'mask_loss'
            loss_dict[key_name] = mask_loss

            # Get mask accuracy if needed
            if analysis_dict is not None:
                mask_acc = 1.0 if len(tgt_dict['masks']) == 0 else 0.0
                mask_acc = torch.tensor(mask_acc, dtype=mask_loss.dtype, device=device)

                key_name = f'mask_acc_{id}' if id is not None else 'mask_acc'
                analysis_dict[key_name] = 100 * mask_acc

            return loss_dict, analysis_dict

        # Get mask logits
        if self.process_all_qrys:
            mask_logits = storage_dict['seg_mask_logits'][matched_qry_ids]
        else:
            mask_logits = self.get_mask_logits(qry_feats, matched_qry_ids, storage_dict)

        # Get mask targets
        tgt_masks = tgt_dict['masks']

        if self.mask_type == 'image':
            fH, fW = mask_logits.size()[1:]

            mask_targets = tgt_masks[matched_tgt_ids].float().unsqueeze(dim=1)
            mask_targets = F.interpolate(mask_targets, size=(fH, fW), mode='bilinear', align_corners=False)
            mask_targets = mask_targets.squeeze(dim=1)

        elif self.mask_type == 'roi':
            images = storage_dict['images']
            pred_boxes = storage_dict['pred_boxes'][matched_qry_ids]

            roi_boxes = pred_boxes.to_format('xyxy').to_img_scale(images).boxes.detach()
            roi_boxes = torch.cat([matched_tgt_ids[:, None], roi_boxes], dim=1)

            mask_targets = tgt_masks.float().unsqueeze(dim=1)
            mask_targets = self.tgt_roi_ext([mask_targets], roi_boxes)
            mask_targets = mask_targets.squeeze(dim=1)

        else:
            error_msg = f"Invalid mask type in BaseSegHead (got '{self.mask_type}')."
            raise ValueError(error_msg)

        # Get mask logits and target in desired points
        if self.loss_sample is not None:
            mask_logits, mask_targets = self.loss_sample(mask_logits, mask_targets)

        elif self.mask_update and self.loss_updated_only:
            update_mask = storage_dict[self.update_mask_key]
            update_mask = update_mask[matched_qry_ids].squeeze(dim=1)

            mask_logits = mask_logits[update_mask]
            mask_targets = mask_targets[update_mask]

        # Get mask loss
        if mask_logits.numel() == 0:
            mask_loss = 0.0 * qry_feats.sum() + sum(0.0 * p.flatten()[0] for p in self.parameters())
        else:
            mask_loss = self.loss(mask_logits, mask_targets)

        if self.mask_update and self.loss_updated_only:
            mask_loss *= len(matched_qry_ids)

        key_name = f'mask_loss_{id}' if id is not None else 'mask_loss'
        loss_dict[key_name] = mask_loss

        # Get mask accuracy if needed
        if analysis_dict is not None:

            if self.loss_sample is not None or (self.mask_update and self.loss_updated_only):
                mask_preds = mask_logits.sigmoid() > self.mask_thr
                mask_targets = mask_targets > 0.5

                if mask_preds.numel() > 0:
                    mask_acc = (mask_preds == mask_targets).sum() / mask_preds.numel()
                else:
                    mask_acc = torch.tensor(1.0, device=device)

            else:
                mask_preds = mask_logits.flatten(1).sigmoid() > self.mask_thr
                mask_targets = mask_targets.flatten(1) > 0.5

                mask_inter = (mask_preds * mask_targets).sum(dim=1)
                mask_union = mask_preds.sum(dim=1) + mask_targets.sum(dim=1) - mask_inter

                mask_acc = mask_inter / mask_union.clamp_(min=1)
                mask_acc = mask_acc.mean()

            key_name = f'mask_acc_{id}' if id is not None else 'mask_acc'
            analysis_dict[key_name] = 100 * mask_acc

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
            analysis_dict[key_name] = seg_bnd_acc

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


@MODELS.register_module()
class TopDownSegHead(nn.Module):
    """
    Class implementing the TopDownSegHead module.

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
        mask_thr (float): Value containing the mask threshold used to determine the segmentation masks.
        pred_mask_type (str): String containing the type of predicted segmentation mask.
        metadata (detectron2.data.Metadata): Metadata instance containing additional dataset information.
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
                 roi_ins_cfg=None, get_segs=True, dup_attrs=None, max_segs=None, pred_mask_type='instance',
                 matcher_cfg=None, roi_sizes=None, apply_ids=None, **kwargs):
        """
        Initializes the TopDownSegHead module.

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
            dup_attrs (Dict): Attribute dictionary specifying the duplicate removal mechanism (default=None).
            max_segs (int): Integer with the maximum number of returned segmentation predictions (default=None).
            pred_mask_type (str): String containing the type of predicted segmentation mask (default='instance').
            matcher_cfg (Dict): Configuration dictionary specifying the matcher module (default=None).
            roi_sizes (Tuple): Tuple of size [seg_iters] containing the RoI sizes (default=None).
            apply_ids (List): List with integers determining when the head should be applied (default=None).
            kwargs (Dict): Dictionary of unused keyword arguments.
        """

        # Initialization of default nn.Module
        super().__init__()

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
        self.pred_mask_type = pred_mask_type
        self.metadata = metadata
        self.seg_loss_weights = seg_loss_weights
        self.ref_loss_weights = ref_loss_weights
        self.apply_ids = apply_ids

    @torch.no_grad()
    def compute_segs(self, qry_feats, storage_dict, pred_dicts, **kwargs):
        """
        Method computing the segmentation predictions.

        Args:
            qry_feats (FloatTensor): Query features of shape [num_qrys, qry_feat_size].

            storage_dict (Dict): Storage dictionary containing at least following keys:
                - images (Images): images structure containing the batched images of size [batch_size];
                - cls_logits (FloatTensor): classification logits of shape [num_qrys, num_labels];
                - cum_feats_batch (LongTensor): cumulative number of queries per batch entry [batch_size+1];
                - pred_boxes (Boxes): predicted 2D bounding boxes of size [num_qrys].

            pred_dicts (List): List of size [num_pred_dicts] collecting various prediction dictionaries.
            kwargs (Dict): Dictionary of unused keyword arguments.

        Returns:
            pred_dicts (List): List with prediction dictionaries containing following additional entry:
                pred_dict (Dict): Prediction dictionary containing following keys:
                    - labels (LongTensor): predicted class indices of shape [num_preds];
                    - mask_scores (FloatTensor): predicted segmentation mask scores of shape [num_preds, iH, iW];
                    - scores (FloatTensor): normalized prediction scores of shape [num_preds];
                    - batch_ids (LongTensor): batch indices of predictions of shape [num_preds].

        Raises:
            ValueError: Error when an invalid type of duplicate removal mechanism is provided.
        """

        # Retrieve various items from storage dictionary
        images = storage_dict['images']
        cls_logits = storage_dict['cls_logits']
        cum_feats_batch = storage_dict['cum_feats_batch']
        pred_boxes = storage_dict['pred_boxes']

        # Get image size, number of queries, number of classes and device
        iW, iH = images.size()
        num_qrys = cls_logits.size(dim=0)
        num_classes = cls_logits.size(dim=1) - 1
        device = cls_logits.device

        # Get prediction query indices,labels and scores
        pred_qry_ids = torch.arange(num_qrys, device=device)[:, None].expand(-1, num_classes).reshape(-1)
        pred_labels = torch.arange(num_classes, device=device)[None, :].expand(num_qrys, -1).reshape(-1)
        pred_scores = cls_logits[:, :-1].sigmoid().view(-1)

        # Get prediction boxes in desired format
        pred_boxes = pred_boxes.to_format('xyxy').to_img_scale(images).boxes

        # Get batch indices
        batch_size = len(cum_feats_batch) - 1
        batch_ids = torch.arange(batch_size, device=device)
        batch_ids = batch_ids.repeat_interleave(cum_feats_batch.diff())
        batch_ids = batch_ids[:, None].expand(-1, num_classes).reshape(-1)

        # Initialize prediction dictionary
        pred_keys = ('labels', 'mask_scores', 'scores', 'batch_ids')
        pred_dict = {pred_key: [] for pred_key in pred_keys}
        pred_dict['mask_thr'] = self.mask_thr

        # Iterate over every batch entry
        for i in range(batch_size):

            # Get predictions corresponding to batch entry
            batch_mask = batch_ids == i

            pred_qry_ids_i = pred_qry_ids[batch_mask]
            pred_labels_i = pred_labels[batch_mask]
            pred_scores_i = pred_scores[batch_mask]

            # Remove duplicate predictions if needed
            if self.dup_attrs is not None:
                dup_removal_type = self.dup_attrs['type']

                if dup_removal_type == 'nms':
                    num_candidates = self.dup_attrs['nms_candidates']
                    num_preds = len(pred_scores_i)
                    candidate_ids = pred_scores_i.topk(min(num_candidates, num_preds))[1]

                    pred_qry_ids_i = pred_qry_ids_i[candidate_ids]
                    pred_labels_i = pred_labels_i[candidate_ids]
                    pred_scores_i = pred_scores_i[candidate_ids]

                    pred_boxes_i = pred_boxes[pred_qry_ids_i]
                    iou_thr = self.dup_attrs['nms_thr']
                    non_dup_ids = batched_nms(pred_boxes_i, pred_scores_i, pred_labels_i, iou_thr)

                    pred_qry_ids_i = pred_qry_ids_i[non_dup_ids]
                    pred_labels_i = pred_labels_i[non_dup_ids]
                    pred_scores_i = pred_scores_i[non_dup_ids]

                else:
                    error_msg = f"Invalid type of duplicate removal mechanism (got '{dup_removal_type}')."
                    raise ValueError(error_msg)

            # Only keep top predictions if needed
            if self.max_segs is not None:
                if len(pred_scores_i) > self.max_segs:
                    top_pred_ids = pred_scores_i.topk(self.max_segs)[1]

                    pred_qry_ids_i = pred_qry_ids_i[top_pred_ids]
                    pred_labels_i = pred_labels_i[top_pred_ids]
                    pred_scores_i = pred_scores_i[top_pred_ids]

            # Get query indices for which to compute segmentations
            seg_qry_ids, pred_inv_ids = pred_qry_ids_i.unique(sorted=True, return_inverse=True)

            # Get segmentation and refinement predictions for desired queries
            self.get_preds(qry_feats, storage_dict, seg_qry_ids=seg_qry_ids, **kwargs)

            # Retrieve various items related to segmentation predictions from storage dictionary
            roi_ids_list = storage_dict['roi_ids_list']
            pos_ids_list = storage_dict['pos_ids_list']
            seg_logits_list = storage_dict['seg_logits_list']

            # Get prediction masks
            num_rois = len(seg_qry_ids)
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
            pred_boxes_i = pred_boxes[seg_qry_ids]

            mask_scores_i = _do_paste_mask(mask_scores_i, pred_boxes_i, iH, iW, skip_empty=False)[0]
            mask_scores_i = mask_scores_i[pred_inv_ids]

            pred_masks_i = mask_scores_i > self.mask_thr
            pred_scores_i = pred_scores_i * (pred_masks_i * mask_scores_i).flatten(1).sum(dim=1)
            pred_scores_i = pred_scores_i / (pred_masks_i.flatten(1).sum(dim=1) + 1e-6)

            if self.pred_mask_type == 'panoptic':
                mask_scores_i = pred_scores_i[:, None, None] * mask_scores_i

            # Add predictions to prediction dictionary
            pred_dict['labels'].append(pred_labels_i)
            pred_dict['mask_scores'].append(mask_scores_i)
            pred_dict['scores'].append(pred_scores_i)
            pred_dict['batch_ids'].append(torch.full_like(pred_labels_i, i))

        # Concatenate predictions of different batch entries
        pred_dict.update({k: torch.cat(v, dim=0) for k, v in pred_dict.items() if k != 'mask_thr'})

        # Add prediction dictionary to list of prediction dictionaries
        pred_dicts.append(pred_dict)

        return pred_dicts

    @torch.no_grad()
    def draw_segs(self, storage_dict, images_dict, pred_dicts, tgt_dict=None, vis_score_thr=0.4, id=None, **kwargs):
        """
        Draws predicted and target segmentations on the corresponding images.

        Segmentations must have a score of at least the score threshold to be drawn. Target segmentations get a default
        100% score.

        Args:
            storage_dict (Dict): Storage dictionary containing at least following key:
                - images (Images): images structure containing the batched images of size [batch_size].

            images_dict (Dict): Dictionary with annotated images of predictions/targets.

            pred_dicts (List): List with prediction dictionaries containing as last entry:
                pred_dict (Dict): Prediction dictionary containing following keys:
                    - labels (LongTensor): predicted class indices of shape [num_preds];
                    - mask_scores (FloatTensor): predicted segmentation mask scores of shape [num_preds, iH, iW];
                    - scores (FloatTensor): normalized prediction scores of shape [num_preds];
                    - batch_ids (LongTensor): batch indices of predictions of shape [num_preds].

            tgt_dict (Dict): Optional target dictionary containing at least following keys when given:
                - labels (LongTensor): target class indices of shape [num_targets];
                - masks (BoolTensor): target segmentation masks of shape [num_targets, iH, iW];
                - sizes (LongTensor): cumulative number of targets per batch entry of size [batch_size+1].

            vis_score_thr (float): Threshold indicating the minimum score for an instance to be drawn (default=0.4).
            id (int or str): Integer or string containing the head id (default=None).
            kwargs (Dict): Dictionary of unused keyword arguments.

        Returns:
            images_dict (Dict): Dictionary containing additional images annotated with segmentations.

        Raises:
            ValueError: Error when an invalid prediction mask type is provided.
        """

        # Retrieve images from storage dictionary and get batch size
        images = storage_dict['images']
        batch_size = len(images)

        # Initialize list of draw dictionaries and list of dictionary names
        draw_dicts = []
        dict_names = []

        # Get prediction draw dictionary and dictionary name
        pred_dict = pred_dicts[-1]
        draw_dict = {}

        if self.pred_mask_type == 'instance':
            pred_scores = pred_dict['scores']
            sufficient_score = pred_scores >= vis_score_thr

            draw_dict['labels'] = pred_dict['labels'][sufficient_score]
            draw_dict['masks'] = pred_dict['mask_scores'][sufficient_score] > 0.5
            draw_dict['scores'] = pred_scores[sufficient_score]

            pred_batch_ids = pred_dict['batch_ids'][sufficient_score]
            pred_sizes = [0] + [(pred_batch_ids == i).sum() for i in range(batch_size)]
            pred_sizes = torch.tensor(pred_sizes, device=pred_scores.device).cumsum(dim=0)
            draw_dict['sizes'] = pred_sizes

        elif self.pred_mask_type == 'panoptic':
            thing_ids = tuple(self.metadata.thing_dataset_id_to_contiguous_id.values())

            pred_labels = pred_dict['labels']
            mask_scores = pred_dict['mask_scores']
            pred_batch_ids = pred_dict['batch_ids']

            pred_sizes = [0] + [(pred_batch_ids == i).sum() for i in range(batch_size)]
            pred_sizes = torch.tensor(pred_sizes, device=pred_labels.device).cumsum(dim=0)
            draw_dict['sizes'] = pred_sizes

            mask_list = []
            segments = []

            for i0, i1 in zip(pred_sizes[:-1], pred_sizes[1:]):
                mask_i = mask_scores[i0:i1].argmax(dim=0)
                mask_list.append(mask_i)

                for j in range(i1-i0):
                    cat_id = pred_labels[i0+j]
                    is_thing = cat_id in thing_ids

                    segment_ij = {'id': j, 'category_id': cat_id, 'isthing': is_thing}
                    segments.append(segment_ij)

            draw_dict['masks'] = torch.stack(mask_list, dim=0)
            draw_dict['segments'] = segments

        else:
            error_msg = f"Invalid prediction mask type in TopDownSegHead (got '{self.pred_mask_type}')."
            raise ValueError(error_msg)

        draw_dicts.append(draw_dict)
        dict_name = f"seg_pred_{id}" if id is not None else "seg_pred"
        dict_names.append(dict_name)

        # Get target draw dictionary and dictionary name if needed
        if tgt_dict is not None and not any('seg_tgt' in key for key in images_dict.keys()):
            draw_dict = {}

            if self.pred_mask_type == 'instance':
                draw_dict['labels'] = tgt_dict['labels']
                draw_dict['masks'] = tgt_dict['masks']
                draw_dict['scores'] = torch.ones_like(tgt_dict['labels'], dtype=torch.float)
                draw_dict['sizes'] = tgt_dict['sizes']

            elif self.pred_mask_type == 'panoptic':
                tgt_labels = tgt_dict['labels']
                tgt_masks = tgt_dict['masks']
                tgt_sizes = tgt_dict['sizes']

                mask_list = []
                segments = []

                for i0, i1 in zip(tgt_sizes[:-1], tgt_sizes[1:]):
                    mask_i = tgt_masks[i0:i1].int().argmax(dim=0)
                    mask_list.append(mask_i)

                    for j in range(i1-i0):
                        cat_id = tgt_labels[i0+j]
                        is_thing = cat_id in thing_ids

                        segment_ij = {'id': j, 'category_id': cat_id, 'isthing': is_thing}
                        segments.append(segment_ij)

                draw_dict['masks'] = torch.stack(mask_list, dim=0)
                draw_dict['segments'] = segments
                draw_dict['sizes'] = tgt_sizes

            else:
                error_msg = f"Invalid prediction mask type in TopDownSegHead (got '{self.pred_mask_type}')."
                raise ValueError(error_msg)

            draw_dicts.append(draw_dict)
            dict_names.append('seg_tgt')

        # Get image sizes without padding in (width, height) format
        img_sizes = images.size(mode='without_padding')

        # Get image sizes with padding in (height, width) format
        img_sizes_pad = images.size(mode='with_padding')
        img_sizes_pad = (img_sizes_pad[1], img_sizes_pad[0])

        # Draw 2D object detections on images and add them to images dictionary
        for dict_name, draw_dict in zip(dict_names, draw_dicts):
            sizes = draw_dict['sizes']

            for i, i0, i1 in zip(range(batch_size), sizes[:-1], sizes[1:]):
                img_size = img_sizes[i]
                img_size = (img_size[1], img_size[0])

                image = images.images[i].clone()
                image = T.crop(image, 0, 0, *img_size)
                image = image.permute(1, 2, 0) * 255
                image = image.to(torch.uint8).cpu().numpy()

                if self.pred_mask_type == 'instance':
                    visualizer = Visualizer(image, metadata=self.metadata, instance_mode=ColorMode.IMAGE)

                    if i1 > i0:
                        labels_i = draw_dict['labels'][i0:i1].cpu().numpy()
                        scores_i = draw_dict['scores'][i0:i1].cpu().numpy()

                        masks_i = draw_dict['masks'][i0:i1]
                        masks_i = T.resize(masks_i, img_sizes_pad)
                        masks_i = T.crop(masks_i, 0, 0, *img_size)
                        masks_i = masks_i.cpu().numpy()

                        instances = Instances(img_size, pred_classes=labels_i, pred_masks=masks_i, scores=scores_i)
                        visualizer.draw_instance_predictions(instances)

                elif self.pred_mask_type == 'panoptic':
                    mask_i = draw_dict['masks'][i].cpu()
                    segments_i = draw_dict['segments'][i0:i1]

                    visualizer = Visualizer(image, metadata=self.metadata, instance_mode=ColorMode.SEGMENTATION)
                    visualizer.draw_panoptic_seg(mask_i, segments_i)

                else:
                    error_msg = f"Invalid prediction mask type in TopDownSegHead (got '{self.pred_mask_type}')."
                    raise ValueError(error_msg)

                annotated_image = visualizer.output.get_image()
                images_dict[f'{dict_name}_{i}'] = annotated_image

        return images_dict

    def get_preds(self, qry_feats, storage_dict, seg_qry_ids, **kwargs):
        """
        Method computing the segmentation and refinement logits for the desired queries.

        Args:
            qry_feats (FloatTensor): Query features of shape [num_qrys, qry_feat_size].

            storage_dict (Dict): Storage dictionary containing at least following keys:
                - cum_feats_batch (LongTensor): cumulative number of queries per batch entry [batch_size+1];
                - feat_maps (List): list of size [num_maps] with feature maps of shape [batch_size, feat_size, fH, fW];
                - images (Images): images structure containing the batched images of size [batch_size];
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

        # Get device
        device = qry_feats.device

        # Retrieve various items from storage dictionary
        cum_feats_batch = storage_dict['cum_feats_batch']
        key_feat_maps = storage_dict['feat_maps']
        images = storage_dict['images']
        qry_boxes = storage_dict['pred_boxes'].clone()

        # Get batch indices
        batch_size = len(cum_feats_batch) - 1
        batch_ids = torch.arange(batch_size, device=device)
        batch_ids = batch_ids.repeat_interleave(cum_feats_batch.diff())

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
        Forward prediction method of the TopDownSegHead module.

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

    def forward_loss(self, qry_feats, storage_dict, tgt_dict, loss_dict, analysis_dict, id=None, **kwargs):
        """
        Forward loss method of the TopDownSegHead module.

        Args:
            qry_feats (FloatTensor): Query features of shape [num_qrys, qry_feat_size].

            storage_dict (Dict): Storage dictionary containing following keys (after matching):
                - images (Images): images structure containing the batched images of size [batch_size];
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

        # Get device
        device = qry_feats.device

        # Perform matching if matcher is available
        if self.matcher is not None:
            self.matcher(storage_dict=storage_dict, tgt_dict=tgt_dict, analysis_dict=analysis_dict, **kwargs)

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
            error_msg = "The TopDownSegHead does not support a single query to be matched with multiple targets."
            raise ValueError(error_msg)

        # Get segmentation and refinement predictions for desired queries
        self.get_preds(qry_feats, storage_dict, seg_qry_ids=matched_qry_ids, **kwargs)

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

    def forward(self, mode, **kwargs):
        """
        Forward method of the TopDownSegHead module.

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
