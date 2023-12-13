"""
Dynamic Resolution Segmentation (DRS) head.
"""

from detectron2.layers import batched_nms
import torch
from torch import nn
import torch.nn.functional as F

from models.build import build_model, MODELS
from models.functional.loss import reduce_losses, update_loss_module
from models.functional.utils import maps_to_seq, seq_to_maps
from models.heads.seg.base import BaseSegHead
from structures.boxes import mask_to_box


@MODELS.register_module()
class DRSHead(BaseSegHead):
    """
    Class implementing the DRSHead module.

    Attributes:
        seg_qst_dicts (List): List [num_qsts] of segmentation question dictionaries, each possibly containing:
            - name (str): string containing the name of the segmentation question;
            - loss_reduction (str): string containing the loss reduction mechanism.

        qry (nn.Module): Optional module updating the query features.
        key (nn.Module): Optional module updating the key features.
        key_map_ids (List): List with feature map indices from which to compute initial mask logits.
        update_mask_key (str): String with key to retrieve update mask from the storage dictionary.
        get_bnd_masks (bool): Boolean indicating whether to get the segmentation boundary masks.
        get_gain_masks (bool): Boolean indicating whether to get the segmentation gain masks.
        get_unc_masks (bool): Boolean indicating whether to get the segmentation uncertainty masks.
        bnd_thr (int): Integer containing the relative boundary threshold per query (or None).
        rew_thr (int): Integer containing the relative reward threshold per query.
        gain_thr (int): Integer containing the relative gain threshold per query.
        unc_thr (int): Integer containing the relative uncertainty threshold per query.
        get_segs (bool): Boolean indicating whether to get segmentation predictions.
        box_mask_key (str): String with key to retrieve box mask from the storage dictionary.

        score_attrs (Dict): Dictionary specifying the scoring mechanism possibly containing following keys:
            - cls_power (float): value containing the classification score power;
            - box_power (float): value containing the box score power;
            - mask_power (float): value containing the mask score power.

        dup_attrs (Dict): Dictionary specifying the duplicate removal or rescoring mechanism possibly containing:
            - type (str): string containing the type of duplicate removal or rescoring mechanism;
            - needs_masks (bool): boolean indicating whether the duplicate mechanism needs segmentation masks;
            - nms_candidates (int): integer containing the maximum number of candidate detections retained before NMS;
            - nms_thr (float): IoU threshold used during NMS or Soft-NMS to remove or rescore duplicate detections.

        max_segs (int): Optional integer with the maximum number of returned segmentation predictions.
        mask_decay (float): Value containing the mask certainty decay factor.
        mask_thr (float): Value containing the normalized (instance) segmentation mask threshold.

        pan_post_attrs (Dict): Dictionary specifying the panoptic post-processing mechanism possibly containing:
            - score_thr (float): value containing the instance score threshold (or None);
            - nms_thr (float): value containing the IoU threshold used during mask IoU (or None);
            - pan_mask_thr (float): value containing the normalized panoptic segmentation mask threshold;
            - ins_pan_thr (float): value containing the IoU threshold between instance and panoptic masks;
            - area_thr (int): integer containing the mask area threshold (or None).

        matcher (nn.Module): Optional matcher module determining the target segmentation maps.
        loss_modules (nn.ModuleDict): Dictionary of modules computing the losses for the different questions.
        apply_ids (List): List with integers determining when the head should be applied.
    """

    def __init__(self, seg_qst_dicts, metadata, qry_cfg=None, key_cfg=None, key_map_ids=None, update_mask_key=None,
                 get_bnd_masks=False, get_gain_masks=False, get_unc_masks=False, bnd_thr=None, rew_thr=100,
                 gain_thr=100, unc_thr=100, get_segs=True, seg_type='instance', box_mask_key=None, score_attrs=None,
                 dup_attrs=None, max_segs=None, mask_decay=1e-6, mask_thr=0.5, pan_post_attrs=None, matcher_cfg=None,
                 apply_ids=None, **kwargs):
        """
        Initializes the DRSHead module.

        Args:
            seg_qst_dicts (List): List [num_qsts] of segmentation question dictionaries, each possibly containing:
                - name (str): string containing the name of the segmentation question;
                - loss_cfg (Dict): configuration dictionary specifying the loss module;
                - loss_reduction (str): string containing the loss reduction mechanism.

            metadata (detectron2.data.Metadata): Metadata instance containing additional dataset information.
            qry_cfg (Dict): Configuration dictionary specifying the query module (default=None).
            key_cfg (Dict): Configuration dictionary specifying the key module (default=None).
            key_map_ids (List): List with feature map indices from which to compute initial mask logits (default=None).
            update_mask_key (str): String with key to retrieve update mask from the storage dictionary (default=None).
            get_bnd_masks (bool): Boolean indicating whether to get the segmentation boundary masks (default=False).
            get_gain_masks (bool): Boolean indicating whether to get the segmentation gain masks (default=False).
            get_unc_masks (bool): Boolean indicating whether to get the segmentation uncertainty masks (default=False).
            bnd_thr (int): Integer containing the relative boundary threshold per query (default=None).
            rew_thr (int): Integer containing the relative reward threshold per query (default=100).
            gain_thr (int): Integer containing the relative gain threshold per query (default=100).
            unc_thr (int): Integer containing the relative uncertainty threshold per query (default=100).
            get_segs (bool): Boolean indicating whether to get segmentation predictions (default=True).
            seg_type (str): String containing the type of segmentation task (default='instance').
            box_mask_key (str): String with key to retrieve box mask from the storage dictionary (default=None).
            score_attrs (Dict): Attribute dictionary specifying the scoring mechanism (default=None).
            dup_attrs (Dict): Dictionary specifying the duplicate removal or rescoring mechanism (default=None).
            max_segs (int): Integer with the maximum number of returned segmentation predictions (default=None).
            mask_decay (float): Value containing the mask certainty decay factor (default=1e-6).
            mask_thr (float): Value containing the normalized segmentation mask threshold (default=0.5).
            pan_post_attrs (Dict): Attribute dictionary specifying the panoptic post-processing (default=None).
            matcher_cfg (Dict): Configuration dictionary specifying the matcher module (default=None).
            apply_ids (List): List with integers determining when the head should be applied (default=None).
            kwargs (Dict): Dictionary of unused keyword arguments.

        Raises:
            ValueError: Error when the mask certainty decay factor is not between 0 and 1.
        """

        # Check mask certainty decay factor
        if (mask_decay < 0) or (mask_decay > 1):
            error_msg = f"The mask certainty decay factor should be between 0 and 1 (got '{mask_decay}')."
            raise ValueError(error_msg)

        # Initialization of BaseSegHead module
        super().__init__(seg_type, metadata)

        # Build query and key modules if needed
        self.qry = build_model(qry_cfg, sequential=True) if qry_cfg is not None else None
        self.key = build_model(key_cfg, sequential=True) if key_cfg is not None else None

        # Build matcher module if needed
        self.matcher = build_model(matcher_cfg) if matcher_cfg is not None else None

        # Build loss modules
        self.loss_modules = nn.ModuleDict()

        for seg_qst_dict in seg_qst_dicts:
            loss_cfg = seg_qst_dict.pop('loss_cfg', None)

            if loss_cfg is not None:
                loss_module = build_model(loss_cfg)
                loss_module, loss_reduction_from_module = update_loss_module(loss_module)

                qst_name = seg_qst_dict['name']
                self.loss_modules[qst_name] = loss_module
                seg_qst_dict.setdefault('loss_reduction', loss_reduction_from_module)

        # Set remaining attributes
        self.seg_qst_dicts = seg_qst_dicts
        self.key_map_ids = key_map_ids if key_map_ids is not None else [0]
        self.update_mask_key = update_mask_key
        self.get_bnd_masks = get_bnd_masks
        self.get_gain_masks = get_gain_masks
        self.get_unc_masks = get_unc_masks
        self.bnd_thr = bnd_thr
        self.rew_thr = rew_thr
        self.gain_thr = gain_thr
        self.unc_thr = unc_thr
        self.get_segs = get_segs
        self.box_mask_key = box_mask_key
        self.score_attrs = score_attrs if score_attrs is not None else dict()
        self.dup_attrs = dup_attrs if dup_attrs is not None else dict()
        self.max_segs = max_segs
        self.mask_decay = mask_decay
        self.mask_thr = mask_thr
        self.pan_post_attrs = pan_post_attrs if pan_post_attrs is not None else dict()
        self.apply_ids = apply_ids

    def get_answers(self, storage_dict):
        """
        Method computing the answers for the different segmentation questions.

        Args:
            storage_dict (Dict): Storage dictionary (possibly) containing following keys:
                - qry_feats (FloatTensor): query features of shape [num_qrys, qry_feat_size];
                - batch_ids (LongTensor): batch indices of queries of shape [num_qrys];
                - key_feats (FloatTensor): key features of shape [batch_size, num_keys, key_feat_size];
                - images (Images): Images structure containing the batched images of size [batch_size];
                - map_offs (LongTensor): cumulative number of features per feature map of shape [num_maps+1];
                - {self.update_mask_key} (BoolTensor): update mask of shape [num_qrys, num_keys];
                - seg_answers (FloatTensor): segmentation answers of shape [num_qsts, num_qrys, num_keys].

        Returns:
            answers (FloatTensor): Answers to segmentation questions of shape [num_qsts, num_qrys, num_keys].
            qry_ids (LongTensor): Query indices with new segmentation answers of shape [num_new_answers].
            key_ids (LongTensor): Key indices with new segmentation answers of shape [num_new_answers].

        Raises:
            NotImplementedError: Error when having the high-resolution question in update mode.
        """

        # Retrieve various items from storage dictionary
        qry_feats = storage_dict['qry_feats']
        batch_ids = storage_dict['batch_ids']
        key_feats = storage_dict['key_feats']
        images = storage_dict['images']

        # Update query features if needed
        if self.qry is not None:
            qry_feats = self.qry(qry_feats, storage_dict=storage_dict)

        # Get batch size and device
        batch_size = len(images)
        device = qry_feats.device

        # Get question names
        qst_names = [seg_qst_dict['name'] for seg_qst_dict in self.seg_qst_dicts]

        # Get answers to segmentation questions
        if 'seg_answers' not in storage_dict:
            num_qsts, num_qrys, num_keys = (len(self.seg_qst_dicts), len(qry_feats), key_feats.size(dim=1))
            answers = torch.zeros(num_qsts, num_qrys, num_keys, dtype=torch.float, device=device)

            map_offs = storage_dict['map_offs']
            key_ids = sum((list(range(map_offs[i], map_offs[i+1])) for i in self.key_map_ids), [])
            key_ids = torch.tensor(key_ids, device=device)

            num_keys = len(key_ids)
            key_feats = key_feats[:, key_ids]

            if self.key is not None:
                key_feats = self.key(key_feats, storage_dict=storage_dict)

            qry_ids_list = []
            key_ids_list = []

            for i in range(batch_size):
                qry_ids = (batch_ids == i).nonzero()[:, 0]

                qry_feats_i = qry_feats[qry_ids]
                key_feats_i = key_feats[i]

                num_qrys_i, feat_size = qry_feats_i.size()
                exp_qry_ids = qry_ids[:, None].expand(-1, num_keys)
                exp_key_ids = key_ids[None, :].expand(num_qrys_i, -1)

                qry_feats_i = qry_feats_i.view(num_qrys_i, num_qsts, feat_size // num_qsts)
                qry_feats_i = qry_feats_i.permute(1, 0, 2).contiguous()

                key_feats_i = key_feats_i.view(num_keys, num_qsts, feat_size // num_qsts)
                key_feats_i = key_feats_i.permute(1, 2, 0).contiguous()

                answers_i = qry_feats_i @ key_feats_i
                answers[:, exp_qry_ids, exp_key_ids] = answers_i

                qry_ids_list.append(exp_qry_ids)
                key_ids_list.append(exp_key_ids)

            if 'high_res' in qst_names:
                high_res_qst_id = qst_names.index('high_res')
                high_res_gains = answers[high_res_qst_id]
                storage_dict['seg_high_res_gains'] = high_res_gains

                key_map_ids = torch.as_tensor(self.key_map_ids, device=device)
                key_map_ids = key_map_ids.msort()
                assert (key_map_ids.diff() == 1).all().item()

                min_key_map_id = key_map_ids[0].item()
                insert_map_id = min_key_map_id - 1

                if insert_map_id >= 0:
                    map_shapes = storage_dict['map_shapes']
                    high_res_gains = high_res_gains.unsqueeze(dim=2)

                    high_res_gain_maps = seq_to_maps(high_res_gains, map_shapes)
                    high_res_gain_map = high_res_gain_maps[min_key_map_id].clone()

                    mH, mW = map_shapes[insert_map_id].tolist()
                    padding = (mH % 2, mW % 2)
                    output_padding = (mH % 2, mW % 2)

                    kernel = torch.ones(1, 1, 2, 2, device=device)
                    conv_kwargs = {'stride': 2, 'padding': padding, 'output_padding': output_padding}
                    high_res_gain_map = F.conv_transpose2d(high_res_gain_map, kernel, **conv_kwargs)

                    gain_qst_id = qst_names.index('gain')
                    gains = answers[gain_qst_id].unsqueeze(dim=2)

                    gain_maps = seq_to_maps(gains, map_shapes)
                    gain_maps[insert_map_id] = high_res_gain_map

                    gains = maps_to_seq(gain_maps).squeeze(dim=2)
                    answers[gain_qst_id] = gains

                keep_qst_mask = torch.ones(num_qsts, dtype=torch.bool, device=device)
                keep_qst_mask[high_res_qst_id] = False
                answers = answers[keep_qst_mask]

        else:
            update_mask = storage_dict[self.update_mask_key]
            pred_mask = update_mask.clone()

            answers = storage_dict['seg_answers']
            num_qsts = len(self.seg_qst_dicts)

            mask_qst_id = qst_names.index('mask')
            prev_mask_logits = answers[mask_qst_id].clone().detach()
            storage_dict['seg_prev_mask_logits'] = prev_mask_logits

            qry_ids_list = []
            key_ids_list = []

            for i in range(batch_size):
                qry_ids = (batch_ids == i).nonzero()[:, 0]

                pred_mask_i = pred_mask[qry_ids]
                local_qry_ids, key_ids = pred_mask_i.nonzero(as_tuple=True)
                qry_ids = qry_ids[local_qry_ids]

                qry_feats_i = qry_feats[qry_ids]
                key_feats_i = key_feats[i][key_ids]

                if self.key is not None:
                    key_feats_i = self.key(key_feats_i, storage_dict=storage_dict)

                num_preds, feat_size = qry_feats_i.size()
                qry_feats_i = qry_feats_i.view(num_preds * num_qsts, 1, feat_size // num_qsts)
                key_feats_i = key_feats_i.view(num_preds * num_qsts, feat_size // num_qsts, 1)

                pred_answers = (qry_feats_i @ key_feats_i).view(num_preds, num_qsts).t()
                answers[:, qry_ids, key_ids] = pred_answers

                qry_ids_list.append(qry_ids)
                key_ids_list.append(key_ids)

            if 'high_res' in qst_names:
                error_msg = 'Update mode currently does not support the high-resolution question.'
                raise NotImplementedError(error_msg)

        # Get query and key indices for which new answer was computed
        qry_ids = torch.cat(qry_ids_list, dim=0)
        key_ids = torch.cat(key_ids_list, dim=0)

        return answers, qry_ids, key_ids

    @torch.no_grad()
    def compute_segs(self, storage_dict, pred_dicts, **kwargs):
        """
        Method computing the segmentation predictions.

        Args:
            storage_dict (Dict): Storage dictionary (possibly) containing following keys:
                - batch_ids (LongTensor): batch indices of queries of shape [num_qrys];
                - images (Images): Images structure containing the batched images of size [batch_size];
                - cls_logits (FloatTensor): classification logits of shape [num_qrys, num_labels];
                - pred_boxes (Boxes): predicted 2D bounding boxes of size [num_box_qrys];
                - box_scores (FloatTensor): unnormalized 2D bounding box scores of shape [num_box_qrys];
                - seg_answers (FloatTensor): answers to segmentation questions of shape [num_qsts, num_qrys, num_keys].

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

        # Get image width and height with padding
        images = storage_dict['images']
        iW, iH = images.size()

        # Get batch size and device
        batch_size = len(images)
        device = images.images.device

        # Get number of queries and number of classes
        cls_logits = storage_dict['cls_logits']
        num_qrys, num_labels = cls_logits.size()
        num_classes = num_labels - 1

        # Get query and batch indices
        qry_ids = torch.arange(num_qrys, device=device)
        batch_ids = storage_dict['batch_ids']

        if self.seg_type == 'instance':
            qry_ids = qry_ids[:, None].expand(-1, num_classes).reshape(-1)
            batch_ids = batch_ids[:, None].expand(-1, num_classes).reshape(-1)

        # Get prediction labels and scores
        if self.seg_type == 'instance':
            pred_labels = torch.arange(num_classes, device=device)[None, :].expand(num_qrys, -1).reshape(-1)
            pred_scores = cls_logits[:, :-1].sigmoid().view(-1)

        else:
            pred_scores, pred_labels = cls_logits[:, :-1].sigmoid().max(dim=1)

        cls_power = self.score_attrs.get('cls_power', 1.0)
        pred_scores = pred_scores ** cls_power

        # Get box ids if needed
        if self.box_mask_key is not None:
            box_mask = storage_dict[self.box_mask_key]

            if box_mask.dtype != torch.bool:
                box_ids = box_mask
                box_mask = torch.zeros(num_qrys, dtype=torch.bool, device=device)
                box_mask[box_ids] = True

            if self.seg_type == 'instance':
                box_mask = box_mask[:, None].expand(-1, num_classes).flatten()

        # Update prediction scores with box scores if needed
        box_scores = storage_dict.get('box_scores', None)

        if box_scores is not None:
            box_power = self.score_attrs.get('box_power', 1.0)
            box_scores = box_scores.sigmoid() ** box_power

            if self.seg_type == 'instance':
                box_scores = box_scores[:, None].expand(-1, num_classes).flatten()

            if self.box_mask_key is not None:
                pred_scores[box_mask] *= box_scores
            else:
                pred_scores *= box_scores

        # Get segmentation mask logits
        qst_names = [seg_qst_dict['name'] for seg_qst_dict in self.seg_qst_dicts]
        qst_id = qst_names.index('mask')
        mask_logits = storage_dict['seg_answers'][qst_id]

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
            dup_type = self.dup_attrs.get('type', None)
            dup_needs_masks = self.dup_attrs.get('needs_masks', False)

            if dup_type is not None and not dup_needs_masks:
                if dup_type == 'box_nms':

                    if self.box_mask_key is not None:
                        box_mask_i = box_mask[batch_mask]
                        non_box_mask = ~box_mask_i

                        qry_ids_non_box = qry_ids_i[non_box_mask]
                        pred_labels_non_box = pred_labels_i[non_box_mask]
                        pred_scores_non_box = pred_scores_i[non_box_mask]

                        qry_ids_i = qry_ids_i[box_mask_i]
                        pred_labels_i = pred_labels_i[box_mask_i]
                        pred_scores_i = pred_scores_i[box_mask_i]

                    num_preds = len(pred_scores_i)
                    num_candidates = self.dup_attrs.get('nms_candidates', 1000)
                    candidate_ids = pred_scores_i.topk(min(num_candidates, num_preds))[1]

                    qry_ids_i = qry_ids_i[candidate_ids]
                    pred_labels_i = pred_labels_i[candidate_ids]
                    pred_scores_i = pred_scores_i[candidate_ids]

                    pred_boxes = storage_dict['pred_boxes'].clone()
                    pred_boxes_i = pred_boxes[qry_ids_i].to_format('xyxy')

                    iou_thr = self.dup_attrs.get('nms_thr', 0.65)
                    non_dup_ids = batched_nms(pred_boxes_i.boxes, pred_scores_i, pred_labels_i, iou_thr)

                    qry_ids_i = qry_ids_i[non_dup_ids]
                    pred_labels_i = pred_labels_i[non_dup_ids]
                    pred_scores_i = pred_scores_i[non_dup_ids]

                    if self.box_mask_key is not None:
                        qry_ids_i = torch.cat([qry_ids_i, qry_ids_non_box], dim=0)
                        pred_labels_i = torch.cat([pred_labels_i, pred_labels_non_box], dim=0)
                        pred_scores_i = torch.cat([pred_scores_i, pred_scores_non_box], dim=0)

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

            # Get mask scores and instance segmentation masks
            qry_ids_i, non_unique_ids = qry_ids_i.unique(sorted=False, return_inverse=True)
            mask_logits_i = mask_logits[qry_ids_i].unsqueeze(dim=2)

            mask_logits_maps = seq_to_maps(mask_logits_i, storage_dict['map_shapes'])
            mask_logits_i = mask_logits_maps[-1]
            decay_logits = mask_logits_i.clone()

            for mask_logits_map in reversed(mask_logits_maps[:-1]):
                decay_logits = self.mask_decay * decay_logits

                fH, fW = mask_logits_map.shape[-2:]
                decay_logits = F.interpolate(decay_logits, size=(fH, fW), mode='bilinear', align_corners=False)
                mask_logits_i = F.interpolate(mask_logits_i, size=(fH, fW), mode='bilinear', align_corners=False)

                decay_logits = torch.stack([decay_logits, mask_logits_map], dim=0)
                mask_logits_i = torch.stack([mask_logits_i, mask_logits_map], dim=0)

                max_ids = decay_logits.abs().argmax(dim=0, keepdim=True)
                decay_logits = decay_logits.gather(dim=0, index=max_ids).squeeze(dim=0)
                mask_logits_i = mask_logits_i.gather(dim=0, index=max_ids).squeeze(dim=0)

            mask_scores_i = mask_logits_i.sigmoid()
            mask_scores_i = F.interpolate(mask_scores_i, size=(iH, iW), mode='bilinear', align_corners=False)
            mask_scores_i = mask_scores_i.squeeze(dim=1)

            mask_scores_i = mask_scores_i[non_unique_ids]
            ins_seg_masks = mask_scores_i > self.mask_thr

            # Update prediction scores based on mask scores if needed
            if self.seg_type == 'instance':
                mask_areas = ins_seg_masks.flatten(1).sum(dim=1).clamp_(min=1)
                avg_mask_scores = (ins_seg_masks * mask_scores_i).flatten(1).sum(dim=1) / mask_areas

                mask_power = self.score_attrs.get('mask_power', 1.0)
                pred_scores_i *= avg_mask_scores ** mask_power

            # Remove duplicate predictions if needed
            if dup_type is not None and dup_needs_masks:
                if dup_type == 'box_nms':

                    if self.box_mask_key is not None:
                        box_mask_i = box_mask[batch_mask]
                        non_box_mask = ~box_mask_i

                        qry_ids_non_box = qry_ids_i[non_box_mask]
                        pred_labels_non_box = pred_labels_i[non_box_mask]
                        pred_scores_non_box = pred_scores_i[non_box_mask]

                        qry_ids_i = qry_ids_i[box_mask_i]
                        pred_labels_i = pred_labels_i[box_mask_i]
                        pred_scores_i = pred_scores_i[box_mask_i]

                    num_preds = len(pred_scores_i)
                    num_candidates = self.dup_attrs.get('nms_candidates', 1000)
                    candidate_ids = pred_scores_i.topk(min(num_candidates, num_preds))[1]

                    pred_labels_i = pred_labels_i[candidate_ids]
                    mask_scores_i = mask_scores_i[candidate_ids]
                    ins_seg_masks = ins_seg_masks[candidate_ids]
                    pred_scores_i = pred_scores_i[candidate_ids]

                    pred_boxes = mask_to_box(ins_seg_masks).to_format('xyxy')
                    iou_thr = self.dup_attrs.get('nms_thr', 0.65)
                    non_dup_ids = batched_nms(pred_boxes.boxes, pred_scores_i, pred_labels_i, iou_thr)

                    pred_labels_i = pred_labels_i[non_dup_ids]
                    mask_scores_i = mask_scores_i[non_dup_ids]
                    pred_scores_i = pred_scores_i[non_dup_ids]

                    if self.box_mask_key is not None:
                        qry_ids_i = torch.cat([qry_ids_i, qry_ids_non_box], dim=0)
                        pred_labels_i = torch.cat([pred_labels_i, pred_labels_non_box], dim=0)
                        pred_scores_i = torch.cat([pred_scores_i, pred_scores_non_box], dim=0)

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
        Forward prediction method of the DRSHead module.

        Args:
            storage_dict (Dict): Storage dictionary (potentially) containing following keys:
                - batch_ids (LongTensor): batch indices of queries of shape [num_qrys];
                - key_feats (FloatTensor): key features of shape [batch_size, num_keys, key_feat_size];
                - images (Images): Images structure containing the batched images of size [batch_size];
                - feat_maps (List): list of size [num_maps] with feature maps of shape [batch_size, feat_size, fH, fW];
                - map_shapes (LongTensor): feature map shapes in (H, W) format of shape [num_maps, 2];
                - map_offs (LongTensor): cumulative number of features per feature map of shape [num_maps+1];
                - seg_answers (FloatTensor): answers to segmentation questions of shape [num_qsts, num_qrys, num_keys].

            images_dict (Dict): Dictionary with annotated images of predictions/targets (default=None).
            kwargs (Dict): Dictionary of keyword arguments passed to underlying methods.

        Returns:
            storage_dict (Dict): Storage dictionary (possibly) containing following additional keys:
                - key_feats (FloatTensor): key features of shape [batch_size, num_keys, key_feat_size];
                - map_shapes (LongTensor): feature map shapes in (H, W) format of shape [num_maps, 2];
                - map_offs (LongTensor): cumulative number of features per feature map of shape [num_maps+1];
                - seg_answers (FloatTensor): answers to segmentation questions of shape [num_qsts, num_qrys, num_keys];
                - seg_qry_ids (LongTensor): query indices with new segmentation answers of shape [*];
                - seg_key_ids (LongTensor): key indices with new segmentation answers of shape [*];
                - seg_qry_bnd_mask (BoolTensor): query-based boundary mask of shape [num_qrys, num_keys];
                - seg_qry_gain_mask (BoolTensor): query-based gain mask of shape [num_qrys, num_keys];
                - seg_qry_unc_mask (BoolTensor): query-based uncertainty mask of shape [num_qrys, num_keys];
                - seg_img_bnd_mask (BoolTensor): image-based boundary mask of shape [batch_size, num_keys];
                - seg_img_gain_mask (BoolTensor): image-based gain mask of shape [batch_size, num_keys];
                - seg_img_unc_mask (BoolTensor): image-based uncertainty mask of shape [batch_size, num_keys].

            images_dict (Dict): Dictionary containing additional images annotated with segmentations (if given).
        """

        # Retrieve desired items from storage dictionary
        feat_maps = storage_dict['feat_maps']
        images = storage_dict['images']

        # Get batch size and device
        batch_size = len(images)
        device = feat_maps[0].device

        # Get key features if needed
        if 'key_feats' not in storage_dict:
            key_feats = maps_to_seq(feat_maps)
            storage_dict['key_feats'] = key_feats

        # Add map shapes to storage dictionary if needed
        if 'map_shapes' not in storage_dict:
            map_shapes = [feat_map.shape[-2:] for feat_map in feat_maps]
            storage_dict['map_shapes'] = torch.tensor(map_shapes, device=device)

        # Add map offsets to storage dictionary if needed
        if 'map_offs' not in storage_dict:
            map_offs = storage_dict['map_shapes'].prod(dim=1).cumsum(dim=0)
            storage_dict['map_offs'] = torch.cat([map_offs.new_zeros([1]), map_offs], dim=0)

        # Get answers, query indices and key indices
        answers, qry_ids, key_ids = self.get_answers(storage_dict)

        # Update storage dictionary
        storage_dict['seg_answers'] = answers
        storage_dict['seg_qry_ids'] = qry_ids
        storage_dict['seg_key_ids'] = key_ids

        # Get boundary masks if needed
        if self.get_bnd_masks:
            qst_names = [seg_qst_dict['name'] for seg_qst_dict in self.seg_qst_dicts]
            qst_id = qst_names.index('mask')
            mask_logits = answers[qst_id]

            seg_mask = mask_logits.sigmoid().unsqueeze(dim=2) > self.mask_thr
            seg_masks = seq_to_maps(seg_mask, storage_dict['map_shapes'])

            kernel = mask_logits.new_ones([1, 1, 3, 3])
            qry_bnd_masks = []

            for seg_mask in seg_masks:
                qry_bnd_mask = F.conv2d(seg_mask.float(), kernel, padding=1)
                qry_bnd_mask = (qry_bnd_mask > 0.5) & (qry_bnd_mask < 8.5)
                qry_bnd_masks.append(qry_bnd_mask)

            qry_bnd_mask = maps_to_seq(qry_bnd_masks)
            qry_bnd_mask = qry_bnd_mask.squeeze(dim=2)

            if self.bnd_thr is not None:
                qry_bnd_mask = qry_bnd_mask.float()
                qry_bnd_mask *= torch.randn_like(qry_bnd_mask).abs()

                num_qrys = len(qry_bnd_mask)
                qry_ids = torch.arange(num_qrys, device=device)[:, None].expand(-1, self.bnd_thr)
                topk_vals, key_ids = qry_bnd_mask.topk(self.bnd_thr, dim=1, sorted=False)

                valid_mask = topk_vals > 0
                qry_ids = qry_ids[valid_mask]
                key_ids = key_ids[valid_mask]

                qry_bnd_mask = torch.zeros_like(qry_bnd_mask, dtype=torch.bool)
                qry_bnd_mask[qry_ids, key_ids] = True

            storage_dict['seg_qry_bnd_mask'] = qry_bnd_mask
            batch_ids = storage_dict['batch_ids']
            img_bnd_masks = []

            for i in range(batch_size):
                img_bnd_mask = qry_bnd_mask[batch_ids == i].any(dim=0)
                img_bnd_masks.append(img_bnd_mask)

            img_bnd_mask = torch.stack(img_bnd_masks, dim=0)
            storage_dict['seg_img_bnd_mask'] = img_bnd_mask

        # Get gain masks if needed
        if self.get_gain_masks:
            qst_names = [seg_qst_dict['name'] for seg_qst_dict in self.seg_qst_dicts]

            reward_qst_id = qst_names.index('reward')
            reward_logits = answers[reward_qst_id]

            num_qrys = len(reward_logits)
            qry_ids = torch.arange(num_qrys, device=device)[:, None].expand(-1, self.rew_thr)
            key_ids = reward_logits.topk(self.rew_thr, dim=1, sorted=False)[1]

            gain_qst_id = qst_names.index('gain')
            gain_logits = answers[gain_qst_id, qry_ids, key_ids]

            qry_ids = torch.arange(num_qrys, device=device)[:, None].expand(-1, self.gain_thr)
            local_key_ids = gain_logits.topk(self.gain_thr, dim=1, sorted=False)[1]
            key_ids = key_ids.gather(dim=1, index=local_key_ids)

            qry_gain_mask = torch.zeros_like(reward_logits, dtype=torch.bool)
            qry_gain_mask[qry_ids, key_ids] = True
            storage_dict['seg_qry_gain_mask'] = qry_gain_mask

            batch_ids = storage_dict['batch_ids']
            img_gain_masks = []

            for i in range(batch_size):
                img_gain_mask = qry_gain_mask[batch_ids == i].any(dim=0)
                img_gain_masks.append(img_gain_mask)

            img_gain_mask = torch.stack(img_gain_masks, dim=0)
            storage_dict['seg_img_gain_mask'] = img_gain_mask

        # Get uncertainty masks if needed
        if self.get_unc_masks:
            qst_names = [seg_qst_dict['name'] for seg_qst_dict in self.seg_qst_dicts]
            qst_id = qst_names.index('mask')
            mask_logits = answers[qst_id]

            num_qrys = len(mask_logits)
            qry_ids = torch.arange(num_qrys, device=device)[:, None].expand(-1, self.unc_thr)

            unc_vals = -mask_logits.abs()
            unc_vals[unc_vals == 0] = -1e6
            key_ids = unc_vals.topk(self.unc_thr, dim=1, sorted=False)[1]

            qry_unc_mask = torch.zeros_like(mask_logits, dtype=torch.bool)
            qry_unc_mask[qry_ids, key_ids] = True
            storage_dict['seg_qry_unc_mask'] = qry_unc_mask

            batch_ids = storage_dict['batch_ids']
            img_unc_masks = []

            for i in range(batch_size):
                img_unc_mask = qry_unc_mask[batch_ids == i].any(dim=0)
                img_unc_masks.append(img_unc_mask)

            img_unc_mask = torch.stack(img_unc_masks, dim=0)
            storage_dict['seg_img_unc_mask'] = img_unc_mask

        # Get segmentation predictions if needed
        if self.get_segs and not self.training:
            self.compute_segs(storage_dict=storage_dict, **kwargs)

        # Draw predicted and target segmentations if needed
        if self.get_segs and images_dict is not None:
            self.draw_segs(storage_dict=storage_dict, images_dict=images_dict, **kwargs)

        return storage_dict, images_dict

    def forward_loss(self, storage_dict, tgt_dict, loss_dict, analysis_dict=None, id=None, **kwargs):
        """
        Forward loss method of the DRSHead module.

        Args:
            storage_dict (Dict): Storage dictionary (possibly) containing following keys (after matching):
                - qry_feats (FloatTensor): query features of shape [num_qrys, qry_feat_size];
                - key_feats (FloatTensor): key features of shape [batch_size, num_keys, key_feat_size];
                - map_shapes (LongTensor): feature map shapes in (H, W) format of shape [num_maps, 2];
                - seg_answers (FloatTensor): answers to segmentation questions of shape [num_qsts, num_qrys, num_keys];
                - seg_qry_ids (LongTensor): query indices with new segmentation answers of shape [*];
                - seg_key_ids (LongTensor): key indices with new segmentation answers of shape [*];
                - matched_qry_ids (LongTensor): indices of matched queries of shape [num_pos_qrys];
                - matched_tgt_ids (LongTensor): indices of corresponding matched targets of shape [num_pos_qrys];
                - mask_targets (FloatTensor): segmentation mask targets of shape [num_qrys, num_keys];
                - seg_prev_mask_logits (FloatTensor): previous segmentation mask logits of shape [num_qrys, num_keys];
                - seg_gain_logits (FloatTensor): segmentation gain logits of shape [num_qrys, num_keys];
                - seg_obt_reward_mask (BoolTensor): segmentation obtained-reward mask of shape [num_qrys, num_keys];
                - seg_high_res_gains (FloatTensor): segmentation high-resolution gains of shape [num_qrys, num_keys].

            tgt_dict (Dict): Target dictionary containing at least following key:
                - masks (BoolTensor): target segmentation masks of shape [num_targets, iH, iW].

            loss_dict (Dict): Dictionary containing different weighted loss terms.
            analysis_dict (Dict): Dictionary containing different analyses (default=None).
            id (int or str): Integer or string containing the head id (default=None).
            kwargs (Dict): Dictionary of keyword arguments passed to some underlying modules.

        Returns:
            loss_dict (Dict): Loss dictionary (possibly) containing following additional keys:
                - mask_loss_{id} (FloatTensor): segmentation mask loss of shape [];
                - reward_loss_{id} (FloatTensor): segmentation reward loss of shape [];
                - gain_loss_{id} (FloatTensor): segmentation gain loss of shape [];
                - high_res_loss_{id} (FloatTensor): segmentation high-resolution loss of shape [].

            analysis_dict (Dict): Analysis dictionary (possibly) containing following additional keys (if not None):
                - mask_acc_{id} (FloatTensor): segmentation mask accuracy of shape [];
                - reward_acc_{id} (FloatTensor): reward prediction accuracy of shape [];
                - gain_acc_{id} (FloatTensor): gain prediction accuracy of shape [];
                - high_res_acc_{id} (FloatTensor): high-resolution prediction accuracy of shape [].

        Raises:
            ValueError: Error when a single query is matched with multiple targets.
            ValueError: Error when an invalid segmentation question name is provided.
        """

        # Perform matching if matcher is available
        if self.matcher is not None:
            self.matcher(storage_dict, tgt_dict=tgt_dict, analysis_dict=analysis_dict, **kwargs)

        # Retrieve desired items from storage dictionary
        qry_feats = storage_dict['qry_feats']
        key_feats = storage_dict['key_feats']
        matched_qry_ids = storage_dict['matched_qry_ids']
        matched_tgt_ids = storage_dict['matched_tgt_ids']

        # Get device
        device = qry_feats.device

        # Get zero loss tensor
        zero_loss = 0.0 * qry_feats.flatten()[0] + 0.0 * key_feats.flatten()[0]
        zero_loss += sum(0.0 * p.flatten()[0] for p in self.parameters())

        # Retrieve desired items from storage dictionary
        answers = storage_dict['seg_answers']
        qry_ids = storage_dict['seg_qry_ids']
        key_ids = storage_dict['seg_key_ids']

        # Check that no query is matched with multiple targets
        counts = matched_qry_ids.unique(sorted=False, return_counts=True)[1]

        if torch.any(counts > 1):
            error_msg = "The DRSHead does not support a single query to be matched with multiple targets."
            raise ValueError(error_msg)

        # Only keep matched query and key indices
        num_qrys = len(qry_feats)
        num_matches = len(matched_qry_ids)

        qry_to_match_ids = torch.full([num_qrys], fill_value=-1, dtype=torch.int64, device=device)
        qry_to_match_ids[matched_qry_ids] = torch.arange(num_matches, device=device)
        match_mask = qry_to_match_ids[qry_ids] >= 0

        if qry_ids.dim() == 1:
            qry_ids = qry_ids[match_mask]
            key_ids = key_ids[match_mask]

        else:
            num_keys = qry_ids.size(dim=1)
            qry_ids = qry_ids[match_mask].view(-1, num_keys)
            key_ids = key_ids[match_mask].view(-1, num_keys)

        # Get target indices
        match_ids = qry_to_match_ids[qry_ids]
        tgt_ids = matched_tgt_ids[match_ids]

        # Get mask logits
        qst_names = [seg_qst_dict['name'] for seg_qst_dict in self.seg_qst_dicts]
        mask_qst_id = qst_names.index('mask')
        mask_logits = answers[mask_qst_id, qry_ids, key_ids]

        # Get mask targets
        if 'mask_targets' not in storage_dict:
            tgt_masks = tgt_dict['masks'].float().unsqueeze(dim=1)
            itp_kwargs = {'mode': 'bilinear', 'align_corners': False}

            map_shapes = storage_dict['map_shapes'].tolist()
            mask_targets_maps = []

            for map_shape in map_shapes:
                mask_targets_map = F.interpolate(tgt_masks, size=map_shape, **itp_kwargs)
                mask_targets_maps.append(mask_targets_map)

            mask_targets = maps_to_seq(mask_targets_maps).squeeze(dim=2)
            storage_dict['mask_targets'] = mask_targets

        else:
            mask_targets = storage_dict['mask_targets']

        mask_targets = mask_targets[tgt_ids, key_ids]

        # Iterate over segmentation questions
        for i, seg_qst_dict in enumerate(self.seg_qst_dicts):

            # Get question name
            qst_name = seg_qst_dict['name']

            # Handle mask question
            if qst_name == 'mask':

                # Get loss reduction keyword arguments
                reduction_kwargs = {}
                reduction_kwargs['loss_reduction'] = seg_qst_dict['loss_reduction']
                reduction_kwargs['qry_ids'] = qry_ids
                reduction_kwargs['tgt_ids'] = tgt_ids

                # Get mask loss
                if mask_logits.numel() > 0:
                    loss_module = self.loss_modules[qst_name]
                    mask_losses = loss_module(mask_logits, mask_targets)
                    mask_loss = reduce_losses(mask_losses, **reduction_kwargs)

                else:
                    mask_loss = zero_loss

                # Add mask loss to loss dictionary
                loss_name = f'mask_loss_{id}' if id is not None else 'mask_loss'
                loss_dict[loss_name] = mask_loss

                # Perform mask analyses if needed
                if analysis_dict is not None:

                    # Get mask accuracy
                    acc_mask = (mask_targets == 0) | (mask_targets == 1)

                    mask_preds = mask_logits[acc_mask].sigmoid() > self.mask_thr
                    mask_targets_acc = mask_targets[acc_mask].bool()

                    if mask_preds.numel() > 0:
                        mask_acc = (mask_preds == mask_targets_acc).sum() / mask_preds.numel()
                    else:
                        mask_acc = torch.tensor(0.0, device=device)

                    # Add mask accuracy to analysis dictionary
                    acc_name = loss_name.replace('loss', 'acc')
                    analysis_dict[acc_name] = 100 * mask_acc

                    # Perform previous mask analysis if needed
                    if 'seg_prev_mask_logits' in storage_dict:
                        mask_logits = storage_dict['seg_prev_mask_logits'][qry_ids, key_ids]
                        mask_preds = mask_logits[acc_mask].sigmoid() > self.mask_thr

                        if mask_preds.numel() > 0:
                            mask_acc = (mask_preds == mask_targets_acc).sum() / mask_preds.numel()
                        else:
                            mask_acc = torch.tensor(0.0, device=device)

                        acc_name = f'prev_{acc_name}'
                        analysis_dict[acc_name] = 100 * mask_acc

            # Handle reward question
            elif qst_name == 'reward':

                # Get reward logits
                reward_logits = answers[i, qry_ids, key_ids]

                # Get reward targets
                mask_preds = mask_logits.sigmoid() > self.mask_thr
                reward_targets = (mask_preds != mask_targets) & ((mask_targets == 0) | (mask_targets == 1))
                reward_targets = reward_targets.float()

                # Get loss reduction keyword arguments
                reduction_kwargs = {}
                reduction_kwargs['loss_reduction'] = seg_qst_dict['loss_reduction']
                reduction_kwargs['qry_ids'] = qry_ids
                reduction_kwargs['tgt_ids'] = tgt_ids

                # Get reward loss
                if reward_logits.numel() > 0:
                    loss_module = self.loss_modules[qst_name]
                    reward_losses = loss_module(reward_logits, reward_targets)
                    reward_loss = reduce_losses(reward_losses, **reduction_kwargs)

                else:
                    reward_loss = zero_loss

                # Add reward loss to loss dictionary
                loss_name = f'reward_loss_{id}' if id is not None else 'reward_loss'
                loss_dict[loss_name] = reward_loss

                # Perform reward analyses if needed
                if analysis_dict is not None:

                    # Get reward accuracy
                    reward_preds = reward_logits.sigmoid() > 0.5
                    reward_targets = reward_targets.bool()

                    if reward_preds.numel() > 0:
                        reward_acc = (reward_preds == reward_targets).sum() / reward_preds.numel()
                    else:
                        reward_acc = torch.tensor(0.0, device=device)

                    # Add reward accuracy to analysis dictionary
                    acc_name = loss_name.replace('loss', 'acc')
                    analysis_dict[acc_name] = 100 * reward_acc

            # Handle gain question
            elif qst_name == 'gain':

                # Get new obtained-reward mask
                mask_preds = mask_logits.sigmoid() > self.mask_thr
                new_obt_reward_mask = (mask_preds == mask_targets) & ((mask_targets == 0) | (mask_targets == 1))

                # Get gain loss and analyses if needed
                if 'seg_gain_logits' in storage_dict:

                    # Get gain logits
                    gain_logits = storage_dict['seg_gain_logits'][qry_ids, key_ids]

                    # Get gain targets
                    old_obt_reward_mask = storage_dict['seg_obt_reward_mask'][qry_ids, key_ids]
                    gain_targets = new_obt_reward_mask & (~old_obt_reward_mask)
                    gain_targets = gain_targets.float()

                    # Get loss reduction keyword arguments
                    reduction_kwargs = {}
                    reduction_kwargs['loss_reduction'] = seg_qst_dict['loss_reduction']
                    reduction_kwargs['qry_ids'] = qry_ids
                    reduction_kwargs['tgt_ids'] = tgt_ids

                    # Get gain loss
                    if gain_logits.numel() > 0:
                        loss_module = self.loss_modules[qst_name]
                        gain_losses = loss_module(gain_logits, gain_targets)
                        gain_loss = reduce_losses(gain_losses, **reduction_kwargs)

                    else:
                        gain_loss = zero_loss

                    # Add gain loss to loss dictionary
                    loss_name = f'gain_loss_{id}' if id is not None else 'gain_loss'
                    loss_dict[loss_name] = gain_loss

                    # Perform gain analyses if needed
                    if analysis_dict is not None:

                        # Get gain accuracy
                        gain_preds = gain_logits.sigmoid() > 0.5
                        gain_targets = gain_targets.bool()

                        if gain_preds.numel() > 0:
                            gain_acc = (gain_preds == gain_targets).sum() / gain_preds.numel()
                        else:
                            gain_acc = torch.tensor(0.0, device=device)

                        # Add gain accuracy to analysis dictionary
                        acc_name = loss_name.replace('loss', 'acc')
                        analysis_dict[acc_name] = 100 * gain_acc

                # Add gain logits to storage dictionary
                storage_dict['seg_gain_logits'] = answers[i]

                # Add/update obtained-reward mask to/from storage dictionary
                if 'seg_obt_reward_mask' in storage_dict:
                    obt_reward_mask = storage_dict['seg_obt_reward_mask']
                else:
                    obt_reward_mask = torch.zeros_like(answers[i], dtype=torch.bool)

                obt_reward_mask[qry_ids, key_ids] = new_obt_reward_mask
                storage_dict['seg_obt_reward_mask'] = obt_reward_mask

            # Handle high-resolution question
            elif qst_name == 'high_res':

                # Update segmentation rewards if needed
                key_map_ids = torch.as_tensor(self.key_map_ids, device=device)
                key_map_ids = key_map_ids.msort()
                assert (key_map_ids.diff() == 1).all().item()

                min_key_map_id = key_map_ids[0].item()
                insert_map_id = min_key_map_id - 1

                if insert_map_id >= 0:
                    rewards = storage_dict['seg_rewards'].unsqueeze(dim=2)
                    map_shapes = storage_dict['map_shapes']

                    reward_maps = seq_to_maps(rewards, map_shapes)
                    reward_map = reward_maps[min_key_map_id].clone()

                    mH, mW = map_shapes[insert_map_id].tolist()
                    padding = (mH % 2, mW % 2)
                    output_padding = (mH % 2, mW % 2)

                    kernel = torch.ones(1, 1, 2, 2, device=device)
                    conv_kwargs = {'stride': 2, 'padding': padding, 'output_padding': output_padding}
                    reward_map = F.conv_transpose2d(reward_map, kernel, **conv_kwargs)

                    reward_maps[insert_map_id] = reward_map
                    rewards = maps_to_seq(reward_maps).squeeze(dim=2)
                    storage_dict['seg_rewards'] = rewards

                # Get high-resolution loss and analyses if needed
                if qst_name in self.loss_modules:

                    # Get map shapes
                    map_shapes = storage_dict['map_shapes']

                    # Get high-resolution scores
                    high_res_gains = storage_dict['seg_high_res_gains'].unsqueeze(dim=2)
                    high_res_gain_maps = seq_to_maps(high_res_gains, map_shapes)
                    gain_maps = []

                    for i, high_res_gain_map in enumerate(high_res_gain_maps[1:]):
                        mH, mW = map_shapes[i].tolist()
                        padding = (mH % 2, mW % 2)
                        output_padding = (mH % 2, mW % 2)

                        kernel = torch.ones(1, 1, 2, 2, device=device)
                        conv_kwargs = {'stride': 2, 'padding': padding, 'output_padding': output_padding}

                        gain_map = F.conv_transpose2d(high_res_gain_map, kernel, **conv_kwargs)
                        gain_maps.append(gain_map)

                    gain_maps.append(torch.zeros_like(high_res_gain_maps[-1]))
                    high_res_scores = maps_to_seq(gain_maps).squeeze(dim=2)
                    high_res_scores = high_res_scores[qry_ids, key_ids]

                    # Get high-resolution targets
                    gain_qst_id = qst_names.index('gain')
                    high_res_targets = answers[gain_qst_id].detach()
                    high_res_targets = high_res_targets[qry_ids, key_ids]

                    # Get valid high-resolution scores and targets
                    valid_mask = (high_res_scores != 0) & (high_res_targets != 0)
                    high_res_scores = high_res_scores[valid_mask]
                    high_res_targets = high_res_targets[valid_mask]

                    # Get valid query and target indices
                    valid_qry_ids = qry_ids[valid_mask]
                    valid_tgt_ids = tgt_ids[valid_mask]

                    # Get loss reduction keyword arguments
                    reduction_kwargs = {}
                    reduction_kwargs['loss_reduction'] = seg_qst_dict['loss_reduction']
                    reduction_kwargs['qry_ids'] = valid_qry_ids
                    reduction_kwargs['tgt_ids'] = valid_tgt_ids

                    # Get high-resolution loss
                    if high_res_scores.numel() > 0:
                        loss_module = self.loss_modules[qst_name]
                        high_res_losses = loss_module(high_res_scores, high_res_targets)
                        high_res_loss = reduce_losses(high_res_losses, **reduction_kwargs)

                    else:
                        high_res_loss = zero_loss

                    # Add high-resolution loss to loss dictionary
                    loss_name = f'high_res_loss_{id}' if id is not None else 'high_res_loss'
                    loss_dict[loss_name] = high_res_loss

                    # Perform high-resolution analyses if needed
                    if analysis_dict is not None:

                        # Get gain error
                        if high_res_scores.numel() > 0:
                            high_res_err = (high_res_scores - high_res_targets).abs().mean()
                        else:
                            high_res_err = torch.tensor(0.0, device=device)

                        # Add gain error to analysis dictionary
                        err_name = loss_name.replace('loss', 'err')
                        analysis_dict[err_name] = high_res_err

                else:
                    loss_name = next(iter(loss_dict))
                    loss_dict[loss_name] += zero_loss

            # Handle invalid question
            else:
                error_msg = f"Invalid segmentation question name in DRSHead (got '{qst_name}')."
                raise ValueError(error_msg)

        return loss_dict, analysis_dict
