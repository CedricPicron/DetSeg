"""
Collection of RoI (Region of Interest) heads.
"""

from detectron2.layers import batched_nms
from detectron2.structures.instances import Instances
from detectron2.utils.visualizer import Visualizer
from mmcv import Config
from mmdet.core import BitmapMasks
from mmdet.core.mask.mask_target import mask_target_single
from mmdet.models.roi_heads import StandardRoIHead as MMDetStandardRoIHead
from mmdet.models.roi_heads.mask_heads.fcn_mask_head import _do_paste_mask
import torch
import torchvision.transforms.functional as T

from models.build import build_model, MODELS


@MODELS.register_module()
class StandardRoIHead(MMDetStandardRoIHead):
    """
    Class implementing the StandardRoIHead module.

    The module is based on the StandardRoIHead module from MMDetection.

    Attributes:
        mask_qry (nn.Module): Optional module computing the mask query features.
        get_segs (bool): Boolean indicating whether to get segmentation predictions.

        dup_attrs (Dict): Optional dictionary specifying the duplicate removal mechanism, possibly containing:
            - type (str): string containing the type of duplicate removal mechanism (mandatory);
            - nms_candidates (int): integer containing the maximum number of candidate detections retained before NMS;
            - nms_thr (float): value of IoU threshold used during NMS to remove duplicate detections.

        max_segs (int): Optional integer with the maximum number of returned segmentation predictions.
        metadata (detectron2.data.Metadata): Metadata instance containing additional dataset information.
        matcher (nn.Module): Optional matcher module matching predictions with targets.
    """

    def __init__(self, metadata, mask_qry_cfg=None, get_segs=True, dup_attrs=None, max_segs=None, matcher_cfg=None,
                 **kwargs):
        """
        Initializes the StandardRoIHead module.

        Args:
            metadata (detectron2.data.Metadata): Metadata instance containing additional dataset information.
            mask_qry_cfg (Dict): Configuration dictionary specifying the mask query module (default=None).
            get_segs (bool): Boolean indicating whether to get segmentation predictions (default=True).
            dup_attrs (Dict): Attribute dictionary specifying the duplicate removal mechanism (default=None).
            max_segs (int): Integer with the maximum number of returned segmentation predictions (default=None).
            matcher_cfg (Dict): Configuration dictionary specifying the matcher module (default=None).
            kwargs (Dict): Dictionary of keyword arguments passed to the parent __init__ method.
        """

        # Initialize module using parent __init__ method
        super().__init__(**kwargs)

        # Build mask query module if needed
        self.mask_qry = build_model(mask_qry_cfg) if mask_qry_cfg is not None else None

        # Build matcher module if needed
        self.matcher = build_model(matcher_cfg) if matcher_cfg is not None else None

        # Set additional attributes
        self.get_segs = get_segs
        self.dup_attrs = dup_attrs
        self.max_segs = max_segs
        self.metadata = metadata

    @torch.no_grad()
    def compute_segs(self, storage_dict, pred_dicts, cum_feats_batch=None, **kwargs):
        """
        Method computing the segmentation predictions.

        Args:
            storage_dict (Dict): Storage dictionary containing at least following keys:
                - images (Images): images structure containing the batched images of size [batch_size];
                - cls_logits (FloatTensor): classification logits of shape [num_feats, num_labels];
                - pred_boxes (Boxes): predicted 2D bounding boxes of size [num_feats];
                - mask_logits (FloatTensor): map with mask logits of shape [num_feats, mC, mH, mW].

            pred_dicts (List): List of size [num_pred_dicts] collecting various prediction dictionaries.
            cum_feats_batch (LongTensor): Cumulative number of features per batch entry [batch_size+1] (default=None).

        Returns:
            pred_dicts (List): List with prediction dictionaries containing following additional entry:
                pred_dict (Dict): Prediction dictionary containing following keys:
                    - labels (LongTensor): predicted class indices of shape [num_preds];
                    - masks (BoolTensor): predicted segmentation masks of shape [num_preds, iH, iW];
                    - scores (FloatTensor): normalized prediction scores of shape [num_preds];
                    - batch_ids (LongTensor): batch indices of predictions of shape [num_preds].

        Raises:
            ValueError: Error when an invalid type of duplicate removal mechanism is provided.
        """

        # Retrieve desired items from storage dictionary
        images = storage_dict['images']
        cls_logits = storage_dict['cls_logits']
        pred_boxes = storage_dict['pred_boxes']
        mask_logits = storage_dict['mask_logits']

        # Get image width and height with padding
        iW, iH = images.size()

        # Get number of features, number of classes and device
        num_feats = cls_logits.size(dim=0)
        num_classes = cls_logits.size(dim=1) - 1
        device = cls_logits.device

        # Get feature indices
        feat_ids = torch.arange(num_feats, device=device)[:, None].expand(-1, num_classes).reshape(-1)

        # Get prediction labels and scores
        pred_labels = torch.arange(num_classes, device=device)[None, :].expand(num_feats, -1).reshape(-1)
        pred_scores = cls_logits[:, :-1].sigmoid().view(-1)

        # Get cumulative number of features per batch entry if missing
        if cum_feats_batch is None:
            cum_feats_batch = torch.tensor([0, num_feats], device=device)

        # Get batch indices
        batch_size = len(cum_feats_batch) - 1
        batch_ids = torch.arange(batch_size, device=device)
        batch_ids = batch_ids.repeat_interleave(cum_feats_batch.diff())
        batch_ids = batch_ids[:, None].expand(-1, num_classes).reshape(-1)

        # Initialize prediction dictionary
        pred_keys = ('labels', 'masks', 'scores', 'batch_ids')
        pred_dict = {pred_key: [] for pred_key in pred_keys}

        # Iterate over every batch entry
        for i in range(batch_size):

            # Get predictions corresponding to batch entry
            batch_mask = batch_ids == i

            feat_ids_i = feat_ids[batch_mask]
            pred_labels_i = pred_labels[batch_mask]
            pred_scores_i = pred_scores[batch_mask]

            # Remove duplicate predictions if needed
            if self.dup_attrs is not None:
                dup_removal_type = self.dup_attrs['type']

                if dup_removal_type == 'nms':
                    num_candidates = self.dup_attrs['nms_candidates']
                    num_preds_i = len(pred_scores_i)
                    candidate_ids = pred_scores_i.topk(min(num_candidates, num_preds_i))[1]

                    feat_ids_i = feat_ids_i[candidate_ids]
                    pred_labels_i = pred_labels_i[candidate_ids]
                    pred_scores_i = pred_scores_i[candidate_ids]

                    pred_boxes_i = pred_boxes[feat_ids_i].to_format('xyxy')
                    pred_boxes_i = pred_boxes_i.to_img_scale(images).boxes

                    iou_thr = self.dup_attrs['nms_thr']
                    non_dup_ids = batched_nms(pred_boxes_i, pred_scores_i, pred_labels_i, iou_thr)

                    feat_ids_i = feat_ids_i[non_dup_ids]
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

                    feat_ids_i = feat_ids_i[top_pred_ids]
                    pred_labels_i = pred_labels_i[top_pred_ids]
                    pred_boxes_i = pred_boxes_i[top_pred_ids]
                    pred_scores_i = pred_scores_i[top_pred_ids]

            # Get prediction masks
            mask_logits_i = mask_logits[feat_ids_i]

            if mask_logits_i.size(dim=1) > 1:
                num_masks = len(mask_logits_i)
                mask_logits_i = mask_logits_i[range(num_masks), pred_labels_i]
                mask_logits_i = mask_logits_i[:, None]

            mask_logits_i = _do_paste_mask(mask_logits_i, pred_boxes_i, iH, iW, skip_empty=False)[0]
            pred_masks_i = mask_logits_i > 0

            # Add predictions to prediction dictionary
            pred_dict['labels'].append(pred_labels_i)
            pred_dict['masks'].append(pred_masks_i)
            pred_dict['scores'].append(pred_scores_i)
            pred_dict['batch_ids'].append(torch.full_like(pred_labels_i, i))

        # Concatenate predictions of different batch entries
        pred_dict.update({k: torch.cat(v, dim=0) for k, v in pred_dict.items()})

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
                    - masks (BoolTensor): predicted segmentation masks of shape [num_preds, fH, fW];
                    - scores (FloatTensor): normalized prediction scores of shape [num_preds];
                    - batch_ids (LongTensor): batch indices of predictions of shape [num_preds].

            tgt_dict (Dict): Optional target dictionary containing at least following keys when given:
                - labels (LongTensor): target class indices of shape [num_targets];
                - masks (BoolTensor): target segmentation masks of shape [num_targets, iH, iW];
                - sizes (LongTensor): cumulative number of targets per batch entry of size [batch_size+1].

            vis_score_thr (float): Threshold indicating the minimum score for a segmentation to be drawn (default=0.4).
            id (int): Integer containing the head id (default=None).
            kwargs (Dict): Dictionary of unused keyword arguments.

        Returns:
            images_dict (Dict): Dictionary containing additional images annotated with segmentations.
        """

        # Retrieve images from storage dictionary
        images = storage_dict['images']

        # Initialize list of draw dictionaries and list of dictionary names
        draw_dicts = []
        dict_names = []

        # Get prediction draw dictionary and dictionary name
        pred_dict = pred_dicts[-1]

        pred_scores = pred_dict['scores']
        sufficient_score = pred_scores >= vis_score_thr

        pred_labels = pred_dict['labels'][sufficient_score]
        pred_masks = pred_dict['masks'][sufficient_score]
        pred_scores = pred_scores[sufficient_score]
        pred_batch_ids = pred_dict['batch_ids'][sufficient_score]

        pred_sizes = [0] + [(pred_batch_ids == i).sum() for i in range(len(images))]
        pred_sizes = torch.tensor(pred_sizes, device=pred_scores.device).cumsum(dim=0)

        draw_dict_keys = ['labels', 'masks', 'scores', 'sizes']
        draw_dict_values = [pred_labels, pred_masks, pred_scores, pred_sizes]

        draw_dict = {k: v for k, v in zip(draw_dict_keys, draw_dict_values)}
        draw_dicts.append(draw_dict)

        dict_name = f"seg_pred_{id}" if id is not None else "seg_pred"
        dict_names.append(dict_name)

        # Get target draw dictionary and dictionary name if needed
        if tgt_dict is not None and not any('seg_tgt' in key for key in images_dict.keys()):
            tgt_labels = tgt_dict['labels']
            tgt_masks = tgt_dict['masks']
            tgt_scores = torch.ones_like(tgt_labels, dtype=torch.float)
            tgt_sizes = tgt_dict['sizes']

            draw_dict_values = [tgt_labels, tgt_masks, tgt_scores, tgt_sizes]
            draw_dict = {k: v for k, v in zip(draw_dict_keys, draw_dict_values)}

            draw_dicts.append(draw_dict)
            dict_names.append('seg_tgt')

        # Get number of images and image sizes without padding in (width, height) format
        num_images = len(images)
        img_sizes = images.size(mode='without_padding')

        # Get image sizes with padding in (height, width) format
        img_sizes_pad = images.size(mode='with_padding')
        img_sizes_pad = (img_sizes_pad[1], img_sizes_pad[0])

        # Draw 2D object detections on images and add them to images dictionary
        for dict_name, draw_dict in zip(dict_names, draw_dicts):
            sizes = draw_dict['sizes']

            for image_id, i0, i1 in zip(range(num_images), sizes[:-1], sizes[1:]):
                img_size = img_sizes[image_id]
                img_size = (img_size[1], img_size[0])

                image = images.images[image_id].clone()
                image = T.crop(image, 0, 0, *img_size)
                image = image.permute(1, 2, 0) * 255
                image = image.to(torch.uint8).cpu().numpy()

                metadata = self.metadata
                visualizer = Visualizer(image, metadata=metadata)

                if i1 > i0:
                    img_labels = draw_dict['labels'][i0:i1].cpu().numpy()
                    img_scores = draw_dict['scores'][i0:i1].cpu().numpy()

                    img_masks = draw_dict['masks'][i0:i1]
                    img_masks = T.resize(img_masks, img_sizes_pad)
                    img_masks = T.crop(img_masks, 0, 0, *img_size)
                    img_masks = img_masks.cpu().numpy()

                    instances = Instances(img_size, pred_classes=img_labels, pred_masks=img_masks, scores=img_scores)
                    visualizer.draw_instance_predictions(instances)

                annotated_image = visualizer.output.get_image()
                images_dict[f'{dict_name}_{image_id}'] = annotated_image

        return images_dict

    def forward_pred(self, in_feats, storage_dict, cum_feats_batch=None, images_dict=None, **kwargs):
        """
        Forward prediction method of the StandardRoIHead module.

        Args:
            in_feats (FloatTensor): Input features of shape [num_feats, in_feat_size].

            storage_dict (Dict): Storage dictionary (possibly) requiring following keys:
                - feat_maps (List): list of size [num_maps] with feature maps of shape [batch_size, feat_size, fH, fW];
                - images (Images): images structure containing the batched images of size [batch_size];
                - pred_boxes (Boxes): predicted 2D bounding boxes of size [num_feats].

            cum_feats_batch (LongTensor): Cumulative number of features per batch entry [batch_size+1] (default=None).
            images_dict (Dict): Dictionary with annotated images of predictions/targets (default=None).
            kwargs (Dict): Dictionary of keyword arguments not used by this module.

        Returns:
            storage_dict (Dict): Storage dictionary (possibly) containing following additional key:
                - mask_logits (FloatTensor): map with mask logits of shape [num_feats, mC, mH, mW].

            images_dict (Dict): Dictionary with (possibly) additional images annotated with 2D boxes/segmentations.

        Raises:
            NotImplementedError: Error when the StandardRoIHead module contains a bounding box head.
        """

        # Retrieve desired items from storage dictionary
        feat_maps = storage_dict['feat_maps']
        images = storage_dict['images']

        # Get number of features and device
        num_feats = len(in_feats)
        device = in_feats.device

        # Get cumulative number of features per batch entry if missing
        if cum_feats_batch is None:
            cum_feats_batch = torch.tensor([0, num_feats], device=device)

        # Get batch indices
        batch_size = len(cum_feats_batch) - 1
        batch_ids = torch.arange(batch_size, device=device)
        batch_ids = batch_ids.repeat_interleave(cum_feats_batch.diff())

        # Get box-related predictions
        if self.with_bbox:
            raise NotImplementedError

        # Get mask-related predictions
        if self.with_mask:

            # Get predicted 2D bounding boxes
            pred_boxes = storage_dict['pred_boxes']

            # Get RoI-boxes
            pred_boxes = pred_boxes.clone().to_format('xyxy')
            pred_boxes = pred_boxes.to_img_scale(images).boxes
            roi_boxes = torch.cat([batch_ids[:, None], pred_boxes], dim=1)

            # Get mask query features if needed
            if self.mask_qry is not None:
                mask_qry_feats = self.mask_qry(in_feats)

            # Get mask key features
            mask_feat_maps = feat_maps[:self.mask_roi_extractor.num_inputs]
            mask_key_feats = self.mask_roi_extractor(mask_feat_maps, roi_boxes)

            # Get mask logits
            if self.mask_qry is not None:
                mask_logits = self.mask_head(mask_qry_feats, mask_key_feats)
            else:
                mask_logits = self.mask_head(mask_key_feats)

            # Add mask logits to storage dictionary
            storage_dict['mask_logits'] = mask_logits

            # Get segmentation predictions if needed
            if self.get_segs and not self.training:
                self.compute_segs(storage_dict=storage_dict, cum_feats_batch=cum_feats_batch, **kwargs)

            # Draw predicted and target segmentations if needed
            if images_dict is not None:
                self.draw_segs(storage_dict=storage_dict, images_dict=images_dict, **kwargs)

        return storage_dict, images_dict

    def forward_loss(self, storage_dict, tgt_dict, loss_dict, analysis_dict=None, id=None, **kwargs):
        """
        Forward loss method of the StandardRoIHead module.

        Args:
            storage_dict (Dict): Storage dictionary containing following keys (after matching):
                - images (Images): images structure containing the batched images of size [batch_size];
                - pred_boxes (Boxes): predicted 2D bounding boxes of size [num_feats];
                - mask_logits (FloatTensor): map with mask logits of shape [num_feats, mC, mH, mW];
                - matched_qry_ids (LongTensor): indices of matched queries of shape [num_pos_queries];
                - matched_tgt_ids (LongTensor): indices of corresponding matched targets of shape [num_pos_queries].

            tgt_dict (Dict): Target dictionary containing at least following keys:
                - labels (LongTensor): target class indices of shape [num_targets];
                - masks (BoolTensor): target segmentation masks of shape [num_targets, iH, iW].

            loss_dict (Dict): Dictionary containing different weighted loss terms.
            analysis_dict (Dict): Dictionary containing different analyses (default=None).
            id (int): Integer containing the head id (default=None).
            kwargs (Dict): Dictionary of keyword arguments passed to some underlying modules.

        Returns:
            loss_dict (Dict): Loss dictionary containing following additional key:
                - mask_loss (FloatTensor): mask loss of shape [].

            analysis_dict (Dict): Analysis dictionary containing following additional key (if not None):
                - mask_acc (FloatTensor): mask accuracy of shape [].
        """

        # Perform matching if matcher is available
        if self.matcher is not None:
            self.matcher(storage_dict=storage_dict, tgt_dict=tgt_dict, analysis_dict=analysis_dict, **kwargs)

        # Retrieve desired items from storage dictionary
        images = storage_dict['images']
        pred_boxes = storage_dict['pred_boxes']
        mask_logits = storage_dict['mask_logits']
        matched_qry_ids = storage_dict['matched_qry_ids']
        matched_tgt_ids = storage_dict['matched_tgt_ids']

        # Get device and number of positive matches
        device = mask_logits.device
        num_pos_matches = len(matched_qry_ids)

        # Handle case where there are no positive matches
        if num_pos_matches == 0:

            # Get mask loss
            mask_loss = 0.0 * mask_logits.sum()
            key_name = f'mask_loss_{id}' if id is not None else 'mask_loss'
            loss_dict[key_name] = mask_loss

            # Get mask accuracy if needed
            if analysis_dict is not None:
                mask_acc = 1.0 if len(tgt_dict['masks']) == 0 else 0.0
                mask_acc = torch.tensor(mask_acc, dtype=mask_loss.dtype, device=device)

                key_name = f'mask_acc_{id}' if id is not None else 'mask_acc'
                analysis_dict[key_name] = 100 * mask_acc

            return loss_dict, analysis_dict

        # Get matched mask logits
        mask_logits = mask_logits[matched_qry_ids]

        if mask_logits.size(dim=1) > 1:
            tgt_labels = tgt_dict['labels'][matched_tgt_ids]
            mask_logits = mask_logits[range(num_pos_matches), tgt_labels]

        else:
            mask_logits = mask_logits[:, 0]

        # Get matched mask targets
        roi_boxes = pred_boxes[matched_qry_ids].to_format('xyxy')
        roi_boxes = roi_boxes.to_img_scale(images).boxes

        tgt_masks = tgt_dict['masks'].cpu().numpy()
        tgt_masks = BitmapMasks(tgt_masks, *tgt_masks.shape[-2:])

        mask_size = tuple(mask_logits.size()[-2:])
        mask_tgt_cfg = Config({'mask_size': mask_size})
        mask_targets = mask_target_single(roi_boxes, matched_tgt_ids, tgt_masks, mask_tgt_cfg)

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

    def forward(self, mode, **kwargs):
        """
        Forward method of the StandardRoIHead module.

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
