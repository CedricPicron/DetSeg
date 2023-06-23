"""
Collection of RoI (Region of Interest) heads.
"""

from detectron2.layers import batched_nms
from detectron2.structures.instances import Instances
from detectron2.utils.visualizer import ColorMode, Visualizer
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
import torchvision.transforms.functional as T

from models.build import build_model, MODELS
from models.modules.refine_mask import generate_block_target


@MODELS.register_module()
class StandardRoIHead(MMDetStandardRoIHead):
    """
    Class implementing the StandardRoIHead module.

    The module is based on the StandardRoIHead module from MMDetection.

    Attributes:
        pos_enc (nn.Module): Optional module adding position encodings to the RoI features.
        qry (nn.Module): Optional module updating the query features.
        fuse_qry (nn.Module): Optional module fusing the query features with the RoI features.

        get_segs (bool): Boolean indicating whether to get segmentation predictions.

        dup_attrs (Dict): Optional dictionary specifying the duplicate removal mechanism, possibly containing:
            - type (str): string containing the type of duplicate removal mechanism (mandatory);
            - nms_candidates (int): integer containing the maximum number of candidate detections retained before NMS;
            - nms_thr (float): value of IoU threshold used during NMS to remove duplicate detections.

        max_segs (int): Optional integer with the maximum number of returned segmentation predictions.
        pred_mask_type (str): String containing the type of predicted segmentation mask.
        metadata (detectron2.data.Metadata): Metadata instance containing additional dataset information.
        matcher (nn.Module): Optional matcher module matching predictions with targets.
        apply_ids (List): List with integers determining when the head should be applied.
    """

    def __init__(self, metadata, pos_enc_cfg=None, qry_cfg=None, fuse_qry_cfg=None, get_segs=True, dup_attrs=None,
                 max_segs=None, pred_mask_type='instance', matcher_cfg=None, apply_ids=None, **kwargs):
        """
        Initializes the StandardRoIHead module.

        Args:
            metadata (detectron2.data.Metadata): Metadata instance containing additional dataset information.
            pos_enc_cfg (Dict): Configuration dictionary specifying the position encoder module (default=None).
            qry_cfg (Dict): Configuration dictionary specifying the query module (default=None).
            fuse_qry_cfg (Dict): Configuration dictionary specifying the fuse query module (default=None).
            get_segs (bool): Boolean indicating whether to get segmentation predictions (default=True).
            dup_attrs (Dict): Attribute dictionary specifying the duplicate removal mechanism (default=None).
            max_segs (int): Integer with the maximum number of returned segmentation predictions (default=None).
            pred_mask_type (str): String containing the type of predicted segmentation mask (default='instance').
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
        self.pred_mask_type = pred_mask_type
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
                    - mask_scores (FloatTensor): predicted segmentation mask scores of shape [num_preds, iH, iW];
                    - scores (FloatTensor): normalized prediction scores of shape [num_preds];
                    - batch_ids (LongTensor): batch indices of predictions of shape [num_preds].

        Raises:
            ValueError: Error when an invalid type of duplicate removal mechanism is provided.
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

        # Get number of features, number of classes and device
        num_qrys = cls_logits.size(dim=0)
        num_classes = cls_logits.size(dim=1) - 1
        device = cls_logits.device

        # Get feature indices
        feat_ids = torch.arange(num_qrys, device=device)[:, None].expand(-1, num_classes).reshape(-1)

        # Get prediction labels and scores
        pred_labels = torch.arange(num_classes, device=device)[None, :].expand(num_qrys, -1).reshape(-1)
        pred_scores = cls_logits[:, :-1].sigmoid().view(-1)

        # Get batch size and batch indices
        batch_size = len(images)
        batch_ids = batch_ids[:, None].expand(-1, num_classes).reshape(-1)

        # Initialize prediction dictionary
        pred_keys = ('labels', 'mask_scores', 'scores', 'batch_ids')
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

                qry_feats_i = qry_feats[feat_ids_i]
                qry_feats_i = self.qry(qry_feats_i) if self.qry is not None else qry_feats_i
                qry_feats_i = qry_feats_i[:, :, None, None].expand_as(roi_feats_i)

                fuse_qry_feats_i = torch.cat([qry_feats_i, roi_feats_i], dim=1)
                roi_feats_i = roi_feats_i + self.fuse_qry(fuse_qry_feats_i)

            mask_logits_i = self.mask_head(roi_feats_i)
            mask_logits_i = mask_logits_i[range(len(mask_logits_i)), pred_labels_i]
            mask_logits_i = mask_logits_i[:, None]

            mask_scores_i = mask_logits_i.sigmoid()
            mask_scores_i = _do_paste_mask(mask_scores_i, pred_boxes_i, iH, iW, skip_empty=False)[0]

            pred_masks_i = mask_scores_i > 0.5
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
                - images (Images): Images structure containing the batched images of size [batch_size].

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
            error_msg = f"Invalid prediction mask type in StandardRoIHead (got '{self.pred_mask_type}')."
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
                error_msg = f"Invalid prediction mask type in StandardRoIHead (got '{self.pred_mask_type}')."
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
                    mask_i = draw_dict['masks'][i]
                    mask_i = T.crop(mask_i, 0, 0, *img_size).cpu()
                    segments_i = draw_dict['segments'][i0:i1]

                    visualizer = Visualizer(image, metadata=self.metadata, instance_mode=ColorMode.SEGMENTATION)
                    visualizer.draw_panoptic_seg(mask_i, segments_i)

                else:
                    error_msg = f"Invalid prediction mask type in StandardRoIHead (got '{self.pred_mask_type}')."
                    raise ValueError(error_msg)

                annotated_image = visualizer.output.get_image()
                images_dict[f'{dict_name}_{i}'] = annotated_image

        return images_dict

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
            self.matcher(storage_dict=storage_dict, tgt_dict=tgt_dict, analysis_dict=analysis_dict, **kwargs)

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
                    - mask_scores (FloatTensor): predicted segmentation mask scores of shape [num_preds, iH, iW];
                    - scores (FloatTensor): normalized prediction scores of shape [num_preds];
                    - batch_ids (LongTensor): batch indices of predictions of shape [num_preds].

        Raises:
            ValueError: Error when an invalid type of duplicate removal mechanism is provided.
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

        # Get number of features, number of classes and device
        num_qrys = cls_logits.size(dim=0)
        num_classes = cls_logits.size(dim=1) - 1
        device = cls_logits.device

        # Get feature indices
        feat_ids = torch.arange(num_qrys, device=device)[:, None].expand(-1, num_classes).reshape(-1)

        # Get prediction labels and scores
        pred_labels = torch.arange(num_classes, device=device)[None, :].expand(num_qrys, -1).reshape(-1)
        pred_scores = cls_logits[:, :-1].sigmoid().view(-1)

        # Get batch size and batch indices
        batch_size = len(images)
        batch_ids = batch_ids[:, None].expand(-1, num_classes).reshape(-1)

        # Initialize prediction dictionary
        pred_keys = ('labels', 'mask_scores', 'scores', 'batch_ids')
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

                qry_feats_i = qry_feats[feat_ids_i]
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

            pred_masks_i = mask_scores_i > 0.5
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
            self.matcher(storage_dict=storage_dict, tgt_dict=tgt_dict, analysis_dict=analysis_dict, **kwargs)

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
                    - mask_scores (FloatTensor): predicted segmentation mask scores of shape [num_preds, iH, iW];
                    - scores (FloatTensor): normalized prediction scores of shape [num_preds];
                    - batch_ids (LongTensor): batch indices of predictions of shape [num_preds].

        Raises:
            ValueError: Error when an invalid type of duplicate removal mechanism is provided.
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

        # Get number of features, number of classes and device
        num_qrys = cls_logits.size(dim=0)
        num_classes = cls_logits.size(dim=1) - 1
        device = cls_logits.device

        # Get feature indices
        feat_ids = torch.arange(num_qrys, device=device)[:, None].expand(-1, num_classes).reshape(-1)

        # Get prediction labels and scores
        pred_labels = torch.arange(num_classes, device=device)[None, :].expand(num_qrys, -1).reshape(-1)
        pred_scores = cls_logits[:, :-1].sigmoid().view(-1)

        # Get batch size and batch indices
        batch_size = len(images)
        batch_ids = batch_ids[:, None].expand(-1, num_classes).reshape(-1)

        # Get number of stages and interpolation keyword arguments
        num_stages = len(self.mask_head.stage_sup_size)
        itp_kwargs = {'mode': 'bilinear', 'align_corners': True}

        # Initialize prediction dictionary
        pred_keys = ('labels', 'mask_scores', 'scores', 'batch_ids')
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

                qry_feats_i = qry_feats[feat_ids_i]
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

            pred_masks_i = mask_scores_i > 0.5
            pred_scores_i = pred_scores_i * (pred_masks_i * mask_scores_i).flatten(1).sum(dim=1)
            pred_scores_i = pred_scores_i / (pred_masks_i.flatten(1).sum(dim=1) + 1e-6)

            if self.pred_mask_type == 'panoptic':
                mask_scores_i = pred_scores_i[:, None, None] * mask_scores_i

            # Add predictions to prediction dictionary
            pred_dict['labels'].append(pred_labels_i)
            pred_dict['mask_scores'].append(mask_scores_i)
            pred_dict['scores'].append(pred_scores_i)
            pred_dict['batch_ids'].append(batch_ids_i)

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
            self.matcher(storage_dict=storage_dict, tgt_dict=tgt_dict, analysis_dict=analysis_dict, **kwargs)

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
