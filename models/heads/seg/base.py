"""
Base segmentation head.
"""
from abc import ABCMeta, abstractmethod

from detectron2.layers import batched_nms
from detectron2.structures.instances import Instances
from detectron2.utils.visualizer import ColorMode, Visualizer
import torch
from torch import nn
import torchvision.transforms.functional as T

from models.build import build_model


class BaseSegHead(nn.Module, metaclass=ABCMeta):
    """
    Abstract class implementing the BaseSegHead module.

    Attributes:
        get_segs (bool): Boolean indicating whether to get segmentation predictions.
        seg_type (str): String containing the type of segmentation task.

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

        metadata (detectron2.data.Metadata): Metadata instance containing additional dataset information.
        matcher (nn.Module): Optional module performing matching between predictions and targets (or None).
        apply_ids (List): List with integers determining when the head should be applied.
    """

    def __init__(self, metadata, get_segs=True, seg_type='instance', dup_attrs=None, max_segs=None, mask_thr=0.5,
                 pan_post_attrs=None, matcher_cfg=None, apply_ids=None, **kwargs):
        """
        Initializes the BaseSegHead module.

        Args:
            metadata (detectron2.data.Metadata): Metadata instance containing additional dataset information.
            get_segs (bool): Boolean indicating whether to get segmentation predictions (default=True).
            seg_type (str): String containing the type of segmentation task (default='instance').
            dup_attrs (Dict): Attribute dictionary specifying the duplicate removal mechanism (default=None).
            max_segs (int): Integer with the maximum number of returned segmentation predictions (default=None).
            mask_thr (float): Value containing the normalized (instance) segmentation mask threshold (default=0.5).
            pan_post_attrs (Dict): Attribute dictionary specifying the panoptic post-processing (default=None).
            matcher_cfg (Dict): Configuration dictionary specifying the matcher module (default=None).
            apply_ids (List): List with integers determining when the head should be applied (default=None).
            kwargs (Dict): Dictionary of unused keyword arguments.
        """

        # Initialization of default nn.Module
        nn.Module.__init__(self)

        # Build matcher module if provided
        self.matcher = build_model(matcher_cfg) if matcher_cfg is not None else None

        # Set additional attributes
        self.get_segs = get_segs
        self.seg_type = seg_type
        self.dup_attrs = dup_attrs
        self.max_segs = max_segs
        self.mask_thr = mask_thr
        self.pan_post_attrs = pan_post_attrs if pan_post_attrs is not None else dict()
        self.metadata = metadata
        self.apply_ids = apply_ids

    @abstractmethod
    def get_mask_scores(batch_id, pred_qry_ids, pred_labels, pred_boxes, storage_dict):
        """
        Method computing the segmentation mask scores at image resolution.

        Args:
            batch_id (int): Integer containing the batch index.
            pred_qry_ids (LongTensor): Query indices of predictions of shape [num_preds].
            pred_labels (LongTensor): Class indices of predictions of shape [num_preds].
            pred_boxes (Boxes): 2D bounding boxes of predictions of size [num_preds].
            qry_boxes (Boxes): 2D bounding boxes of queries of size [num_qrys].
            storage_dict (Dict): Dictionary storing various items of interest.

        Returns:
            mask_scores (FloatTensor): Segmentation mask scores of shape [num_preds, iH, iW].
        """
        pass

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
            ValueError: Error when an invalid type of duplicate removal mechanism is provided.
            ValueError: Error when an invalid type of segmentation task if provided.
        """

        # Retrieve desired items from storage dictionary
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
                    num_preds_i = len(pred_scores_i)
                    candidate_ids = pred_scores_i.topk(min(num_candidates, num_preds_i))[1]

                    qry_ids_i = qry_ids_i[candidate_ids]
                    pred_labels_i = pred_labels_i[candidate_ids]
                    pred_scores_i = pred_scores_i[candidate_ids]

                    pred_boxes_i = pred_boxes[qry_ids_i]
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
            in_kwargs = {'batch_id': i, 'pred_qry_ids': qry_ids_i, 'pred_labels': pred_labels_i}
            in_kwargs = {**in_kwargs, 'pred_boxes': pred_boxes_i, 'qry_boxes': pred_boxes}
            in_kwargs = {**in_kwargs, 'storage_dict': storage_dict}

            mask_scores_i = self.get_mask_scores(**in_kwargs)
            ins_seg_masks = mask_scores_i > self.mask_thr

            # Update prediction scores based on mask scores if needed
            if self.seg_type == 'instance':
                pred_scores_i = pred_scores_i * (ins_seg_masks * mask_scores_i).flatten(1).sum(dim=1)
                pred_scores_i = pred_scores_i / (ins_seg_masks.flatten(1).sum(dim=1) + 1e-6)

            # Perform panoptic post-processing if needed
            elif self.seg_type == 'panoptic':

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

    @torch.no_grad()
    def draw_segs(self, storage_dict, images_dict, pred_dicts, tgt_dict=None, vis_score_thr=0.4, id=None, **kwargs):
        """
        Draws predicted and target segmentations on the corresponding images.

        Segmentations must have a score of at least the score threshold to be drawn, and target segmentations get a
        default score of 100%.

        Args:
            storage_dict (Dict): Storage dictionary containing at least following key:
                - images (Images): Images structure containing the batched images of size [batch_size].

            images_dict (Dict): Dictionary with annotated images of predictions/targets.

            pred_dicts (List): List with prediction dictionaries containing as last entry:
                pred_dict (Dict): Prediction dictionary containing following keys:
                    - labels (LongTensor): predicted class indices of shape [num_preds];
                    - masks (BoolTensor): predicted segmentation masks of shape [num_preds, iH, iW];
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
            ValueError: Error when an invalid segmentation type is provided.
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

        if self.seg_type == 'instance':
            pred_scores = pred_dict['scores']
            sufficient_score = pred_scores >= vis_score_thr

            draw_dict['labels'] = pred_dict['labels'][sufficient_score]
            draw_dict['masks'] = pred_dict['masks'][sufficient_score]
            draw_dict['scores'] = pred_scores[sufficient_score]

            pred_batch_ids = pred_dict['batch_ids'][sufficient_score]
            pred_sizes = [0] + [(pred_batch_ids == i).sum() for i in range(batch_size)]
            pred_sizes = torch.tensor(pred_sizes, device=pred_scores.device).cumsum(dim=0)
            draw_dict['sizes'] = pred_sizes

        elif self.seg_type == 'panoptic':
            thing_ids = tuple(self.metadata.thing_dataset_id_to_contiguous_id.values())

            pred_labels = pred_dict['labels']
            pred_masks = pred_dict['masks']
            pred_batch_ids = pred_dict['batch_ids']

            pred_sizes = [0] + [(pred_batch_ids == i).sum() for i in range(batch_size)]
            pred_sizes = torch.tensor(pred_sizes, device=pred_labels.device).cumsum(dim=0)
            draw_dict['sizes'] = pred_sizes

            mask_list = []
            segments = []

            for i0, i1 in zip(pred_sizes[:-1], pred_sizes[1:]):
                max_vals, mask_i = pred_masks[i0:i1].int().max(dim=0)
                mask_i[max_vals == 0] = -1
                mask_list.append(mask_i)

                for j in range(i1-i0):
                    cat_id = pred_labels[i0+j]
                    is_thing = cat_id in thing_ids

                    segment_ij = {'id': j, 'category_id': cat_id, 'isthing': is_thing}
                    segments.append(segment_ij)

            draw_dict['masks'] = torch.stack(mask_list, dim=0)
            draw_dict['segments'] = segments

        else:
            error_msg = f"Invalid segmentation type in BaseSegHead (got '{self.seg_type}')."
            raise ValueError(error_msg)

        draw_dicts.append(draw_dict)
        dict_name = f"seg_pred_{id}" if id is not None else "seg_pred"
        dict_names.append(dict_name)

        # Get target draw dictionary and dictionary name if needed
        if tgt_dict is not None and not any('seg_tgt' in key for key in images_dict.keys()):
            draw_dict = {}

            if self.seg_type == 'instance':
                draw_dict['labels'] = tgt_dict['labels']
                draw_dict['masks'] = tgt_dict['masks']
                draw_dict['scores'] = torch.ones_like(tgt_dict['labels'], dtype=torch.float)
                draw_dict['sizes'] = tgt_dict['sizes']

            elif self.seg_type == 'panoptic':
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
                error_msg = f"Invalid segmentation type in BaseSegHead (got '{self.seg_type}')."
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

                if self.seg_type == 'instance':
                    visualizer = Visualizer(image, metadata=self.metadata, instance_mode=ColorMode.IMAGE)

                    if i1 > i0:
                        labels_i = draw_dict['labels'][i0:i1].cpu().numpy()
                        scores_i = draw_dict['scores'][i0:i1].cpu().numpy()

                        masks_i = draw_dict['masks'][i0:i1]
                        masks_i = T.resize(masks_i, img_sizes_pad, antialias=False)
                        masks_i = T.crop(masks_i, 0, 0, *img_size)
                        masks_i = masks_i.cpu().numpy()

                        instances = Instances(img_size, pred_classes=labels_i, pred_masks=masks_i, scores=scores_i)
                        visualizer.draw_instance_predictions(instances)

                elif self.seg_type == 'panoptic':
                    mask_i = draw_dict['masks'][i]
                    mask_i = T.crop(mask_i, 0, 0, *img_size).cpu()
                    segments_i = draw_dict['segments'][i0:i1]

                    visualizer = Visualizer(image, metadata=self.metadata, instance_mode=ColorMode.SEGMENTATION)
                    visualizer.draw_panoptic_seg(mask_i, segments_i)

                else:
                    error_msg = f"Invalid segmentation type in BaseSegHead (got '{self.seg_type}')."
                    raise ValueError(error_msg)

                annotated_image = visualizer.output.get_image()
                images_dict[f'{dict_name}_{i}'] = annotated_image

        return images_dict

    def forward_pred(self, storage_dict, images_dict=None, **kwargs):
        """
        Forward prediction method of the BaseSegHead module.

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

    @abstractmethod
    def compute_losses(self, storage_dict, tgt_dict, loss_dict, analysis_dict=None, id=None, **kwargs):
        """
        Method computing the losses and collecting analysis metrics.

         Args:
            storage_dict (Dict): Dictionary storing various items of interest.
            tgt_dict (Dict): Target dictionary with ground-truth information used during trainval.
            loss_dict (Dict): Dictionary containing different weighted loss terms.
            analysis_dict (Dict): Dictionary containing different analysis metrics (default=None).
            id (int or str): Integer or string containing the head id (default=None).
            kwargs (Dict): Dictionary of keyword arguments passed to some underlying modules.

        Returns:
            loss_dict (Dict): Loss dictionary (possibly) containing additional weighted loss terms.
            analysis_dict (Dict): Analysis dictionary (possibly) containing additional analysis metrics (or None).
        """

    def forward_loss(self, storage_dict, tgt_dict, analysis_dict=None, **kwargs):
        """
        Forward loss method of the BaseSegHead module.

        Args:
            storage_dict (Dict): Dictionary storing various items of interest.
            tgt_dict (Dict): Target dictionary with ground-truth information used during trainval.
            analysis_dict (Dict): Dictionary containing different analysis metrics (default=None).
            kwargs (Dict): Dictionary of keyword arguments passed to some underlying modules or methods.

        Returns:
            loss_dict (Dict): Loss dictionary (possibly) containing additional weighted loss terms.
            analysis_dict (Dict): Analysis dictionary (possibly) containing additional analysis metrics (or None).
        """

        # Perform matching if matcher is available
        if self.matcher is not None:
            self.matcher(storage_dict, tgt_dict=tgt_dict, analysis_dict=analysis_dict, **kwargs)

        # Compute losses and collect analysis metrics
        in_kwargs = {'storage_dict': storage_dict, 'tgt_dict': tgt_dict, 'analysis_dict': analysis_dict}
        loss_dict, analysis_dict = self.compute_losses(**in_kwargs, **kwargs)

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
