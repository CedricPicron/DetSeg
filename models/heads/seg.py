"""
Collection of segmentation heads.
"""

from detectron2.layers import batched_nms
from detectron2.projects.point_rend.point_features import get_uncertain_point_coords_with_randomness
from detectron2.structures.instances import Instances
from detectron2.utils.visualizer import Visualizer
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
        qry (nn.Module): Module computing the query features.
        key (nn.Module): Module computing the key feature map.
        get_segs (bool): Boolean indicating whether to get segmentation predictions.

        dup_attrs (Dict): Optional dictionary specifying the duplicate removal mechanism, possibly containing:
            - type (str): string containing the type of duplicate removal mechanism (mandatory);
            - nms_candidates (int): integer containing the maximum number of candidate detections retained before NMS;
            - nms_thr (float): value of IoU threshold used during NMS to remove duplicate detections.

        max_segs (int): Optional integer with the maximum number of returned segmentation predictions.
        metadata (detectron2.data.Metadata): Metadata instance containing additional dataset information.
        matcher (nn.Module): Optional matcher module determining the target segmentation maps.

        sample_attrs (Dict): Dictionary specifying the sample procedure, possibly containing following keys:
            - type (str): string containing the type of sample procedure (mandatory);
            - num_points (int): number of points to sample during PointRend sampling;
            - oversample_ratio (float): value of oversample ratio used during PointRend sampling;
            - importance_sample_ratio (float): ratio of importance sampling during PointRend sampling.

        loss (nn.Module): Module computing the segmentation loss.
    """

    def __init__(self, qry_cfg, key_cfg, metadata, sample_attrs, loss_cfg, get_segs=True, dup_attrs=None,
                 max_segs=None, matcher_cfg=None, **kwargs):
        """
        Initializes the BaseSegHead module.

        Args:
            qry_cfg (Dict): Configuration dictionary specifying the query module.
            key_cfg (Dict): Configuration dictionary specifying the key module.
            metadata (detectron2.data.Metadata): Metadata instance containing additional dataset information.
            sample_attrs (Dict): Attribute dictionary specifying the sample procedure during loss computation.
            loss_cfg (Dict): Configuration dictionary specifying the loss module.
            get_segs (bool): Boolean indicating whether to get segmentation predictions (default=True).
            dup_attrs (Dict): Attribute dictionary specifying the duplicate removal mechanism (default=None).
            max_segs (int): Integer with the maximum number of returned segmentation predictions (default=None).
            matcher_cfg (Dict): Configuration dictionary specifying the matcher module (default=None).
            kwargs (Dict): Dictionary of unused keyword arguments.
        """

        # Initialization of default nn.Module
        super().__init__()

        # Build query module
        self.qry = build_model(qry_cfg)

        # Build key module
        self.key = build_model(key_cfg)

        # Build matcher module if needed
        self.matcher = build_model(matcher_cfg) if matcher_cfg is not None else None

        # Build loss module
        self.loss = build_model(loss_cfg)

        # Set remaining attributes
        self.get_segs = get_segs
        self.dup_attrs = dup_attrs
        self.max_segs = max_segs
        self.metadata = metadata
        self.sample_attrs = sample_attrs

    @torch.no_grad()
    def compute_segs(self, storage_dict, pred_dicts, cum_feats_batch=None, **kwargs):
        """
        Method computing the segmentation predictions.

        Args:
            storage_dict (Dict): Storage dictionary containing at least following keys:
                - cls_logits (FloatTensor): classification logits of shape [num_feats, num_labels];
                - seg_logits (FloatTensor): map with segmentation logits of shape [num_feats, fH, fW].

            pred_dicts (List): List of size [num_pred_dicts] collecting various prediction dictionaries.
            cum_feats_batch (LongTensor): Cumulative number of features per batch entry [batch_size+1] (default=None).

        Returns:
            pred_dicts (List): List with prediction dictionaries containing following additional entry:
                pred_dict (Dict): Prediction dictionary containing following keys:
                    - labels (LongTensor): predicted class indices of shape [num_preds];
                    - masks (BoolTensor): predicted segmentation masks of shape [num_preds, fH, fW];
                    - scores (FloatTensor): normalized prediction scores of shape [num_preds];
                    - batch_ids (LongTensor): batch indices of predictions of shape [num_preds].

        Raises:
            ValueError: Error when an invalid type of duplicate removal mechanism is provided.
        """

        # Retrieve classification and segmentation logits
        cls_logits = storage_dict['cls_logits']
        seg_logits = storage_dict['seg_logits']

        # Get prediction masks
        pred_masks = seg_logits > 0
        non_empty_masks = pred_masks.flatten(1).sum(dim=1) > 0
        pred_masks = pred_masks[non_empty_masks]

        # Get smallest boxes containing prediction masks if needed
        if self.dup_attrs is not None:
            dup_removal_type = self.dup_attrs['type']

            if dup_removal_type == 'nms':
                pred_boxes = mask_to_box(pred_masks)

        # Get number of features, number of classes and device
        num_feats = len(pred_masks)
        num_classes = cls_logits.size(dim=1) - 1
        device = cls_logits.device

        # Get feature indices
        feat_ids = torch.arange(num_feats, device=device)[:, None].expand(-1, num_classes).reshape(-1)

        # Get prediction labels and scores
        pred_labels = torch.arange(num_classes, device=device)[None, :].expand(num_feats, -1).reshape(-1)
        pred_scores = cls_logits[non_empty_masks, :-1].sigmoid().view(-1)

        # Get cumulative number of features per batch entry if missing
        if cum_feats_batch is None:
            orig_num_feats = len(cls_logits)
            cum_feats_batch = torch.tensor([0, orig_num_feats], device=device)

        # Get batch indices
        batch_size = len(cum_feats_batch) - 1
        batch_ids = torch.arange(batch_size, device=device)
        batch_ids = batch_ids.repeat_interleave(cum_feats_batch.diff())
        batch_ids = batch_ids[non_empty_masks, None].expand(-1, num_classes).reshape(-1)

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

                if dup_removal_type == 'nms':
                    num_candidates = self.dup_attrs['nms_candidates']
                    num_preds_i = len(pred_scores_i)
                    candidate_ids = pred_scores_i.topk(min(num_candidates, num_preds_i))[1]

                    feat_ids_i = feat_ids_i[candidate_ids]
                    pred_labels_i = pred_labels_i[candidate_ids]
                    pred_scores_i = pred_scores_i[candidate_ids]

                    pred_boxes_i = pred_boxes[feat_ids_i].to_format('xyxy')
                    iou_thr = self.dup_attrs['nms_thr']
                    non_dup_ids = batched_nms(pred_boxes_i.boxes, pred_scores_i, pred_labels_i, iou_thr)

                    feat_ids_i = feat_ids_i[non_dup_ids]
                    pred_labels_i = pred_labels_i[non_dup_ids]
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
                    pred_scores_i = pred_scores_i[top_pred_ids]

            # Get prediction masks
            pred_masks_i = pred_masks[feat_ids_i]

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
        Forward prediction method of the BaseSegHead module.

        Args:
            in_feats (FloatTensor): Input features of shape [num_feats, in_feat_size].

            storage_dict (Dict): Storage dictionary containing at least following key:
                - feat_maps (List): list of size [num_maps] with feature maps of shape [batch_size, feat_size, fH, fW];
                - images (Images): images structure containing the batched images of size [batch_size].

            cum_feats_batch (LongTensor): Cumulative number of features per batch entry [batch_size+1] (default=None).
            images_dict (Dict): Dictionary with annotated images of predictions/targets (default=None).
            kwargs (Dict): Dictionary of keyword arguments not used by this module.

        Returns:
            storage_dict (Dict): Storage dictionary containing following additional key:
                - seg_logits (FloatTensor): map with segmentation logits of shape [num_feats, fH, fW].

            images_dict (Dict): Dictionary containing additional images annotated with segmentations (if given).
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
        base_map_size = storage_dict['images'].size(mode='with_padding')
        base_map_size = (base_map_size[1], base_map_size[0])
        key_feat_map = self.key(in_feat_map, base_map_size=base_map_size)

        # Get segmentation logits
        batch_size = key_feat_map.size(dim=0)
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

        # Get segmentation predictions if needed
        if self.get_segs and not self.training:
            self.compute_segs(storage_dict=storage_dict, cum_feats_batch=cum_feats_batch, **kwargs)

        # Draw predicted and target segmentations if needed
        if images_dict is not None:
            self.draw_segs(storage_dict=storage_dict, images_dict=images_dict, **kwargs)

        return storage_dict, images_dict

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
                - masks (BoolTensor): target segmentation masks of shape [num_targets, iH, iW].

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
        if self.matcher is not None:
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
            seg_preds = seg_logits > 0

            seg_acc = 2 * (seg_preds * seg_targets).sum(dim=1) + 1
            seg_acc = seg_acc / (seg_preds.sum(dim=1) + seg_targets.sum(dim=1) + 1)
            seg_acc = seg_acc.mean()

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


@MODELS.register_module()
class TopDownSegHead(nn.Module):
    """
    Class implementing the TopDownSegHead module.

    Attributes:
        qry (nn.Module): Module computing the query features.
        key (nn.Module): Module computing the key feature map.
        map_offset (int): Integer with map offset used to determine the initial key feature map for each query.
        get_segs (bool): Boolean indicating whether to get segmentation predictions.

        dup_attrs (Dict): Optional dictionary specifying the duplicate removal mechanism, possibly containing:
            - type (str): string containing the type of duplicate removal mechanism (mandatory);
            - nms_candidates (int): integer containing the maximum number of candidate detections retained before NMS;
            - nms_thr (float): value of IoU threshold used during NMS to remove duplicate detections.

        max_segs (int): Optional integer with the maximum number of returned segmentation predictions.
        metadata (detectron2.data.Metadata): Metadata instance containing additional dataset information.
        matcher (nn.Module): Optional matcher module determining the target segmentation maps.
        loss (nn.Module): Module computing the segmentation loss.
    """

    def __init__(self, qry_cfg, key_cfg, map_offset, metadata, loss_cfg, get_segs=True, dup_attrs=None,
                 max_segs=None, matcher_cfg=None, **kwargs):
        """
        Initializes the TopDownSegHead module.

        Args:
            qry_cfg (Dict): Configuration dictionary specifying the query module.
            key_cfg (Dict): Configuration dictionary specifying the key module.
            map_offset (int): Integer with map offset used to determine the initial key feature map for each query.
            metadata (detectron2.data.Metadata): Metadata instance containing additional dataset information.
            loss_cfg (Dict): Configuration dictionary specifying the loss module.
            get_segs (bool): Boolean indicating whether to get segmentation predictions (default=True).
            dup_attrs (Dict): Attribute dictionary specifying the duplicate removal mechanism (default=None).
            max_segs (int): Integer with the maximum number of returned segmentation predictions (default=None).
            matcher_cfg (Dict): Configuration dictionary specifying the matcher module (default=None).
            kwargs (Dict): Dictionary of unused keyword arguments.
        """

        # Initialization of default nn.Module
        super().__init__()

        # Build query module
        self.qry = build_model(qry_cfg)

        # Build key module
        self.key = build_model(key_cfg)

        # Build matcher module if needed
        self.matcher = build_model(matcher_cfg) if matcher_cfg is not None else None

        # Build loss module
        self.loss = build_model(loss_cfg)

        # Set remaining attributes
        self.map_offset = map_offset
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
                - cls_logits (FloatTensor): classification logits of shape [num_feats, num_labels];
                - seg_logits (FloatTensor): map with segmentation logits of shape [num_feats, fH, fW].

            pred_dicts (List): List of size [num_pred_dicts] collecting various prediction dictionaries.
            cum_feats_batch (LongTensor): Cumulative number of features per batch entry [batch_size+1] (default=None).

        Returns:
            pred_dicts (List): List with prediction dictionaries containing following additional entry:
                pred_dict (Dict): Prediction dictionary containing following keys:
                    - labels (LongTensor): predicted class indices of shape [num_preds];
                    - masks (BoolTensor): predicted segmentation masks of shape [num_preds, fH, fW];
                    - scores (FloatTensor): normalized prediction scores of shape [num_preds];
                    - batch_ids (LongTensor): batch indices of predictions of shape [num_preds].

        Raises:
            ValueError: Error when an invalid type of duplicate removal mechanism is provided.
        """

        # Retrieve classification and segmentation logits
        cls_logits = storage_dict['cls_logits']
        seg_logits = storage_dict['seg_logits']

        # Get prediction masks
        pred_masks = seg_logits > 0
        non_empty_masks = pred_masks.flatten(1).sum(dim=1) > 0
        pred_masks = pred_masks[non_empty_masks]

        # Get smallest boxes containing prediction masks if needed
        if self.dup_attrs is not None:
            dup_removal_type = self.dup_attrs['type']

            if dup_removal_type == 'nms':
                pred_boxes = mask_to_box(pred_masks)

        # Get number of features, number of classes and device
        num_feats = len(pred_masks)
        num_classes = cls_logits.size(dim=1) - 1
        device = cls_logits.device

        # Get feature indices
        feat_ids = torch.arange(num_feats, device=device)[:, None].expand(-1, num_classes).reshape(-1)

        # Get prediction labels and scores
        pred_labels = torch.arange(num_classes, device=device)[None, :].expand(num_feats, -1).reshape(-1)
        pred_scores = cls_logits[non_empty_masks, :-1].sigmoid().view(-1)

        # Get cumulative number of features per batch entry if missing
        if cum_feats_batch is None:
            orig_num_feats = len(cls_logits)
            cum_feats_batch = torch.tensor([0, orig_num_feats], device=device)

        # Get batch indices
        batch_size = len(cum_feats_batch) - 1
        batch_ids = torch.arange(batch_size, device=device)
        batch_ids = batch_ids.repeat_interleave(cum_feats_batch.diff())
        batch_ids = batch_ids[non_empty_masks, None].expand(-1, num_classes).reshape(-1)

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

                if dup_removal_type == 'nms':
                    num_candidates = self.dup_attrs['nms_candidates']
                    num_preds_i = len(pred_scores_i)
                    candidate_ids = pred_scores_i.topk(min(num_candidates, num_preds_i))[1]

                    feat_ids_i = feat_ids_i[candidate_ids]
                    pred_labels_i = pred_labels_i[candidate_ids]
                    pred_scores_i = pred_scores_i[candidate_ids]

                    pred_boxes_i = pred_boxes[feat_ids_i].to_format('xyxy')
                    iou_thr = self.dup_attrs['nms_thr']
                    non_dup_ids = batched_nms(pred_boxes_i.boxes, pred_scores_i, pred_labels_i, iou_thr)

                    feat_ids_i = feat_ids_i[non_dup_ids]
                    pred_labels_i = pred_labels_i[non_dup_ids]
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
                    pred_scores_i = pred_scores_i[top_pred_ids]

            # Get prediction masks
            pred_masks_i = pred_masks[feat_ids_i]

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
        Forward prediction method of the TopDownSegHead module.

        Args:
            in_feats (FloatTensor): Input features of shape [num_feats, in_feat_size].

            storage_dict (Dict): Storage dictionary containing at least following key:
                - feat_maps (List): list of size [num_maps] with feature maps of shape [batch_size, feat_size, fH, fW];
                - images (Images): images structure containing the batched images of size [batch_size];
                - feat_ids (LongTensor): indices of selected features resulting in input features of shape [num_feats];
                - pred_boxes (Boxes): predicted 2D bounding boxes obtained from input features of size [num_feats].

            cum_feats_batch (LongTensor): Cumulative number of features per batch entry [batch_size+1] (default=None).
            images_dict (Dict): Dictionary with annotated images of predictions/targets (default=None).
            kwargs (Dict): Dictionary of keyword arguments not used by this module.

        Returns:
            storage_dict (Dict): Storage dictionary containing following additional keys:
                - qry_ids (LongTensor): query indices of query-key pairs of shape [num_qry_key_pairs];
                - key_xy (FloatTensor): normalized key locations of query-key pairs of shape [num_qry_key_pairs, 2];
                - key_wh (FloatTensor): normalized key sizes of query-key pairs of shape [num_qry_key_pairs, 2];
                - seg_logits (FloatTensor): segmentation logits of query-key pairs of shape [num_qry_key_pairs].

            images_dict (Dict): Dictionary containing additional images annotated with segmentations (if given).
        """

        # Get number of features and device
        num_feats = len(in_feats)
        device = in_feats.device

        # Get cumulative number of features per batch entry if missing
        if cum_feats_batch is None:
            cum_feats_batch = torch.tensor([0, num_feats], device=device)

        # Get query features
        qry_feats = self.qry(in_feats)

        # Get key feature maps
        in_feat_maps = storage_dict['feat_maps']
        images = storage_dict['images']

        base_map_size = images.size(mode='with_padding')
        base_map_size = (base_map_size[1], base_map_size[0])

        key_feat_maps = self.key(in_feat_maps, base_map_size=base_map_size)
        num_add_maps = len(key_feat_maps) - len(in_feat_maps)

        # Get cumulative number of features per map
        cum_feats_map = [feat_map.flatten(2).size(dim=2) for feat_map in in_feat_maps]
        cum_feats_map = torch.tensor(cum_feats_map, device=device).cumsum(dim=0)

        # Get query map indices
        qry_feat_ids = storage_dict['feat_ids'] % cum_feats_map[-1]
        qry_map_ids = (qry_feat_ids[:, None] - cum_feats_map[None, :]) >= 0
        qry_map_ids = qry_map_ids.sum(dim=1) - self.map_offset + num_add_maps
        qry_map_ids = torch.clamp(qry_map_ids, min=0)

        # Get query boxes
        qry_boxes = storage_dict['pred_boxes']
        qry_boxes = qry_boxes.normalize(images).to_format('xyxy')

        # Get query-key pairs
        batch_size = len(key_feat_maps[0])
        num_maps = len(key_feat_maps)

        qry_feats_list = []
        key_feats_list = []

        qry_ids_list = []
        key_xy_list = []
        key_wh_list = []

        for i in range(batch_size):
            i0 = cum_feats_batch[i].item()
            i1 = cum_feats_batch[i+1].item()

            for map_id in range(num_maps):
                qry_map_ids_i = qry_map_ids[i0:i1]
                qry_ids = torch.nonzero(qry_map_ids_i == map_id)[:, 0]

                if len(qry_ids) > 0:
                    qry_feats_i = qry_feats[i0:i1][qry_ids]
                    qry_boxes_i = qry_boxes[i0:i1][qry_ids].boxes

                    key_feat_map = key_feat_maps[map_id][i]
                    kH, kW = key_feat_map.size()[-2:]

                    pts_x = torch.linspace(0.5/kW, 1-0.5/kW, steps=kW, device=device)
                    pts_y = torch.linspace(0.5/kH, 1-0.5/kH, steps=kH, device=device)

                    left_mask = pts_x[None, None, :] > qry_boxes_i[:, 0, None, None] - 0.5/kW
                    top_mask = pts_y[None, :, None] > qry_boxes_i[:, 1, None, None] - 0.5/kH
                    right_mask = pts_x[None, None, :] < qry_boxes_i[:, 2, None, None] + 0.5/kW
                    bot_mask = pts_y[None, :, None] < qry_boxes_i[:, 3, None, None] + 0.5/kH

                    key_mask = left_mask & top_mask & right_mask & bot_mask
                    qry_ids, key_ids = torch.nonzero(key_mask.flatten(1), as_tuple=True)

                    qry_feats_i = qry_feats_i[qry_ids]
                    qry_feats_list.append(qry_feats_i)

                    key_feats_i = key_feat_map.flatten(1)
                    key_feats_i = key_feats_i[:, key_ids].t()
                    key_feats_list.append(key_feats_i)

                    qry_ids_i = qry_ids + i0
                    qry_ids_list.append(qry_ids_i)

                    key_xy_i = torch.meshgrid(pts_x, pts_y, indexing='xy')
                    key_xy_i = torch.stack(key_xy_i, dim=0).flatten(1)
                    key_xy_i = key_xy_i[:, key_ids].t()
                    key_xy_list.append(key_xy_i)

                    num_keys = len(key_ids)
                    key_wh_i = torch.tensor([kW, kH], device=device)
                    key_wh_i = key_wh_i[None, :].expand(num_keys, -1)
                    key_wh_list.append(key_wh_i)

        qry_feats = torch.cat(qry_feats_list, dim=0)
        key_feats = torch.cat(key_feats_list, dim=0)

        qry_ids = torch.cat(qry_ids_list, dim=0)
        key_xy = torch.cat(key_xy_list, dim=0)
        key_wh = torch.cat(key_wh_list, dim=0)

        storage_dict['qry_ids'] = qry_ids
        storage_dict['key_xy'] = key_xy
        storage_dict['key_wh'] = key_wh

        # Get segmentation logits of query-key pairs
        seg_logits = (qry_feats * key_feats).sum(dim=1)
        storage_dict['seg_logits'] = seg_logits

        # Get segmentation predictions if needed
        if self.get_segs and not self.training:
            self.compute_segs(storage_dict=storage_dict, cum_feats_batch=cum_feats_batch, **kwargs)

        # Draw predicted and target segmentations if needed
        if images_dict is not None:
            self.draw_segs(storage_dict=storage_dict, images_dict=images_dict, **kwargs)

        return storage_dict, images_dict

    def forward_loss(self, storage_dict, tgt_dict, loss_dict, analysis_dict=None, id=None, **kwargs):
        """
        Forward loss method of the TopDownSegHead module.

        Args:
            storage_dict (Dict): Storage dictionary containing following keys (after matching):
                - qry_ids (LongTensor): query indices of query-key pairs of shape [num_qry_key_pairs];
                - key_xy (FloatTensor): normalized key locations of query-key pairs of shape [num_qry_key_pairs, 2];
                - seg_logits (FloatTensor): segmentation logits of query-key pairs of shape [num_qry_key_pairs].
                - matched_qry_ids (LongTensor): indices of matched queries of shape [num_pos_queries];
                - matched_tgt_ids (LongTensor): indices of corresponding matched targets of shape [num_pos_queries].

            tgt_dict (Dict): Target dictionary containing at least following key:
                - masks (BoolTensor): target segmentation masks of shape [num_targets, iH, iW].

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
        if self.matcher is not None:
            self.matcher(storage_dict=storage_dict, tgt_dict=tgt_dict, analysis_dict=analysis_dict, **kwargs)

        # Retrieve various items of interest from storage dictionary
        qry_ids = storage_dict['qry_ids']
        key_xy = storage_dict['key_xy']
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

        # Get matched segmentation logits and corresponding targets
        counts = None
        qry_ids = None
        key_xy = None
        seg_logits = None
        seg_targets = None

        # Get segmentation loss
        seg_loss = self.loss(seg_logits, seg_targets)
        key_name = f'seg_loss_{id}' if id is not None else 'seg_loss'
        loss_dict[key_name] = seg_loss

        # Get segmentation accuracy if needed
        if analysis_dict is not None:
            seg_preds = seg_logits > 0

            seg_acc = 2 * (seg_preds * seg_targets).sum(dim=1) + 1
            seg_acc = seg_acc / (seg_preds.sum(dim=1) + seg_targets.sum(dim=1) + 1)
            seg_acc = seg_acc.mean()

            key_name = f'seg_acc_{id}' if id is not None else 'seg_acc'
            analysis_dict[key_name] = 100 * seg_acc

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
