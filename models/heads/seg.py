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
from torch_scatter import scatter_sum
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
                - images (Images): images structure containing the batched images of size [batch_size];
                - cls_logits (FloatTensor): classification logits of shape [num_feats, num_labels];
                - seg_logits (FloatTensor): map with segmentation logits of shape [num_feats, fH, fW].

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
        seg_logits = storage_dict['seg_logits']

        # Get image width and height with padding
        iW, iH = images.size()

        # Get prediction masks at feature resolution
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

            # Get prediction masks at image resolution
            seg_logits_i = seg_logits[non_empty_masks][feat_ids_i][:, None]
            seg_logits_i = F.interpolate(seg_logits_i, size=(iH, iW), mode='bilinear', align_corners=False)
            pred_masks_i = seg_logits_i[:, 0] > 0

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

            storage_dict (Dict): Storage dictionary containing at least following keys:
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
        qk_feat_iters (int): Integer containing the number of quey-key feature update iterations.
        qry_key (Sequential): Optional module updating query features (possibly) based on corresponding key features.
        key_qry (Sequential): Optional module updating key features (possibly) based on corresponding query features.
        refine_iters (int): Integer containing the number of refinement iterations.
        refine_grid_size (int): Integer containing the size of the refinement grid.
        tgt_sample_mul (float): Multiplier value determining the target sample locations during refinement.
        get_segs (bool): Boolean indicating whether to get segmentation predictions.

        dup_attrs (Dict): Optional dictionary specifying the duplicate removal mechanism, possibly containing:
            - type (str): string containing the type of duplicate removal mechanism (mandatory);
            - nms_candidates (int): integer containing the maximum number of candidate detections retained before NMS;
            - nms_thr (float): value of IoU threshold used during NMS to remove duplicate detections.

        max_segs (int): Optional integer with the maximum number of returned segmentation predictions.
        mask_thr (float): Value containing the mask threshold used to determine the segmentation masks.
        metadata (detectron2.data.Metadata): Metadata instance containing additional dataset information.
        matcher (nn.Module): Optional matcher module determining the target segmentation maps.
        refined_weight (float): Factor weighting the predictions and losses of refined query-key pairs.
        seg_loss (nn.Module): Module computing the segmentation loss.
    """

    def __init__(self, qry_cfg, key_cfg, map_offset, refine_iters, refine_grid_size, tgt_sample_mul, mask_thr,
                 metadata,  refined_weight, seg_loss_cfg, qk_feat_iters=1, qry_key_cfg=None, key_qry_cfg=None,
                 get_segs=True, dup_attrs=None, max_segs=None, matcher_cfg=None, **kwargs):
        """
        Initializes the TopDownSegHead module.

        Args:
            qry_cfg (Dict): Configuration dictionary specifying the query module.
            key_cfg (Dict): Configuration dictionary specifying the key module.
            map_offset (int): Integer with map offset used to determine the initial key feature map for each query.
            refine_iters (int): Integer containing the number of refinement iterations.
            refine_grid_size (int): Integer containing the size of the refinement grid.
            tgt_sample_mul (float): Multiplier value determining the target sample locations during refinement.
            mask_thr (float): Value containing the mask threshold used to determine the segmentation masks.
            metadata (detectron2.data.Metadata): Metadata instance containing additional dataset information.
            refined_weight (float): Factor weighting the predictions and losses of refined query-key pairs.
            seg_loss_cfg (Dict): Configuration dictionary specifying the segmentation loss module.
            qk_feat_iters (int): Integer containing the number of quey-key feature update iterations (default=1).
            qry_key_cfg (Dict): Configuration dictionary specifying the query-key module (default=None).
            key_qry_cfg (Dict): Configuration dictionary specifying the key-query module (default=None).
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

        # Build query-key module if needed
        self.qry_key = build_model(qry_key_cfg, sequential=True) if qry_key_cfg is not None else None

        # Build key-query module if needed
        self.key_qry = build_model(key_qry_cfg, sequential=True) if key_qry_cfg is not None else None

        # Build matcher module if needed
        self.matcher = build_model(matcher_cfg) if matcher_cfg is not None else None

        # Build segmentation loss module
        self.seg_loss = build_model(seg_loss_cfg)

        # Set remaining attributes
        self.map_offset = map_offset
        self.qk_feat_iters = qk_feat_iters
        self.refine_iters = refine_iters
        self.refine_grid_size = refine_grid_size
        self.tgt_sample_mul = tgt_sample_mul
        self.get_segs = get_segs
        self.dup_attrs = dup_attrs
        self.max_segs = max_segs
        self.mask_thr = mask_thr
        self.metadata = metadata
        self.refined_weight = refined_weight

    @torch.no_grad()
    def compute_segs(self, storage_dict, pred_dicts, cum_feats_batch=None, **kwargs):
        """
        Method computing the segmentation predictions.

        Args:
            storage_dict (Dict): Storage dictionary containing at least following keys:
                - images (Images): images structure containing the batched images of size [batch_size];
                - cls_logits (FloatTensor): classification logits of shape [num_qrys, num_labels];
                - pred_boxes (Boxes): predicted 2D bounding boxes of size [num_feats];
                - qry_ids (LongTensor): query indices of query-key pairs of shape [num_qry_key_pairs];
                - key_xy (FloatTensor): normalized key locations of query-key pairs of shape [num_qry_key_pairs, 2];
                - batch_ids (LongTensor): batch indices of query-key pairs of shape [num_qry_key_pairs];
                - map_ids (LongTensor): map indices of query-key pairs of shape [num_qry_key_pairs];
                - map_shapes (List): list with map shapes in (height, width) format of size [num_key_feat_maps];
                - seg_logits (FloatTensor): segmentation logits of query-key pairs of shape [num_qry_key_pairs];
                - refined_mask (BoolTensor): mask indicating refined query-key pairs of shape [num_qry_key_pairs].

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

        # Retrieve various items from storage dictionary
        images = storage_dict['images']
        cls_logits = storage_dict['cls_logits']
        pred_boxes = storage_dict['pred_boxes']
        qry_ids = storage_dict['qry_ids']
        key_xy = storage_dict['key_xy']
        seg_batch_ids = storage_dict['batch_ids']
        map_ids = storage_dict['map_ids']
        map_shapes = storage_dict['map_shapes']
        seg_logits = storage_dict['seg_logits']
        refined_mask = storage_dict['refined_mask']

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

        # Get additional items common to each iteration
        iW, iH = images.size()
        num_maps = len(map_shapes)

        map_shapes_tensor = torch.tensor(map_shapes, device=device)
        map_shapes_prod = map_shapes_tensor.prod(dim=1)

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

            # Get prediction masks at image resolution
            seg_batch_mask = seg_batch_ids == i

            qry_ids_i = qry_ids[seg_batch_mask]
            key_xy_i = key_xy[seg_batch_mask]
            map_ids_i = map_ids[seg_batch_mask]
            seg_logits_i = seg_logits[seg_batch_mask]
            refined_mask_i = refined_mask[seg_batch_mask]

            num_preds_i = len(feat_ids_i)
            feat_qry_mask = feat_ids_i[:, None] == qry_ids_i[None, :]

            key_xy_i = key_xy_i[None, :, :].expand(num_preds_i, -1, -1)[feat_qry_mask]
            map_ids_i = map_ids_i[None, :].expand(num_preds_i, -1)[feat_qry_mask]
            seg_logits_i = seg_logits_i[None, :].expand(num_preds_i, -1)[feat_qry_mask]
            refined_mask_i = refined_mask_i[None, :].expand(num_preds_i, -1)[feat_qry_mask]

            refined_weights = torch.where(refined_mask_i, self.refined_weight, 1.0)
            seg_logits_i = refined_weights * seg_logits_i

            numel_per_pred = feat_qry_mask.sum(dim=1)
            pred_ids_i = torch.arange(num_preds_i, device=device).repeat_interleave(numel_per_pred, dim=0)

            numel_per_map = num_preds_i * map_shapes_prod
            cum_numel_per_map = torch.cat([numel_per_map.new_zeros([1]), numel_per_map.cumsum(dim=0)], dim=0)

            map_shapes_tensor_i = map_shapes_tensor[map_ids_i]
            x_ids = (key_xy_i[:, 0] * map_shapes_tensor_i[:, 1]).to(torch.int64)
            y_ids = (key_xy_i[:, 1] * map_shapes_tensor_i[:, 0]).to(torch.int64)

            logit_ids = cum_numel_per_map[map_ids_i] + pred_ids_i * map_shapes_prod[map_ids_i]
            logit_ids = logit_ids + y_ids * map_shapes_tensor_i[:, 1] + x_ids

            numel_total = cum_numel_per_map[-1].item()
            pred_logits = torch.zeros(numel_total, device=device)
            pred_logits[logit_ids] += seg_logits_i

            cum_numel_per_map = cum_numel_per_map.tolist()
            pred_logit_maps = [pred_logits[i0:i1] for i0, i1 in zip(cum_numel_per_map[:-1], cum_numel_per_map[1:])]
            pred_logit_maps = [pred_logit_maps[i].view(num_preds_i, 1, *map_shapes[i]) for i in range(num_maps)]
            pred_logits = pred_logit_maps[-1]

            for j in range(num_maps-2, -1, -1):
                pred_logits = F.interpolate(pred_logits, size=map_shapes[j], mode='bilinear', align_corners=False)
                pred_logits = pred_logits + pred_logit_maps[j]

            pred_logits = F.interpolate(pred_logits, size=(iH, iW), mode='bilinear', align_corners=False)
            pred_masks_i = pred_logits[:, 0] > self.mask_thr

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

    def forward_pred(self, in_feats, storage_dict, tgt_dict, cum_feats_batch=None, images_dict=None, **kwargs):
        """
        Forward prediction method of the TopDownSegHead module.

        Args:
            in_feats (FloatTensor): Input features of shape [num_feats, in_feat_size].

            storage_dict (Dict): Storage dictionary containing at least following keys:
                - feat_maps (List): list of size [num_maps] with feature maps of shape [batch_size, feat_size, fH, fW];
                - images (Images): images structure containing the batched images of size [batch_size];
                - feat_ids (LongTensor): indices of selected features resulting in input features of shape [num_feats];
                - pred_boxes (Boxes): predicted 2D bounding boxes obtained from input features of size [num_feats];
                - matched_qry_ids (LongTensor): indices of matched queries of shape [num_pos_queries];
                - matched_tgt_ids (LongTensor): indices of corresponding matched targets of shape [num_pos_queries].

            tgt_dict (Dict): Target dictionary containing at least following key:
                - masks (BoolTensor): target segmentation masks of shape [num_targets, iH, iW].

            cum_feats_batch (LongTensor): Cumulative number of features per batch entry [batch_size+1] (default=None).
            images_dict (Dict): Dictionary with annotated images of predictions/targets (default=None).
            kwargs (Dict): Dictionary of keyword arguments not used by this module.

        Returns:
            storage_dict (Dict): Storage dictionary containing following additional keys:
                - qry_ids (LongTensor): query indices of query-key pairs of shape [num_qry_key_pairs];
                - key_xy (FloatTensor): normalized key locations of query-key pairs of shape [num_qry_key_pairs, 2];
                - batch_ids (LongTensor): batch indices of query-key pairs of shape [num_qry_key_pairs];
                - map_ids (LongTensor): map indices of query-key pairs of shape [num_qry_key_pairs];
                - map_shapes (List): list with map shapes in (height, width) format of size [num_key_feat_maps];
                - seg_logits (FloatTensor): segmentation logits of query-key pairs of shape [num_qry_key_pairs];
                - refined_mask (BoolTensor): mask indicating refined query-key pairs of shape [num_qry_key_pairs].

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
        cum_feats_map = [in_feat_map.flatten(2).size(dim=2) for in_feat_map in in_feat_maps]
        cum_feats_map = torch.tensor(cum_feats_map, device=device).cumsum(dim=0)

        # Get query map indices
        qry_feat_ids = storage_dict['feat_ids'] % cum_feats_map[-1]
        qry_map_ids = (qry_feat_ids[:, None] - cum_feats_map[None, :]) >= 0
        qry_map_ids = qry_map_ids.sum(dim=1) - self.map_offset + num_add_maps
        qry_map_ids = torch.clamp(qry_map_ids, min=0)

        # Get query boxes
        qry_boxes = storage_dict['pred_boxes'].clone()
        qry_boxes = qry_boxes.normalize(images).to_format('xyxy')

        # Get initial query-key pairs with segmentation and refinement logits
        batch_size = len(key_feat_maps[0])
        num_maps = len(key_feat_maps)

        qry_feats_list = []
        key_feats_list = []

        qry_ids_list = []
        key_xy_list = []
        key_wh_list = []

        batch_ids_list = []
        map_ids_list = []

        for i in range(batch_size):
            i0 = cum_feats_batch[i].item()
            i1 = cum_feats_batch[i+1].item()
            qry_map_ids_i = qry_map_ids[i0:i1]

            for map_id in range(num_maps):
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
                    local_qry_ids, key_ids = torch.nonzero(key_mask.flatten(1), as_tuple=True)

                    qry_feats_i = qry_feats_i[local_qry_ids]
                    qry_feats_list.append(qry_feats_i)

                    key_feats_i = key_feat_map.flatten(1).t().contiguous()
                    key_feats_i = key_feats_i[key_ids, :]
                    key_feats_list.append(key_feats_i)

                    qry_ids_i = qry_ids[local_qry_ids] + i0
                    qry_ids_list.append(qry_ids_i)

                    key_xy_i = torch.meshgrid(pts_x, pts_y, indexing='xy')
                    key_xy_i = torch.stack(key_xy_i, dim=0).flatten(1)
                    key_xy_i = key_xy_i[:, key_ids].t()
                    key_xy_list.append(key_xy_i)

                    num_keys = len(key_ids)
                    key_wh_i = torch.tensor([1/kW, 1/kH], device=device)
                    key_wh_i = key_wh_i[None, :].expand(num_keys, -1)
                    key_wh_list.append(key_wh_i)

                    batch_ids_i = torch.full(size=[num_keys], fill_value=i, device=device)
                    batch_ids_list.append(batch_ids_i)

                    map_ids_i = torch.full(size=[num_keys], fill_value=map_id, device=device)
                    map_ids_list.append(map_ids_i)

        qry_feats_i = torch.cat(qry_feats_list, dim=0)
        key_feats_i = torch.cat(key_feats_list, dim=0)

        qry_ids = torch.cat(qry_ids_list, dim=0)
        key_xy = torch.cat(key_xy_list, dim=0)
        key_wh = torch.cat(key_wh_list, dim=0)

        batch_ids = torch.cat(batch_ids_list, dim=0)
        map_ids = torch.cat(map_ids_list, dim=0)

        qry_ids_list = [qry_ids]
        key_xy_list = [key_xy]

        batch_ids_list = [batch_ids]
        map_ids_list = [map_ids]

        for _ in range(self.qk_feat_iters):

            if self.qry_key is not None:
                qry_feats_i = self.qry_key(qry_feats_i, pair_feats=key_feats_i, module_id=0)

            if self.key_qry is not None:
                key_feats_i = self.key_qry(key_feats_i, pair_feats=qry_feats_i, module_id=0)

        seg_logits = (qry_feats_i * key_feats_i).sum(dim=1)
        seg_logits_list = [seg_logits]
        refined_mask_list = []

        # Get refined query-key pairs with segmentation and refinement logits
        matched_qry_ids = storage_dict['matched_qry_ids']
        matched_tgt_ids = storage_dict['matched_tgt_ids']
        tgt_masks = tgt_dict['masks']

        if len(matched_qry_ids) > 0:
            iW, iH = images.size()

            tgt_sample_offsets = torch.arange(2, device=device) - 0.5
            tgt_sample_offsets = torch.meshgrid(tgt_sample_offsets, tgt_sample_offsets, indexing='xy')
            tgt_sample_offsets = torch.stack(tgt_sample_offsets, dim=2).flatten(0, 1)

            grid_size = self.refine_grid_size
            grid_area = grid_size ** 2

            grid_offsets = torch.arange(grid_size, device=device) - (grid_size-1) / 2
            grid_offsets = torch.meshgrid(grid_offsets, grid_offsets, indexing='xy')
            grid_offsets = torch.stack(grid_offsets, dim=2).flatten(0, 1)

            key_feats = [key_feat_map.flatten(2).permute(0, 2, 1) for key_feat_map in key_feat_maps]
            key_feats = torch.cat(key_feats, dim=1)

            map_sizes = [key_feat_map.size()[-2:] for key_feat_map in key_feat_maps]
            map_sizes = [torch.tensor([mW, mH], device=device) for mH, mW in map_sizes]
            map_sizes = torch.stack(map_sizes, dim=0)

            map_offsets = [map_sizes.new_zeros([1]), map_sizes[:-1].prod(dim=1)]
            map_offsets = torch.cat(map_offsets).cumsum(dim=0)

            max_qry_id = max(matched_qry_ids.max().item(), qry_ids.max().item())
            matched_qry_mask = torch.zeros(max_qry_id+1, dtype=torch.bool, device=device)
            matched_qry_mask[matched_qry_ids] = True
            match_mask = matched_qry_mask[qry_ids]

            qry_ids = qry_ids[match_mask]
            key_xy = key_xy[match_mask]
            key_wh = key_wh[match_mask]
            batch_ids = batch_ids[match_mask]
            map_ids = map_ids[match_mask]

            num_matches = len(matched_qry_ids)
            match_ids = torch.zeros(max_qry_id+1, dtype=torch.int64, device=device)
            match_ids[matched_qry_ids] = torch.arange(num_matches, device=device)
            tgt_ids = matched_tgt_ids[match_ids[qry_ids]]

            for i in range(self.refine_iters):
                delta_tgt_sample_xy = self.tgt_sample_mul * tgt_sample_offsets[None, :, :] * key_wh[:, None, :]
                tgt_sample_xy = key_xy[:, None, :] + delta_tgt_sample_xy

                x_ids = (tgt_sample_xy[:, :, 0] * iW).to(torch.int64).clamp(min=0, max=iW-1)
                y_ids = (tgt_sample_xy[:, :, 1] * iH).to(torch.int64).clamp(min=0, max=iH-1)

                tgt_ids = tgt_ids[:, None].expand(-1, 4)
                refine_mask = tgt_masks[tgt_ids, y_ids, x_ids].sum(dim=1)
                refine_mask = (refine_mask > 0) & (refine_mask < grid_area) & (map_ids > 0)

                tgt_ids = tgt_ids[refine_mask].flatten()
                qry_ids = qry_ids[refine_mask].repeat_interleave(grid_area, dim=0)
                qry_ids_list.append(qry_ids)

                key_wh = key_wh[refine_mask] / grid_size
                key_wh = key_wh.repeat_interleave(grid_area, dim=0)

                delta_key_xy = grid_offsets[None, :, :] * key_wh.view(-1, grid_area, 2)
                delta_key_xy = delta_key_xy.flatten(0, 1)

                key_xy = key_xy[refine_mask].repeat_interleave(grid_area, dim=0)
                key_xy = key_xy + delta_key_xy
                key_xy_list.append(key_xy)

                batch_ids = batch_ids[refine_mask]
                batch_ids = batch_ids.repeat_interleave(grid_area, dim=0)
                batch_ids_list.append(batch_ids)

                map_ids = map_ids[refine_mask] - 1
                map_ids = map_ids.repeat_interleave(grid_area, dim=0)
                map_ids_list.append(map_ids)

                map_sizes_i = map_sizes[map_ids]
                feat_ids = torch.floor(key_xy * map_sizes_i).int()
                feat_ids[:, 1] = feat_ids[:, 1] * map_sizes_i[:, 0]
                feat_ids = map_offsets[map_ids] + feat_ids.sum(dim=1)

                qry_feats_i = qry_feats[qry_ids]
                key_feats_i = key_feats[batch_ids, feat_ids, :]

                for _ in range(self.qk_feat_iters):

                    if self.qry_key is not None:
                        qry_feats_i = self.qry_key(qry_feats_i, pair_feats=key_feats_i, module_id=i+1)

                    if self.key_qry is not None:
                        key_feats_i = self.key_qry(key_feats_i, pair_feats=qry_feats_i, module_id=i+1)

                seg_logits = (qry_feats_i * key_feats_i).sum(dim=1)
                seg_logits_list.append(seg_logits)

                if i == 0:
                    refine_ids = torch.nonzero(match_mask, as_tuple=True)[0]
                    refine_ids = refine_ids[torch.nonzero(refine_mask, as_tuple=True)[0]]

                    refine_mask = torch.zeros_like(seg_logits_list[0], dtype=torch.bool)
                    refine_mask[refine_ids] = 1.0

                refined_mask_list.append(refine_mask)

        refine_mask = torch.zeros_like(seg_logits, dtype=torch.bool)
        refined_mask_list.append(refine_mask)

        qry_ids = torch.cat(qry_ids_list, dim=0)
        key_xy = torch.cat(key_xy_list, dim=0)
        batch_ids = torch.cat(batch_ids_list, dim=0)
        map_ids = torch.cat(map_ids_list, dim=0)
        seg_logits = torch.cat(seg_logits_list, dim=0)
        refined_mask = torch.cat(refined_mask_list, dim=0)

        storage_dict['qry_ids'] = qry_ids
        storage_dict['key_xy'] = key_xy
        storage_dict['batch_ids'] = batch_ids
        storage_dict['map_ids'] = map_ids
        storage_dict['map_shapes'] = [key_feat_map.size()[-2:] for key_feat_map in key_feat_maps]
        storage_dict['seg_logits'] = seg_logits
        storage_dict['refined_mask'] = refined_mask

        # Get segmentation predictions if needed
        if self.get_segs and not self.training:
            self.compute_segs(storage_dict=storage_dict, cum_feats_batch=cum_feats_batch, **kwargs)

        # Draw predicted and target segmentations if needed
        if images_dict is not None:
            self.draw_segs(storage_dict=storage_dict, images_dict=images_dict, tgt_dict=tgt_dict, **kwargs)

        return storage_dict, images_dict

    def forward_loss(self, storage_dict, tgt_dict, loss_dict, analysis_dict=None, id=None, **kwargs):
        """
        Forward loss method of the TopDownSegHead module.

        Args:
            storage_dict (Dict): Storage dictionary containing following keys (after matching):
                - qry_ids (LongTensor): query indices of query-key pairs of shape [num_qry_key_pairs];
                - key_xy (FloatTensor): normalized key locations of query-key pairs of shape [num_qry_key_pairs, 2];
                - seg_logits (FloatTensor): segmentation logits of query-key pairs of shape [num_qry_key_pairs];
                - refined_mask (BoolTensor): mask indicating refined query-key pairs of shape [num_qry_key_pairs].;
                - matched_qry_ids (LongTensor): indices of matched queries of shape [num_pos_queries];
                - matched_tgt_ids (LongTensor): indices of corresponding matched targets of shape [num_pos_queries].

            tgt_dict (Dict): Target dictionary containing at least following key:
                - masks (BoolTensor): target segmentation masks of shape [num_targets, iH, iW].

            loss_dict (Dict): Dictionary containing different weighted loss terms.
            analysis_dict (Dict): Dictionary containing different analyses (default=None).
            id (int): Integer containing the head id (default=None).
            kwargs (Dict): Dictionary of keyword arguments passed to some underlying modules.

        Returns:
            loss_dict (Dict): Loss dictionary containing following additional keys:
                - seg_loss (FloatTensor): segmentation loss of shape [].

            analysis_dict (Dict): Analysis dictionary containing following additional keys (if not None):
                - seg_acc (FloatTensor): segmentation accuracy of shape [].

        Raises:
            ValueError: Error when a single query is matched with multiple targets.
        """

        # Perform matching if matcher is available
        if self.matcher is not None:
            self.matcher(storage_dict=storage_dict, tgt_dict=tgt_dict, analysis_dict=analysis_dict, **kwargs)

        # Retrieve various items of interest from storage dictionary
        qry_ids = storage_dict['qry_ids']
        key_xy = storage_dict['key_xy']
        seg_logits = storage_dict['seg_logits']
        refined_mask = storage_dict['refined_mask']

        matched_qry_ids = storage_dict['matched_qry_ids']
        matched_tgt_ids = storage_dict['matched_tgt_ids']

        # Get device
        device = seg_logits.device

        # Handle case where there are no positive matches
        if len(matched_qry_ids) == 0:

            # Get segmentation loss
            seg_loss = 0.0 * seg_logits.sum() + sum(0.0 * p.flatten()[0] for p in self.parameters())
            key_name = f'seg_loss_{id}' if id is not None else 'seg_loss'
            loss_dict[key_name] = seg_loss

            # Perform analyses if needed
            if analysis_dict is not None:

                # Get segmentation accuracy
                seg_acc = 1.0 if len(tgt_dict['masks']) == 0 else 0.0
                seg_acc = torch.tensor(seg_acc, dtype=seg_loss.dtype, device=device)

                key_name = f'seg_acc_{id}' if id is not None else 'seg_acc'
                analysis_dict[key_name] = 100 * seg_acc

            return loss_dict, analysis_dict

        # Get matched segmentation logits and corresponding targets
        counts = matched_qry_ids.unique(sorted=False, return_counts=True)[1]

        if torch.any(counts > 1):
            error_msg = "The TopDownSegHead does not support a single query to be matched with multiple targets."
            raise ValueError(error_msg)

        max_qry_id = max(matched_qry_ids.max().item(), qry_ids.max().item())
        matched_qry_mask = torch.zeros(max_qry_id+1, dtype=torch.bool, device=device)
        matched_qry_mask[matched_qry_ids] = True

        matched_qry_mask = matched_qry_mask[qry_ids]
        seg_logits = seg_logits[matched_qry_mask]

        num_matches = len(matched_qry_ids)
        match_ids = torch.zeros(max_qry_id+1, dtype=torch.int64, device=device)
        match_ids[matched_qry_ids] = torch.arange(num_matches, device=device)

        qry_ids = qry_ids[matched_qry_mask]
        match_ids = match_ids[qry_ids]
        tgt_ids = matched_tgt_ids[match_ids]

        tgt_masks = tgt_dict['masks']
        iH, iW = tgt_masks.size()[-2:]

        key_xy = key_xy[matched_qry_mask]
        x_ids = (key_xy[:, 0] * iW).to(torch.int64)
        y_ids = (key_xy[:, 1] * iH).to(torch.int64)

        seg_targets = tgt_masks[tgt_ids, y_ids, x_ids]
        seg_targets = seg_targets.float()

        # Get segmentation loss
        refined_mask = refined_mask[matched_qry_mask]
        loss_weights = torch.where(refined_mask, self.refined_weight, 1.0)

        inv_ids = torch.unique(qry_ids, sorted=False, return_inverse=True)[1]
        loss_weights = loss_weights / scatter_sum(loss_weights, inv_ids)[inv_ids]

        seg_loss = self.seg_loss(seg_logits, seg_targets, weight=loss_weights)
        seg_loss = seg_loss + sum(0.0 * p.flatten()[0] for p in self.parameters())

        key_name = f'seg_loss_{id}' if id is not None else 'seg_loss'
        loss_dict[key_name] = seg_loss

        # Perform analyses if needed
        if analysis_dict is not None:

            # Get segmentation accuracy
            seg_preds = seg_logits > 0
            seg_targets = seg_targets.bool()
            seg_acc = (seg_preds == seg_targets).sum() / len(seg_preds)

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
