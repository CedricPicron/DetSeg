"""
Collection of segmentation heads.
"""

from detectron2.layers import batched_nms
from detectron2.projects.point_rend.point_features import get_uncertain_point_coords_with_randomness
from detectron2.structures.instances import Instances
from detectron2.utils.visualizer import Visualizer
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
        apply_ids (List): List with integers determining when the head should be applied.
    """

    def __init__(self, qry_cfg, key_cfg, metadata, sample_attrs, loss_cfg, get_segs=True, dup_attrs=None,
                 max_segs=None, matcher_cfg=None, apply_ids=None, **kwargs):
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
            apply_ids (List): List with integers determining when the head should be applied (default=None).
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
        self.apply_ids = apply_ids

    @torch.no_grad()
    def compute_segs(self, storage_dict, pred_dicts, **kwargs):
        """
        Method computing the segmentation predictions.

        Args:
            storage_dict (Dict): Storage dictionary containing at least following keys:
                - images (Images): images structure containing the batched images of size [batch_size];
                - cls_logits (FloatTensor): classification logits of shape [num_feats, num_labels];
                - cum_feats_batch (LongTensor): cumulative number of features per batch entry [batch_size+1];
                - seg_logits (FloatTensor): map with segmentation logits of shape [num_feats, fH, fW].

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
        """

        # Retrieve desired items from storage dictionary
        images = storage_dict['images']
        cls_logits = storage_dict['cls_logits']
        cum_feats_batch = storage_dict['cum_feats_batch']
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

    def forward_pred(self, qry_feats, storage_dict, images_dict=None, **kwargs):
        """
        Forward prediction method of the BaseSegHead module.

        Args:
            qry_feats (FloatTensor): Query features of shape [num_feats, qry_feat_size].

            storage_dict (Dict): Storage dictionary containing at least following keys:
                - feat_maps (List): list of size [num_maps] with feature maps of shape [batch_size, feat_size, fH, fW];
                - images (Images): images structure containing the batched images of size [batch_size];
                - cum_feats_batch (LongTensor): cumulative number of features per batch entry [batch_size+1].

            images_dict (Dict): Dictionary with annotated images of predictions/targets (default=None).
            kwargs (Dict): Dictionary of keyword arguments not used by this module.

        Returns:
            storage_dict (Dict): Storage dictionary containing following additional key:
                - seg_logits (FloatTensor): map with segmentation logits of shape [num_feats, fH, fW].

            images_dict (Dict): Dictionary containing additional images annotated with segmentations (if given).
        """

        # Get query segmentation features
        qry_feats = self.qry(qry_feats)

        # Get key feature map
        in_feat_map = storage_dict['feat_maps'][0]
        base_map_size = storage_dict['images'].size(mode='with_padding')
        base_map_size = (base_map_size[1], base_map_size[0])
        key_feat_map = self.key(in_feat_map, base_map_size=base_map_size)

        # Get segmentation logits
        batch_size = key_feat_map.size(dim=0)
        cum_feats_batch = storage_dict['cum_feats_batch']
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
            self.compute_segs(storage_dict=storage_dict, **kwargs)

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

        map_offset (int): Integer with map offset used to determine the coarse key feature map for each query.
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
        metadata (detectron2.data.Metadata): Metadata instance containing additional dataset information.
        matcher (nn.Module): Optional matcher module determining the target segmentation maps.
        tgt_roi_ext (nn.ModuleList): List [seg_iters] of modules extracting the RoI-based target segmentation masks.
        seg_loss (nn.Module): Module computing the segmentation loss.
        seg_loss_weights (Tuple): Tuple of size [seg_iters] containing the segmentation loss weights.
        ref_loss (nn.Module): Module computing the refinement loss.
        ref_loss_weights (Tuple): Tuple of size [seg_iters] containing the refinement loss weights.
        apply_ids (List): List with integers determining when the head should be applied.
    """

    def __init__(self, roi_ext_cfg, seg_cfg, ref_cfg, fuse_td_cfg, fuse_key_cfg, trans_cfg, proc_cfg, map_offset,
                 key_min_id, key_max_id, seg_iters, refines_per_iter, mask_thr, metadata, seg_loss_cfg,
                 seg_loss_weights, ref_loss_cfg, ref_loss_weights, key_2d_cfg=None, pos_enc_cfg=None, qry_cfg=None,
                 fuse_qry_cfg=None, roi_ins_cfg=None, get_segs=True, dup_attrs=None, max_segs=None, matcher_cfg=None,
                 apply_ids=None, **kwargs):
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
            map_offset (int): Integer with map offset used to determine the coarse key feature map for each query.
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
            matcher_cfg (Dict): Configuration dictionary specifying the matcher module (default=None).
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

        # Build target RoI extractor
        tgt_roi_ext_cfg = dict(type='mmdet.SingleRoIExtractor')
        tgt_roi_ext_cfg['roi_layer'] = dict(type='RoIAlign', sampling_ratio=0)
        tgt_roi_ext_cfg['out_channels'] = 1
        tgt_roi_ext_cfg['featmap_strides'] = [1]
        self.tgt_roi_ext = nn.ModuleList()

        for i in range(seg_iters):
            tgt_roi_ext_cfg['roi_layer']['output_size'] = 2**i * roi_ext_cfg['roi_layer']['output_size']
            self.tgt_roi_ext.append(build_model(tgt_roi_ext_cfg))

        # Build segmentation and refinement loss modules
        self.seg_loss = build_model(seg_loss_cfg)
        self.ref_loss = build_model(ref_loss_cfg)

        # Set remaining attributes
        self.map_offset = map_offset
        self.key_min_id = key_min_id
        self.key_max_id = key_max_id
        self.seg_iters = seg_iters
        self.refines_per_iter = refines_per_iter
        self.get_segs = get_segs
        self.dup_attrs = dup_attrs
        self.max_segs = max_segs
        self.mask_thr = mask_thr
        self.metadata = metadata
        self.seg_loss_weights = seg_loss_weights
        self.ref_loss_weights = ref_loss_weights
        self.apply_ids = apply_ids

    @torch.no_grad()
    def compute_segs(self, qry_feats, storage_dict, pred_dicts, **kwargs):
        """
        Method computing the segmentation predictions.

        Args:
            qry_feats (FloatTensor): Query features of shape [num_feats, qry_feat_size].

            storage_dict (Dict): Storage dictionary containing at least following keys:
                - images (Images): images structure containing the batched images of size [batch_size];
                - cls_logits (FloatTensor): classification logits of shape [num_qrys, num_labels];
                - cum_feats_batch (LongTensor): cumulative number of features per batch entry [batch_size+1];
                - pred_boxes (Boxes): predicted 2D bounding boxes of size [num_feats].

            pred_dicts (List): List of size [num_pred_dicts] collecting various prediction dictionaries.
            kwargs (Dict): Dictionary of unused keyword arguments.

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
        cum_feats_batch = storage_dict['cum_feats_batch']
        pred_boxes = storage_dict['pred_boxes']

        # Get image size, number of features, number of classes and device
        iW, iH = images.size()
        num_feats = cls_logits.size(dim=0)
        num_classes = cls_logits.size(dim=1) - 1
        device = cls_logits.device

        # Get prediction query indices,labels and scores
        pred_qry_ids = torch.arange(num_feats, device=device)[:, None].expand(-1, num_classes).reshape(-1)
        pred_labels = torch.arange(num_classes, device=device)[None, :].expand(num_feats, -1).reshape(-1)
        pred_scores = cls_logits[:, :-1].sigmoid().view(-1)

        # Get prediction boxes in desired format
        pred_boxes = pred_boxes.to_format('xyxy').to_img_scale(images[0]).boxes

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
            roi_size = storage_dict['roi_size']
            roi_ids_list = storage_dict['roi_ids_list']
            pos_ids_list = storage_dict['pos_ids_list']
            seg_logits_list = storage_dict['seg_logits_list']

            # Get prediction masks
            num_rois = len(seg_qry_ids)
            mask_logits = torch.zeros(num_rois, 1, *roi_size, device=device)

            for j in range(self.seg_iters):
                roi_ids = roi_ids_list[j]
                pos_ids = pos_ids_list[j]

                seg_logits = seg_logits_list[j]
                mask_logits[roi_ids, 0, pos_ids[:, 1], pos_ids[:, 0]] = seg_logits

                if j < self.seg_iters-1:
                    roi_size = tuple(2*size for size in roi_size)
                    mask_logits = F.interpolate(mask_logits, roi_size, mode='bilinear', align_corners=False)

            mask_scores = mask_logits.sigmoid()
            pred_boxes_i = pred_boxes[seg_qry_ids]

            mask_scores = _do_paste_mask(mask_scores, pred_boxes_i, iH, iW, skip_empty=False)[0]
            mask_scores = mask_scores[pred_inv_ids]
            pred_masks_i = mask_scores > self.mask_thr

            pred_scores_i = pred_scores_i * (pred_masks_i * mask_scores).flatten(1).sum(dim=1)
            pred_scores_i = pred_scores_i / (pred_masks_i.flatten(1).sum(dim=1) + 1e-6)

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

    def get_preds(self, qry_feats, storage_dict, seg_qry_ids, **kwargs):
        """
        Method computing the segmentation and refinement logits for the desired queries.

        Args:
            qry_feats (FloatTensor): Query features of shape [num_feats, qry_feat_size].

            storage_dict (Dict): Storage dictionary containing at least following keys:
                - cum_feats_batch (LongTensor): cumulative number of features per batch entry [batch_size+1];
                - feat_maps (List): list of size [num_maps] with feature maps of shape [batch_size, feat_size, fH, fW];
                - images (Images): images structure containing the batched images of size [batch_size];
                - feat_ids (LongTensor): indices of selected features resulting in query features of shape [num_feats];
                - pred_boxes (Boxes): predicted 2D bounding boxes obtained from query features of size [num_feats].

            seg_qry_ids (LongTensor): Query indices for which to compute segmentations of shape [num_segs].
            kwargs (Dict): Dictionary of keyword arguments not used by this module.

        Returns:
            storage_dict (Dict): Storage dictionary containing following additional keys:
                - roi_size (Tuple): tuple [2] containing the size of the initial RoI in (height, width) format;
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
        roi_boxes = qry_boxes.to_format('xyxy').to_img_scale(images[0]).boxes.detach()
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
        roi_feats = self.roi_ins(roi_feats) if self.roi_ins is not None else roi_feats
        num_rois, feat_size, rH, rW = roi_feats.size()

        # Get map indices from which RoI features were extracted
        map_ids = self.roi_ext.map_roi_levels(roi_boxes, self.roi_ext.num_inputs)
        max_map_id = map_ids.max().item()
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
        seg_boxes = qry_boxes.to_format('xywh').normalize(images[0]).boxes

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

            # Refine graph if needed
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
        storage_dict['roi_size'] = (rH, rW)
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
            qry_feats (FloatTensor): Query features of shape [num_feats, qry_feat_size].

            storage_dict (Dict): Storage dictionary containing following keys (after matching):
                - images (Images): images structure containing the batched images of size [batch_size];
                - pred_boxes (Boxes): predicted 2D bounding boxes obtained from query features of size [num_feats];
                - matched_qry_ids (LongTensor): indices of matched queries of shape [num_pos_queries];
                - matched_tgt_ids (LongTensor): indices of corresponding matched targets of shape [num_pos_queries].

            tgt_dict (Dict): Target dictionary containing at least following key:
                - masks (BoolTensor): target segmentation masks of shape [num_targets, iH, iW].

            loss_dict (Dict): Dictionary containing different weighted loss terms.
            analysis_dict (Dict): Dictionary containing different analyses.
            id (int): Integer containing the head id (default=None).
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
        roi_boxes = roi_boxes.to_format('xyxy').to_img_scale(images[0]).boxes.detach()
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
