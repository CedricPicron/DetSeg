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
        qry (nn.Module): Optional module updating the query features.
        key_2d (nn.Module): Optional module updating the 2D key feature maps.

        key (nn.Module): Optional module updating the initial segmentation features.
        pos_enc (nn.Module): Optional module adding position features to the initial segmentation features.
        fuse_qry (nn.Module): Optional module fusing the query features with the initial segmentation features.

        proc (nn.ModuleList): List [seg_iters] of modules processing the segmentation features.
        seg (nn.ModuleList): List [seg_iters] of modules computing segmentation logits from segmentation features.
        ref (nn.ModuleList): List [seg_iters] of modules computing refinement logits from segmentation features.

        fuse_td (nn.ModuleList): List [seg_iters-1] of modules fusing top-down features with segmentation features.
        fuse_key (nn.ModuleList): List [seg_iters-1] of modules fusing key features with segmentation features.
        trans (nn.ModuleList): List [seg_iters-1] of modules transitioning segmentation features to new feature space.

        map_offset (int): Integer with map offset used to determine the coarse key feature map for each query.
        key_min_id (int): Integer containing the downsampling index of the highest resolution key feature map.
        key_max_id (int): Integer containing the downsampling index of the lowest resolution key feature map.
        seg_iters (int): Integer containing the number of segmentation iterations.
        refines_per_iter (int): Integer containing the number of refinements per refinement iteration.
        get_segs (bool): Boolean indicating whether to get segmentation predictions.

        dup_attrs (Dict): Optional dictionary specifying the duplicate removal mechanism, possibly containing:
            - type (str): string containing the type of duplicate removal mechanism (mandatory);
            - nms_candidates (int): integer containing the maximum number of candidate detections retained before NMS;
            - nms_thr (float): value of IoU threshold used during NMS to remove duplicate detections.

        max_segs (int): Optional integer with the maximum number of returned segmentation predictions.
        mask_thr (float): Value containing the mask threshold used to determine the segmentation masks.
        metadata (detectron2.data.Metadata): Metadata instance containing additional dataset information.
        matcher (nn.Module): Optional matcher module determining the target segmentation maps.
        seg_loss (nn.Module): Module computing the segmentation loss.
        seg_loss_weights (Tuple): Tuple of size [seg_iters] containing the segmentation loss weights.
        ref_loss (nn.Module): Module computing the refinement loss.
        ref_loss_weights (Tuple): Tuple of size [seg_iters] containing the refinement loss weights.
    """

    def __init__(self, proc_cfg, seg_cfg, ref_cfg, fuse_td_cfg, fuse_key_cfg, trans_cfg, map_offset, key_min_id,
                 key_max_id, seg_iters, refines_per_iter, mask_thr, metadata, seg_loss_cfg, seg_loss_weights,
                 ref_loss_cfg, ref_loss_weights, qry_cfg=None, key_2d_cfg=None, key_cfg=None, pos_enc_cfg=None,
                 fuse_qry_cfg=None, get_segs=True, dup_attrs=None, max_segs=None, matcher_cfg=None, **kwargs):
        """
        Initializes the TopDownSegHead module.

        Args:
            proc_cfg (Dict): Configuration dictionary specifying the processing module.
            seg_cfg (Dict): Configuration dictionary specifying the segmentation module.
            ref_cfg (Dict): Configuration dictionary specifying the refinement module.
            fuse_td_cfg (Dict): Configuration dictionary specifying the fuse top-down module.
            fuse_key_cfg (Dict): Configuration dictionary specifying the fuse key module.
            trans_cfg (Dict): Configuration dictionary specifying the transition module.
            map_offset (int): Integer with map offset used to determine the coarse key feature map for each query.
            key_min_id (int): Integer containing the downsampling index of the highest resolution key feature map.
            key_max_id (int): Integer containing the downsampling index of the lowest resolution key feature map.
            seg_iters (int): Integer containing the number of segmentation iterations.
            refines_per_iter (int): Integer containing the number of refinements per refinement iteration.
            mask_thr (float): Value containing the mask threshold used to determine the segmentation masks.
            metadata (detectron2.data.Metadata): Metadata instance containing additional dataset information.
            seg_loss_cfg (Dict): Configuration dictionary specifying the segmentation loss module.
            seg_loss_weights (Tuple): Tuple of size [seg_iters] containing the segmentation loss weights.
            ref_loss_cfg (Dict): Configuration dictionary specifying the refinement loss module.
            ref_loss_weights (Tuple): Tuple of size [seg_iters] containing the refinement loss weights.
            qry_cfg (Dict): Configuration dictionary specifying the query module (default=None).
            key_2d_cfg (Dict): Configuration dictionary specifying the key 2D module (default=None).
            key_cfg (Dict): Configuration dictionary specifying the key module (default=None).
            pos_enc_cfg (Dict): Configuration dictionary specifying the position encoder module (default=None).
            fuse_qry_cfg (Dict): Configuration dictionary specifying the fuse query module (default=None).
            get_segs (bool): Boolean indicating whether to get segmentation predictions (default=True).
            dup_attrs (Dict): Attribute dictionary specifying the duplicate removal mechanism (default=None).
            max_segs (int): Integer with the maximum number of returned segmentation predictions (default=None).
            matcher_cfg (Dict): Configuration dictionary specifying the matcher module (default=None).
            kwargs (Dict): Dictionary of unused keyword arguments.
        """

        # Initialization of default nn.Module
        super().__init__()

        # Build various modules used to obtain segmentation and refinement logits from inputs
        self.qry = build_model(qry_cfg) if qry_cfg is not None else None
        self.key_2d = build_model(key_2d_cfg) if key_2d_cfg is not None else None

        self.key = build_model(key_cfg) if key_cfg is not None else None
        self.pos_enc = build_model(pos_enc_cfg) if pos_enc_cfg is not None else None
        self.fuse_qry = build_model(fuse_qry_cfg) if fuse_qry_cfg is not None else None

        self.proc = nn.ModuleList([build_model(cfg_i) for cfg_i in proc_cfg])
        self.seg = nn.ModuleList([build_model(cfg_i) for cfg_i in seg_cfg])
        self.ref = nn.ModuleList([build_model(cfg_i) for cfg_i in ref_cfg])

        self.fuse_td = nn.ModuleList([build_model(cfg_i) for cfg_i in fuse_td_cfg])
        self.fuse_key = nn.ModuleList([build_model(cfg_i) for cfg_i in fuse_key_cfg])
        self.trans = nn.ModuleList([build_model(cfg_i) for cfg_i in trans_cfg])

        # Build matcher module if needed
        self.matcher = build_model(matcher_cfg) if matcher_cfg is not None else None

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

                    pred_boxes_i = pred_boxes[pred_qry_ids_i].to_format('xyxy')
                    iou_thr = self.dup_attrs['nms_thr']
                    non_dup_ids = batched_nms(pred_boxes_i.boxes, pred_scores_i, pred_labels_i, iou_thr)

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

            # Compute segmentation predictions for desired queries
            self.forward_pred(qry_feats, storage_dict, seg_qry_ids=seg_qry_ids, **kwargs)

            # Retrieve various items related to segmentation predictions from storage dictionary
            qry_ids = storage_dict['qry_ids']
            seg_xy = storage_dict['seg_xy']
            map_ids = storage_dict['map_ids']
            map_shapes = storage_dict['map_shapes']
            seg_logits = storage_dict['seg_logits']
            refined_mask = storage_dict['refined_mask']

            # Set segmentation logits to zero at locations that were refined
            seg_logits[refined_mask] = 0.0

            # Get prediction masks at image resolution
            pred_qry_mask = pred_inv_ids[:, None] == qry_ids[None, :]
            pred_ids = torch.nonzero(pred_qry_mask)[:, 0]

            num_preds = len(pred_qry_ids_i)
            seg_xy = seg_xy[None, :, :].expand(num_preds, -1, -1)[pred_qry_mask]
            map_ids = map_ids[None, :].expand(num_preds, -1)[pred_qry_mask]
            seg_logits = seg_logits[None, :].expand(num_preds, -1)[pred_qry_mask]

            num_maps = len(map_shapes)
            map_shapes_tensor = torch.tensor(map_shapes, device=device)
            map_shapes_prod = map_shapes_tensor.prod(dim=1)

            numel_per_map = num_preds * map_shapes_prod
            cum_numel_per_map = torch.cat([numel_per_map.new_zeros([1]), numel_per_map.cumsum(dim=0)], dim=0)

            map_shapes_tensor = map_shapes_tensor[map_ids]
            x_ids = (seg_xy[:, 0] * map_shapes_tensor[:, 1]).to(torch.int64)
            y_ids = (seg_xy[:, 1] * map_shapes_tensor[:, 0]).to(torch.int64)

            prob_ids = cum_numel_per_map[map_ids] + pred_ids * map_shapes_prod[map_ids]
            prob_ids = prob_ids + y_ids * map_shapes_tensor[:, 1] + x_ids

            numel_total = cum_numel_per_map[-1].item()
            pred_probs = torch.zeros(numel_total, device=device)
            pred_probs[prob_ids] = seg_logits.sigmoid()

            cum_numel_per_map = cum_numel_per_map.tolist()
            pred_prob_maps = [pred_probs[i0:i1] for i0, i1 in zip(cum_numel_per_map[:-1], cum_numel_per_map[1:])]
            pred_prob_maps = [pred_prob_maps[i].view(num_preds, 1, *map_shapes[i]) for i in range(num_maps)]
            pred_probs = pred_prob_maps[-1]

            for j in range(num_maps-2, -1, -1):
                pred_probs = F.interpolate(pred_probs, size=map_shapes[j], mode='bilinear', align_corners=False)
                pred_prob_map = pred_prob_maps[j]

                insert_mask = pred_prob_map > 0
                insert_vals = pred_prob_map[insert_mask]
                pred_probs[insert_mask] = insert_vals

            pred_probs = F.interpolate(pred_probs, size=(iH, iW), mode='bilinear', align_corners=False)
            pred_masks_i = pred_probs[:, 0] > self.mask_thr

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

    def forward_pred(self, qry_feats, storage_dict, images_dict=None, seg_qry_ids=None, **kwargs):
        """
        Forward prediction method of the TopDownSegHead module.

        Args:
            qry_feats (FloatTensor): Query features of shape [num_feats, qry_feat_size].

            storage_dict (Dict): Storage dictionary containing at least following keys:
                - cum_feats_batch (LongTensor): cumulative number of features per batch entry [batch_size+1];
                - feat_maps (List): list of size [num_maps] with feature maps of shape [batch_size, feat_size, fH, fW];
                - images (Images): images structure containing the batched images of size [batch_size];
                - feat_ids (LongTensor): indices of selected features resulting in query features of shape [num_feats];
                - pred_boxes (Boxes): predicted 2D bounding boxes obtained from query features of size [num_feats].

            images_dict (Dict): Dictionary with annotated images of predictions/targets (default=None).
            seg_qry_ids (Dict): Indices determining for which queries to compute segmetations (default=None).
            kwargs (Dict): Dictionary of keyword arguments not used by this module.

        Returns:
            storage_dict (Dict): Storage dictionary (possibly) containing following additional keys:
                - qry_ids (LongTensor): query indices (post-selection) of predictions of shape [num_preds];
                - seg_xy (FloatTensor): normalized key locations of predictions of shape [num_preds, 2];
                - batch_ids (LongTensor): batch indices of predictions of shape [num_preds];
                - map_ids (LongTensor): map indices of predictions of shape [num_preds];
                - map_shapes (List): list with map shapes in (height, width) format of size [self.key_max_id+1];
                - seg_logits (FloatTensor): segmentation logits of shape [num_preds];
                - ref_logits (FloatTensor): refinement logits of shape [num_preds];
                - num_stage_preds (List): list with number of predictions per stage of size [self.seg_iters];
                - refined_mask (BoolTensor): mask indicating refined predictions of shape [num_preds].

            images_dict (Dict): Dictionary (possibly) containing additional images annotated with segmentations.
        """

        # Handle case where no segmentation queries are provided
        if seg_qry_ids is None:

            if self.training:
                return storage_dict, images_dict

            if self.get_segs:
                self.compute_segs(qry_feats, storage_dict=storage_dict, **kwargs)

            if images_dict is not None:
                self.draw_segs(storage_dict=storage_dict, images_dict=images_dict, **kwargs)

            return storage_dict, images_dict

        # Get device
        device = qry_feats.device

        # Retrieve various items from storage dictionary
        cum_feats_batch = storage_dict['cum_feats_batch']
        key_feat_maps = storage_dict['feat_maps']
        images = storage_dict['images']
        feat_ids = storage_dict['feat_ids']
        qry_boxes = storage_dict['pred_boxes'].clone()

        # Get batch indices
        batch_size = len(cum_feats_batch) - 1
        batch_ids = torch.arange(batch_size, device=device)
        batch_ids = batch_ids.repeat_interleave(cum_feats_batch.diff())

        # Select for which queries to compute segmentations
        qry_feats = qry_feats[seg_qry_ids]
        batch_ids = batch_ids[seg_qry_ids]
        feat_ids = feat_ids[seg_qry_ids]
        qry_boxes = qry_boxes[seg_qry_ids]

        # Get updated query features
        qry_feats = self.qry(qry_feats) if self.qry is not None else qry_feats

        # Get updated key feature maps
        key_feat_maps = self.key_2d(key_feat_maps) if self.key_2d is not None else key_feat_maps

        # Get coarse map indices to sample keys from for each query
        cum_feats_map = [key_feat_map.flatten(2).size(dim=2) for key_feat_map in key_feat_maps]
        cum_feats_map = torch.tensor(cum_feats_map, device=device).cumsum(dim=0)

        qry_feat_ids = feat_ids % cum_feats_map[-1]
        coa_map_ids = (qry_feat_ids[:, None] - cum_feats_map[None, :]) >= 0
        coa_map_ids = coa_map_ids.sum(dim=1) - self.map_offset
        coa_map_ids = torch.clamp(coa_map_ids, min=0)

        # Get batch and map masks
        num_maps = len(key_feat_maps)
        batch_masks = [batch_ids == i for i in range(batch_size)]
        map_masks = [coa_map_ids == j for j in range(num_maps)]

        # Normalize and clip query boxes
        boxes_per_img = [batch_mask.sum().item() for batch_mask in batch_masks]
        qry_boxes.boxes_per_img = torch.tensor(boxes_per_img, device=device)
        qry_boxes = qry_boxes.normalize(images).clip((0.0, 0.0, 1.0, 1.0), eps=1e-6)[0]

        # Initialize adjacency offset and empty lists
        core_masks_list = []
        adj_offset = 0
        adj_ids_list = []
        qry_ids_list = []
        seg_xy_list = []
        seg_wh_list = []
        batch_ids_list = []
        map_ids_list = []
        seg_feats_list = []

        # Extract segmentation features per image and map
        for i in range(batch_size):
            for j in range(num_maps):

                # Get query indices of interest
                qry_ids_ij = torch.nonzero(batch_masks[i] & map_masks[j])[:, 0]
                num_qrys_ij = len(qry_ids_ij)

                if num_qrys_ij == 0:
                    continue

                # Get query boxes
                qry_boxes_ij = qry_boxes[qry_ids_ij].to_format('xyxy').boxes

                # Get padded key features in desired format
                key_feat_map = key_feat_maps[j][i]
                kH, kW = key_feat_map.size()[-2:]

                key_feat_map = F.pad(key_feat_map, (1, 1, 1, 1), mode='constant', value=0.0)
                key_feats = key_feat_map.flatten(1).t().contiguous()

                # Get various masks
                pts_x = torch.linspace(-0.5/kW, 1+0.5/kW, steps=kW+2, device=device)
                pts_y = torch.linspace(-0.5/kH, 1+0.5/kH, steps=kH+2, device=device)

                grid_x = pts_x[None, None, :].expand(-1, kH+2, -1)
                grid_y = pts_y[None, :, None].expand(-1, -1, kW+2)

                left_mask_0 = grid_x > qry_boxes_ij[:, 0, None, None] - 1.5/kW
                top_mask_0 = grid_y > qry_boxes_ij[:, 1, None, None] - 1.5/kH
                right_mask_0 = grid_x < qry_boxes_ij[:, 2, None, None] + 1.5/kW
                bot_mask_0 = grid_y < qry_boxes_ij[:, 3, None, None] + 1.5/kH

                left_mask_1 = left_mask_0.clone()
                top_mask_1 = top_mask_0.clone()
                right_mask_1 = right_mask_0.clone()
                bot_mask_1 = bot_mask_0.clone()

                qry_ids = torch.arange(num_qrys_ij, device=device)
                left_ids = (pts_x[None, :] > qry_boxes_ij[:, 0, None] - 1.5/kW).int().argmax(dim=1)
                top_ids = (pts_y[None, :] > qry_boxes_ij[:, 1, None] - 1.5/kH).int().argmax(dim=1)
                right_ids = (pts_x[None, :] < qry_boxes_ij[:, 2, None] + 1.5/kW).int().argmin(dim=1) - 1
                bot_ids = (pts_y[None, :] < qry_boxes_ij[:, 3, None] + 1.5/kH).int().argmin(dim=1) - 1

                left_mask_1[qry_ids, :, left_ids] = False
                top_mask_1[qry_ids, top_ids, :] = False
                right_mask_1[qry_ids, :, right_ids] = False
                bot_mask_1[qry_ids, bot_ids, :] = False

                left_mask_2 = left_mask_1.clone()
                top_mask_2 = top_mask_1.clone()
                right_mask_2 = right_mask_1.clone()
                bot_mask_2 = bot_mask_1.clone()

                left_mask_2[qry_ids, :, left_ids+1] = False
                top_mask_2[qry_ids, top_ids+1, :] = False
                right_mask_2[qry_ids, :, right_ids-1] = False
                bot_mask_2[qry_ids, bot_ids-1, :] = False

                left_mask_0 = left_mask_0.flatten(1)
                left_mask_1 = left_mask_1.flatten(1)
                left_mask_2 = left_mask_2.flatten(1)

                top_mask_0 = top_mask_0.flatten(1)
                top_mask_1 = top_mask_1.flatten(1)
                top_mask_2 = top_mask_2.flatten(1)

                right_mask_0 = right_mask_0.flatten(1)
                right_mask_1 = right_mask_1.flatten(1)
                right_mask_2 = right_mask_2.flatten(1)

                bot_mask_0 = bot_mask_0.flatten(1)
                bot_mask_1 = bot_mask_1.flatten(1)
                bot_mask_2 = bot_mask_2.flatten(1)

                # Get key and local query indices
                key_mask = left_mask_0 & top_mask_0 & right_mask_0 & bot_mask_0
                local_qry_ids, key_ids = torch.nonzero(key_mask, as_tuple=True)
                num_keys = len(key_ids)

                # Get core mask
                core_mask = left_mask_1 & top_mask_1 & right_mask_1 & bot_mask_1
                core_mask = core_mask[local_qry_ids, key_ids]
                core_masks_list.append(core_mask)

                # Get adjacency indices
                i00 = (left_mask_0 & top_mask_0 & right_mask_2 & bot_mask_2)[local_qry_ids, key_ids].nonzero()
                i01 = (left_mask_1 & top_mask_0 & right_mask_1 & bot_mask_2)[local_qry_ids, key_ids].nonzero()
                i02 = (left_mask_2 & top_mask_0 & right_mask_0 & bot_mask_2)[local_qry_ids, key_ids].nonzero()
                i10 = (left_mask_0 & top_mask_1 & right_mask_2 & bot_mask_1)[local_qry_ids, key_ids].nonzero()
                i11 = core_mask.nonzero()
                i12 = (left_mask_2 & top_mask_1 & right_mask_0 & bot_mask_1)[local_qry_ids, key_ids].nonzero()
                i20 = (left_mask_0 & top_mask_2 & right_mask_2 & bot_mask_0)[local_qry_ids, key_ids].nonzero()
                i21 = (left_mask_1 & top_mask_2 & right_mask_1 & bot_mask_0)[local_qry_ids, key_ids].nonzero()
                i22 = (left_mask_2 & top_mask_2 & right_mask_0 & bot_mask_0)[local_qry_ids, key_ids].nonzero()
                adj_ids = torch.cat([i00, i01, i02, i10, i11, i12, i20, i21, i22], dim=1)

                adj_ids += adj_offset
                adj_ids_list.append(adj_ids)
                adj_offset += num_keys

                # Get query indices
                qry_ids_ij = qry_ids_ij[local_qry_ids]
                qry_ids_list.append(qry_ids_ij)

                # Get segmentation locations
                seg_xy = torch.meshgrid(pts_x, pts_y, indexing='xy')
                seg_xy = torch.stack(seg_xy, dim=0).flatten(1)
                seg_xy = seg_xy[:, key_ids].t()
                seg_xy_list.append(seg_xy)

                # Get segmentation widths and heights
                num_core_keys = core_mask.sum().item()
                seg_wh = torch.tensor([1/kW, 1/kH], device=device)
                seg_wh = seg_wh[None, :].expand(num_core_keys, -1)
                seg_wh_list.append(seg_wh)

                # Get batch indices
                batch_ids_i = torch.full(size=[num_core_keys], fill_value=i, device=device)
                batch_ids_list.append(batch_ids_i)

                # Get map indices
                map_ids_i = torch.full(size=[num_core_keys], fill_value=j, device=device)
                map_ids_list.append(map_ids_i)

                # Get segmentation features
                seg_feats = key_feats[key_ids, :]
                seg_feats_list.append(seg_feats)

        # Concatenate entries from different images and maps
        core_mask = torch.cat(core_masks_list, dim=0)
        adj_ids = torch.cat(adj_ids_list, dim=0)
        qry_ids = torch.cat(qry_ids_list, dim=0)
        seg_xy = torch.cat(seg_xy_list, dim=0)
        seg_wh = torch.cat(seg_wh_list, dim=0)
        batch_ids = torch.cat(batch_ids_list, dim=0)
        map_ids = torch.cat(map_ids_list, dim=0) + self.key_min_id
        seg_feats = torch.cat(seg_feats_list, dim=0)

        # Get ids and base_offs tensors used during update of adjacency indices
        ids = torch.tensor([[0, 1, 1, 3, 4, 4, 3, 4, 4],
                            [1, 1, 2, 4, 4, 5, 4, 4, 5],
                            [3, 4, 4, 3, 4, 4, 6, 7, 7],
                            [4, 4, 5, 4, 4, 5, 7, 7, 8]], device=device)

        base_offs = torch.tensor([[0, 1, 0, 2, 3, 2, 0, 1, 0],
                                  [1, 0, 1, 3, 2, 3, 1, 0, 1],
                                  [2, 3, 2, 0, 1, 0, 2, 3, 2],
                                  [3, 2, 3, 1, 0, 1, 3, 2, 3]], device=device)

        # Get grid offsets
        grid_offsets = torch.arange(2, device=device) - 0.5
        grid_offsets = torch.meshgrid(grid_offsets, grid_offsets, indexing='xy')
        grid_offsets = torch.stack(grid_offsets, dim=2).flatten(0, 1)

        # Get concatenated key features and key feature size
        cat_key_feats = [key_feat_map.flatten(2).permute(0, 2, 1) for key_feat_map in key_feat_maps]
        cat_key_feats = torch.cat(cat_key_feats, dim=1)
        key_feat_size = cat_key_feats.size(dim=2)

        # Get map shapes, sizes and offsets
        base_map_size = images.size(mode='with_padding')
        base_map_size = (base_map_size[1], base_map_size[0])

        map_shape = base_map_size
        map_shapes = [base_map_size]

        for _ in range(self.key_max_id):
            map_shape = ((map_shape[0] + 1) // 2, (map_shape[1] + 1) // 2)
            map_shapes.append(map_shape)

        map_sizes = [torch.tensor([mW, mH], device=device) for mH, mW in map_shapes]
        map_sizes = torch.stack(map_sizes, dim=0)

        map_offsets = [map_sizes.new_zeros([1]), map_sizes[:-1].prod(dim=1)]
        map_offsets = torch.cat(map_offsets).cumsum(dim=0)
        delta_key_map_offset = map_offsets[self.key_min_id]

        # Get updated segmentation features
        seg_feats = self.key(seg_feats) if self.key is not None else seg_feats

        # Add position encodings if needed
        if self.pos_enc is not None:
            qry_boxes = qry_boxes[qry_ids].to_format('xywh').boxes
            norm_seg_xy = (seg_xy - qry_boxes[:, :2]) / qry_boxes[:, 2:]
            seg_feats = seg_feats + self.pos_enc(norm_seg_xy)

        # Fuse query features if needed
        if self.fuse_qry is not None:
            fuse_qry_feats = torch.cat([qry_feats[qry_ids], seg_feats], dim=1)
            seg_feats = seg_feats + self.fuse_qry(fuse_qry_feats)

        # Only keep core query indices and segmentation locations
        qry_ids = qry_ids[core_mask]
        seg_xy = seg_xy[core_mask]

        # Store desired items in lists
        qry_ids_list = [qry_ids]
        seg_xy_list = [seg_xy]
        batch_ids_list = [batch_ids]
        map_ids_list = [map_ids]

        # Initialize empty lists
        seg_logits_list = []
        ref_logits_list = []
        num_stage_preds = []
        refined_mask_list = []

        # Perform segmentation iterations
        for i in range(self.seg_iters):

            # Process segmentation features
            seg_feats = self.proc[i](seg_feats, mask=core_mask, adj_ids=adj_ids)

            # Get segmentation and refinement logits
            core_seg_feats = seg_feats[core_mask]
            seg_logits = self.seg[i](core_seg_feats)
            ref_logits = self.ref[i](core_seg_feats)

            seg_logits_list.append(seg_logits)
            ref_logits_list.append(ref_logits)

            # Save number of predictions for this refinement stage
            num_stage_preds.append(len(seg_logits))

            # Refine graph if needed
            if i < self.seg_iters-1:

                # Get refine mask
                core_refine_mask = map_ids > 0

                if core_refine_mask.sum().item() > self.refines_per_iter:
                    refine_ids = torch.topk(ref_logits[core_refine_mask], self.refines_per_iter, sorted=False)[1]
                    refine_ids = core_refine_mask.nonzero()[refine_ids, 0]

                    core_refine_mask = torch.zeros_like(core_refine_mask)
                    core_refine_mask[refine_ids] = True

                refine_mask = core_mask.clone()
                refine_mask[core_mask] = core_refine_mask
                refined_mask_list.append(core_refine_mask)

                # Update adjacency indices
                adj_ids = adj_ids[core_refine_mask]
                adj_ids = adj_ids[:, ids]

                shifts = torch.where(refine_mask, 3, 0).cumsum(dim=0)
                shifts = shifts[adj_ids]

                adj_refine_mask = refine_mask[adj_ids]
                offs = base_offs[None, :, :].expand_as(adj_refine_mask).clone()
                offs[~adj_refine_mask] = 0

                adj_ids = adj_ids + shifts - offs
                adj_ids = adj_ids.flatten(0, 1)
                used_ids, adj_ids = adj_ids.unique(sorted=True, return_inverse=True)

                # Update core mask
                repeats = torch.where(refine_mask, 4, 1)
                core_mask = refine_mask.repeat_interleave(repeats, dim=0)
                core_mask = core_mask[used_ids]

                # Update query indices
                qry_ids = qry_ids[core_refine_mask].repeat_interleave(4, dim=0)
                qry_ids_list.append(qry_ids)

                # Update segmentation widths and heights
                seg_wh = seg_wh[core_refine_mask] / 2
                seg_wh = seg_wh.repeat_interleave(4, dim=0)

                # Update segmentation locations
                delta_seg_xy = grid_offsets[None, :, :] * seg_wh.view(-1, 4, 2)
                delta_seg_xy = delta_seg_xy.flatten(0, 1)

                seg_xy = seg_xy[core_refine_mask].repeat_interleave(4, dim=0)
                seg_xy = seg_xy + delta_seg_xy
                seg_xy_list.append(seg_xy)

                # Update batch indices
                batch_ids = batch_ids[core_refine_mask]
                batch_ids = batch_ids.repeat_interleave(4, dim=0)
                batch_ids_list.append(batch_ids)

                # Update map indices
                map_ids = map_ids[core_refine_mask] - 1
                map_ids = map_ids.repeat_interleave(4, dim=0)
                map_ids_list.append(map_ids)

                # Fuse top-down features
                map_sizes_i = map_sizes[map_ids]
                map_offsets_i = map_offsets[map_ids]

                feat_ids = torch.floor(seg_xy * map_sizes_i).int()
                feat_ids[:, 1] = feat_ids[:, 1] * map_sizes_i[:, 0]
                feat_ids = (map_offsets_i - delta_key_map_offset) + feat_ids.sum(dim=1)

                fuse_td_feats = self.fuse_td[i](seg_feats[refine_mask])
                seg_feats = seg_feats.repeat_interleave(repeats, dim=0)[used_ids]
                seg_feats[core_mask] += fuse_td_feats

                # Fuse key features
                select_mask = map_ids >= self.key_min_id
                key_feats = torch.zeros(len(select_mask), key_feat_size, device=device)
                key_feats[select_mask] = cat_key_feats[batch_ids[select_mask], feat_ids[select_mask], :]

                fuse_key_feats = torch.cat([seg_feats[core_mask], key_feats], dim=1)
                seg_feats[core_mask] += self.fuse_key[i](fuse_key_feats)

                # Transition segmentation features
                seg_feats = self.trans[i](seg_feats)

        # Get final refine mask
        core_refine_mask = torch.zeros_like(ref_logits, dtype=torch.bool)
        refined_mask_list.append(core_refine_mask)

        # Concatenate entries from different refinement iterations
        qry_ids = torch.cat(qry_ids_list, dim=0)
        seg_xy = torch.cat(seg_xy_list, dim=0)
        batch_ids = torch.cat(batch_ids_list, dim=0)
        map_ids = torch.cat(map_ids_list, dim=0)
        seg_logits = torch.cat(seg_logits_list, dim=0)
        ref_logits = torch.cat(ref_logits_list, dim=0)
        refined_mask = torch.cat(refined_mask_list, dim=0)

        # Store desired items in storage dictionary
        storage_dict['qry_ids'] = qry_ids
        storage_dict['seg_xy'] = seg_xy
        storage_dict['batch_ids'] = batch_ids
        storage_dict['map_ids'] = map_ids
        storage_dict['map_shapes'] = map_shapes
        storage_dict['seg_logits'] = seg_logits
        storage_dict['ref_logits'] = ref_logits
        storage_dict['num_stage_preds'] = num_stage_preds
        storage_dict['refined_mask'] = refined_mask

        return storage_dict, images_dict

    def forward_loss(self, qry_feats, storage_dict, tgt_dict, loss_dict, analysis_dict, id=None, **kwargs):
        """
        Forward loss method of the TopDownSegHead module.

        Args:
            qry_feats (FloatTensor): Query features of shape [num_feats, qry_feat_size].

            storage_dict (Dict): Storage dictionary containing following keys (after matching):
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
                - seg_loss (FloatTensor): segmentation loss over all stages of shape [];
                - ref_loss (FloatTensor): refinement loss over all stages of shape [].

            analysis_dict (Dict): Analysis dictionary containing following additional keys (if not None):
                - seg_loss_{i} (FloatTensor): segmentation loss of stage {i} of shape [];
                - ref_loss_{i} (FloatTensor): refinement loss of stage {i} of shape [];
                - seg_acc_{i} (FloatTensor): segmentation accuracy of stage {i} of shape [];
                - seg_acc (FloatTensor): segmentation accuracy over all stages of shape [];
                - ref_acc_{i} (FloatTensor): refinement accuracy of stage {i} of shape [];
                - ref_acc (FloatTensor): refinement accuracy over all stages of shape [].

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

            # Get refinement loss
            ref_loss = 0.0 * qry_feats.sum()

            for i in range(self.seg_iters):
                key_name = f'ref_loss_{id}_{i}' if id is not None else f'ref_loss_{i}'
                analysis_dict[key_name] = ref_loss.detach()

            key_name = f'ref_loss_{id}' if id is not None else 'ref_loss'
            loss_dict[key_name] = ref_loss

            # Get segmentation and refinement accuracies
            with torch.no_grad():

                # Get segmentation accuracies
                seg_acc = 1.0 if len(tgt_dict['masks']) == 0 else 0.0
                seg_acc = torch.tensor(seg_acc, dtype=seg_loss.dtype, device=device)

                for i in range(self.seg_iters):
                    key_name = f'seg_acc_{id}_{i}' if id is not None else f'seg_acc_{i}'
                    analysis_dict[key_name] = 100 * seg_acc

                key_name = f'seg_acc_{id}' if id is not None else 'seg_acc'
                analysis_dict[key_name] = 100 * seg_acc

                # Get refinement accuracies
                ref_acc = 1.0 if len(tgt_dict['masks']) == 0 else 0.0
                ref_acc = torch.tensor(ref_acc, dtype=ref_loss.dtype, device=device)

                for i in range(self.seg_iters):
                    key_name = f'ref_acc_{id}_{i}' if id is not None else f'ref_acc_{i}'
                    analysis_dict[key_name] = 100 * ref_acc

                key_name = f'ref_acc_{id}' if id is not None else 'ref_acc'
                analysis_dict[key_name] = 100 * ref_acc

            return loss_dict, analysis_dict

        # Check that no query is matched with multiple targets
        counts = matched_qry_ids.unique(sorted=False, return_counts=True)[1]

        if torch.any(counts > 1):
            error_msg = "The TopDownSegHead does not support a single query to be matched with multiple targets."
            raise ValueError(error_msg)

        # Compute segmentation predictions for desired queries
        self.forward_pred(qry_feats, storage_dict, seg_qry_ids=matched_qry_ids, **kwargs)

        # Retrieve various items related to segmentation predictions from storage dictionary
        qry_ids = storage_dict['qry_ids']
        seg_xy = storage_dict['seg_xy']
        map_ids = storage_dict['map_ids']
        map_shapes = storage_dict['map_shapes']
        seg_logits = storage_dict['seg_logits']
        ref_logits = storage_dict['ref_logits']
        num_stage_preds = storage_dict['num_stage_preds']

        # Update matched query indices
        matched_qry_ids = torch.arange(num_matches, device=device)

        # Get target segmentation values
        tgt_masks = tgt_dict['masks']
        tgt_maps = tgt_masks[:, None, :, :].float()
        tgt_maps_list = [tgt_maps]

        for _ in range(self.key_max_id):
            tgt_maps = F.avg_pool2d(tgt_maps, kernel_size=2, stride=2, ceil_mode=True)
            tgt_maps_list.append(tgt_maps)

        tgt_seg_vals_list = [(tgt_maps > 0).byte() + (tgt_maps >= 1.0).byte() for tgt_maps in tgt_maps_list]
        tgt_seg_vals = torch.cat([tgt_seg_vals_i.flatten(1) for tgt_seg_vals_i in tgt_seg_vals_list], dim=1)

        # Get segmentation and refinement logits with corresponding targets
        map_sizes = [torch.tensor([mW, mH], device=device) for mH, mW in map_shapes]
        map_sizes = torch.stack(map_sizes, dim=0)

        map_offsets = [map_sizes.new_zeros([1]), map_sizes[:-1].prod(dim=1)]
        map_offsets = torch.cat(map_offsets).cumsum(dim=0)

        map_sizes = map_sizes[map_ids]
        map_offsets = map_offsets[map_ids]

        x_ids = (seg_xy[:, 0] * map_sizes[:, 0]).int()
        y_ids = (seg_xy[:, 1] * map_sizes[:, 1]).int()
        hw_ids = map_offsets + y_ids * map_sizes[:, 0] + x_ids

        tgt_ids = matched_tgt_ids[qry_ids]
        targets = tgt_seg_vals[tgt_ids, hw_ids]
        seg_mask = targets != 1

        seg_logits = seg_logits[seg_mask]
        seg_targets = targets[seg_mask].float() / 2
        ref_targets = (~seg_mask).float()

        # Get segmentation loss
        seg_masks = seg_mask.split(num_stage_preds)
        seg_num_stage_preds = [seg_mask.sum().item() for seg_mask in seg_masks]

        seg_logits_list = seg_logits.split(seg_num_stage_preds)
        seg_targets_list = seg_targets.split(seg_num_stage_preds)

        seg_zip = zip(seg_logits_list, seg_targets_list)
        seg_loss = sum(0.0 * p.flatten()[0] for p in self.parameters())

        for i, (seg_logits_i, seg_targets_i) in enumerate(seg_zip):

            if len(seg_logits_i) > 0:
                seg_loss_i = self.seg_loss(seg_logits_i, seg_targets_i)
                seg_loss_i *= self.seg_loss_weights[i] * num_matches
                seg_loss += seg_loss_i

            else:
                seg_loss_i = torch.tensor(0.0, device=device)

            key_name = f'seg_loss_{id}_{i}' if id is not None else f'seg_loss_{i}'
            analysis_dict[key_name] = seg_loss_i.detach()

        key_name = f'seg_loss_{id}' if id is not None else 'seg_loss'
        loss_dict[key_name] = seg_loss

        # Get refinement loss
        ref_logits_list = ref_logits.split(num_stage_preds)
        ref_targets_list = ref_targets.split(num_stage_preds)

        ref_zip = zip(ref_logits_list, ref_targets_list)
        ref_loss = sum(0.0 * p.flatten()[0] for p in self.parameters())

        for i, (ref_logits_i, ref_targets_i) in enumerate(ref_zip):

            if len(ref_logits_i) > 0:
                ref_loss_i = self.ref_loss(ref_logits_i, ref_targets_i)
                ref_loss_i *= self.ref_loss_weights[i] * num_matches
                ref_loss += ref_loss_i

            else:
                ref_loss_i = torch.tensor(0.0, device=device)

            key_name = f'ref_loss_{id}_{i}' if id is not None else f'ref_loss_{i}'
            analysis_dict[key_name] = ref_loss_i.detach()

        key_name = f'ref_loss_{id}' if id is not None else 'ref_loss'
        loss_dict[key_name] = ref_loss

        # Get segmentation and refinement accuracies
        with torch.no_grad():

            # Get segmentation accuracies
            seg_preds = seg_logits > 0
            seg_targets = seg_targets.bool()

            seg_preds_list = seg_preds.split(seg_num_stage_preds)
            seg_targets_list = seg_targets.split(seg_num_stage_preds)

            for i, (seg_preds_i, seg_targets_i) in enumerate(zip(seg_preds_list, seg_targets_list)):
                if len(seg_preds_i) > 0:
                    seg_acc_i = (seg_preds_i == seg_targets_i).sum() / len(seg_preds_i)
                else:
                    seg_acc_i = torch.tensor(1.0, dtype=torch.float, device=device)

                key_name = f'seg_acc_{id}_{i}' if id is not None else f'seg_acc_{i}'
                analysis_dict[key_name] = 100 * seg_acc_i

            if len(seg_preds) > 0:
                seg_acc = (seg_preds == seg_targets).sum() / len(seg_preds)
            else:
                seg_acc = torch.tensor(1.0, dtype=torch.float, device=device)

            key_name = f'seg_acc_{id}' if id is not None else 'seg_acc'
            analysis_dict[key_name] = 100 * seg_acc

            # Get refinement accuracies
            ref_preds = ref_logits > 0
            ref_targets = ref_targets.bool()

            ref_targets_list = ref_targets.split(num_stage_preds)
            ref_num_stage_preds = [ref_targets_i.sum().item() for ref_targets_i in ref_targets_list]

            ref_preds = ref_preds[ref_targets]
            ref_targets = ref_targets[ref_targets]

            ref_preds_list = ref_preds.split(ref_num_stage_preds)
            ref_targets_list = ref_targets.split(ref_num_stage_preds)

            for i, (ref_preds_i, ref_targets_i) in enumerate(zip(ref_preds_list, ref_targets_list)):
                if len(ref_preds_i) > 0:
                    ref_acc_i = (ref_preds_i == ref_targets_i).sum() / len(ref_preds_i)
                else:
                    ref_acc_i = torch.tensor(1.0, dtype=torch.float, device=device)

                key_name = f'ref_acc_{id}_{i}' if id is not None else f'ref_acc_{i}'
                analysis_dict[key_name] = 100 * ref_acc_i

            if len(ref_preds) > 0:
                ref_acc = (ref_preds == ref_targets).sum() / len(ref_preds)
            else:
                ref_acc = torch.tensor(1.0, dtype=torch.float, device=device)

            key_name = f'ref_acc_{id}' if id is not None else 'ref_acc'
            analysis_dict[key_name] = 100 * ref_acc

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
