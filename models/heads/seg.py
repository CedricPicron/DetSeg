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

    def forward_pred(self, qry_feats, storage_dict, cum_feats_batch=None, images_dict=None, **kwargs):
        """
        Forward prediction method of the BaseSegHead module.

        Args:
            qry_feats (FloatTensor): Query features of shape [num_feats, qry_feat_size].

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
        num_feats = len(qry_feats)
        device = qry_feats.device

        # Get cumulative number of features per batch entry if missing
        if cum_feats_batch is None:
            cum_feats_batch = torch.tensor([0, num_feats], device=device)

        # Get query segmentation features
        qry_feats = self.qry(qry_feats)

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
        qry (nn.Module): Optional module updating the query features before concatenation with coarse key features.
        key_2d (nn.Module): Optional module updating the 2D key feature maps.

        coa_key (nn.Module): Optional module computing the coarse key features.
        pos_enc (nn.Module): Optional module computing position features added to the coarse key features.
        coa_in (nn.Module): Optional coarse input projection module updating the segmentation features.
        coa_conv (nn.Module): Optional coarse convolution module updating the segmentation features.
        coa_out (nn.Module): Optional coarse output projection module updating the segmentation features.

        seg (nn.Module): Module computing the segmentation logits from the segmentation features.
        ref (nn.Module): Module computing the refinement logits from the segmentation features.

        td (nn.Module): Optional module computing the top-down features from the segmentation features.
        fine_key (nn.Module): Optional module computing the fine key features.
        fine_core (nn.Module): Optional module updating the fine core features.
        fine_in (nn.Module): Optional fine input projection module updating the segmentation features.
        fine_conv (nn.Module): Optional fine convolution module updating the segmentation features.
        fine_out (nn.Module): Optional fine output projection module updating the segmentation features.

        map_offset (int): Integer with map offset used to determine the coarse key feature map for each query.
        key_min_id (int): Integer containing the downsampling index of the highest resolution key feature map.
        key_max_id (int): Integer containing the downsampling index of the lowest resolution key feature map.
        refine_iters (int): Integer containing the number of refinement iterations.
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
        ref_loss (nn.Module): Module computing the refinement loss.
    """

    def __init__(self, seg_cfg, ref_cfg, map_offset, key_min_id, key_max_id, refine_iters, refines_per_iter, mask_thr,
                 metadata, seg_loss_cfg, ref_loss_cfg, qry_cfg=None, key_2d_cfg=None, coa_key_cfg=None,
                 pos_enc_cfg=None, coa_in_cfg=None, coa_conv_cfg=None, coa_out_cfg=None, td_cfg=None,
                 fine_key_cfg=None, fine_core_cfg=None, fine_in_cfg=None, fine_conv_cfg=None, fine_out_cfg=None,
                 get_segs=True, dup_attrs=None, max_segs=None, matcher_cfg=None, **kwargs):
        """
        Initializes the TopDownSegHead module.

        Args:
            seg_cfg (Dict): Configuration dictionary specifying the segmentation module.
            ref_cfg (Dict): Configuration dictionary specifying the refinement module.
            map_offset (int): Integer with map offset used to determine the coarse key feature map for each query.
            key_min_id (int): Integer containing the downsampling index of the highest resolution key feature map.
            key_max_id (int): Integer containing the downsampling index of the lowest resolution key feature map.
            refine_iters (int): Integer containing the number of refinement iterations.
            refines_per_iter (int): Integer containing the number of refinements per refinement iteration.
            mask_thr (float): Value containing the mask threshold used to determine the segmentation masks.
            metadata (detectron2.data.Metadata): Metadata instance containing additional dataset information.
            seg_loss_cfg (Dict): Configuration dictionary specifying the segmentation loss module.
            ref_loss_cfg (Dict): Configuration dictionary specifying the refinement loss module.
            qry_cfg (Dict): Configuration dictionary specifying the query module (default=None).
            key_2d_cfg (Dict): Configuration dictionary specifying the key 2D module (default=None).
            coa_key_cfg (Dict): Configuration dictionary specifying the coarse key module (default=None.)
            pos_enc_cfg (Dict): Configuration dictionary specifying the position encoder module (default=None).
            coa_in_cfg (Dict): Configuration dictionary specifying the coarse input projection module (default=None).
            coa_conv_cfg (Dict): Configuration dictionary specifying the coarse convolution module (default=None).
            coa_out_cfg (Dict): Configuration dictionary specifying the coarse output projection module (default=None).
            td_cfg (Dict): Configuration dictionary specifying the top-down module (default=None).
            fine_key_cfg (Dict): Configuration dictionary specifying the fine key module (default=None).
            fine_core_cfg (Dict): Configuration dictionary specifying the fine core module (default=None).
            fine_in_cfg (Dict): Configuration dictionary specifying the fine input projection module (default=None).
            fine_conv_cfg (Dict): Configuration dictionary specifying the fine convolution module (default=None).
            fine_out_cfg (Dict): Configuration dictionary specifying the fine output projection module (default=None).
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

        self.coa_key = build_model(coa_key_cfg) if coa_key_cfg is not None else None
        self.pos_enc = build_model(pos_enc_cfg) if pos_enc_cfg is not None else None
        self.coa_in = build_model(coa_in_cfg) if coa_in_cfg is not None else None
        self.coa_conv = build_model(coa_conv_cfg) if coa_conv_cfg is not None else None
        self.coa_out = build_model(coa_out_cfg) if coa_out_cfg is not None else None

        self.seg = build_model(seg_cfg)
        self.ref = build_model(ref_cfg)

        self.td = build_model(td_cfg) if td_cfg is not None else None
        self.fine_key = build_model(fine_key_cfg) if fine_key_cfg is not None else None
        self.fine_core = build_model(fine_core_cfg) if fine_core_cfg is not None else None
        self.fine_in = build_model(fine_in_cfg) if fine_in_cfg is not None else None
        self.fine_conv = build_model(fine_conv_cfg) if fine_conv_cfg is not None else None
        self.fine_out = build_model(fine_out_cfg) if fine_out_cfg is not None else None

        # Build matcher module if needed
        self.matcher = build_model(matcher_cfg) if matcher_cfg is not None else None

        # Build segmentation and refinement loss modules
        self.seg_loss = build_model(seg_loss_cfg)
        self.ref_loss = build_model(ref_loss_cfg)

        # Set remaining attributes
        self.map_offset = map_offset
        self.key_min_id = key_min_id
        self.key_max_id = key_max_id
        self.refine_iters = refine_iters
        self.refines_per_iter = refines_per_iter
        self.get_segs = get_segs
        self.dup_attrs = dup_attrs
        self.max_segs = max_segs
        self.mask_thr = mask_thr
        self.metadata = metadata

    @torch.no_grad()
    def compute_segs(self, qry_feats, storage_dict, pred_dicts, cum_feats_batch=None, **kwargs):
        """
        Method computing the segmentation predictions.

        Args:
            qry_feats (FloatTensor): Query features of shape [num_feats, qry_feat_size].

            storage_dict (Dict): Storage dictionary containing at least following keys:
                - images (Images): images structure containing the batched images of size [batch_size];
                - cls_logits (FloatTensor): classification logits of shape [num_qrys, num_labels];
                - pred_boxes (Boxes): predicted 2D bounding boxes of size [num_feats].

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

        # Get image size, number of features, number of classes and device
        iW, iH = images.size()
        num_feats = cls_logits.size(dim=0)
        num_classes = cls_logits.size(dim=1) - 1
        device = cls_logits.device

        # Get prediction query indices,labels and scores
        pred_qry_ids = torch.arange(num_feats, device=device)[:, None].expand(-1, num_classes).reshape(-1)
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
            forward_pred_kwargs = {'cum_feats_batch': cum_feats_batch, 'seg_qry_ids': seg_qry_ids}
            self.forward_pred(qry_feats, storage_dict, **forward_pred_kwargs, **kwargs)

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

    def forward_pred(self, qry_feats, storage_dict, cum_feats_batch=None, images_dict=None, seg_qry_ids=None,
                     **kwargs):
        """
        Forward prediction method of the TopDownSegHead module.

        Args:
            qry_feats (FloatTensor): Query features of shape [num_feats, qry_feat_size].

            storage_dict (Dict): Storage dictionary containing at least following keys:
                - feat_maps (List): list of size [num_maps] with feature maps of shape [batch_size, feat_size, fH, fW];
                - images (Images): images structure containing the batched images of size [batch_size];
                - feat_ids (LongTensor): indices of selected features resulting in query features of shape [num_feats];
                - pred_boxes (Boxes): predicted 2D bounding boxes obtained from query features of size [num_feats].

            cum_feats_batch (LongTensor): Cumulative number of features per batch entry [batch_size+1] (default=None).
            images_dict (Dict): Dictionary with annotated images of predictions/targets (default=None).
            seg_qry_ids (Dict): Indices determining for which queries to compute segmetations (default=None).
            kwargs (Dict): Dictionary of keyword arguments not used by this module.

        Returns:
            storage_dict (Dict): Storage dictionary (possibly) containing following additional keys:
                - qry_ids (LongTensor): query indices (post-selection) of query-key pairs of shape [num_qry_key_pairs];
                - seg_xy (FloatTensor): normalized key locations of query-key pairs of shape [num_qry_key_pairs, 2];
                - batch_ids (LongTensor): batch indices of query-key pairs of shape [num_qry_key_pairs];
                - map_ids (LongTensor): map indices of query-key pairs of shape [num_qry_key_pairs];
                - map_shapes (List): list with map shapes in (height, width) format of size [self.key_max_id + 1];
                - seg_logits (FloatTensor): segmentation logits of query-key pairs of shape [num_qry_key_pairs];
                - ref_logits (FloatTensor): refinement logits of query-key pairs of shape [num_qry_key_pairs];
                - num_stage_preds (List): list with number of predictions per stage of size [num_stages];
                - refined_mask (BoolTensor): mask indicating refined query-key pairs of shape [num_qry_key_pairs].

            images_dict (Dict): Dictionary (possibly) containing additional images annotated with segmentations.
        """

        # Handle case where no segmentation queries are provided
        if seg_qry_ids is None:

            if self.training:
                return storage_dict, images_dict

            if self.get_segs:
                self.compute_segs(qry_feats, storage_dict=storage_dict, cum_feats_batch=cum_feats_batch, **kwargs)

            if images_dict is not None:
                self.draw_segs(storage_dict=storage_dict, images_dict=images_dict, **kwargs)

            return storage_dict, images_dict

        # Get number of features and device
        num_feats = len(qry_feats)
        device = qry_feats.device

        # Get cumulative number of features per batch entry if missing
        if cum_feats_batch is None:
            cum_feats_batch = torch.tensor([0, num_feats], device=device)

        # Get batch indices
        batch_size = len(cum_feats_batch) - 1
        batch_ids = torch.arange(batch_size, device=device)
        batch_ids = batch_ids.repeat_interleave(cum_feats_batch.diff())

        # Retrieve various items from storage dictionary
        key_feat_maps = storage_dict['feat_maps']
        images = storage_dict['images']
        feat_ids = storage_dict['feat_ids']
        qry_boxes = storage_dict['pred_boxes'].clone()

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

        # Get coarse segmentation and refinement logits
        qry_ids_list = []
        seg_xy_list = []
        seg_wh_list = []
        batch_ids_list = []
        map_ids_list = []
        seg_feats_list = []

        for i in range(batch_size):
            for j in range(num_maps):
                qry_ids_ij = torch.nonzero(batch_masks[i] & map_masks[j])[:, 0]

                if len(qry_ids_ij) > 0:
                    qry_boxes_ij = qry_boxes[qry_ids_ij].normalize(images[i])
                    qry_boxes_ij = qry_boxes_ij.to_format('xyxy')

                    key_feat_map = key_feat_maps[j][i]
                    kH, kW = key_feat_map.size()[-2:]

                    key_feat_map = F.pad(key_feat_map, (1, 1, 1, 1), mode='constant', value=0.0)
                    key_feats = key_feat_map.flatten(1).t().contiguous()

                    pts_x = torch.linspace(-0.5/kW, 1+0.5/kW, steps=kW, device=device)
                    pts_y = torch.linspace(-0.5/kH, 1+0.5/kH, steps=kH, device=device)

                    left_mask_0 = pts_x[None, None, :] > qry_boxes_ij.boxes[:, 0, None, None] - 1.5/kW
                    top_mask_0 = pts_y[None, :, None] > qry_boxes_ij.boxes[:, 1, None, None] - 1.5/kH
                    right_mask_0 = pts_x[None, None, :] < qry_boxes_ij.boxes[:, 2, None, None] + 1.5/kW
                    bot_mask_0 = pts_y[None, :, None] < qry_boxes_ij.boxes[:, 3, None, None] + 1.5/kH

                    left_mask_1 = pts_x[None, None, :] > qry_boxes_ij.boxes[:, 0, None, None] - 0.5/kW
                    top_mask_1 = pts_y[None, :, None] > qry_boxes_ij.boxes[:, 1, None, None] - 0.5/kH
                    right_mask_1 = pts_x[None, None, :] < qry_boxes_ij.boxes[:, 2, None, None] + 0.5/kW
                    bot_mask_1 = pts_y[None, :, None] < qry_boxes_ij.boxes[:, 3, None, None] + 0.5/kH

                    left_mask_2 = pts_x[None, None, :] > qry_boxes_ij.boxes[:, 0, None, None] + 0.5/kW
                    top_mask_2 = pts_y[None, :, None] > qry_boxes_ij.boxes[:, 1, None, None] + 0.5/kH
                    right_mask_2 = pts_x[None, None, :] < qry_boxes_ij.boxes[:, 2, None, None] - 0.5/kW
                    bot_mask_2 = pts_y[None, :, None] < qry_boxes_ij.boxes[:, 3, None, None] - 0.5/kH

                    key_mask = left_mask_0 & top_mask_0 & right_mask_0 & bot_mask_0
                    local_qry_ids, key_ids = torch.nonzero(key_mask.flatten(1), as_tuple=True)

                    core_mask = left_mask_1 & top_mask_1 & right_mask_1 & bot_mask_1
                    core_mask = core_mask[local_qry_ids, key_ids]
                    aux_mask = ~core_mask

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

                    # Get query indices
                    qry_ids_ij = qry_ids_ij[local_qry_ids]
                    qry_ids_list.append(qry_ids_ij)

                    # Get segmentation locations
                    seg_xy = torch.meshgrid(pts_x, pts_y, indexing='xy')
                    seg_xy = torch.stack(seg_xy, dim=0).flatten(1)
                    seg_xy = seg_xy[:, key_ids].t()
                    seg_xy_list.append(seg_xy)

                    # Get segmentation widths and heights
                    num_keys = len(key_ids)
                    seg_wh_i = torch.tensor([1/kW, 1/kH], device=device)
                    seg_wh_i = seg_wh_i[None, :].expand(num_keys, -1)
                    seg_wh_list.append(seg_wh_i)

                    # Get batch indices
                    batch_ids_i = torch.full(size=[num_keys], fill_value=i, device=device)
                    batch_ids_list.append(batch_ids_i)

                    # Get map indices
                    map_ids_i = torch.full(size=[num_keys], fill_value=j, device=device)
                    map_ids_list.append(map_ids_i)

                    # Get segmentation features
                    seg_feats = key_feats[key_ids, :]
                    seg_feats = self.coa_key(seg_feats) if self.coa_key is not None else seg_feats

                    if self.pos_enc is not None:
                        qry_boxes_ij = qry_boxes_ij[local_qry_ids].to_format('xywh').boxes
                        norm_seg_xy = (seg_xy - qry_boxes_ij[:, :2]) / qry_boxes_ij[:, 2:]
                        seg_feats = seg_feats + self.pos_enc(norm_seg_xy)

                    qry_feats_ij = qry_feats[qry_ids_ij]
                    seg_feats = torch.cat([qry_feats_ij, seg_feats], dim=1)
                    seg_feats = self.coa_in(seg_feats) if self.coa_in is not None else seg_feats

                    if self.coa_conv is not None:
                        seg_feats = self.coa_conv(seg_feats, core_mask, adj_ids, aux_mask=aux_mask)

                    seg_feats = self.coa_out(seg_feats) if self.coa_out is not None else seg_feats
                    seg_feats_list.append(seg_feats)

        qry_ids = torch.cat(qry_ids_list, dim=0)
        seg_xy = torch.cat(seg_xy_list, dim=0)
        seg_wh = torch.cat(seg_wh_list, dim=0)
        batch_ids = torch.cat(batch_ids_list, dim=0)
        map_ids = torch.cat(map_ids_list, dim=0) + self.key_min_id
        seg_feats = torch.cat(seg_feats_list, dim=0)

        core_seg_feats = seg_feats[core_mask]
        seg_logits = self.seg(core_seg_feats)
        ref_logits = self.ref(core_seg_feats)

        qry_ids_list = [qry_ids]
        seg_xy_list = [seg_xy]
        batch_ids_list = [batch_ids]
        map_ids_list = [map_ids]
        seg_logits_list = [seg_logits]
        ref_logits_list = [ref_logits]
        num_stage_preds = [len(qry_ids)]
        refined_mask_list = []

        # Get fine segmentation and refinement logits
        grid_offsets = torch.arange(2, device=device) - 0.5
        grid_offsets = torch.meshgrid(grid_offsets, grid_offsets, indexing='xy')
        grid_offsets = torch.stack(grid_offsets, dim=2).flatten(0, 1)

        cat_key_feats = [key_feat_map.flatten(2).permute(0, 2, 1) for key_feat_map in key_feat_maps]
        cat_key_feats = torch.cat(cat_key_feats, dim=1)

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

        for i in range(self.refine_iters):
            refine_mask = core_mask & (map_ids > 0)

            if refine_mask.sum().item() > self.refines_per_iter:
                refine_ids = torch.topk(ref_logits[refine_mask], self.refines_per_iter, sorted=False)[1]
                refine_ids = refine_mask.nonzero()[refine_ids, 0]

                refine_mask = torch.zeros_like(refine_mask)
                refine_mask[refine_ids] = True

            refined_mask_list.append(refine_mask)

            shifts = torch.where(refine_mask, 3, 0).cumsum(dim=0)
            adj_ids = adj_ids[refine_mask[core_mask]]
            adj_ids += shifts[adj_ids]

            num_refines = len(adj_ids)
            adj_ids = adj_ids.repeat_interleave(4, dim=0)

            adj_ids[:, 0] += None
            adj_ids[:, 5] += torch.arange(-3, 1, device=device).repeat_interleave(num_refines, dim=0)

            repeats = torch.where(refine_mask, 4, 1)
            core_mask = refine_mask.repeat_interleave(repeats, dim=0)
            aux_mask = ~core_mask

            qry_ids = qry_ids[refine_mask].repeat_interleave(4, dim=0)
            qry_ids_list.append(qry_ids)

            seg_wh = seg_wh[refine_mask] / 2
            seg_wh = seg_wh.repeat_interleave(4, dim=0)

            delta_seg_xy = grid_offsets[None, :, :] * seg_wh.view(-1, 4, 2)
            delta_seg_xy = delta_seg_xy.flatten(0, 1)

            seg_xy = seg_xy[refine_mask].repeat_interleave(4, dim=0)
            seg_xy = seg_xy + delta_seg_xy
            seg_xy_list.append(seg_xy)

            batch_ids = batch_ids[refine_mask]
            batch_ids = batch_ids.repeat_interleave(4, dim=0)
            batch_ids_list.append(batch_ids)

            map_ids = map_ids[refine_mask] - 1
            map_ids = map_ids.repeat_interleave(4, dim=0)
            map_ids_list.append(map_ids)

            map_sizes_i = map_sizes[map_ids]
            map_offsets_i = map_offsets[map_ids]

            feat_ids = torch.floor(seg_xy * map_sizes_i).int()
            feat_ids[:, 1] = feat_ids[:, 1] * map_sizes_i[:, 0]
            feat_ids = (map_offsets_i - delta_key_map_offset) + feat_ids.sum(dim=1)

            core_feats = seg_feats[refine_mask]
            core_feats = self.td(core_feats) if self.td is not None else core_feats.repeat_interleave(4, dim=0)

            select_mask = map_ids >= self.key_min_id
            key_feats = torch.zeros_like(core_feats)
            key_feats[select_mask] = cat_key_feats[batch_ids[select_mask], feat_ids[select_mask], :]
            key_feats = self.fine_key(key_feats) if self.fine_key is not None else key_feats

            core_feats = torch.cat([core_feats, key_feats], dim=1)
            core_feats = self.core_in(core_feats) if self.core_in is not None else core_feats

            seg_feats = self.fine_in(seg_feats) if self.fine_in is not None else seg_feats
            seg_feats[core_mask] += core_feats

            if self.fine_conv is not None:
                seg_feats = self.fine_conv(seg_feats, core_mask, adj_ids, aux_mask=aux_mask)

            seg_feats = self.fine_out(seg_feats) if self.fine_out is not None else seg_feats
            core_seg_feats = seg_feats[core_mask]

            seg_logits = self.seg(core_seg_feats)
            ref_logits = self.ref(core_seg_feats)

            seg_logits_list.append(seg_logits)
            ref_logits_list.append(ref_logits)
            num_stage_preds.append(len(qry_ids))

        refine_mask = torch.zeros_like(ref_logits, dtype=torch.bool)
        refined_mask_list.append(refine_mask)

        qry_ids = torch.cat(qry_ids_list, dim=0)
        seg_xy = torch.cat(seg_xy_list, dim=0)
        batch_ids = torch.cat(batch_ids_list, dim=0)
        map_ids = torch.cat(map_ids_list, dim=0)
        seg_logits = torch.cat(seg_logits_list, dim=0)
        ref_logits = torch.cat(ref_logits_list, dim=0)
        refined_mask = torch.cat(refined_mask_list, dim=0)

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

            for i in range(self.refine_iters+1):
                key_name = f'seg_loss_{id}_{i}' if id is not None else f'seg_loss_{i}'
                analysis_dict[key_name] = seg_loss.detach()

            key_name = f'seg_loss_{id}' if id is not None else 'seg_loss'
            loss_dict[key_name] = seg_loss

            # Get refinement loss
            ref_loss = 0.0 * qry_feats.sum()

            for i in range(self.refine_iters+1):
                key_name = f'ref_loss_{id}_{i}' if id is not None else f'ref_loss_{i}'
                analysis_dict[key_name] = ref_loss.detach()

            key_name = f'ref_loss_{id}' if id is not None else 'ref_loss'
            loss_dict[key_name] = ref_loss

            # Get segmentation and refinement accuracies
            with torch.no_grad():

                # Get segmentation accuracies
                seg_acc = 1.0 if len(tgt_dict['masks']) == 0 else 0.0
                seg_acc = torch.tensor(seg_acc, dtype=seg_loss.dtype, device=device)

                for i in range(self.refine_iters+1):
                    key_name = f'seg_acc_{id}_{i}' if id is not None else f'seg_acc_{i}'
                    analysis_dict[key_name] = 100 * seg_acc

                key_name = f'seg_acc_{id}' if id is not None else 'seg_acc'
                analysis_dict[key_name] = 100 * seg_acc

                # Get refinement accuracies
                ref_acc = 1.0 if len(tgt_dict['masks']) == 0 else 0.0
                ref_acc = torch.tensor(ref_acc, dtype=ref_loss.dtype, device=device)

                for i in range(self.refine_iters+1):
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
        inv_ids, counts = torch.unique(qry_ids[seg_mask], sorted=False, return_inverse=True, return_counts=True)[1:]
        seg_weights = (1/counts)[inv_ids]

        seg_masks = seg_mask.split(num_stage_preds)
        seg_num_stage_preds = [seg_mask.sum().item() for seg_mask in seg_masks]

        seg_logits_list = seg_logits.split(seg_num_stage_preds)
        seg_targets_list = seg_targets.split(seg_num_stage_preds)
        seg_weights_list = seg_weights.split(seg_num_stage_preds)

        seg_zip = zip(seg_logits_list, seg_targets_list, seg_weights_list)
        seg_loss = sum(0.0 * p.flatten()[0] for p in self.parameters())

        for i, (seg_logits_i, seg_targets_i, seg_weights_i) in enumerate(seg_zip):
            seg_loss_i = self.seg_loss(seg_logits_i, seg_targets_i, weight=seg_weights_i)
            seg_loss += seg_loss_i

            key_name = f'seg_loss_{id}_{i}' if id is not None else f'seg_loss_{i}'
            analysis_dict[key_name] = seg_loss_i.detach()

        key_name = f'seg_loss_{id}' if id is not None else 'seg_loss'
        loss_dict[key_name] = seg_loss

        # Get refinement loss
        inv_ids, counts = torch.unique(qry_ids, sorted=False, return_inverse=True, return_counts=True)[1:]
        ref_weights = (1/counts)[inv_ids]

        ref_logits_list = ref_logits.split(num_stage_preds)
        ref_targets_list = ref_targets.split(num_stage_preds)
        ref_weights_list = ref_weights.split(num_stage_preds)

        ref_zip = zip(ref_logits_list, ref_targets_list, ref_weights_list)
        ref_loss = sum(0.0 * p.flatten()[0] for p in self.parameters())

        for i, (ref_logits_i, ref_targets_i, ref_weights_i) in enumerate(ref_zip):
            ref_loss_i = self.ref_loss(ref_logits_i, ref_targets_i, weight=ref_weights_i)
            ref_loss += ref_loss_i

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
                seg_acc_i = (seg_preds_i == seg_targets_i).sum() / len(seg_preds_i)

                key_name = f'seg_acc_{id}_{i}' if id is not None else f'seg_acc_{i}'
                analysis_dict[key_name] = 100 * seg_acc_i

            seg_acc = (seg_preds == seg_targets).sum() / len(seg_preds)
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
