"""
Map-Based Detector (MBD) head.
"""
from collections import OrderedDict
from copy import deepcopy

from fvcore.nn import sigmoid_focal_loss
import torch
from torch import nn
import torch.nn.functional as F

from models.functional.loss import dice_loss
from models.functional.net import get_net
from models.modules.container import Sequential
from structures.boxes import box_iou, get_box_deltas, mask_to_box


class MBD (nn.Module):
    """
    Class implementing the Map-Based Detector (MBD) head.

    Attributes:
        sbd (SBD): State-based detector (SBD) module computing the object features.

        rae (Sequential): Relative anchor encoding (RAE) module computing encodings from anchor differences.
        aae (Sequential): Absolute anchor encoding (AAE) module computing encodings from normalized anchors.

        ca (Sequential): Cross-attention (CA) module sampling from feature maps with object features as context.
        ca_type (str): String containing the type of cross-attention used by the MBD head.

        match_thr (float): Threshold determining the minimum box IoU for positive matching.

        use_gt_seg (bool): Boolean indicating whether to use ground-truth segmentation masks during training.
        seg_types (List): List with strings containing the types of segmentation loss functions.
        seg_alpha (float): Alpha value used by the segmentation sigmoid focal loss.
        seg_gamma (float): Gamma value used by the segmentation sigmoid focal loss.
        seg_weights (List): List with factors weighting the different segmentation losses.

        pred_thr (float): Threshold determining the minimum probability for a positive pixel prediction.

        metadata (detectron2.data.Metadata): Metadata instance containing additional dataset information.
    """

    def __init__(self, sbd, rae_dict, aae_dict, ca_dict, match_dict, loss_dict, pred_dict, metadata):
        """
        Initializes the MBD module.

        Args:
            sbd (SBD): State-based detector (SBD) module computing the object features.

            rae_dict (Dict): Relative anchor encoding (RAE) network dictionary containing following keys:
                - type (str): string containing the type of hidden RAE (HRAE) network;
                - layers (int): integer containing the number of HRAE network layers;
                - in_size (int): input feature size of the HRAE network;
                - hidden_size (int): hidden feature size of the HRAE network;
                - out_size (int): output feature size of the HRAE network;
                - norm (str): string containing the type of normalization of the HRAE network;
                - act_fn (str): string containing the type of activation function of the HRAE network;
                - skip (bool): boolean indicating whether layers of the HRAE network contain skip connections.

            aae_dict (Dict): Absolute anchor encoding (AAE) network dictionary containing following keys:
                - type (str): string containing the type of hidden AAE (HAAE) network;
                - layers (int): integer containing the number of HAAE network layers;
                - in_size (int): input feature size of the HAAE network;
                - hidden_size (int): hidden feature size of the HAAE network;
                - out_size (int): output feature size of the HAAE network;
                - norm (str): string containing the type of normalization of the HAAE network;
                - act_fn (str): string containing the type of activation function of the HAAE network;
                - skip (bool): boolean indicating whether layers of the HAAE network contain skip connections.

            ca_dict: Cross-attention (CA) network dictionary containing following keys:
                - type (str): string containing the type of CA network;
                - layers (int): integer containing the number of CA network layers;
                - in_size (int): input feature size of the CA network;
                - sample_size (int): sample feature size of the CA network;
                - out_size (int): output feature size of the CA network;
                - norm (str): string containing the type of normalization of the CA network;
                - act_fn (str): string containing the type of activation function of the CA network;
                - skip (bool): boolean indicating whether layers of the CA network contain skip connections;
                - version (int): integer containing the version of the CA network;
                - num_heads (int): integer containing the number of attention heads of the CA network;
                - num_levels (int): integer containing the number of map levels for the CA network to sample from;
                - num_points (int): integer containing the number of points of the CA network;
                - rad_pts (int): integer containing the number of radial points of the CA network;
                - ang_pts (int): integer containing the number of angular points of the CA network;
                - lvl_pts (int): integer containing the number of level points of the CA network;
                - dup_pts (int): integer containing the number of duplicate points of the CA network;
                - qk_size (int): query and key feature size of the CA network;
                - val_size (int): value feature size of the CA network;
                - val_with_pos (bool): boolean indicating whether position info is added to CA value features;
                - norm_z (float): factor normalizing the CA sample offsets in the Z-direction;
                - step_size (float): size of the CA sample steps relative to the sample step normalization;
                - step_norm_xy (str): string containing the normalization type of CA sample steps in the XY-direction;
                - step_norm_z (float): value normalizing the CA sample steps in the Z-direction;
                - num_particles (int): integer containing the number of particles per CA head;
                - sample_insert (bool): boolean indicating whether to insert CA sample information in a maps structure;
                - insert_size (int): integer containing the size of features to be inserted during CA sample insertion.

            match_dict (Dict): Matching dictionary containing following key:
                - match_thr (float): threshold determining the minimum box IoU for positive matching.

            loss_dict (Dict): Loss dictionary containing following keys:
                - use_gt_seg (bool): boolean indicating whether to use ground-truth segmentation masks during training;
                - seg_types (List): list with strings containing the types of segmentation loss functions;
                - seg_alpha (float): alpha value used by the segmentation sigmoid focal loss;
                - seg_gamma (float): gamma value used by the segmentation sigmoid focal loss;
                - seg_weights (List): list with factors weighting the different segmentation losses.

            pred_dict (Dict): Prediction dictionary containing following key:
                - pred_thr (float): threshold determining the minimum probability for a positive pixel prediction.

        metadata (detectron2.data.Metadata): Metadata instance containing additional dataset information.
        """

        # Initialization of default nn.Module
        super().__init__()

        # Set SBD attribute
        self.sbd = sbd

        # Initialization of relative anchor encoding (RAE) network
        irae = nn.Linear(4, rae_dict['in_size'])
        hrae = get_net(rae_dict)
        self.rae = Sequential(OrderedDict([('in', irae), ('hidden', hrae)]))

        # Initialization of absolute anchor encoding (AAE) network
        iaae = nn.Linear(4, aae_dict['in_size'])
        haae = get_net(aae_dict)
        self.aae = Sequential(OrderedDict([('in', iaae), ('hidden', haae)]))

        # Set CA-related attributes
        self.ca = get_net(ca_dict)
        self.ca_type = ca_dict['type']

        if self.ca_type == 'deformable_attn':
            last_msda_layer = self.ca[-1].msda
            last_msda_layer.no_out_feats_computation()

        elif self.ca_type == 'particle_attn':
            last_pa_layer = self.ca[-1].pa
            last_pa_layer.no_out_feats_computation()
            last_pa_layer.no_sample_locations_update()

        # Set matching attributes
        for k, v in match_dict.items():
            setattr(self, k, v)

        # Set loss attributes
        for k, v in loss_dict.items():
            setattr(self, k, v)

        # Set prediction attributes
        for k, v in pred_dict.items():
            setattr(self, k, v)

        # Set metadata attribute
        metadata.stuff_classes = metadata.thing_classes
        metadata.stuff_colors = metadata.thing_colors
        self.metadata = metadata

    def get_loss(self, pred_maps, tgt_dict, sbd_boxes, images):
        """
        Compute segmentation loss from prediction maps and perform additional accuracy-related analyses.

        Args:
            pred_maps (FloatTensor): Maps with prediction logits of shape [num_objs_total, 1, sH, sW].

            tgt_dict (Dict): Target dictionary potentially containing following keys:
                - boxes (Boxes): structure containing axis-aligned bounding boxes of size [num_targets_total];
                - sizes (LongTensor): tensor of shape [batch_size+1] with the cumulative target sizes of batch entries;
                - masks (ByteTensor): optional padded segmentation masks of shape [num_targets_total, max_iH, max_iW].

            sbd_boxes (List): List [batch_size] of Boxes structures predicted by the SBD module of size [num_objs].
            images (Images): Images structure containing the batched images.

        Returns:
            loss_dict (Dict): Loss dictionary containing following key:
                - mbd_seg_loss (FloatTensor): tensor containing the segmentation loss of shape [1].

            analysis_dict (Dict): Analysis dictionary containing following keys:
                - mbd_box_acc (FloatTensor): tensor containing the box accuracy (in percentage) of shape [1];
                - mbd_seg_acc (FloatTensor): tensor containing the segmentation accuracy (in percentage) of shape [1].

        Raises:
            NotImplementedError: Error when target masks should be obtained without using ground-truth segmentations.
            ValueError: Error when the number of segmentation loss types and corresponding loss weights is different.
            ValueError: Error when invalid segmentation loss type is provided.
        """

        # Initialize loss and analysis dictionaries
        dtype = pred_maps.dtype
        device = pred_maps.device

        loss_dict = {k: torch.zeros(1, dtype=dtype, device=device) for k in ['mbd_seg_loss']}
        analysis_dict = {k: torch.zeros(1, dtype=dtype, device=device) for k in ['mbd_box_acc', 'mbd_seg_acc']}

        # Get number of targets
        num_tgts = len(tgt_dict['boxes'])

        # Handle case where there are no targets
        if num_tgts == 0:
            loss_dict['mbd_seg_loss'] += 0.0 * pred_maps.sum()
            analysis_dict['mbd_box_acc'] += 100.0
            analysis_dict['mbd_seg_acc'] += 100.0

            return loss_dict, analysis_dict

        # Get target boxes per image
        tgt_sizes = tgt_dict['sizes']
        tgt_boxes = [tgt_dict['boxes'][i0:i1] for i0, i1 in zip(tgt_sizes[:-1], tgt_sizes[1:])]

        # Perform prediction-target matching
        pred_offset = 0
        tgt_offset = 0

        pred_ids = []
        tgt_ids = []

        for sbd_boxes_i, tgt_boxes_i in zip(sbd_boxes, tgt_boxes):
            if len(tgt_boxes_i) == 0:
                continue

            iou_matrix = box_iou(sbd_boxes_i, tgt_boxes_i)
            max_ious, tgt_ids_i = torch.max(iou_matrix, dim=1)
            pred_mask = max_ious >= self.match_thr

            pred_ids_i = torch.arange(len(sbd_boxes_i), device=device)
            pred_ids_i = pred_ids_i[pred_mask] + pred_offset
            tgt_ids_i = tgt_ids_i[pred_mask] + tgt_offset

            pred_offset += len(sbd_boxes_i)
            tgt_offset += len(tgt_boxes_i)

            pred_ids.append(pred_ids_i)
            tgt_ids.append(tgt_ids_i)

        pred_ids = torch.cat(pred_ids, dim=0)
        tgt_ids = torch.cat(tgt_ids, dim=0)

        # Handle case where there are no matches
        if len(pred_ids) == 0:
            loss_dict['mbd_seg_loss'] += 0.0 * pred_maps.sum()

            return loss_dict, analysis_dict

        # Get target masks
        if self.use_gt_seg:
            tgt_masks = tgt_dict['masks']

        else:
            raise NotImplementedError

        # Match prediction maps with target masks
        pred_maps = pred_maps[pred_ids]
        tgt_masks = tgt_masks[tgt_ids]

        # Downsample target masks to size of prediction masps
        downsample_size = pred_maps.shape[-2:]
        tgt_maps = tgt_masks.unsqueeze(dim=1).to(dtype=torch.float)
        tgt_masks = F.interpolate(tgt_maps, size=downsample_size, mode='bilinear', align_corners=False) >= 0.5
        tgt_masks = tgt_masks.squeeze(dim=1)

        # Get prediction and target maps
        pred_maps = pred_maps.squeeze(dim=1)
        tgt_maps = tgt_masks.to(dtype=pred_maps.dtype)

        # Get segmentation loss
        if len(self.seg_types) != len(self.seg_weights):
            error_msg = "The number of segmentation loss types and corresponding loss weights must be equal."
            raise ValueError(error_msg)

        for seg_type, seg_weight in zip(self.seg_types, self.seg_weights):
            if seg_type == 'dice':
                seg_loss = dice_loss(pred_maps, tgt_maps, reduction='sum')
                loss_dict['mbd_seg_loss'] += seg_weight * seg_loss

            elif seg_type == 'sigmoid_focal':
                focal_kwargs = {'alpha': self.seg_alpha, 'gamma': self.seg_gamma, 'reduction': 'none'}
                seg_losses = sigmoid_focal_loss(pred_maps, tgt_maps, **focal_kwargs)

                seg_loss = seg_losses.flatten(start_dim=1).mean(dim=1).sum()
                loss_dict['mbd_seg_loss'] += seg_weight * seg_loss

            else:
                error_msg = f"Invalid segmentation loss type '{seg_type}'."
                raise ValueError(error_msg)

        # Get bounding box and segmenation accuracies
        with torch.no_grad():

            # Get prediction masks
            pred_masks = pred_maps.sigmoid() >= self.pred_thr

            # Get bounding box accuracy
            pred_boxes = mask_to_box(pred_masks)
            tgt_boxes = tgt_dict['boxes'].clone().normalize(images)
            tgt_boxes = tgt_boxes[tgt_ids]

            box_acc = box_iou(pred_boxes, tgt_boxes).diag().mean()
            analysis_dict['mbd_box_acc'] += 100.0 * box_acc

            # Get segmentation accuracy
            seg_acc = torch.eq(pred_masks, tgt_masks).sum() / pred_masks.numel()
            analysis_dict['mbd_seg_acc'] += 100.0 * seg_acc

        return loss_dict, analysis_dict

    def forward(self, feat_maps, tgt_dict=None, images=None, visualize=False, **kwargs):
        """
        Forward method of the MBD head.

        Args:
            feat_maps (List): List of size [num_maps] with feature maps of shape [batch_size, feat_size, fH, fW].

            tgt_dict (Dict): Target dictionary potentially containing following keys:
                - boxes (Boxes): structure containing axis-aligned bounding boxes of size [num_targets_total];
                - sizes (LongTensor): tensor of shape [batch_size+1] with the cumulative target sizes of batch entries;
                - masks (ByteTensor): optional padded segmentation masks of shape [num_targets_total, max_iH, max_iW].

            images (Images): Images structure containing the batched images (default=None).
            visualize (bool): Boolean indicating whether to compute dictionary with visualizations (default=False).
            kwargs (Dict): Dictionary of keyword arguments not used by this module, but passed to some sub-modules.

        Returns:
            * If MBD module in training mode (i.e. during training):
                loss_dict (Dict): Dictionary of different weighted loss terms used during training.
                analysis_dict (Dict): Dictionary of different analyses used for logging purposes only.

            * If MBD module not in training mode and tgt_dict is not None (i.e. during validation):
                pred_dicts (List): List with SBD and MBD prediction dictionaries.
                loss_dict (Dict): Dictionary of different weighted loss terms used during training.
                analysis_dict (Dict): Dictionary of different analyses used for logging purposes only.

            * If MBD module not in training mode and tgt_dict is None (i.e. during testing):
                pred_dicts (List): List with SBD and MBD prediction dictionaries.
                analysis_dict (Dict): Dictionary of different analyses used for logging purposes only.

        Raises:
            NotImplementedError: Error when visualizations are requested.
            ValueError: Error when no Images structure is provided.
        """

        # Check whether visualizations are requested
        if visualize:
            raise NotImplementedError

        # Check whether Images structure is provided
        if images is None:
            error_msg = "An Images structure containing the batched images must be provided."
            raise ValueError(error_msg)

        # Get batch size
        batch_size = len(feat_maps[0])

        # Apply SBD module and extract output
        sbd_output = self.sbd(feat_maps, tgt_dict, images, stand_alone=False, visualize=visualize, **kwargs)
        obj_feats, obj_anchors, sample_feats, sample_feat_ids = sbd_output[-4:]

        if self.training:
            loss_dict, sbd_analysis_dict, pred_dict = sbd_output[:3]
        elif tgt_dict is not None:
            pred_dicts, loss_dict, sbd_analysis_dict = sbd_output[:3]
        else:
            pred_dicts, sbd_analysis_dict = sbd_output[:2]

        # Get desired prediction, loss and analysis dictionaries
        if not self.training:
            pred_dict = pred_dicts[-1]

        analysis_dict = {f'sbd_{k}': v for k, v in sbd_analysis_dict.items() if 'dod_' not in k}
        analysis_dict.update({k: v for k, v in sbd_analysis_dict.items() if 'dod_' in k})

        # Get SBD boxes
        batch_ids = pred_dict['batch_ids']
        sbd_boxes = pred_dict['boxes']
        sbd_boxes = [sbd_boxes[batch_ids == i] for i in range(batch_size)]

        # Update object features by adding relative anchor encodings (RAE)
        box_deltas = [get_box_deltas(obj_anchors[i], sbd_boxes[i]) for i in range(batch_size)]
        obj_feats = [obj_feats[i] + self.rae(box_deltas[i]) for i in range(batch_size)]

        # Get normalized SBD boxes and corresponding absolute anchor encodings (AAE)
        norm_boxes = [sbd_boxes[i].clone().normalize(images[i]).to_format('cxcywh') for i in range(batch_size)]
        abs_anchor_encs = [self.aae(norm_boxes[i].boxes) for i in range(batch_size)]

        # Get keyword arguments for cross-attention layers
        ca_kwargs = [{} for _ in range(batch_size)]

        if self.ca_type in ('deformable_attn', 'particle_attn'):
            device = sample_feats.device
            num_map_feats = sample_feats.shape[1]

            sample_map_shapes = torch.tensor([feat_map.shape[-2:] for feat_map in feat_maps], device=device)
            sample_map_start_ids = sample_map_shapes.prod(dim=1).cumsum(dim=0)[:-1]
            sample_map_start_ids = torch.cat([sample_map_shapes.new_zeros((1,)), sample_map_start_ids], dim=0)

            for i in range(batch_size):
                num_objs = len(obj_feats[i])
                map_feats = torch.zeros(num_objs, num_map_feats, 1, device=device)

                ca_kwargs[i]['sample_priors'] = norm_boxes[i].boxes
                ca_kwargs[i]['sample_feats'] = sample_feats[i]
                ca_kwargs[i]['sample_map_shapes'] = sample_map_shapes
                ca_kwargs[i]['sample_map_start_ids'] = sample_map_start_ids
                ca_kwargs[i]['add_encs'] = abs_anchor_encs[i]
                ca_kwargs[i]['storage_dict'] = {'map_feats': map_feats}

                map_mask = (sample_feat_ids[i][:, None] - sample_map_start_ids) >= 0
                map_ids = map_mask.sum(dim=1) - 1
                ca_kwargs[i]['map_ids'] = map_ids

        # Apply cross-attention (CA) module
        [self.ca(obj_feats[i], **ca_kwargs[i]) for i in range(batch_size)]

        # Concatenate map features across batch entries
        map_feats = torch.cat([ca_kwargs[i]['storage_dict']['map_feats'] for i in range(batch_size)], dim=0)

        # Get multi-scale prediction maps
        map_sizes = sample_map_shapes.prod(dim=1).tolist()
        ms_pred_maps = map_feats.transpose(1, 2).split(map_sizes, dim=2)

        # Get single-scale prediction maps
        num_levels = len(sample_map_shapes)
        pred_maps = ms_pred_maps[-1].view(-1, 1, *sample_map_shapes[-1])
        kernel = torch.tensor([[[[1/16, 1/8, 1/16], [1/8, 1/4, 1/8], [1/16, 1/8, 1/16]]]], device=device)

        for i in range(num_levels-2, -1, -1):
            output_padding = ((sample_map_shapes[i] + 1) % 2).tolist()
            pred_maps = F.conv_transpose2d(pred_maps, kernel, stride=2, padding=1, output_padding=output_padding)
            pred_maps = pred_maps + ms_pred_maps[i].view(-1, 1, *sample_map_shapes[i])

        # Get MBD losses and corresponding analyses if desired
        if tgt_dict is not None:
            sbd_boxes = [sbd_boxes[i].to_img_scale(images[i]) for i in range(batch_size)]
            local_loss_dict, local_analysis_dict = self.get_loss(pred_maps, tgt_dict, sbd_boxes, images)

            loss_dict.update(local_loss_dict)
            analysis_dict.update(local_analysis_dict)

        # Get and append prediction dictionary if desired
        if not self.training:

            # Get prediction boxes
            pred_masks = pred_maps.squeeze(dim=1).sigmoid() >= self.pred_thr
            boxes_per_img = pred_dicts[-1]['boxes'].boxes_per_img
            pred_boxes = mask_to_box(pred_masks, boxes_per_img=boxes_per_img)

            # Get prediction dictionary and append to SBD prediction dictionaries
            pred_dict = deepcopy(pred_dicts[-1])
            pred_dict['boxes'] = pred_boxes
            pred_dicts.append(pred_dict)

        # Return desired dictionaries
        if self.training:
            return loss_dict, analysis_dict
        elif tgt_dict is not None:
            return pred_dicts, loss_dict, analysis_dict
        else:
            return pred_dicts, analysis_dict
