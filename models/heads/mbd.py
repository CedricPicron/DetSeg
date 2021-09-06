"""
Map-Based Detector (MBD) head.
"""
from collections import OrderedDict

from fvcore.nn import sigmoid_focal_loss
import torch
from torch import nn
import torch.nn.functional as F

from models.functional.loss import dice_loss
from models.functional.net import get_net
from models.modules.container import Sequential
from structures.boxes import box_iou, get_box_deltas


class MBD (nn.Module):
    """
    Class implementing the Map-Based Detector (MBD) head.

    Attributes:
        sbd (SBD): State-based detector (SBD) module computing the object features.
        train_sbd (bool): Boolean indicating whether underlying SBD module should be trained.

        rae (Sequential): Relative anchor encoding (RAE) module computing encodings from anchor differences.
        aae (Sequential): Absolute anchor encoding (AAE) module computing encodings from normalized anchors.

        ca (Sequential): Cross-attention (CA) module sampling from feature maps with object features as context.
        ca_type (str): String containing the type of cross-attention used by the MBD head.

        metadata (detectron2.data.Metadata): Metadata instance containing additional dataset information.
    """

    def __init__(self, sbd_dict, rae_dict, aae_dict, ca_dict, metadata):
        """
        Initializes the MBD module.

        Args:
            sbd_dict (Dict): State-based detector (SBD) dictionary containing following keys:
                - sbd (SBD): state-based detector (SBD) module computing the object features;
                - train_sbd (bool): boolean indicating whether underlying SBD module should be trained.

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
                - qk_size (int): query and key feature size of the CA network;
                - val_size (int): value feature size of the CA network;
                - val_with_pos (bool): boolean indicating whether position info is added to CA value features;
                - step_size (float): size of the CA sample steps relative to the sample step normalization;
                - step_norm_xy (str): string containing the normalization type of CA sample steps in the XY-direction;
                - step_norm_z (float): value normalizing the CA sample steps in the Z-direction;
                - num_particles (int): integer containing the number of particles per CA head;
                - sample_insert (bool): boolean indicating whether to insert CA sample information in a maps structure;
                - insert_size (int): integer containing the size of features to be inserted during CA sample insertion.

        metadata (detectron2.data.Metadata): Metadata instance containing additional dataset information.
        """

        # Initialization of default nn.Module
        super().__init__()

        # Set SBD-related attributes
        self.sbd = sbd_dict['sbd']
        self.train_sbd = sbd_dict['train_sbd']
        self.sbd.requires_grad_(self.train_sbd)

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

        # Set metadata attribute
        metadata.stuff_classes = metadata.thing_classes
        metadata.stuff_colors = metadata.thing_colors
        self.metadata = metadata

    def get_loss(self, pred_maps, tgt_dict, sbd_boxes):
        """
        Compute segmentation loss from prediction maps and perform additional accuracy-related analyses.

        Args:
            pred_maps (FloatTensor): Maps with prediction logits of shape [num_objs_total, 1, sH, sW].

            tgt_dict (Dict): Target dictionary potentially containing following keys:
                - boxes (List): list of size [batch_size] with Boxes structures of size [num_targets];
                - masks (ByteTensor): optional padded segmentation masks of shape [num_targets_total, max_iH, max_iW].

            sbd_boxes (List): List [batch_size] of Boxes structures predicted by the SBD module of size [num_objs].

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

        # Perform prediction-target matching
        pred_ids = []
        tgt_ids = []
        offset = 0

        for sbd_boxes_i, tgt_boxes_i in zip(sbd_boxes, tgt_dict['boxes']):
            pred_ids_i = torch.arange(len(sbd_boxes_i), device=device)

            iou_matrix = box_iou(sbd_boxes_i, tgt_boxes_i)
            max_ious, tgt_ids_i = torch.max(iou_matrix, dim=1)

            pred_mask = max_ious >= self.match.thr
            pred_ids_i = pred_ids_i[pred_mask]
            tgt_ids_i = tgt_ids_i[pred_mask] + offset

            pred_ids.append(pred_ids_i)
            tgt_ids.append(tgt_ids_i)
            offset += len(tgt_boxes_i)

        pred_ids = torch.cat(pred_ids, dim=0)
        tgt_ids = torch.cat(tgt_ids, dim=0)

        # Handle case where there are no matches
        if len(pred_ids) == 0:
            loss_dict['mbd_seg_loss'] = 0.0 * pred_maps.sum()
            loss_dict['mbd_box_acc'] = 100.0
            loss_dict['mbd_seg_acc'] = 100.0

        # Get target masks
        if self.use_gt_seg:
            tgt_masks = tgt_dict['masks']

        else:
            raise NotImplementedError

        # Match prediction maps with target masks
        pred_maps = pred_maps[pred_ids]
        tgt_masks = tgt_masks[tgt_ids].to(dtype=pred_maps.dtype)

        # Upsample prediction maps to size of target masks
        upsample_size = tgt_masks.shape[-2:]
        pred_maps = F.interpolate(pred_maps, size=upsample_size, mode='bilinear', align_corners=False)
        pred_maps = pred_maps.squeeze(dim=1)

        # Get segmentation loss
        if len(self.seg_types) != len(self.seg_weights):
            error_msg = "The number of segmentation loss types and corresponding loss weights must be equal."
            raise ValueError(error_msg)

        for seg_type, seg_weight in zip(self.seg_types, self.seg_weights):
            if seg_type == 'dice':
                seg_loss = dice_loss(pred_maps, tgt_masks, reduction='sum')
                loss_dict['mbd_seg_loss'] += seg_weight * seg_loss

            elif seg_type == 'sigmoid_focal':
                focal_kwargs = {'alpha': self.seg_alpha, 'gamma': self.seg_gamma, 'reduction': 'none'}
                seg_losses = sigmoid_focal_loss(pred_maps, tgt_masks, **focal_kwargs)

                seg_loss = seg_losses.flatten(start_dim=1).mean(dim=1).sum()
                loss_dict['mbd_seg_loss'] += seg_weight * seg_loss

            else:
                error_msg = f"Invalid segmentation loss type '{seg_type}'."
                raise ValueError(error_msg)

        return loss_dict, analysis_dict

    def forward(self, feat_maps, tgt_dict=None, images=None, visualize=False, **kwargs):
        """
        Forward method of the MBD head.

        Args:
            feat_maps (List): List of size [num_maps] with feature maps of shape [batch_size, feat_size, fH, fW].

            tgt_dict (Dict): Optional target dictionary used during trainval containing at least following keys:
                - labels (LongTensor): tensor of shape [num_targets_total] containing the class indices;
                - boxes (Boxes): structure containing axis-aligned bounding boxes of size [num_targets_total];
                - sizes (LongTensor): tensor of shape [batch_size+1] with the cumulative target sizes of batch entries.

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

        elif not self.train_sbd:
            loss_dict = {}

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
                ca_kwargs[i]['storage_dict'] = {'map_feats': map_feats}
                ca_kwargs[i]['add_encs'] = abs_anchor_encs[i]

                if self.ca_type == 'particle_attn':
                    ca_kwargs[i]['sample_feat_ids'] = sample_feat_ids[i]

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
            local_loss_dict, local_analysis_dict = self.get_loss(pred_maps, tgt_dict, sbd_boxes)

            loss_dict.update(local_loss_dict)
            analysis_dict.update(local_analysis_dict)

        # Return desired dictionaries
        if self.training:
            return loss_dict, analysis_dict
        elif tgt_dict is not None:
            return pred_dicts, loss_dict, analysis_dict
        else:
            return pred_dicts, analysis_dict
