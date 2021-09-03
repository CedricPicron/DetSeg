"""
Map-Based Detector (MBD) head.
"""
from collections import OrderedDict

import torch
from torch import nn

from models.modules.container import Sequential
from .sbd import SBD
from structures.boxes import get_box_deltas


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
        hrae = SBD.get_net(rae_dict)
        self.rae = Sequential(OrderedDict([('in', irae), ('hidden', hrae)]))

        # Initialization of absolute anchor encoding (AAE) network
        iaae = nn.Linear(4, aae_dict['in_size'])
        haae = SBD.get_net(aae_dict)
        self.aae = Sequential(OrderedDict([('in', iaae), ('hidden', haae)]))

        # Set CA-related attributes
        self.ca = SBD.get_net(ca_dict)
        self.ca_type = ca_dict['type']

        if self.ca_type == 'particle_attn':
            last_pa_layer = self.ca[-1].pa
            last_pa_layer.no_sample_locations_update()

        # Set metadata attribute
        metadata.stuff_classes = metadata.thing_classes
        metadata.stuff_colors = metadata.thing_colors
        self.metadata = metadata

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
                map_feats = torch.zeros(num_objs, num_map_feats, 2, device=device)

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

        # Return desired dictionaries
        if self.training:
            return loss_dict, analysis_dict
        elif tgt_dict is not None:
            return pred_dicts, loss_dict, analysis_dict
        else:
            return pred_dicts, analysis_dict
