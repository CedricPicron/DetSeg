"""
State-Based Detector (SBD) head.
"""
from collections import OrderedDict
from copy import deepcopy

import torch
from torch import nn

from models.functional.position import sine_pos_encodings
from models.modules.mlp import OneStepMLP, TwoStepMLP


class SBD(nn.Module):
    """
    Class implementing the State-Based Detector (SBD) head.

    Attributes:
        dod (DOD): Dense object discovery (DOD) module selecting promising features corresponding to objects.
        osi (nn.Sequential): Object state initialization (OSI) module computing the initial object states.
        cls (nn.Sequential): Classification (CLS) module computing classification predictions from object states.
        box (nn.Sequential): Bounding box (BOX) module computing bounding box predictions from object states.
    """

    def __init__(self, dod, osi_dict, cls_dict, box_dict):
        """
        Initializes the SBD module.

        Args:
            dod (DOD): Dense object discovery (DOD) module selecting promising features corresponding to objects.

            osi_dict (Dict): Object state initialization (OSI) network dictionary containing following keys:
                - type (str): string containing the type of OSI network;
                - layers (int): integer containing the number of OSI network layers;
                - in_size (int): input feature size of the OSI network;
                - hidden_size (int): hidden feature size of the OSI network;
                - out_size (int): output feature size of the OSI network;
                - norm (str): string containing the type of normalization of the OSI network;
                - act_fn (str): string containing the type of activation function of the OSI network;
                - skip (bool): boolean indicating whether layers of the OSI network contain skip connections.

            cls_dict (Dict): Classification (CLS) network dictionary containing following keys:
                - type (str): string containing the type of hidden CLS (HCLS) network;
                - layers (int): integer containing the number of HCLS network layers;
                - in_size (int): input feature size of the HCLS network;
                - hidden_size (int): hidden feature size of the HCLS network;
                - out_size (int): output feature size of the HCLS network;
                - norm (str): string containing the type of normalization of the HCLS network;
                - act_fn (str): string containing the type of activation function of the HCLS network;
                - skip (bool): boolean indicating whether layers of the HCLS network contain skip connections;
                - num_classes (int): integer containing the number of object classes (without background).

            box_dict (Dict): Bounding box (BOX) network dictionary containing following keys:
                - type (str): string containing the type of hidden BOX (HBOX) network;
                - layers (int): integer containing the number of HBOX network layers;
                - in_size (int): input feature size of the HBOX network;
                - hidden_size (int): hidden feature size of the HBOX network;
                - out_size (int): output feature size of the HBOX network;
                - norm (str): string containing the type of normalization of the HBOX network;
                - act_fn (str): string containing the type of activation function of the HBOX network;
                - skip (bool): boolean indicating whether layers of the HBOX network contain skip connections;
                - sigmoid (bool): boolean indicating whether to use sigmoid function at the end of the BOX network.
        """

        # Initialization of default nn.Module
        super().__init__()

        # Set DOD attribute
        self.dod = dod

        # Initialization of object state initialization (OSI) network
        self.osi = SBD.get_net(osi_dict)

        # Initialization of classification prediction (CLS) network
        num_classes = cls_dict.pop('num_classes')
        hcls = SBD.get_net(cls_dict)
        ocls = nn.Linear(cls_dict['out_size'], num_classes+1)
        self.cls = nn.Sequential(OrderedDict([('hidden', hcls), ('out', ocls)]))

        # Initialization of bounding box prediction (BOX) network
        sigmoid = box_dict.pop('sigmoid')
        hbox = SBD.get_net(box_dict)
        obox = nn.Linear(cls_dict['out_size'], 4)

        if sigmoid:
            self.box = nn.Sequential(OrderedDict([('hidden', hbox), ('out', obox), ('sigmoid', nn.Sigmoid())]))
        else:
            self.box = nn.Sequential(OrderedDict([('hidden', hbox), ('out', obox)]))

    @staticmethod
    def get_net(net_dict):
        """
        Get network from network dictionary.

        Args:
            net_dict (Dict): Network dictionary containing following keys:
                - type (str): string containing the type of network;
                - layers (int): integer containing the number of network layers;
                - in_size (int): input feature size of the network;
                - hidden_size (int): hidden feature size of the network;
                - out_size (int): output feature size of the network;
                - norm (str): string containing the type of normalization of the network;
                - act_fn (str): string containing the type of activation function of the etwork;
                - skip (bool): boolean indicating whether layers of the network contain skip connections.

        Returns:
            net (nn.Sequential): Module implementing the network specified by the given network dictionary.

        Raises:
            ValueError: Error when unsupported type of network is provided.
            ValueError: Error when the number of layers in non-positive.
        """

        if net_dict['type'] == 'one_step_mlp':
            net_args = (net_dict['in_size'], net_dict['out_size'])
            net_kwargs = {k: v for k, v in net_dict.items() if k in ('norm', 'act_fn', 'skip')}
            net_layer = OneStepMLP(*net_args, **net_kwargs)

        elif net_dict['type'] == 'two_step_mlp':
            net_args = (net_dict['in_size'], net_dict['hidden_size'], net_dict['out_size'])
            net_kwargs = {'norm1': net_dict['norm'], 'act_fn2': net_dict['act_fn'], 'skip': net_dict['skip']}
            net_layer = TwoStepMLP(*net_args, **net_kwargs)

        else:
            error_msg = f"The provided network type '{net_dict['type']}' is not supported."
            raise ValueError(error_msg)

        if net_dict['layers'] > 0:
            net = nn.Sequential(*[deepcopy(net_layer) for _ in range(net_dict['layers'])])
        else:
            error_msg = f"The number of network layers must be positive, but got {net_dict['layers']}."
            raise ValueError(error_msg)

        return net

    @staticmethod
    @torch.no_grad()
    def get_xyz(feat_maps):
        """
        Get (x, y, z) coordinates corresponding to features from the given feature maps.

        The method assumes the feature maps are sorted from high to low resolution.

        Args:
            feat_maps (List): List of size [num_maps] with feature maps of shape [batch_size, feat_size, fH, fW].

        Returns:
            feat_xyz (FloatTensor): The (x, y, z) feature coordinates of shape [num_feats, 3].
        """

        # Get (x, y) coordinates
        _, xy_maps = sine_pos_encodings(feat_maps, normalize=True)
        feat_xy = torch.cat([xy_map.flatten(1).t() for xy_map in xy_maps], dim=0)

        # Get z-coordinates
        num_maps = len(feat_maps)
        feat_z = torch.arange(num_maps).to(feat_xy)
        feat_z = feat_z / (num_maps-1)

        # Concatenate (x, y) coordinates with z-coordinates
        map_numel = torch.tensor([feat_map.flatten(2).shape[-1] for feat_map in feat_maps]).to(feat_xy.device)
        feat_z = torch.repeat_interleave(feat_z, map_numel, dim=0)
        feat_xyz = torch.cat([feat_xy, feat_z[:, None]], dim=1)

        return feat_xyz

    def forward(self, feat_maps, tgt_dict=None, images=None, visualize=False, **kwargs):
        """
        Forward method of the SBD head.

        Args:
            feat_maps (List): List of size [num_maps] with feature maps of shape [batch_size, feat_size, fH, fW].

            tgt_dict (Dict): Optional target dictionary used during trainval containing at least following keys:
                - labels (LongTensor): tensor of shape [num_targets_total] containing the class indices;
                - boxes (Boxes): structure containing axis-aligned bounding boxes of size [num_targets_total];
                - sizes (LongTensor): tensor of shape [batch_size+1] with the cumulative target sizes of batch entries.

            images (Images): Images structure containing the batched images (default=None).
            visualize (bool): Boolean indicating whether to compute dictionary with visualizations (default=False).
            kwargs (Dict): Dictionary of keyword arguments not used by this module, but passed to some sub-modules.

        Raises:
            NotImplementedError: Error when visualizations are requested.
        """

        # Check whether visualizations are requested
        if visualize:
            raise NotImplementedError

        # Get batch size and device
        batch_size = len(feat_maps[0])
        device = feat_maps[0].device

        # Sort feature maps from high to low resolution
        map_numel = torch.tensor([feat_map.flatten(2).shape[-1] for feat_map in feat_maps])
        sort_ids = torch.argsort(map_numel, descending=True).tolist()
        feat_maps = [feat_maps[i] for i in sort_ids]

        # Get features and corresponding (x, y, z) coordinates
        feats = torch.cat([feat_map.flatten(2).permute(0, 2, 1) for feat_map in feat_maps], dim=1)
        feat_xyz = SBD.get_xyz(feat_maps)

        # Apply DOD module and extract output
        dod_output = self.dod(feat_maps, tgt_dict=tgt_dict, images=images, stand_alone=False, **kwargs)

        if tgt_dict is None:
            sel_ids, analysis_dict = dod_output
        elif self.dod.tgt_mode == 'ext_dynamic':
            logits, obj_probs, sel_ids, pos_masks, neg_masks, tgt_sorted_ids = dod_output
        else:
            sel_ids, pos_masks, loss_dict, analysis_dict = dod_output

        # Get selected features with corresponding images indices and coordinates
        num_sel = torch.tensor([len(sel_ids_i) for sel_ids_i in sel_ids], device=device)
        img_ids = torch.arange(batch_size, device=device).repeat_interleave(num_sel)
        sel_feats = feats[img_ids, torch.cat(sel_ids, dim=0), :]
        obj_xyz = torch.cat([feat_xyz[sel_ids_i, :] for sel_ids_i in sel_ids], dim=0)

        # Get initial object states
        obj_states = self.osi(sel_feats)
        num_states = len(obj_states)
        analysis_dict['num_states_0'] = num_states / batch_size

        # Get initial predictions
        cls_preds = self.cls(obj_states)
        box_preds = self.box(obj_states)

        # Get initial loss
        if tgt_dict is not None:
            feat_wh = torch.tensor([[1/s for s in feat_map.shape[:1:-1]] for feat_map in feat_maps]).to(feat_xyz)
