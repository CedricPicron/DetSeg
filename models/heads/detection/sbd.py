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

        match_mode (str): String containing the prediction-target matching mode.
    """

    def __init__(self, dod, osi_dict, cls_dict, box_dict, match_dict):
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

            match_dict (Dict): Matching dictionary containing following keys:
                - mode (str): string containing the prediction-target matching mode.
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

        # Set matching attributes
        self.match_mode = match_dict['mode']

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

    def get_loss(self, cls_preds, box_preds, pred_xyz, feat_wh, pred_feat_ids=None, pos_masks=None):
        """
        Compute classification and bounding box losses from predictions.

        Args:
            cls_preds (List): List [batch_size] of classification predictions of shape [num_preds, num_classes+1].
            box_preds (List): List [batch_size] of bounding box predictions of shape [num_preds, 4].
            pred_xyz (List): List [batch_size] of prediction coordinates of shape [num_preds, 3].
            feat_wh (FloatTensor): Tensor containing the feature widths and heights of shape [num_maps, 2].
            pred_feat_ids (List): List [batch_size] with feature indices of shape [num_preds] (default=None).
            pos_masks (List): List [batch_size] of positive DOD masks of shape [num_feats, num_targets] (default=None).

        Returns:
            loss_dict (Dict): Loss dictionary containing following keys:
                - cls_loss (FloatTensor): tensor containing the weighted classification loss of shape [1];
                - box_loss (FloatTensor): tensor containing the weighted bounding box loss of shape [1].

            pos_pred_ids (List): List [batch_size] with indices of positive predictions of shape [num_pos_preds].
            tgt_found (List): List [batch_size] with masks of found targets of shape [num_targets].

        Raises:
            ValueError: Error when invalid prediction-target matching mode is provided.
        """

        # Get batch size
        batch_size = len(cls_preds)

        # Perform prediction-target matching
        pred_ids = []
        tgt_ids = []
        pos_pred_ids = []
        tgt_found = []

        if self.match_mode == 'dod_based':
            for i in range(batch_size):
                pred_pos_mask = pos_masks[i][pred_feat_ids[i]]
                pred_ids_i, tgt_ids_i = torch.nonzero(pred_pos_mask, as_tuple=True)
                pos_pred_ids_i = torch.unique_consecutive(pred_ids_i)

                num_tgts = pred_pos_mask.shape[1]
                tgt_found_i = torch.zeros(num_tgts, dtype=torch.bool, device=tgt_ids_i.device)
                tgt_found_i[tgt_ids_i] = True

                pred_ids.append(pred_ids_i)
                tgt_ids.append(tgt_ids_i)
                pos_pred_ids.append(pos_pred_ids_i)
                tgt_found.append(tgt_found_i)

        else:
            error_msg = f"Invalid prediction-target matching mode '{self.match_mode}'."
            raise ValueError(error_msg)

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

        # Get batch size
        batch_size = len(feat_maps[0])

        # Sort feature maps from high to low resolution
        map_numel = torch.tensor([feat_map.flatten(2).shape[-1] for feat_map in feat_maps])
        sort_ids = torch.argsort(map_numel, descending=True).tolist()
        feat_maps = [feat_maps[i] for i in sort_ids]

        # Get features and corresponding (x, y, z) coordinates
        feats = torch.cat([feat_map.flatten(2).permute(0, 2, 1) for feat_map in feat_maps], dim=1)
        feat_xyz = SBD.get_xyz(feat_maps)

        # Apply DOD module and extract output
        dod_kwargs = {'tgt_dict': tgt_dict, 'images': images, 'stand_alone': False}
        dod_output = self.dod(feat_maps, **dod_kwargs, **kwargs)

        if tgt_dict is None:
            sel_ids, analysis_dict = dod_output
        elif self.dod.tgt_mode == 'ext_dynamic':
            logits, obj_probs, sel_ids, pos_masks, neg_masks, tgt_sorted_ids, analysis_dict = dod_output
        else:
            sel_ids, pos_masks, dod_loss_dict, analysis_dict = dod_output

        # Get selected features and corresponding (x, y, z) coordinates
        sel_feats = [feats[i, sel_ids[i], :] for i in range(batch_size)]
        obj_xyz = [feat_xyz[sel_ids_i, :] for sel_ids_i in sel_ids]

        # Get initial object states
        obj_states = [self.osi(sel_feats_i) for sel_feats_i in sel_feats]
        num_states = sum(len(obj_states_i) for obj_states_i in obj_states)
        analysis_dict['num_states_0'] = num_states / batch_size

        # Get initial predictions
        cls_preds = [self.cls(obj_states_i) for obj_states_i in obj_states]
        box_preds = [self.box(obj_states_i) for obj_states_i in obj_states]

        # Get initial loss
        if tgt_dict is not None:

            # Get feature widths and heights
            feat_wh = torch.tensor([[1/s for s in feat_map.shape[:1:-1]] for feat_map in feat_maps]).to(feat_xyz)

            # Get initial classification and bounding box losses
            loss_kwargs = {'pred_feat_ids': sel_ids, 'pos_masks': pos_masks} if self.match_mode == 'dod_based' else {}
            loss_dict, pred_feat_ids, tgt_found = self.get_loss(cls_preds, box_preds, obj_xyz, feat_wh, **loss_kwargs)
            loss_dict = {f'{k}_0': v for k, v in loss_dict.items()}

            # Get DOD losses if in DOD external dynamic target mode
            if self.dod.tgt_mode == 'ext_dynamic':

                # Get external dictionary
                pos_feat_ids = [sel_ids[i][pred_feat_ids[i]] for i in range(batch_size)]
                ext_dict = {'logits': logits, 'obj_probs': obj_probs, 'pos_feat_ids': pos_feat_ids}
                ext_dict = {**ext_dict, 'tgt_found': tgt_found, 'pos_masks': pos_masks, 'neg_masks': neg_masks}
                ext_dict = {**ext_dict, 'tgt_sorted_ids': tgt_sorted_ids}

                # Get DOD loss and analysis dictionary
                dod_loss_dict, dod_analysis_dict = self.dod(feat_maps, ext_dict=ext_dict, **dod_kwargs, **kwargs)
                analysis_dict.update(dod_analysis_dict)

            # Add DOD losses to main loss dictionary
            loss_dict.update(dod_loss_dict)
