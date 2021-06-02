"""
State-Based Detector (SBD) head.
"""
from copy import deepcopy

import torch
from torch import nn

from models.functional.position import sine_pos_encodings
from models.modules.mlp import OneStepMLP, TwoStepMLP
from structures.boxes import Boxes, get_feat_boxes


class SBD(nn.Module):
    """
    Class implementing the State-Based Detector (SBD) head.

    Attributes:
        dod (DOD): Dense object discovery (DOD) module predicting promising features corresponding to objects.
        sel_mode (str): String containing the feature selection mode.
        sel_abs_thr (float): Absolute threshold determining the selected features.
        sel_rel_thr (int): Relative threshold determining the selected features.

        hsi (nn.Sequential): Hidden state initialization (HSI) module computing the initial hidden states.
    """

    def __init__(self, sel_dict, hsi_dict):
        """
        Initializes the SBD module.

        Args:
            sel_dict (Dict): Feature selection dictionary containing following keys:
                - dod (DOD): module predicting promising features corresponding to objects;
                - mode (str): string containing the feature selection mode;
                - abs_thr (float): absolute threshold determining the selected features;
                - rel_thr (int): relative threshold determining the selected features.

            hsi_dict (Dict): Hidden state initialization (HSI) dictionary containing following keys:
                - type (str): string containing the type of HSI network;
                - in_size (int): input feature size of the HSI network;
                - hidden_size (int): hidden feature size of the HSI network;
                - out_size (int): output feature size of the HSI network;
                - norm (str): string containing the type of normalization of the HSI network;
                - act_fn (str): string containing the type of activation function of the HSI network;
                - skip (bool): boolean indicating whether layers of the HSI network contain skip connections.

        Raises:
            ValueError: Error when unsupported type of HSI network is provided.
            ValueError: Error when the number of HSI layers in non-positive.
        """

        # Initialization of default nn.Module
        super().__init__()

        # Set feature selection attributes
        self.dod = sel_dict['dod']
        self.sel_mode = sel_dict['mode']
        self.sel_abs_thr = sel_dict['abs_thr']
        self.sel_rel_thr = sel_dict['rel_thr']

        # Initialization of hidden state initialization (HSI) network
        if hsi_dict['type'] == 'one_step_mlp':
            hsi_args = (hsi_dict['in_size'], hsi_dict['out_size'])
            hsi_kwargs = {k: v for k, v in hsi_dict.items() if k in ('norm', 'act_fn', 'skip')}
            hsi_layer = OneStepMLP(*hsi_args, **hsi_kwargs)

        elif hsi_dict['type'] == 'two_step_mlp':
            hsi_args = (hsi_dict['in_size'], hsi_dict['hidden_size'], hsi_dict['out_size'])
            hsi_kwargs = {'norm1': hsi_dict['norm'], 'act_fn2': hsi_dict['act_fn'], 'skip': hsi_dict['skip']}
            hsi_layer = TwoStepMLP(*hsi_args, **hsi_kwargs)

        else:
            error_msg = f"The provided HSI network type '{hsi_dict['type']}' is not supported."
            raise ValueError(error_msg)

        if hsi_dict['layers'] > 0:
            self.hsi = nn.Sequential(*[deepcopy(hsi_layer) for _ in range(hsi_dict['layers'])])
        else:
            error_msg = f"The number of HSI layers must be positive, but got {hsi_dict['layers']}."

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
            kwargs (Dict): Dictionary of keyword arguments not used by this head module.

        Raises:
            NotImplementedError: Error when visualizations are requested.
            ValueError: Error when invalid feature selection mode is provided.
        """

        # Check whether visualizations are requested
        if visualize:
            raise NotImplementedError

        # Get batch size
        batch_size = len(feat_maps[0])

        # Get position-augmented appearance (PAA) features
        pos_maps, _ = sine_pos_encodings(feat_maps)
        paa_maps = [feat_map+pos_map for feat_map, pos_map in zip(feat_maps, pos_maps)]
        paa_feats = torch.cat([paa_map.flatten(2).permute(0, 2, 1) for paa_map in paa_maps], dim=1)

        # Select promising features with corresponding initial box states
        dod_logits, obj_probs, analysis_dict = self.dod(feat_maps, mode='pred')
        feat_boxes = get_feat_boxes(feat_maps)

        if self.sel_mode == 'abs':
            sel_mask = obj_probs >= self.sel_abs_thr
            sel_feats = paa_feats[sel_mask]
            box_states = Boxes.cat([feat_boxes[sel_mask_i] for sel_mask_i in sel_mask])

        elif self.sel_mode == 'rel':
            sel_ids = torch.argsort(obj_probs, dim=1)[:, :self.sel_rel_thr]
            batch_ids = torch.arange(batch_size).to(sel_ids).repeat_interleave(self.sel_rel_thr)
            sel_feats = paa_feats[batch_ids, sel_ids.flatten(), :]
            box_states = Boxes.cat([feat_boxes[sel_ids_i] for sel_ids_i in sel_ids])

        else:
            error_msg = f"Invalid feature selection mode '{self.sel_mode}'."
            raise ValueError(error_msg)

        num_states = len(box_states)
        analysis_dict['num_states_0'] = num_states / batch_size

        # Get initial hidden and classification states
        hid_states = self.hsi(sel_feats)
        cls_states = torch.zeros(num_states, self.num_classes).to(hid_states)
