"""
Base Reinforced Decoder (BRD) detection head.
"""
from copy import deepcopy

import torch
from torch import nn

from models.functional.position import sine_pos_encodings
from models.modules.attention import SelfAttn1d
from models.modules.mlp import FFN, MLP
from models.modules.policy import PolicyNet


class BRD(nn.Module):
    """
    Class implementing the Base Reinforced Decoder (BRD) module.

    Attributes:
        policy (PolicyNet): Policy network computing action masks and initial action losses.
        decoder (nn.Sequential): Sequence of decoder layers, with each layer having a self-attention and FFN operation.
        cls_head (MLP): Module computing the classification logits from object features.
        box_head (MLP): Module computing the bounding box logits from object features.
    """

    def __init__(self, feat_size, policy_dict, decoder_dict, head_dict):
        """
        Initializes the BRD module.

        Args:
            feat_size (int): Integer containing the feature size.

            policy_dict (Dict): Policy dictionary, potentially containing following keys:
                - num_hidden_layers (int): number of hidden layers of the policy head;
                - inference_samples (int): maximum number of samples during inference;
                - num_groups (int): number of groups used for group normalization.

            decoder_dict (Dict): Decoder dictionary containing following keys:
                - num_heads (int): number of attention heads used during the self-attention operation;
                - hidden_size (int): integer containing the hidden feature size used during the FFN operation;
                - num_layers (int): number of consecutive decoder layers.

            head_dict (Dict): Head dictionary containing following keys:
                - num_classes (int): integer containing the number of object classes (without background);
                - hidden_size (int): integer containing the hidden feature size used during the MLP operation;
                - num_hidden_layers (int): number of hidden layers of the MLP head.
        """

        # Initialization of default nn.Module
        super().__init__()

        # Initialize policy network
        self.policy = PolicyNet(feat_size, input_type='pyramid', **policy_dict)

        # Initialize decoder
        self_attn = SelfAttn1d(feat_size, decoder_dict['num_heads'])
        ffn = FFN(feat_size, decoder_dict['hidden_size'])
        decoder_layer = nn.Sequential(self_attn, ffn)

        num_decoder_layers = decoder_dict['num_layers']
        self.decoder = nn.Sequential(*[deepcopy(decoder_layer) for _ in range(num_decoder_layers)])

        # Initialize classification and bounding box heads
        num_classes = head_dict['num_classes']
        self.cls_head = MLP(feat_size, out_size=num_classes, **head_dict)
        self.box_head = MLP(feat_size, out_size=4, **head_dict)

    def forward(self, feat_maps, feat_masks=None, **kwargs):
        """
        Forward method of the BRD module.

        Args:
            feat_maps (List): List of size [num_maps] with feature maps of shape [batch_size, feat_size, fH, fW].
            feat_masks (List): Optional list [num_maps] with masks of active features of shape [batch_size, fH, fW].
        """

        # Assume no padded regions when feature masks are missing
        if feat_masks is None:
            tensor_kwargs = {'dtype': torch.bool, 'device': feat_maps[0].device}
            feat_masks = [torch.ones(*feat_map[:, 0].shape, **tensor_kwargs) for feat_map in feat_maps]

        # Get position-augmented features
        pos_maps = sine_pos_encodings((feat_maps, feat_masks), input_type='pyramid')
        aug_maps = [feat_map+pos_map for feat_map, pos_map in zip(feat_maps, pos_maps)]
        aug_feats = torch.cat([aug_map.flatten(2).permute(0, 2, 1) for aug_map in aug_maps], dim=1)

        # Apply policy network to obtain object features
        if self.training:
            sample_masks, action_losses = self.policy(feat_maps)
            obj_feats = [aug_feats[i][sample_masks[i]] for i in range(len(aug_feats))]

        else:
            sample_ids = self.policy(feat_maps)
            obj_feats = [aug_feats[i][sample_ids[i]] for i in range(aug_feats)]

        # Process object features with decoder
        obj_feats = self.decoder(obj_feats)

        # Get classification and bounding box logits
        cls_logits = self.cls_head(obj_feats)
        box_logits = self.box_head(obj_feats)

        return cls_logits, box_logits
