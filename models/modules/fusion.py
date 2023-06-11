"""
Collection of fusion modules.
"""

import torch
from torch import nn

from models.build import MODELS


@MODELS.register_module()
class AttnFusion(nn.Module):
    """
    Class implementing the AttnFusion module.

    Attributes:
        attn (nn.MultiheadAttention): Module performing the multi-head attention operation.
    """

    def __init__(self, **kwargs):
        """
        Initializes the AttnFusion module.

        Args:
            kwargs (Dict): Dictionary passed to the __init__ method of the underlying attention module.
        """

        # Initialization of default nn.Module
        super().__init__()

        # Get attention module
        kwargs.pop('batch_first', None)
        self.attn = nn.MultiheadAttention(batch_first=True, **kwargs)

    def forward(self, in_feats, feats_list, **kwargs):
        """
        Forward method of the AttnFusion module.

        Args:
            in_feats (FloatTensor): Input features of shape [num_groups, feat_size].
            feats_list (List): List of features to be fused of shape [num_groups, *, feat_size].
            kwargs (Dict): Dictionary of unused keyword arguments.

        Returns:
            out_feats (FloatTensor): Output features of shape [num_groups, feat_size].
        """

        # Get query features
        qry_feats = in_feats[:, None, :]

        # Get key-value features
        kv_feats_list = []

        for feats_i in feats_list:
            if feats_i.dim() == 2:
                kv_feats_list.append(feats_i[:, None, :])
            else:
                kv_feats_list.append(feats_i)

        kv_feats = torch.cat(kv_feats_list, dim=1)

        # Get output features
        out_feats = self.attn(qry_feats, kv_feats, kv_feats)[0]
        out_feats = out_feats.squeeze(dim=1)

        return out_feats


@MODELS.register_module()
class FusionInit(nn.Module):
    """
    Class implementing the FusionInit module.

    Attributes:
        init_type (str): String containing the fusion initialization type.
        init_id (int): Integer with the initialization index for the 'index' initialization type.
    """

    def __init__(self, init_type, init_id=None):
        """
        Initializes the FusionInit module.

        Args:
            init_type (str): String containing the fusion initialization type.
            init_id (int): Integer with the initialization index for the 'index' initialization type (default=None).
        """

        # Initialization of default nn.Module
        super().__init__()

        # Set attributes
        self.init_type = init_type
        self.init_id = init_id

    def forward(self, in_feats_list, **kwargs):
        """
        Forward method of the FusionInit module.

        Args:
            in_feats_list (List): List of input features to be fused of shape [num_groups, *, feat_size].
            kwargs (Dict): Dictionary of unused keyword arguments.

        Returns:
            init_feats (FloatTensor): Tensor containing the initial fusion features of shape [num_groups, feat_size].

        Raises:
            ValueError: Error when an invalid initialization type is provided.
        """

        # Get initial fusion features
        if self.init_type == 'index':
            init_feats = in_feats_list[self.init_id]

            if init_feats.dim() == 3:
                init_feats = init_feats.mean(dim=1)

        elif self.init_type == 'mean':
            init_feats_list = []

            for feats_i in in_feats_list:
                if feats_i.dim() == 2:
                    init_feats_list.append(feats_i[:, None, :])
                else:
                    init_feats_list.append(feats_i)

            init_feats = torch.cat(init_feats_list, dim=1)
            init_feats = init_feats.mean(dim=1)

        elif self.init_type == 'zero':
            num_groups = in_feats_list[0].size(dim=0)
            feat_size = in_feats_list[0].size(dim=-1)
            device = in_feats_list[0].device
            init_feats = torch.zeros(num_groups, feat_size, device=device)

        else:
            error_msg = f"Invalid initialization type in FusionInit module (got '{self.init_type}')."
            raise ValueError(error_msg)

        return init_feats
