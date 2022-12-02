"""
Collection of deprecated attention-based modules.
"""

import torch
from torch import nn
import torch.nn.functional as F

from models.build import MODELS


@MODELS.register_module()
class LegacySelfAttn1d(nn.Module):
    """
    Class implementing the LegacySelfAttn1d module.

    The module performs multi-head self-attention on sets of 1D features.

    Attributes:
        feat_size (int): Integer containing the feature size.
        num_heads (int): Number of attention heads.

        layer_norm (nn.LayerNorm): Layer normalization module before attention mechanism.
        in_proj_qk (nn.Linear): Linear input projection module projecting normalized features to queries and keys.
        in_proj_v (nn.Linear): Linear input projection module projecting normalized features to values.
        out_proj (nn.Linear): Linear output projection module projecting weighted values to delta output features.
    """

    def __init__(self, feat_size, num_heads):
        """
        Initializes the LegacySelfAttn1d module.

        Args:
            feat_size (int): Integer containing the feature size.
            num_heads (int): Number of attention heads.

        Raises:
            ValueError: Raised when the number of heads does not divide the feature size.
        """

        # Initialization of default nn.Module
        super().__init__()

        # Check inputs
        if feat_size % num_heads != 0:
            raise ValueError(f"The number of heads ({num_heads}) should divide the feature size ({feat_size}).")

        # Set non-learnable attributes
        self.feat_size = feat_size
        self.num_heads = num_heads

        # Initialize layer normalization module
        self.layer_norm = nn.LayerNorm(feat_size)

        # Initalize input and output projection modules
        self.in_proj_qk = nn.Linear(feat_size, 2*feat_size)
        self.in_proj_v = nn.Linear(feat_size, feat_size)
        self.out_proj = nn.Linear(feat_size, feat_size)

        # Set default initial values of module parameters
        self.reset_parameters()

    def reset_parameters(self):
        """
        Resets module parameters to default initial values.
        """

        nn.init.xavier_uniform_(self.in_proj_qk.weight)
        nn.init.xavier_uniform_(self.in_proj_v.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.in_proj_qk.bias)
        nn.init.zeros_(self.in_proj_v.bias)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, in_feat_list, pos_feat_list=None):
        """
        Forward method of the LegacySelfAttn1d module.

        Args:
            in_feat_list (List): List [num_feat_sets] containing input features of shape [num_features, feat_size].
            pos_feat_list (List): List [num_feat_sets] containing position features of shape [num_features, feat_size].

        Returns:
            out_feat_list (List): List [num_feat_sets] containing output features of shape [num_features, feat_size].
        """

        # Initialize empty output features list
        out_feat_list = []

        # Get zero position features if position features are missing
        if pos_feat_list is None:
            pos_feat_list = [torch.zeros_like(in_feats) for in_feats in in_feat_list]

        # Perform self-attention on every set of input features
        for in_feats, pos_feats in zip(in_feat_list, pos_feat_list):

            # Get normalized features
            norm_feats = self.layer_norm(in_feats)

            # Get position-enhanced normalized features
            pos_norm_feats = norm_feats + pos_feats

            # Get queries and keys
            f = self.feat_size
            head_size = f//self.num_heads

            qk_feats = self.in_proj_qk(pos_norm_feats)
            queries = qk_feats[:, :f].view(-1, self.num_heads, head_size).permute(1, 0, 2)
            keys = qk_feats[:, f:2*f].view(-1, self.num_heads, head_size).permute(1, 2, 0)

            # Get initial values
            values = self.in_proj_v(norm_feats)
            values = values.view(-1, self.num_heads, head_size).permute(1, 0, 2)

            # Get weighted values
            scale = float(head_size)**-0.5
            attn_weights = F.softmax(scale*torch.bmm(queries, keys), dim=2)
            weighted_values = torch.bmm(attn_weights, values)
            weighted_values = weighted_values.permute(1, 0, 2).reshape(-1, self.feat_size)

            # Get output features
            delta_feats = self.out_proj(weighted_values)
            out_feats = in_feats + delta_feats
            out_feat_list.append(out_feats)

        return out_feat_list
