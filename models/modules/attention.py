"""
Collection of attention-based modules.
"""

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class Attn2d(nn.Module):
    """
    Class implementing the Attn2d module.

    The module performs multi-head attention on 2D feature maps.

    Attributes:
        in_channels (Tuple): Number of channels of input maps used for query and key/value computation respectively.
        out_channels (int): Number of channels of output map (i.e. number of channels of value map).
        kernel_size (Tuple): Size of the attention region in (height, width) format.
        stride (Tuple): Stride of the attention region in (height, width) format.
        padding (Tuple): Key/value map padding in (height, width) format.
        dilation (Tuple): Dilation of the attention region in (height, width) format.
        num_heads (int): Number of attention heads.
        padding_mode (str): Padding mode from {'constant', 'reflect', 'replicate', 'circular'}.
        attn_mode (str): Attention mode from {'self', 'cross'}.
        pos_attn (bool): Whehter or not local position features are learned.
        q_stride (Tuple): Stride of the query elements in (height, width) format.
        qk_channels (int): Number of channels of query and key maps.
        qk_norm (callable): Query-key normalization function.

        If attn_mode is 'self':
            proj (nn.Conv2d): Module projecting input map to initial query and key/value maps during self-attention.

        If attn_mode is 'cross':
            q_proj (nn.Conv2d): Module projecting first input map to initial query map during cross-attention.
            kv_proj (nn.Conv2d): Module projecting second input map to initial key/value maps during cross-attention.

        If pos_attn is True:
            pos_feats (Parameter): Learned local position features added to the keys before query/key comparison.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, num_heads=1, bias=True,
                 padding_mode='zeros', attn_mode='self', pos_attn=True, q_stride=1, qk_channels=None,
                 qk_norm='softmax'):
        """
        Initializes the Attn2d module.

        Args:
            in_channels (int or List): Number of channels of input maps.
            out_channels (int): Number of channels of output map (i.e. number of channels of value map).
            kernel_size (int or List): Size of the attention region in (height, width) format.
            stride (int or List): Stride of the attention region in (height, width) format (default=1).
            padding (int or List): Key/value map padding in (height, width) format (default=0).
            dilation (int or List): Dilation of the attention region in (height, width) format (default=1).
            num_heads (int): Number of attention heads (default=1).
            bias (bool): Whether or not input projections contain learnable biases (default=True).
            padding_mode (str): Padding mode from {'zeros', 'reflect', 'replicate', 'circular'} (default='zeros').
            attn_mode (str): Attention mode from {'self', 'cross'} (default='self').
            pos_attn (bool): Whehter or not local position features are learned (default=True).
            q_stride (int or List): Stride of the query elements in (height, width) format (default=1).
            qk_channels (int or None): Number of channels of query and key maps (default=None).
            qk_norm (str): Query-key normalization function name from {'softmax', 'sigmoid'} (default='softmax').

        Raises:
            ValueError: Error when invalid attention mode is provided.
            ValueError: Error when two different number of input channels are given in 'self' attention mode.
            ValueError: Error when the number of heads does not divide the number of output channels.
            ValueError: Error when the number of heads does not divide the number of query-key channels.
        """

        # Initialization of default nn.Module
        super().__init__()

        # Check inputs (1)
        if attn_mode not in ['self', 'cross']:
            raise ValueError(f"Attention mode must be 'self' or 'cross' (got {attn_mode}).")

        if attn_mode == 'self' and isinstance(in_channels, tuple):
            if in_channels[0] != in_channels[1]:
                raise ValueError(f"In-channels must be equal in 'self' attention mode (got {in_channels}).")

        if qk_norm not in ['softmax', 'sigmoid']:
            raise ValueError(f"Query-key normalization function name must be 'softmax' or 'sigmoid' (got {qk_norm}).")

        # Set non-learnable attributes
        self.in_channels = (in_channels, in_channels) if isinstance(in_channels, int) else tuple(in_channels)
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else tuple(kernel_size)
        self.stride = (stride, stride) if isinstance(stride, int) else tuple(stride)
        self.padding = (padding, padding) if isinstance(padding, int) else tuple(padding)
        self.dilation = (dilation, dilation) if isinstance(dilation, int) else tuple(dilation)
        self.num_heads = num_heads
        self.padding_mode = 'constant' if padding_mode == 'zeros' else padding_mode
        self.attn_mode = attn_mode
        self.pos_attn = pos_attn
        self.q_stride = (q_stride, q_stride) if isinstance(q_stride, int) else tuple(q_stride)
        self.qk_channels = qk_channels if qk_channels is not None else out_channels

        if qk_norm == 'softmax':
            def softmax(x): return F.softmax(x, dim=-1)
            self.qk_norm = softmax

        elif qk_norm == 'sigmoid':
            self.qk_norm = torch.sigmoid

        # Check inputs (2)
        if self.out_channels % self.num_heads != 0:
            msg = f"The number of heads ({num_heads}) should divide the number of output channels ({out_channels})."
            raise ValueError(msg)

        if self.qk_channels % self.num_heads != 0:
            msg = f"The number of heads ({num_heads}) should divide the number of qk-channels ({self.qk_channels})."
            raise ValueError(msg)

        # Initialize linear projection modules
        if attn_mode == 'self':
            self.proj = nn.Conv2d(self.in_channels[0], 2*self.qk_channels+self.out_channels, kernel_size=1, bias=bias)
        else:
            self.q_proj = nn.Conv2d(self.in_channels[0], self.qk_channels, kernel_size=1, bias=bias)
            self.kv_proj = nn.Conv2d(self.in_channels[1], self.qk_channels+self.out_channels, kernel_size=1, bias=bias)

        # Initialize local position features if requested
        if pos_attn:
            self.pos_feats = Parameter(torch.empty(self.qk_channels, self.kernel_size[0]*self.kernel_size[1]))

        # Set default initial values of module parameters
        self.reset_parameters()

    def reset_parameters(self):
        """
        Resets module parameters to default initial values.
        """

        for name, parameter in self.named_parameters():
            if 'proj' in name:
                if 'weight' in name:
                    nn.init.xavier_uniform_(parameter)
                elif 'bias' in name:
                    nn.init.zeros_(parameter)

            elif 'pos_feats' in name:
                nn.init.zeros_(parameter)

    def forward(self, *in_feat_maps, **kwargs):
        """
        Forward method of the Attn2d module.

        Args:
            If attn_mode is 'self':
                in_feat_maps[0]: Input feature map of shape [batch_size, in_channels[0], fH, fW].

            If attn_mode is 'cross':
                in_feat_maps[0]: Query input feature map of shape [batch_size, in_channels[0], fH, fW].
                in_feat_maps[1]: Key/value input feature map of shape [batch_size, in_channels[1], fH, fW].

        Returns:
            out_feat_map (FloatTensor): Output feature map of shape [batch_size, out_channels, fH, fW].
        """

        # Get initial query, key and value maps
        q_channels = self.qk_channels
        q_sH, q_sW = self.q_stride

        if self.attn_mode == 'self':
            qkv_map = self.proj(in_feat_maps[0])
            query_map = qkv_map[:, :q_channels, ::q_sH, ::q_sW]
            kv_map = qkv_map[:, q_channels:, :, :]

        else:
            query_map = self.q_proj(in_feat_maps[0][:, :, ::q_sH, ::q_sW])
            kv_map = self.kv_proj(in_feat_maps[1])

        # Process query map
        scale = float(self.qk_channels//self.num_heads)**-0.25
        query_map = scale*query_map.permute(0, 2, 3, 1)
        query_map = query_map.view(*query_map.shape[:3], self.num_heads, 1, -1)

        # Process key/value map
        kv_size = kv_map.shape[-2:]

        if sum(self.padding) > 0:
            padding_size = (self.padding[1], self.padding[1], self.padding[0], self.padding[0])
            kv_map = F.pad(kv_map, padding_size, mode=self.padding_mode)

        kv_map = kv_map.permute(0, 2, 3, 1)
        sizes = [(size+attn_stride-1)//attn_stride for size, attn_stride in zip(kv_size, self.stride)]
        center_strides = [stride*attn_stride for stride, attn_stride in zip(kv_map.stride()[1:3], self.stride)]
        points_strides = [stride*dilation for stride, dilation in zip(kv_map.stride()[1:3], self.dilation)]

        kv_channels = self.qk_channels + self.out_channels
        sizes = [kv_map.shape[0], *sizes, kv_channels, *self.kernel_size]
        strides = [kv_map.stride()[0], *center_strides, kv_map.stride()[3], *points_strides]
        kv_map = kv_map.as_strided(sizes, strides).reshape(*sizes[:-2], -1)

        # Process key map
        key_map = kv_map[:, :, :, :self.qk_channels, :]
        key_map = torch.add(self.pos_feats, key_map, alpha=scale) if self.pos_attn else scale*key_map
        key_map = key_map.view(*key_map.shape[:3], self.num_heads, self.qk_channels//self.num_heads, -1)

        # Process value map
        value_map = kv_map[:, :, :, self.qk_channels:, :]
        value_map = value_map.view(*value_map.shape[:3], self.num_heads, self.out_channels//self.num_heads, -1)

        # Get output feature map
        attn_weights = self.qk_norm(torch.matmul(query_map, key_map))
        out_feat_map = torch.sum(attn_weights * value_map, dim=-1).view(*value_map.shape[:3], -1)
        out_feat_map = out_feat_map.permute(0, 3, 1, 2)

        return out_feat_map


class SelfAttn1d(nn.Module):
    """
    Class implementing the SelfAttn1d module.

    The module performs multi-head self-attention on sets of 1D features.

    Attributes:
        feat_size (int): Integer containing the feature size.
        num_heads (int): Number of attention heads.

        layer_norm (nn.LayerNorm): Layer normalization module before attention mechanism.
        in_proj (nn.Linear): Linear input projection module projecting normalized features to queries, keys and values.
        out_proj (nn.Linear): Linear output projection module projecting weighted values to delta output features.
    """

    def __init__(self, feat_size, num_heads):
        """
        Initializes the SelfAttn1d module.

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
        self.in_proj = nn.Linear(feat_size, 3*feat_size)
        self.out_proj = nn.Linear(feat_size, feat_size)

        # Set default initial values of module parameters
        self.reset_parameters()

    def reset_parameters(self):
        """
        Resets module parameters to default initial values.
        """

        nn.init.xavier_uniform_(self.in_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.in_proj.bias)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, in_feat_list):
        """
        Forward method of the SelfAttn1d module.

        Args:
            in_feat_list (List): List [num_feat_sets] containing input features of shape [num_features, feat_size].

        Returns:
            out_feat_list (List): List [num_feat_sets] containing output features of shape [num_features, feat_size].
        """

        # Initialize empty output features list
        out_feat_list = []

        # Perform self-attention on every set of input features
        for in_feats in in_feat_list:

            # Apply layer normalization
            norm_feats = self.layer_norm(in_feats)

            # Get queries, keys and values
            f = self.feat_size
            head_size = f//self.num_heads

            qkv_feats = self.in_proj(norm_feats)
            queries = qkv_feats[:, :f].view(-1, self.num_heads, head_size).permute(1, 0, 2)
            keys = qkv_feats[:, f:2*f].view(-1, self.num_heads, head_size).permute(1, 2, 0)
            values = qkv_feats[:, 2*f:3*f].view(-1, self.num_heads, head_size).permute(1, 0, 2)

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
