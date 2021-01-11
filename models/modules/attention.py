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
        out_channels (int): Number of channels of output map.
        kernel_size (Tuple): Size of the attention region in (height, width) format.
        stride (Tuple): Stride of the attention region in (height, width) format.
        padding (Tuple): Key/value map padding in (height, width) format.
        dilation (Tuple): Dilation of the attention region in (height, width) format.
        num_heads (int): Number of attention heads.
        padding_mode (str): Padding mode from {'constant', 'reflect', 'replicate', 'circular'}.
        attn_mode (str): Attention mode from {'self', 'cross'}.
        pos_attn (bool): Whehter or not local position features are learned.
        q_stride (Tuple): Stride of the query elements in (height, width) format.
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
                 padding_mode='zeros', attn_mode='self', pos_attn=True, q_stride=1, qk_norm='softmax'):
        """
        Initializes the Attn2d module.

        Args:
            in_channels (int or List): Number of channels of input maps.
            out_channels (int): Number of channels of output map.
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
            qk_norm (str): Query-key normalization function name from {'softmax', 'sigmoid'} (default='softmax').

        Raises:
            ValueError: Error when invalid attention mode is provided.
            ValueError: Error when two different number of input channels are given in 'self' attention mode.
        """

        # Initialization of default nn.Module
        super().__init__()

        # Check inputs
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

        if qk_norm == 'softmax':
            def softmax(x): return F.softmax(x, dim=-1)
            self.qk_norm = softmax

        elif qk_norm == 'sigmoid':
            self.qk_norm = torch.sigmoid

        # Initialize linear projection modules
        if attn_mode == 'self':
            self.proj = nn.Conv2d(in_channels[0], 3*out_channels, kernel_size=1, bias=bias)
        else:
            self.q_proj = nn.Conv2d(in_channels[0], out_channels, kernel_size=1, bias=bias)
            self.kv_proj = nn.Conv2d(in_channels[1], 2*out_channels, kernel_size=1, bias=bias)

        # Initialize local position features if requested
        if pos_attn:
            self.pos_feats = Parameter(torch.empty(out_channels, kernel_size[0]*kernel_size[1]))

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

        # Some renaming for readability
        f = self.out_channels
        q_sH, q_sW = self.q_stride

        # Get initial query, key and value maps
        if self.attn_mode == 'self':
            qkv_map = self.proj(in_feat_maps[0])
            query_map = qkv_map[:, :f, ::q_sH, ::q_sW]
            kv_map = qkv_map[:, f:3*f, :, :]

        else:
            query_map = self.q_proj(in_feat_maps[0][:, :, ::q_sH, ::q_sW])
            kv_map = self.kv_proj(in_feat_maps[1])

        # Process query map
        scale = float(f//self.num_heads)**-0.25
        query_map = scale*query_map.permute(0, 2, 3, 1)
        query_map = query_map.view(*query_map.shape[:3], self.num_heads, 1, -1)

        # Process key/value map
        if sum(self.padding) > 0:
            padding_size = (self.padding[1], self.padding[1], self.padding[0], self.padding[0])
            kv_map = F.pad(kv_map, padding_size, mode=self.padding_mode)

        kv_map = kv_map.permute(0, 2, 3, 1)
        sizes = [(size+attn_stride-1)//attn_stride for size, attn_stride in zip(kv_map.shape[1:3], self.stride)]
        center_strides = [stride*attn_stride for stride, attn_stride in zip(kv_map.stride()[1:3], self.stride)]
        points_strides = [stride*dilation for stride, dilation in zip(kv_map.stride()[1:3], self.dilation)]

        sizes = [kv_map.shape[0], *sizes, 2*f, *self.kernel_size]
        strides = [kv_map.stride()[0], *center_strides, kv_map.stride()[3], *points_strides]
        kv_map = kv_map.as_strided(sizes, strides).reshape(*sizes[:-2], -1)

        # Process key map
        key_map = kv_map[:, :, :, :f, :]
        key_map = torch.add(self.pos_feats, key_map, alpha=scale) if self.pos_attn else scale*key_map
        key_map = key_map.view(*key_map.shape[:3], self.num_heads, f//self.num_heads, -1)

        # Process value map
        value_map = kv_map[:, :, :, f:2*f, :]
        value_map = value_map.view(*key_map.shape)

        # Get output feature map
        attn_weights = self.qk_norm(torch.matmul(query_map, key_map))
        out_feat_map = torch.sum(attn_weights * value_map, dim=-1).view(*value_map.shape[:3], -1)
        out_feat_map = out_feat_map.permute(0, 3, 1, 2)

        return out_feat_map
