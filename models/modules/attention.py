"""
Collection of attention-based modules.
"""
import math

from deformable_detr.models.ops.modules import MSDeformAttn as MSDA
from deformable_detr.models.ops.functions import MSDeformAttnFunction as MSDAF
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from models.ops.sampler.functional import naive_maps_sampler_2d, naive_maps_sampler_3d


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


class DeformableAttn(nn.Module):
    """
    Class implementing the DeformableAttn module.

    Attributes:
        norm (nn.Module): Optional normalization module of the DeformableAttn module.
        act_fn (nn.Module): Optional module with the activation function of the DeformableAttn module.
        msda (nn.Module): Multi-scale deformable attention module of the DeformableAttn module.
        skip (bool): Boolean indicating whether skip connection is used or not.
    """

    def __init__(self, in_size, sample_size, out_size=-1, norm='', act_fn='', skip=True, version=0, num_levels=5,
                 num_heads=8, num_points=4, qk_size=-1, value_size=-1, val_with_pos=False):
        """
        Initializes the DeformableAttn module.

        Args:
            in_size (int): Size of input features.
            sample_size (int): Size of sample features.
            out_size (int): Size of output features (default=-1).
            norm (str): String containing the type of normalization (default='').
            act_fn (str): String containing the type of activation function (default='').
            skip (bool): Boolean indicating whether skip connection is used or not (default=True).
            version (int): Integer containing the version of the MSDA module (default=0).
            num_levels (int): Integer containing the number of map levels to sample from (default=5).
            num_heads (int): Integer containing the number of attention heads (default=8).
            num_points (int): Integer containing the number of sampling points per head and per level (default=4).
            qk_size (int): Size of query and key features (default=-1).
            value_size (int): Size of value features (default=-1).
            val_with_pos (bool): Boolean indicating whether position info is added to value features (default=False).

        Raises:
            ValueError: Error when unsupported type of normalization is provided.
            ValueError: Error when unsupported type of activation function is provided.
            ValueError: Error when input and output feature sizes are different when skip connection is used.
            ValueError: Error when the output feature size is not specified when no skip connection is used.
            ValueError: Error when invalid MSDA version number is provided.
        """

        # Initialization of default nn.Module
        super().__init__()

        # Initialization of optional normalization module
        if not norm:
            pass
        elif norm == 'layer':
            self.norm = nn.LayerNorm(in_size)
        else:
            error_msg = f"The DeformableAttn module does not support the '{norm}' normalization type."
            raise ValueError(error_msg)

        # Initialization of optional module with activation function
        if not act_fn:
            pass
        elif act_fn == 'gelu':
            self.act_fn = nn.GELU()
        elif act_fn == 'relu':
            self.act_fn = nn.ReLU(inplace=False) if not norm and skip else nn.ReLU(inplace=True)
        else:
            error_msg = f"The DeformableAttn module does not support the '{act_fn}' activation function."

        # Get and check output feature size
        if skip and out_size == -1:
            out_size = in_size

        elif skip and in_size != out_size:
            error_msg = f"Input ({in_size}) and output ({out_size}) sizes must match when skip connection is used."
            raise ValueError(error_msg)

        elif not skip and out_size == -1:
            error_msg = "The output feature size must be specified when no skip connection is used."
            raise ValueError(error_msg)

        # Initialization of multi-scale deformable attention module
        if version == 0:
            self.msda = MSDA(in_size, num_levels, num_heads, num_points)
            self.msda.output_proj = nn.Linear(in_size, out_size)
            nn.init.xavier_uniform_(self.msda.output_proj.weight)
            nn.init.zeros_(self.msda.output_proj.bias)

        elif version == 1:
            self.msda = MSDAv1(in_size, sample_size, out_size, num_levels, num_heads, num_points, value_size)

        elif version == 2:
            self.msda = MSDAv2(in_size, sample_size, out_size, num_levels, num_heads, num_points, value_size,
                               val_with_pos)

        elif version == 3:
            self.msda = MSDAv3(in_size, sample_size, out_size, num_levels, num_heads, num_points, qk_size, value_size,
                               val_with_pos)

        elif version == 4:
            self.msda = MSDAv4(in_size, sample_size, out_size, num_levels, num_heads, num_points, value_size,
                               val_with_pos)

        else:
            error_msg = f"Invalid MSDA version number '{version}'."
            raise ValueError(error_msg)

        # Set skip attribute
        self.skip = skip

    def forward(self, in_feats, sample_priors, sample_feats, sample_map_shapes, sample_map_start_ids,
                add_encs=None, mul_encs=None, sample_mask=None, **kwargs):
        """
        Forward method of the DeformableAttn module.

        Args:
            in_feats (FloatTensor): Input features of shape [*, num_in_feats, in_size].
            sample_priors (FloatTensor): Sample priors of shape [*, num_in_feats, {2, 4}].
            sample_feats (FloatTensor): Sample features of shape [*, num_sample_feats, sample_size].
            sample_map_shapes (LongTensor): Map shapes corresponding to samples of shape [num_levels, 2].
            sample_map_start_ids (LongTensor): Start indices of sample maps of shape [num_levels].
            add_encs (FloatTensor): Encodings added to queries of shape [*, num_in_feats, in_size] (default=None).
            mul_encs (FloatTensor): Encodings multiplied by queries of shape [*, num_in_feats, in_size] (default=None).
            sample_mask (BoolTensor): Inactive samples mask of shape [*, num_sample_feats] (default=None).
            kwargs (Dict): Dictionary of keyword arguments not used by this module.

        Returns:
            out_feats (FloatTensor): Output features of shape [*, num_in_feats, out_size].
        """

        # Apply optional normalization and activation function modules
        delta_feats = in_feats
        delta_feats = self.norm(delta_feats) if hasattr(self, 'norm') else delta_feats
        delta_feats = self.act_fn(delta_feats) if hasattr(self, 'act_fn') else delta_feats

        # Apply multi-scale deformable attention module
        orig_shape = delta_feats.shape
        delta_feats = delta_feats.view(-1, *orig_shape[-2:])

        if mul_encs is not None:
            delta_feats = delta_feats * mul_encs.view(-1, *orig_shape[-2:])

        if add_encs is not None:
            delta_feats = delta_feats + add_encs.view(-1, *orig_shape[-2:])

        num_levels = len(sample_map_start_ids)
        sample_priors = sample_priors.view(-1, *sample_priors.shape[-2:])
        sample_priors = sample_priors[:, :, None, :].expand(-1, -1, num_levels, -1)
        sample_feats = sample_feats.view(-1, *sample_feats.shape[-2:])

        msda_args = (delta_feats, sample_priors, sample_feats, sample_map_shapes, sample_map_start_ids)
        msda_kwargs = {'input_padding_mask': sample_mask}
        delta_feats = self.msda(*msda_args, **msda_kwargs).view(*orig_shape[:-1], -1)

        # Get output features
        out_feats = in_feats + delta_feats if self.skip else delta_feats

        return out_feats


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


class MSDAv1(nn.Module):
    """
    Class implementing the MSDAv1 module.

    Attributes:
        sampling_offsets (nn.Linear): Module computing the sampling offsets from the input features.
        attention_weights (nn.Linear): Module computing the attention weights from the input features.
        value_proj (nn.Linear): Module computing value features from sample features.
        out_proj (nn.Linear): Module computing output features from weighted value features.

        num_levels (int): Integer containing the number of map levels to sample from.
        num_heads (int): Integer containing the number of attention heads.
        num_points (int): Integer containing the number of sampling points per head and per level.
    """

    def __init__(self, in_size, sample_size, out_size=-1, num_levels=5, num_heads=8, num_points=4, value_size=-1):
        """
        Initializes the MSDAv1 module.

        Args:
            in_size (int): Size of input features.
            sample_size (int): Size of sample features.
            out_size (int): Size of output features (default=-1).
            num_levels (int): Integer containing the number of map levels to sample from (default=5).
            num_heads (int): Integer containing the number of attention heads (default=8).
            num_points (int): Integer containing the number of sampling points per head and per level (default=4).
            value_size (int): Size of value features (default=-1).

        Raises:
            ValueError: Error when the input feature size does not divide the number of heads.
            ValueError: Error when the value feature size does not divide the number of heads.
        """

        # Initialization of default nn.Module
        super().__init__()

        # Check divisibility query size by number of heads
        if in_size % num_heads != 0:
            error_msg = f"The input feature size ({in_size}) must divide the number of heads ({num_heads})."
            raise ValueError(error_msg)

        # Initialize module computing the sample offsets
        self.sampling_offsets = nn.Linear(in_size, num_heads * num_levels * num_points * 2)
        nn.init.zeros_(self.sampling_offsets.weight)

        thetas = torch.arange(num_heads, dtype=torch.float) * (2.0 * math.pi / num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], dim=1)
        grid_init = grid_init / grid_init.abs().max(dim=1, keepdim=True)[0]
        grid_init = grid_init.view(num_heads, 1, 1, 2).repeat(1, num_levels, 1, 1)

        sizes = torch.arange(1, num_points+1, dtype=torch.float).view(1, 1, num_points, 1)
        grid_init = sizes * grid_init
        self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))

        # Initialize module computing the unnormalized attention weights
        self.attention_weights = nn.Linear(in_size, num_heads * num_levels * num_points)
        nn.init.zeros_(self.attention_weights.weight)
        nn.init.zeros_(self.attention_weights.bias)

        # Get and check size of value features
        if value_size == -1:
            value_size = in_size

        elif value_size % num_heads != 0:
            error_msg = f"The value feature size ({value_size}) must divide the number of heads ({num_heads})."
            raise ValueError(error_msg)

        # Initialize module computing the value features
        self.value_proj = nn.Linear(sample_size, value_size)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.zeros_(self.value_proj.bias)

        # Initialize module computing the output features
        out_size = in_size if out_size == -1 else out_size
        self.out_proj = nn.Linear(value_size, out_size)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

        # Set attributes related to number of levels, heads and points
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points

    def forward(self, in_feats, sample_priors, sample_feats, sample_map_shapes, sample_map_start_ids, **kwargs):
        """
        Forward method of the MSDAv1 module.

        Args:
            in_feats (FloatTensor): Input features of shape [batch_size, num_in_feats, in_size].
            sample_priors (FloatTensor): Sample priors of shape [batch_size, num_in_feats, num_levels, {2, 4}].
            sample_feats (FloatTensor): Sample features of shape [batch_size, num_sample_feats, sample_size].
            sample_map_shapes (LongTensor): Map shapes corresponding to samples of shape [num_levels, 2].
            sample_map_start_ids (LongTensor): Start indices of sample maps of shape [num_levels].
            kwargs (Dict): Dictionary of keyword arguments not used by this module.

        Returns:
            out_feats (FloatTensor): Output features of shape [batch_size, num_in_feats, out_size].

        Raises:
            ValueError: Error when the last dimension of 'sample_priors' is different from 2 or 4.
        """

        # Get shapes of input tensors
        batch_size, num_in_feats = in_feats.shape[:2]
        common_shape = (batch_size, num_in_feats, self.num_heads)

        # Get sample offsets
        sample_offsets = self.sampling_offsets(in_feats).view(*common_shape, self.num_levels, self.num_points, 2)

        # Get sample locations
        if sample_priors.shape[-1] == 2:
            offset_normalizers = sample_map_shapes.fliplr()[None, None, None, :, None, :]
            sample_locations = sample_priors[:, :, None, :, None, :]
            sample_locations = sample_locations + sample_offsets / offset_normalizers

        elif sample_priors.shape[-1] == 4:
            offset_factors = 0.5 * sample_priors[:, :, None, :, None, 2:] / self.num_points
            sample_locations = sample_priors[:, :, None, :, None, :2]
            sample_locations = sample_locations + sample_offsets * offset_factors

        else:
            error_msg = f"Last dimension of 'sample_priors' must be 2 or 4, but got {sample_priors.shape[-1]}."
            raise ValueError(error_msg)

        # Get attention weights
        attn_weights = self.attention_weights(in_feats).view(*common_shape, self.num_levels * self.num_points)
        attn_weights = F.softmax(attn_weights, dim=3).view(*common_shape, self.num_levels, self.num_points)

        # Get value features
        value_feats = self.value_proj(sample_feats)

        # Apply MSDA function
        value_size = value_feats.shape[-1]
        value_feats = value_feats.view(batch_size, -1, self.num_heads, value_size // self.num_heads)

        msdaf_args = (value_feats, sample_map_shapes, sample_map_start_ids, sample_locations, attn_weights, 64)
        weighted_value_feats = MSDAF.apply(*msdaf_args)

        # Get output features
        out_feats = self.out_proj(weighted_value_feats)

        return out_feats


class MSDAv2(nn.Module):
    """
    Class implementing the MSDAv2 module.

    Attributes:
        sampling_offsets (nn.Linear): Module computing the sampling offsets from the input features.
        attention_weights (nn.Linear): Module computing the attention weights from the input features.
        value_proj (nn.Linear): Module computing value features from sample features.
        val_pos_encs (nn.Linear): Optional module computing the value position encodings.
        out_proj (nn.Linear): Module computing output features from weighted value features.

        num_levels (int): Integer containing the number of map levels to sample from.
        num_heads (int): Integer containing the number of attention heads.
        num_points (int): Integer containing the number of sampling points per head and per level.
    """

    def __init__(self, in_size, sample_size, out_size=-1, num_levels=5, num_heads=8, num_points=4, value_size=-1,
                 val_with_pos=False):
        """
        Initializes the MSDAv2 module.

        Args:
            in_size (int): Size of input features.
            sample_size (int): Size of sample features.
            out_size (int): Size of output features (default=-1).
            num_levels (int): Integer containing the number of map levels to sample from (default=5).
            num_heads (int): Integer containing the number of attention heads (default=8).
            num_points (int): Integer containing the number of sampling points per head and per level (default=4).
            value_size (int): Size of value features (default=-1).
            val_with_pos (bool): Boolean indicating whether position info is added to value features (default=False).

        Raises:
            ValueError: Error when the input feature size does not divide the number of heads.
            ValueError: Error when the value feature size does not divide the number of heads.
        """

        # Initialization of default nn.Module
        super().__init__()

        # Check divisibility query size by number of heads
        if in_size % num_heads != 0:
            error_msg = f"The input feature size ({in_size}) must divide the number of heads ({num_heads})."
            raise ValueError(error_msg)

        # Initialize module computing the sample offsets
        self.sampling_offsets = nn.Linear(in_size, num_heads * num_levels * num_points * 2)
        nn.init.zeros_(self.sampling_offsets.weight)

        thetas = torch.arange(num_heads, dtype=torch.float) * (2.0 * math.pi / num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], dim=1)
        grid_init = grid_init / grid_init.abs().max(dim=1, keepdim=True)[0]
        grid_init = grid_init.view(num_heads, 1, 1, 2).repeat(1, num_levels, 1, 1)

        sizes = torch.arange(1, num_points+1, dtype=torch.float).view(1, 1, num_points, 1)
        grid_init = sizes * grid_init
        self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))

        # Initialize module computing the unnormalized attention weights
        self.attention_weights = nn.Linear(in_size, num_heads * num_levels * num_points)
        nn.init.zeros_(self.attention_weights.weight)
        nn.init.zeros_(self.attention_weights.bias)

        # Get and check size of value features
        if value_size == -1:
            value_size = in_size

        elif value_size % num_heads != 0:
            error_msg = f"The value feature size ({value_size}) must divide the number of heads ({num_heads})."
            raise ValueError(error_msg)

        # Initialize module computing the value features
        self.value_proj = nn.Linear(sample_size, value_size)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.zeros_(self.value_proj.bias)

        # Initialize module computing the value position encodings if requested
        if val_with_pos:
            self.val_pos_encs = nn.Linear(3, value_size // num_heads)

        # Initialize module computing the output features
        out_size = in_size if out_size == -1 else out_size
        self.out_proj = nn.Linear(value_size, out_size)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

        # Set attributes related to number of levels, heads and points
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points

    def forward(self, in_feats, sample_priors, sample_feats, sample_map_shapes, sample_map_start_ids, **kwargs):
        """
        Forward method of the MSDAv2 module.

        Args:
            in_feats (FloatTensor): Input features of shape [batch_size, num_in_feats, in_size].
            sample_priors (FloatTensor): Sample priors of shape [batch_size, num_in_feats, num_levels, {2, 4}].
            sample_feats (FloatTensor): Sample features of shape [batch_size, num_sample_feats, sample_size].
            sample_map_shapes (LongTensor): Map shapes corresponding to samples of shape [num_levels, 2].
            sample_map_start_ids (LongTensor): Start indices of sample maps of shape [num_levels].
            kwargs (Dict): Dictionary of keyword arguments not used by this module.

        Returns:
            out_feats (FloatTensor): Output features of shape [batch_size, num_in_feats, out_size].

        Raises:
            ValueError: Error when the last dimension of 'sample_priors' is different from 2 or 4.
        """

        # Get shapes of input tensors
        batch_size, num_in_feats = in_feats.shape[:2]
        common_shape = (batch_size, num_in_feats, self.num_heads)

        # Get sample offsets
        sample_offsets = self.sampling_offsets(in_feats).view(*common_shape, self.num_levels, self.num_points, 2)

        # Get sample locations
        if sample_priors.shape[-1] == 2:
            offset_normalizers = sample_map_shapes.fliplr()[None, None, None, :, None, :]
            sample_locations = sample_priors[:, :, None, :, None, :]
            sample_locations = sample_locations + sample_offsets / offset_normalizers

        elif sample_priors.shape[-1] == 4:
            offset_factors = 0.5 * sample_priors[:, :, None, :, None, 2:] / self.num_points
            sample_locations = sample_priors[:, :, None, :, None, :2]
            sample_locations = sample_locations + sample_offsets * offset_factors

        else:
            error_msg = f"Last dimension of 'sample_priors' must be 2 or 4, but got {sample_priors.shape[-1]}."
            raise ValueError(error_msg)

        # Get attention weights
        attn_weights = self.attention_weights(in_feats).view(*common_shape, self.num_levels * self.num_points)
        attn_weights = F.softmax(attn_weights, dim=3)

        # Get value features
        value_feats = self.value_proj(sample_feats)

        # Get sampled value features
        value_size = value_feats.shape[-1]
        value_feats = value_feats.view(batch_size, -1, self.num_heads, value_size // self.num_heads)
        value_feats = value_feats.transpose(1, 2).view(batch_size * self.num_heads, -1, value_size // self.num_heads)

        sample_map_shapes = sample_map_shapes.fliplr()
        sample_locations = sample_locations.transpose(1, 2).reshape(batch_size * self.num_heads, -1, 2)

        sample_map_ids = torch.arange(self.num_levels, device=sample_locations.device)
        sample_map_ids = sample_map_ids[None, None, :, None]
        sample_map_ids = sample_map_ids.expand(batch_size * self.num_heads, num_in_feats, -1, self.num_points)
        sample_map_ids = sample_map_ids.reshape(batch_size * self.num_heads, -1)

        sampler_args = (value_feats, sample_map_shapes, sample_map_start_ids, sample_locations, sample_map_ids)
        sampled_feats = naive_maps_sampler_2d(*sampler_args)

        sampled_feats = sampled_feats.view(batch_size, self.num_heads, num_in_feats, -1, value_size // self.num_heads)
        sampled_feats = sampled_feats.transpose(1, 2)

        # Get and add position encodings to sampled value features if needed
        if hasattr(self, 'val_pos_encs'):
            sample_xy = 0.5 * sample_offsets / self.num_points
            sample_z = sample_map_ids.view(batch_size, self.num_heads, num_in_feats, -1, self.num_points, 1)
            sample_z = sample_z.transpose(1, 2) / (self.num_levels-1)

            sample_xyz = torch.cat([sample_xy, sample_z], dim=5).flatten(3, 4)
            sampled_feats = sampled_feats + self.val_pos_encs(sample_xyz)

        # Get weighted value features
        weighted_feats = attn_weights[:, :, :, :, None] * sampled_feats
        weighted_feats = weighted_feats.sum(dim=3).view(batch_size, num_in_feats, value_size)

        # Get output features
        out_feats = self.out_proj(weighted_feats)

        return out_feats


class MSDAv3(nn.Module):
    """
    Class implementing the MSDAv3 module.

    Attributes:
        sampling_offsets (nn.Linear): Module computing the sampling offsets from the input features.
        query_proj (nn.Linear): Module computing query features from input features.
        kv_proj (nn.Linear): Module computing key-value features from sample features.
        point_encs (nn.Parameter): Parameter tensor containing the point encodings.
        val_pos_encs (nn.Linear): Optional module computing the value position encodings.
        out_proj (nn.Linear): Module computing output features from weighted value features.

        num_levels (int): Integer containing the number of map levels to sample from.
        num_heads (int): Integer containing the number of attention heads.
        num_points (int): Integer containing the number of sampling points per head and per level.
    """

    def __init__(self, in_size, sample_size, out_size=-1, num_levels=5, num_heads=8, num_points=4, qk_size=-1,
                 value_size=-1, val_with_pos=False):
        """
        Initializes the MSDAv3 module.

        Args:
            in_size (int): Size of input features.
            sample_size (int): Size of sample features.
            out_size (int): Size of output features (default=-1).
            num_levels (int): Integer containing the number of map levels to sample from (default=5).
            num_heads (int): Integer containing the number of attention heads (default=8).
            num_points (int): Integer containing the number of sampling points per head and per level (default=4).
            qk_size (int): Size of query and key features (default=-1).
            value_size (int): Size of value features (default=-1).
            val_with_pos (bool): Boolean indicating whether position info is added to value features (default=False).

        Raises:
            ValueError: Error when the input feature size does not divide the number of heads.
            ValueError: Error when the query and key feature size does not divide the number of heads.
            ValueError: Error when the value feature size does not divide the number of heads.
        """

        # Initialization of default nn.Module
        super().__init__()

        # Check divisibility input size by number of heads
        if in_size % num_heads != 0:
            error_msg = f"The input feature size ({in_size}) must divide the number of heads ({num_heads})."
            raise ValueError(error_msg)

        # Initialize module computing the sample offsets
        self.sampling_offsets = nn.Linear(in_size, num_heads * num_levels * num_points * 2)
        nn.init.zeros_(self.sampling_offsets.weight)

        thetas = torch.arange(num_heads, dtype=torch.float) * (2.0 * math.pi / num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], dim=1)
        grid_init = grid_init / grid_init.abs().max(dim=1, keepdim=True)[0]
        grid_init = grid_init.view(num_heads, 1, 1, 2).repeat(1, num_levels, 1, 1)

        sizes = torch.arange(1, num_points+1, dtype=torch.float).view(1, 1, num_points, 1)
        grid_init = sizes * grid_init
        self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))

        # Get and check size of query and key features
        if qk_size == -1:
            qk_size = in_size

        elif qk_size % num_heads != 0:
            error_msg = f"The query and key feature size ({qk_size}) must divide the number of heads ({num_heads})."
            raise ValueError(error_msg)

        # Initialize module computing the query features
        self.query_proj = nn.Linear(in_size, qk_size)
        nn.init.xavier_uniform_(self.query_proj.weight)
        nn.init.zeros_(self.query_proj.bias)

        # Get and check size of value features
        if value_size == -1:
            value_size = in_size

        elif value_size % num_heads != 0:
            error_msg = f"The value feature size ({value_size}) must divide the number of heads ({num_heads})."
            raise ValueError(error_msg)

        # Initialize module computing the key-value features
        kv_size = qk_size + value_size
        self.kv_proj = nn.Linear(sample_size, kv_size)
        nn.init.xavier_uniform_(self.kv_proj.weight)
        nn.init.zeros_(self.kv_proj.bias)

        # Initialize point encodings
        self.point_encs = nn.Parameter(torch.zeros(num_heads, num_levels * num_points, qk_size // num_heads))

        # Initialize module computing the value position encodings if requested
        if val_with_pos:
            self.val_pos_encs = nn.Linear(3, value_size // num_heads)

        # Initialize module computing the output features
        out_size = in_size if out_size == -1 else out_size
        self.out_proj = nn.Linear(value_size, out_size)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

        # Set attributes related to number of levels, heads and points
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points

    def forward(self, in_feats, sample_priors, sample_feats, sample_map_shapes, sample_map_start_ids, **kwargs):
        """
        Forward method of the MSDAv3 module.

        Args:
            in_feats (FloatTensor): Input features of shape [batch_size, num_in_feats, in_size].
            sample_priors (FloatTensor): Sample priors of shape [batch_size, num_in_feats, num_levels, {2, 4}].
            sample_feats (FloatTensor): Sample features of shape [batch_size, num_sample_feats, sample_size].
            sample_map_shapes (LongTensor): Map shapes corresponding to samples of shape [num_levels, 2].
            sample_map_start_ids (LongTensor): Start indices of sample maps of shape [num_levels].
            kwargs (Dict): Dictionary of keyword arguments not used by this module.

        Returns:
            out_feats (FloatTensor): Output features of shape [batch_size, num_in_feats, out_size].

        Raises:
            ValueError: Error when the last dimension of 'sample_priors' is different from 2 or 4.
        """

        # Get shapes of input tensors
        batch_size, num_in_feats = in_feats.shape[:2]
        common_shape = (batch_size, num_in_feats, self.num_heads)

        # Get sample offsets
        sample_offsets = self.sampling_offsets(in_feats).view(*common_shape, self.num_levels, self.num_points, 2)

        # Get sample locations
        if sample_priors.shape[-1] == 2:
            offset_normalizers = sample_map_shapes.fliplr()[None, None, None, :, None, :]
            sample_locations = sample_priors[:, :, None, :, None, :]
            sample_locations = sample_locations + sample_offsets / offset_normalizers

        elif sample_priors.shape[-1] == 4:
            offset_factors = 0.5 * sample_priors[:, :, None, :, None, 2:] / self.num_points
            sample_locations = sample_priors[:, :, None, :, None, :2]
            sample_locations = sample_locations + sample_offsets * offset_factors

        else:
            error_msg = f"Last dimension of 'sample_priors' must be 2 or 4, but got {sample_priors.shape[-1]}."
            raise ValueError(error_msg)

        # Get query and key-value features
        query_feats = self.query_proj(in_feats).view(*common_shape, 1, -1)
        kv_feats = self.kv_proj(sample_feats)

        # Get sampled key-value features
        kv_size = kv_feats.shape[-1]
        kv_feats = kv_feats.view(batch_size, -1, self.num_heads, kv_size // self.num_heads)
        kv_feats = kv_feats.transpose(1, 2).view(batch_size * self.num_heads, -1, kv_size // self.num_heads)

        sample_map_shapes = sample_map_shapes.fliplr()
        sample_locations = sample_locations.transpose(1, 2).reshape(batch_size * self.num_heads, -1, 2)

        sample_map_ids = torch.arange(self.num_levels, device=sample_locations.device)
        sample_map_ids = sample_map_ids[None, None, :, None]
        sample_map_ids = sample_map_ids.expand(batch_size * self.num_heads, num_in_feats, -1, self.num_points)
        sample_map_ids = sample_map_ids.reshape(batch_size * self.num_heads, -1)

        sampler_args = (kv_feats, sample_map_shapes, sample_map_start_ids, sample_locations, sample_map_ids)
        sampled_feats = naive_maps_sampler_2d(*sampler_args)

        sampled_feats = sampled_feats.view(batch_size, self.num_heads, num_in_feats, -1, kv_size // self.num_heads)
        sampled_feats = sampled_feats.transpose(1, 2)

        # Get sampled key and value features
        head_qk_size = query_feats.shape[-1]
        sampled_key_feats = sampled_feats[:, :, :, :, :head_qk_size]
        sampled_value_feats = sampled_feats[:, :, :, :, head_qk_size:]

        # Add point encodings to sampled key features
        sampled_key_feats = sampled_key_feats + self.point_encs

        # Get attention weights
        query_feats = query_feats / math.sqrt(head_qk_size)
        attn_weights = torch.matmul(query_feats, sampled_key_feats.transpose(3, 4)).squeeze(dim=3)
        attn_weights = F.softmax(attn_weights, dim=3)

        # Get and add position encodings to sampled value features if needed
        if hasattr(self, 'val_pos_encs'):
            sample_xy = 0.5 * sample_offsets / self.num_points
            sample_z = sample_map_ids.view(batch_size, self.num_heads, num_in_feats, -1, self.num_points, 1)
            sample_z = sample_z.transpose(1, 2) / (self.num_levels-1)

            sample_xyz = torch.cat([sample_xy, sample_z], dim=5).flatten(3, 4)
            sampled_value_feats = sampled_value_feats + self.val_pos_encs(sample_xyz)

        # Get weighted value features
        weighted_feats = attn_weights[:, :, :, :, None] * sampled_value_feats
        weighted_feats = weighted_feats.sum(dim=3).view(batch_size, num_in_feats, -1)

        # Get output features
        out_feats = self.out_proj(weighted_feats)

        return out_feats


class MSDAv4(nn.Module):
    """
    Class implementing the MSDAv4 module.

    Attributes:
        sampling_offsets (nn.Linear): Module computing the sampling offsets from the input features.
        attention_weights (nn.Linear): Module computing the attention weights from the input features.
        value_proj (nn.Linear): Module computing value features from sample features.
        val_pos_encs (nn.Linear): Optional module computing the value position encodings.
        out_proj (nn.Linear): Module computing output features from weighted value features.

        num_levels (int): Integer containing the number of map levels to sample from.
        num_heads (int): Integer containing the number of attention heads.
        num_points (int): Integer containing the number of sampling points per head and per level.
    """

    def __init__(self, in_size, sample_size, out_size=-1, num_levels=5, num_heads=8, num_points=4, value_size=-1,
                 val_with_pos=False):
        """
        Initializes the MSDAv4 module.

        Args:
            in_size (int): Size of input features.
            sample_size (int): Size of sample features.
            out_size (int): Size of output features (default=-1).
            num_levels (int): Integer containing the number of map levels to sample from (default=5).
            num_heads (int): Integer containing the number of attention heads (default=8).
            num_points (int): Integer containing the number of sampling points per head and per level (default=4).
            value_size (int): Size of value features (default=-1).
            val_with_pos (bool): Boolean indicating whether position info is added to value features (default=False).

        Raises:
            ValueError: Error when the input feature size does not divide the number of heads.
            ValueError: Error when the value feature size does not divide the number of heads.
        """

        # Initialization of default nn.Module
        super().__init__()

        # Check divisibility query size by number of heads
        if in_size % num_heads != 0:
            error_msg = f"The input feature size ({in_size}) must divide the number of heads ({num_heads})."
            raise ValueError(error_msg)

        # Initialize module computing the sample offsets
        self.sampling_offsets = nn.Linear(in_size, num_heads * num_levels * num_points * 3)
        nn.init.zeros_(self.sampling_offsets.weight)

        thetas = torch.arange(num_heads, dtype=torch.float) * (2.0 * math.pi / num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin(), torch.zeros_like(thetas)], dim=1)
        grid_init = grid_init / grid_init.abs().max(dim=1, keepdim=True)[0]
        grid_init = grid_init.view(num_heads, 1, 1, 3).repeat(1, num_levels, 1, 1)

        sizes = torch.arange(1, num_points+1, dtype=torch.float).view(1, 1, num_points, 1)
        grid_init = sizes * grid_init
        self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))

        # Initialize module computing the unnormalized attention weights
        self.attention_weights = nn.Linear(in_size, num_heads * num_levels * num_points)
        nn.init.zeros_(self.attention_weights.weight)
        nn.init.zeros_(self.attention_weights.bias)

        # Get and check size of value features
        if value_size == -1:
            value_size = in_size

        elif value_size % num_heads != 0:
            error_msg = f"The value feature size ({value_size}) must divide the number of heads ({num_heads})."
            raise ValueError(error_msg)

        # Initialize module computing the value features
        self.value_proj = nn.Linear(sample_size, value_size)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.zeros_(self.value_proj.bias)

        # Initialize module computing the value position encodings if requested
        if val_with_pos:
            self.val_pos_encs = nn.Linear(3, value_size // num_heads)

        # Initialize module computing the output features
        out_size = in_size if out_size == -1 else out_size
        self.out_proj = nn.Linear(value_size, out_size)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

        # Set attributes related to number of levels, heads and points
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points

    def forward(self, in_feats, sample_priors, sample_feats, sample_map_shapes, sample_map_start_ids, **kwargs):
        """
        Forward method of the MSDAv4 module.

        Args:
            in_feats (FloatTensor): Input features of shape [batch_size, num_in_feats, in_size].
            sample_priors (FloatTensor): Sample priors of shape [batch_size, num_in_feats, num_levels, {2, 4}].
            sample_feats (FloatTensor): Sample features of shape [batch_size, num_sample_feats, sample_size].
            sample_map_shapes (LongTensor): Map shapes corresponding to samples of shape [num_levels, 2].
            sample_map_start_ids (LongTensor): Start indices of sample maps of shape [num_levels].
            kwargs (Dict): Dictionary of keyword arguments not used by this module.

        Returns:
            out_feats (FloatTensor): Output features of shape [batch_size, num_in_feats, out_size].

        Raises:
            ValueError: Error when the last dimension of 'sample_priors' is different from 2 or 4.
        """

        # Get shapes of input tensors
        batch_size, num_in_feats = in_feats.shape[:2]
        common_shape = (batch_size, num_in_feats, self.num_heads)
        num_levels = len(sample_map_shapes)

        # Get sample offsets
        sample_offsets = self.sampling_offsets(in_feats).view(*common_shape, self.num_levels, self.num_points, 3)

        # Get sample locations
        sample_z = torch.linspace(0, 1, num_levels, dtype=sample_priors.dtype, device=sample_priors.device)
        sample_z = sample_z.view(1, 1, 1, num_levels, 1, 1).expand(batch_size, num_in_feats, -1, -1, -1, -1)

        if sample_priors.shape[-1] == 2:
            offset_normalizers = sample_map_shapes.fliplr()[None, None, None, :, None, :]
            sample_offsets[:, :, :, :, :, :2] = sample_offsets[:, :, :, :, :, :2] / offset_normalizers

            sample_locations = torch.cat([sample_priors[:, :, None, :, None, :], sample_z], dim=5)
            sample_locations = sample_locations + sample_offsets

        elif sample_priors.shape[-1] == 4:
            offset_factors = 0.5 * sample_priors[:, :, None, :, None, 2:] / self.num_points
            sample_offsets[:, :, :, :, :, :2] = sample_offsets[:, :, :, :, :, :2] * offset_factors

            sample_locations = torch.cat([sample_priors[:, :, None, :, None, :2], sample_z], dim=5)
            sample_locations = sample_locations + sample_offsets

        else:
            error_msg = f"Last dimension of 'sample_priors' must be 2 or 4, but got {sample_priors.shape[-1]}."
            raise ValueError(error_msg)

        # Get attention weights
        attn_weights = self.attention_weights(in_feats).view(*common_shape, self.num_levels * self.num_points)
        attn_weights = F.softmax(attn_weights, dim=3)

        # Get value features
        value_feats = self.value_proj(sample_feats)

        # Get sampled value features
        value_size = value_feats.shape[-1]
        value_feats = value_feats.view(batch_size, -1, self.num_heads, value_size // self.num_heads)
        value_feats = value_feats.transpose(1, 2).view(batch_size * self.num_heads, -1, value_size // self.num_heads)

        sample_map_shapes = sample_map_shapes.fliplr()
        sample_locations = sample_locations.transpose(1, 2).reshape(batch_size * self.num_heads, -1, 3)

        sampled_feats = naive_maps_sampler_3d(value_feats, sample_map_shapes, sample_map_start_ids, sample_locations)
        sampled_feats = sampled_feats.view(batch_size, self.num_heads, num_in_feats, -1, value_size // self.num_heads)
        sampled_feats = sampled_feats.transpose(1, 2)

        # Get and add position encodings to sampled value features if needed
        if hasattr(self, 'val_pos_encs'):
            sample_xy = 0.5 * sample_offsets[:, :, :, :, :, :2] / self.num_points
            sample_z = sample_locations[:, :, 2]
            sample_z = sample_z.view(batch_size, self.num_heads, num_in_feats, self.num_levels, self.num_points, 1)
            sample_z = sample_z.transpose(1, 2)

            sample_xyz = torch.cat([sample_xy, sample_z], dim=5).flatten(3, 4)
            sampled_feats = sampled_feats + self.val_pos_encs(sample_xyz)

        # Get weighted value features
        weighted_feats = attn_weights[:, :, :, :, None] * sampled_feats
        weighted_feats = weighted_feats.sum(dim=3).view(batch_size, num_in_feats, value_size)

        # Get output features
        out_feats = self.out_proj(weighted_feats)

        return out_feats


class SelfAttn1d(nn.Module):
    """
    Class implementing the SelfAttn1d module.

    Attributes:
        norm (nn.Module): Optional normalization module of the SelfAttn1d module.
        act_fn (nn.Module): Optional module with the activation function of the SelfAttn1d module.
        mha (nn.MultiheadAttention): Multi-head attention module of the SelfAttn1d module.
        skip (bool): Boolean indicating whether skip connection is used or not.
    """

    def __init__(self, in_size, out_size=-1, norm='', act_fn='', skip=True, num_heads=8):
        """
        Initializes the SelfAttn1d module.

        Args:
            in_size (int): Size of input features.
            out_size (int): Size of output features (default=-1).
            norm (str): String containing the type of normalization (default='').
            act_fn (str): String containing the type of activation function (default='').
            skip (bool): Boolean indicating whether skip connection is used or not (default=True).
            num_heads (int): Integer containing the number of attention heads (default=8).

        Raises:
            ValueError: Error when unsupported type of normalization is provided.
            ValueError: Error when unsupported type of activation function is provided.
            ValueError: Error when input and output feature sizes are different when skip connection is used.
            ValueError: Error when the output feature size is not specified when no skip connection is used.
        """

        # Initialization of default nn.Module
        super().__init__()

        # Initialization of optional normalization module
        if not norm:
            pass
        elif norm == 'layer':
            self.norm = nn.LayerNorm(in_size)
        else:
            error_msg = f"The SelfAttn1d module does not support the '{norm}' normalization type."
            raise ValueError(error_msg)

        # Initialization of optional module with activation function
        if not act_fn:
            pass
        elif act_fn == 'gelu':
            self.act_fn = nn.GELU()
        elif act_fn == 'relu':
            self.act_fn = nn.ReLU(inplace=False) if not norm and skip else nn.ReLU(inplace=True)
        else:
            error_msg = f"The SelfAttn1d module does not support the '{act_fn}' activation function."

        # Get and check output feature size
        if skip and out_size == -1:
            out_size = in_size

        elif skip and in_size != out_size:
            error_msg = f"Input ({in_size}) and output ({out_size}) sizes must match when skip connection is used."
            raise ValueError(error_msg)

        elif not skip and out_size == -1:
            error_msg = "The output feature size must be specified when no skip connection is used."
            raise ValueError(error_msg)

        # Initialization of multi-head attention module
        self.mha = nn.MultiheadAttention(in_size, num_heads)
        self.mha.out_proj = nn.Linear(in_size, out_size)
        nn.init.zeros_(self.mha.out_proj.bias)

        # Set skip attribute
        self.skip = skip

    def forward(self, in_feats, add_encs=None, mul_encs=None, **kwargs):
        """
        Forward method of the SelfAttn1d module.

        Args:
            in_feats (FloatTensor): Input features of shape [*, num_feats, in_size].
            add_encs (FloatTensor): Encodings added to queries of shape [*, num_in_feats, in_size] (default=None).
            mul_encs (FloatTensor): Encodings multiplied by queries of shape [*, num_in_feats, in_size] (default=None).
            kwargs (Dict): Dictionary of keyword arguments not used by this module.

        Returns:
            out_feats (FloatTensor): Output features of shape [*, num_feats, out_size].
        """

        # Apply optional normalization and activation function modules
        delta_feats = in_feats
        delta_feats = self.norm(delta_feats) if hasattr(self, 'norm') else delta_feats
        delta_feats = self.act_fn(delta_feats) if hasattr(self, 'act_fn') else delta_feats

        # Apply multi-head attention module
        orig_shape = delta_feats.shape
        delta_feats = delta_feats.view(-1, *orig_shape[-2:]).transpose(0, 1)
        values = delta_feats

        if mul_encs is not None:
            delta_feats = delta_feats * mul_encs.view(-1, *orig_shape[-2:]).transpose(0, 1)

        if add_encs is not None:
            delta_feats = delta_feats + add_encs.view(-1, *orig_shape[-2:]).transpose(0, 1)

        queries = keys = delta_feats
        delta_feats = self.mha(queries, keys, values, need_weights=False)[0]
        delta_feats = delta_feats.transpose(0, 1).view(*orig_shape[:-1], -1)

        # Get output features
        out_feats = in_feats + delta_feats if self.skip else delta_feats

        return out_feats
