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

from models.build import build_model, MODELS
from models.extensions.deformable.modules import MSDA3D
from models.extensions.deformable.python.insert import pytorch_maps_insert_2d, pytorch_maps_insert_3d
from models.extensions.deformable.python.sample import pytorch_maps_sample_2d, pytorch_maps_sample_3d
from structures.boxes import Boxes


@MODELS.register_module()
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
        val_map = kv_map[:, :, :, self.qk_channels:, :]
        val_map = val_map.view(*val_map.shape[:3], self.num_heads, self.out_channels//self.num_heads, -1)

        # Get output feature map
        attn_weights = self.qk_norm(torch.matmul(query_map, key_map))
        out_feat_map = torch.sum(attn_weights * val_map, dim=-1).view(*val_map.shape[:3], -1)
        out_feat_map = out_feat_map.permute(0, 3, 1, 2)

        return out_feat_map


@MODELS.register_module()
class BoxCrossAttn(nn.Module):
    """
    Class implementing the BoxCrossAttn module.

    Attributes:
        attn (nn.Module): Module performing the actual box-based cross-attention.
    """

    def __init__(self, attn_cfg):
        """
        Initializes the BoxCrossAttn module.

        Args:
            attn_cfg (Dict): Configuration dictionary specifying the underlying box-based cross-attention module.
        """

        # Initialization of default nn.Module
        super().__init__()

        # Build underlying box-based cross-attention module
        self.attn = build_model(attn_cfg)

    def forward(self, in_feats, storage_dict, cum_feats_batch=None):
        """
        Forward method of the BoxCrossAttn module.

        Args:
            in_feats (FloatTensor): Input features of shape [num_feats, in_size].

            storage_dict (Dict): Dictionary storing all kinds of key-value pairs, possibly containing following keys:
                - feat_maps (List): list of size [num_maps] with feature maps of shape [batch_size, feat_size, fH, fW];
                - images (Images): images structure of size [batch_size] containing the batched images;
                - prior_boxes (Boxes): structure with prior bounding boxes of size [num_feats];
                - add_encs (FloatTensor): encodings added to queries and keys of shape [num_feats, in_size].

            cum_feats_batch (LongTensor): Cumulative number of features per batch entry [batch_size+1] (default=None).

        Returns:
            out_feats (FloatTensor): Output features of shape [num_feats, out_size].

        Raises:
            ValueError: Error when storage dictionary does not have key-value pair to obtain sample priors.
            ValueError: Error when storage dictionary does not have key-value pair to obtain sample features.
            ValueError: Error when storage dictionary does not have key-value pair to obtain sample map shapes.
            ValueError: Error when storage dictionary does not have key-value pair to obtain sample map start indices.
        """

        # Get device
        device = in_feats.device

        # Get cumulative number of features per batch entry if missing
        if cum_feats_batch is None:
            cum_feats_batch = torch.tensor([0, len(in_feats)], device=device)

        # Add sample priors to storage dictionary if needed
        if 'sample_priors' not in storage_dict:

            if 'prior_boxes' in storage_dict:
                sample_priors = storage_dict['prior_boxes'].clone()

            else:
                keys = ['prior_boxes']
                error_msg = f"Storage dictionary must contain a key from {keys} to obtain sample priors."
                raise ValueError(error_msg)

            images = storage_dict['images']
            sample_priors_list = []

            for i, image in enumerate(images):
                i0 = cum_feats_batch[i].item()
                i1 = cum_feats_batch[i+1].item()

                sample_priors_i = sample_priors[i0:i1].normalize(image)
                sample_priors_list.append(sample_priors_i)

            sample_priors = Boxes.cat(sample_priors_list)
            sample_priors = sample_priors.to_format('cxcywh')
            storage_dict['sample_priors'] = sample_priors.boxes

        # Add sample features to storage dictionary if needed
        if 'sample_feats' not in storage_dict:

            if 'feat_maps' in storage_dict:
                feat_maps = storage_dict['feat_maps']
                sample_feats = torch.cat([feat_map.flatten(2).permute(0, 2, 1) for feat_map in feat_maps], dim=1)
                storage_dict['sample_feats'] = sample_feats

            else:
                keys = ['feat_maps']
                error_msg = f"Storage dictionary must contain a key from {keys} to obtain sample features."
                raise ValueError(error_msg)

        # Add sample map shapes to storage dictionary if needed
        if 'sample_map_shapes' not in storage_dict:

            if 'feat_maps' in storage_dict:
                feat_maps = storage_dict['feat_maps']
                sample_map_shapes = torch.tensor([feat_map.shape[-2:] for feat_map in feat_maps], device=device)
                storage_dict['sample_map_shapes'] = sample_map_shapes

            else:
                keys = ['feat_maps']
                error_msg = f"Storage dictionary must contain a key from {keys} to obtain sample map shapes."
                raise ValueError(error_msg)

        # Add sample map start indices to storage dictionary if needed
        if 'sample_map_start_ids' not in storage_dict:

            if 'feat_maps' in storage_dict:
                sample_map_start_ids = storage_dict['sample_map_shapes'].prod(dim=1).cumsum(dim=0)[:-1]
                sample_map_start_ids = torch.cat([sample_map_start_ids.new_zeros([1]), sample_map_start_ids], dim=0)
                storage_dict['sample_map_start_ids'] = sample_map_start_ids

            else:
                keys = ['feat_maps']
                error_msg = f"Storage dictionary must contain a key from {keys} to obtain sample map start indices."
                raise ValueError(error_msg)

        # Perform box-based cross-attention
        attn_kwargs = {k: v for k, v in storage_dict.items() if k in ('sample_map_shapes', 'sample_map_start_ids')}
        batch_size = len(cum_feats_batch) - 1
        out_feats_list = []

        for i in range(batch_size):
            i0 = cum_feats_batch[i].item()
            i1 = cum_feats_batch[i+1].item()

            attn_kwargs['sample_priors'] = storage_dict['sample_priors'][i0:i1]
            attn_kwargs['sample_feats'] = storage_dict['sample_feats'][i]

            if storage_dict.get('add_encs', None) is not None:
                attn_kwargs['add_encs'] = storage_dict['add_encs'][i0:i1]

            out_feats_i = self.attn(in_feats[i0:i1], **attn_kwargs)
            out_feats_list.append(out_feats_i)

        out_feats = torch.cat(out_feats_list, dim=0)

        return out_feats


@MODELS.register_module()
class DeformableAttn(nn.Module):
    """
    Class implementing the DeformableAttn module.

    Attributes:
        norm (nn.Module): Optional normalization module of the DeformableAttn module.
        act_fn (nn.Module): Optional module with the activation function of the DeformableAttn module.
        msda (nn.Module): Multi-scale deformable attention module of the DeformableAttn module.
        skip (bool): Boolean indicating whether skip connection is used or not.
        version (int): Integer containing the version of the MSDA module.
    """

    def __init__(self, in_size, sample_size, out_size=-1, norm='', act_fn='', skip=True, version=0, num_heads=8,
                 num_levels=5, num_points=4, rad_pts=4, ang_pts=1, lvl_pts=1, dup_pts=1, qk_size=-1, val_size=-1,
                 val_with_pos=False, norm_z=1.0, sample_insert=False, insert_size=1):
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
            num_heads (int): Integer containing the number of attention heads (default=8).
            num_levels (int): Integer containing the number of map levels to sample from (default=5).
            num_points (int): Integer containing the number of sampling points per head and level (default=4).
            rad_pts (int): Integer containing the number of radial sampling points per head and level (default=4).
            ang_pts (int): Integer containing the number of angular sampling points per head and level (default=1).
            lvl_pts (int): Integer containing the number of level sampling points per head (default=1).
            dup_pts (int): Integer containing the number of duplicate sampling points per head and level (default=1).
            qk_size (int): Size of query and key features (default=-1).
            val_size (int): Size of value features (default=-1).
            val_with_pos (bool): Boolean indicating whether position info is added to value features (default=False).
            norm_z (float): Factor normalizing the sample offsets in the Z-direction (default=1.0).
            sample_insert (bool): Boolean indicating whether to insert sample info in a maps structure (default=False).
            insert_size (int): Integer containing size of features to be inserted during sample insertion (default=1).

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
            self.msda = MSDAv1(in_size, sample_size, out_size, num_heads, num_levels, num_points, val_size)

        elif version == 2:
            self.msda = MSDAv2(in_size, sample_size, out_size, num_heads, num_levels, num_points, val_size,
                               val_with_pos, sample_insert, insert_size)

        elif version == 3:
            self.msda = MSDAv3(in_size, sample_size, out_size, num_heads, num_levels, num_points, qk_size, val_size,
                               val_with_pos, sample_insert, insert_size)

        elif version == 4:
            self.msda = MSDAv4(in_size, sample_size, out_size, num_heads, num_levels, num_points, val_size,
                               val_with_pos, norm_z, sample_insert, insert_size)

        elif version == 5:
            self.msda = MSDAv5(in_size, sample_size, out_size, num_heads, num_levels, rad_pts, ang_pts, dup_pts,
                               val_size, val_with_pos, norm_z, sample_insert, insert_size)

        elif version == 6:
            self.msda = MSDAv6(in_size, sample_size, out_size, num_heads, num_levels, rad_pts, ang_pts, lvl_pts,
                               dup_pts, val_size, val_with_pos, norm_z, sample_insert, insert_size)

        elif version == 7:
            self.msda = MSDA3D(in_size, sample_size, out_size, num_heads, rad_pts, lvl_pts, val_size)

        else:
            error_msg = f"Invalid MSDA version number '{version}'."
            raise ValueError(error_msg)

        # Set skip and version attribute
        self.skip = skip
        self.version = version

    def forward(self, in_feats, sample_priors, sample_feats, sample_map_shapes, sample_map_start_ids,
                add_encs=None, mul_encs=None, map_ids=None, sample_mask=None, storage_dict=None, **kwargs):
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
            map_ids (LongTensor): Map indices of input features of shape [*, num_in_feats] (default=None).
            sample_mask (BoolTensor): Inactive samples mask of shape [*, num_sample_feats] (default=None).
            storage_dict (Dict): Dictionary storing additional arguments (default=None).
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

        sample_priors = sample_priors.view(-1, *sample_priors.shape[-2:])
        sample_feats = sample_feats.view(-1, *sample_feats.shape[-2:])

        if self.version <= 6:
            num_levels = len(sample_map_start_ids)
            sample_priors = sample_priors[:, :, None, :].expand(-1, -1, num_levels, -1)

        if map_ids is not None:
            map_ids = map_ids.view(-1, map_ids.shape[-1])

        msda_args = (delta_feats, sample_priors, sample_feats, sample_map_shapes, sample_map_start_ids)
        msda_kwargs = {'map_ids': map_ids, 'input_padding_mask': sample_mask, 'storage_dict': storage_dict}
        delta_feats = self.msda(*msda_args, **msda_kwargs).view(*orig_shape[:-1], -1)

        # Get output features
        out_feats = in_feats + delta_feats if self.skip else delta_feats

        return out_feats


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


class MSDAv1(nn.Module):
    """
    Class implementing the MSDAv1 module.

    Attributes:
        num_heads (int): Integer containing the number of attention heads.
        num_levels (int): Integer containing the number of map levels to sample from.
        num_points (int): Integer containing the number of sampling points per head and level.

        sampling_offsets (nn.Linear): Module computing the sampling offsets from the input features.
        attn_weights (nn.Linear): Module computing the attention weights from the input features.
        val_proj (nn.Linear): Module computing value features from sample features.
        out_proj (nn.Linear): Module computing output features from weighted value features.
    """

    def __init__(self, in_size, sample_size, out_size=-1, num_heads=8, num_levels=5, num_points=4, val_size=-1):
        """
        Initializes the MSDAv1 module.

        Args:
            in_size (int): Size of input features.
            sample_size (int): Size of sample features.
            out_size (int): Size of output features (default=-1).
            num_heads (int): Integer containing the number of attention heads (default=8).
            num_levels (int): Integer containing the number of map levels to sample from (default=5).
            num_points (int): Integer containing the number of sampling points per head and level (default=4).
            val_size (int): Size of value features (default=-1).

        Raises:
            ValueError: Error when the input feature size does not divide the number of heads.
            ValueError: Error when the value feature size does not divide the number of heads.
        """

        # Initialization of default nn.Module
        super().__init__()

        # Set attributes related to the number of heads, levels and points
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points

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

        # Initialize module computing the unnormalized attention weights
        self.attn_weights = nn.Linear(in_size, num_heads * num_levels * num_points)
        nn.init.zeros_(self.attn_weights.weight)
        nn.init.zeros_(self.attn_weights.bias)

        # Get and check size of value features
        if val_size == -1:
            val_size = in_size

        elif val_size % num_heads != 0:
            error_msg = f"The value feature size ({val_size}) must divide the number of heads ({num_heads})."
            raise ValueError(error_msg)

        # Initialize module computing the value features
        self.val_proj = nn.Linear(sample_size, val_size)
        nn.init.xavier_uniform_(self.val_proj.weight)
        nn.init.zeros_(self.val_proj.bias)

        # Initialize module computing the output features
        out_size = in_size if out_size == -1 else out_size
        self.out_proj = nn.Linear(val_size, out_size)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

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
        attn_weights = self.attn_weights(in_feats).view(*common_shape, self.num_levels * self.num_points)
        attn_weights = F.softmax(attn_weights, dim=3).view(*common_shape, self.num_levels, self.num_points)

        # Get value features
        val_feats = self.val_proj(sample_feats)

        # Apply MSDA function
        val_size = val_feats.shape[-1]
        val_feats = val_feats.view(batch_size, -1, self.num_heads, val_size // self.num_heads)

        msdaf_args = (val_feats, sample_map_shapes, sample_map_start_ids, sample_locations, attn_weights, 64)
        weighted_val_feats = MSDAF.apply(*msdaf_args)

        # Get output features
        out_feats = self.out_proj(weighted_val_feats)

        return out_feats


class MSDAv2(nn.Module):
    """
    Class implementing the MSDAv2 module.

    Attributes:
        num_heads (int): Integer containing the number of attention heads.
        num_levels (int): Integer containing the number of map levels to sample from.
        num_points (int): Integer containing the number of sampling points per head and level.

        sampling_offsets (nn.Linear): Module computing the sampling offsets from the input features.
        val_proj (nn.Linear): Module computing value features from sample features.
        val_pos_encs (nn.Linear): Optional module computing the value position encodings.

        sample_insert (bool): Boolean indicating whether to insert sample info in a maps structure.

        If sample_insert is True:
            insert_weight (nn.Parameter): Parameter containing the weight matrix used during sample insertion.
            insert_bias (nn.Parameter): Parameter containing the bias vector used during sample insertion.

        compute_out_feats (bool): Boolean indicating whether output features should be computed.

        If compute_out_feats is True:
            attn_weights (nn.Linear): Module computing the attention weights from the input features.
            out_proj (nn.Linear): Module computing output features from weighted value features.
    """

    def __init__(self, in_size, sample_size, out_size=-1, num_heads=8, num_levels=5, num_points=4, val_size=-1,
                 val_with_pos=False, sample_insert=False, insert_size=1):
        """
        Initializes the MSDAv2 module.

        Args:
            in_size (int): Size of input features.
            sample_size (int): Size of sample features.
            out_size (int): Size of output features (default=-1).
            num_heads (int): Integer containing the number of attention heads (default=8).
            num_levels (int): Integer containing the number of map levels to sample from (default=5).
            num_points (int): Integer containing the number of sampling points per head and level (default=4).
            val_size (int): Size of value features (default=-1).
            val_with_pos (bool): Boolean indicating whether position info is added to value features (default=False).
            sample_insert (bool): Boolean indicating whether to insert sample info in a maps structure (default=False).
            insert_size (int): Integer containing size of features to be inserted during sample insertion (default=1).

        Raises:
            ValueError: Error when the input feature size does not divide the number of heads.
            ValueError: Error when the value feature size does not divide the number of heads.
        """

        # Initialization of default nn.Module
        super().__init__()

        # Set attributes related to the number of heads, levels and points
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points

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

        # Get and check size of value features
        if val_size == -1:
            val_size = in_size

        elif val_size % num_heads != 0:
            error_msg = f"The value feature size ({val_size}) must divide the number of heads ({num_heads})."
            raise ValueError(error_msg)

        # Initialize module computing the value features
        self.val_proj = nn.Linear(sample_size, val_size)
        nn.init.xavier_uniform_(self.val_proj.weight)
        nn.init.zeros_(self.val_proj.bias)

        # Initialize module computing the value position encodings if requested
        if val_with_pos:
            self.val_pos_encs = nn.Linear(3, val_size // num_heads)

        # Set attributes related to sample insertion
        self.sample_insert = sample_insert

        if sample_insert:
            self.insert_weight = nn.Parameter(torch.zeros(num_heads, val_size // num_heads, insert_size))
            self.insert_bias = nn.Parameter(torch.zeros(num_heads, 1, insert_size))

        # Set attribute determining whether output features should be computed
        self.compute_out_feats = True

        # Initialize module computing the unnormalized attention weights
        self.attn_weights = nn.Linear(in_size, num_heads * num_levels * num_points)
        nn.init.zeros_(self.attn_weights.weight)
        nn.init.zeros_(self.attn_weights.bias)

        # Initialize module computing the output features
        out_size = in_size if out_size == -1 else out_size
        self.out_proj = nn.Linear(val_size, out_size)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def no_out_feats_computation(self):
        """
        Method changing the module to not compute output features.

        Raises:
            RuntimeError: Error when sample insertion is False.
        """

        # Check whether sample insertion is True
        if self.sample_insert:

            # Change attribute that no output features should be computed
            self.compute_out_feats = False

            # Delete all attributes related to the computation of output features
            delattr(self, 'attn_weights')
            delattr(self, 'out_proj')

        else:
            error_msg = "Sample insertion should be True when not computing output features."
            raise RuntimeError(error_msg)

    def forward(self, in_feats, sample_priors, sample_feats, sample_map_shapes, sample_map_start_ids, storage_dict=None,
                **kwargs):
        """
        Forward method of the MSDAv2 module.

        Args:
            in_feats (FloatTensor): Input features of shape [batch_size, num_in_feats, in_size].
            sample_priors (FloatTensor): Sample priors of shape [batch_size, num_in_feats, num_levels, {2, 4}].
            sample_feats (FloatTensor): Sample features of shape [batch_size, num_sample_feats, sample_size].
            sample_map_shapes (LongTensor): Map shapes corresponding to samples of shape [num_levels, 2].
            sample_map_start_ids (LongTensor): Start indices of sample maps of shape [num_levels].
            storage_dict (Dict): Dictionary storing additional arguments (default=None).
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

        # Get value features
        val_feats = self.val_proj(sample_feats)

        # Get sampled value features
        val_size = val_feats.shape[-1]
        val_feats = val_feats.view(batch_size, -1, self.num_heads, val_size // self.num_heads)
        val_feats = val_feats.transpose(1, 2).reshape(batch_size * self.num_heads, -1, val_size // self.num_heads)

        sample_map_shapes = sample_map_shapes.fliplr()
        sample_locations = sample_locations.transpose(1, 2).reshape(batch_size * self.num_heads, -1, 2)

        sample_map_ids = torch.arange(self.num_levels, device=sample_locations.device)
        sample_map_ids = sample_map_ids[None, None, :, None]
        sample_map_ids = sample_map_ids.expand(batch_size * self.num_heads, num_in_feats, -1, self.num_points)
        sample_map_ids = sample_map_ids.reshape(batch_size * self.num_heads, -1)

        sample_args = (val_feats, sample_map_shapes, sample_map_start_ids, sample_locations, sample_map_ids)
        sampled_feats = pytorch_maps_sample_2d(*sample_args)

        sampled_feats = sampled_feats.view(batch_size, self.num_heads, num_in_feats, -1, val_size // self.num_heads)
        sampled_feats = sampled_feats.transpose(1, 2)

        # Get and add position encodings to sampled value features if needed
        if hasattr(self, 'val_pos_encs'):
            sample_xy = 0.5 * sample_offsets / self.num_points
            sample_z = sample_map_ids.view(batch_size, self.num_heads, num_in_feats, -1, self.num_points, 1)
            sample_z = sample_z.transpose(1, 2) / (self.num_levels-1)

            sample_xyz = torch.cat([sample_xy, sample_z], dim=5).flatten(3, 4)
            sampled_feats = sampled_feats + self.val_pos_encs(sample_xyz)

        # Perform sample insertion if needed
        if self.sample_insert:
            insert_feats = sampled_feats.permute(2, 0, 1, 3, 4)
            insert_feats = insert_feats.view(self.num_heads, -1, val_size // self.num_heads)
            insert_feats = torch.bmm(insert_feats, self.insert_weight) + self.insert_bias

            insert_size = insert_feats.shape[2]
            insert_feats = insert_feats.view(self.num_heads, batch_size*num_in_feats, -1, insert_size)
            insert_feats = insert_feats.transpose(0, 1).reshape(batch_size*num_in_feats, -1, insert_size)

            insert_xy = sample_locations.view(batch_size, self.num_heads, num_in_feats, -1, 2)
            insert_xy = insert_xy.transpose(1, 2).reshape(batch_size*num_in_feats, -1, 2)

            insert_map_ids = sample_map_ids.view(batch_size, self.num_heads, num_in_feats, -1)
            insert_map_ids = insert_map_ids.transpose(1, 2).reshape(batch_size*num_in_feats, -1)

            insert_args = (storage_dict['map_feats'], sample_map_shapes, sample_map_start_ids)
            insert_args = (*insert_args, insert_feats, insert_xy, insert_map_ids)

            map_feats = pytorch_maps_insert_2d(*insert_args)
            storage_dict['map_feats'] = map_feats

        # Get output features
        if self.compute_out_feats:

            # Get attention weights
            attn_weights = self.attn_weights(in_feats).view(*common_shape, self.num_levels * self.num_points)
            attn_weights = F.softmax(attn_weights, dim=3)

            # Get weighted value features
            weighted_feats = attn_weights[:, :, :, :, None] * sampled_feats
            weighted_feats = weighted_feats.sum(dim=3).view(batch_size, num_in_feats, val_size)

            # Get non-zero output features
            out_feats = self.out_proj(weighted_feats)

        else:
            out_feats = torch.zeros_like(in_feats)

        return out_feats


class MSDAv3(nn.Module):
    """
    Class implementing the MSDAv3 module.

    Attributes:
        num_heads (int): Integer containing the number of attention heads.
        num_levels (int): Integer containing the number of map levels to sample from.
        num_points (int): Integer containing the number of sampling points per head and level.

        sampling_offsets (nn.Linear): Module computing the sampling offsets from the input features.
        kv_proj (nn.Linear): Module computing key-value features from sample features.
        val_pos_encs (nn.Linear): Optional module computing the value position encodings.

        sample_insert (bool): Boolean indicating whether to insert sample info in a maps structure.

        If sample_insert is True:
            insert_weight (nn.Parameter): Parameter containing the weight matrix used during sample insertion.
            insert_bias (nn.Parameter): Parameter containing the bias vector used during sample insertion.

        compute_out_feats (bool): Boolean indicating whether output features should be computed.

        If compute_out_feats is True:
            query_proj (nn.Linear): Module computing query features from input features.
            point_encs (nn.Parameter): Parameter tensor containing the point encodings.
            out_proj (nn.Linear): Module computing output features from weighted value features.
    """

    def __init__(self, in_size, sample_size, out_size=-1, num_heads=8, num_levels=5, num_points=4, qk_size=-1,
                 val_size=-1, val_with_pos=False, sample_insert=False, insert_size=1):
        """
        Initializes the MSDAv3 module.

        Args:
            in_size (int): Size of input features.
            sample_size (int): Size of sample features.
            out_size (int): Size of output features (default=-1).
            num_heads (int): Integer containing the number of attention heads (default=8).
            num_levels (int): Integer containing the number of map levels to sample from (default=5).
            num_points (int): Integer containing the number of sampling points per head and level (default=4).
            qk_size (int): Size of query and key features (default=-1).
            val_size (int): Size of value features (default=-1).
            val_with_pos (bool): Boolean indicating whether position info is added to value features (default=False).
            sample_insert (bool): Boolean indicating whether to insert sample info in a maps structure (default=False).
            insert_size (int): Integer containing size of features to be inserted during sample insertion (default=1).

        Raises:
            ValueError: Error when the input feature size does not divide the number of heads.
            ValueError: Error when the value feature size does not divide the number of heads.
            ValueError: Error when the query and key feature size does not divide the number of heads.
        """

        # Initialization of default nn.Module
        super().__init__()

        # Set attributes related to the number of heads, levels and points
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points

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

        # Get and check size of value features
        if val_size == -1:
            val_size = in_size

        elif val_size % num_heads != 0:
            error_msg = f"The value feature size ({val_size}) must divide the number of heads ({num_heads})."
            raise ValueError(error_msg)

        # Initialize module computing the key-value features
        kv_size = qk_size + val_size
        self.kv_proj = nn.Linear(sample_size, kv_size)
        nn.init.xavier_uniform_(self.kv_proj.weight)
        nn.init.zeros_(self.kv_proj.bias)

        # Initialize module computing the value position encodings if requested
        if val_with_pos:
            self.val_pos_encs = nn.Linear(3, val_size // num_heads)

        # Set attributes related to sample insertion
        self.sample_insert = sample_insert

        if sample_insert:
            self.insert_weight = nn.Parameter(torch.zeros(num_heads, val_size // num_heads, insert_size))
            self.insert_bias = nn.Parameter(torch.zeros(num_heads, 1, insert_size))

        # Set attribute determining whether output features should be computed
        self.compute_out_feats = True

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

        # Initialize point encodings
        self.point_encs = nn.Parameter(torch.zeros(num_heads, num_levels * num_points, qk_size // num_heads))

        # Initialize module computing the output features
        out_size = in_size if out_size == -1 else out_size
        self.out_proj = nn.Linear(val_size, out_size)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def no_out_feats_computation(self):
        """
        Method changing the module to not compute output features.

        Raises:
            RuntimeError: Error when sample insertion is False.
        """

        # Check whether sample insertion is True
        if self.sample_insert:

            # Change attribute that no output features should be computed
            self.compute_out_feats = False

            # Change key-value projection module to only compute value features
            kv_size, sample_size = self.kv_proj.weight.shape
            qk_size = self.query_proj.bias.shape[0]
            val_size = kv_size - qk_size

            self.kv_proj = nn.Linear(sample_size, val_size)
            nn.init.xavier_uniform_(self.kv_proj.weight)
            nn.init.zeros_(self.kv_proj.bias)

            # Delete all attributes related to the computation of output features
            delattr(self, 'query_proj')
            delattr(self, 'point_encs')
            delattr(self, 'out_proj')

        else:
            error_msg = "Sample insertion should be True when not computing output features."
            raise RuntimeError(error_msg)

    def forward(self, in_feats, sample_priors, sample_feats, sample_map_shapes, sample_map_start_ids, storage_dict=None,
                **kwargs):
        """
        Forward method of the MSDAv3 module.

        Args:
            in_feats (FloatTensor): Input features of shape [batch_size, num_in_feats, in_size].
            sample_priors (FloatTensor): Sample priors of shape [batch_size, num_in_feats, num_levels, {2, 4}].
            sample_feats (FloatTensor): Sample features of shape [batch_size, num_sample_feats, sample_size].
            sample_map_shapes (LongTensor): Map shapes corresponding to samples of shape [num_levels, 2].
            sample_map_start_ids (LongTensor): Start indices of sample maps of shape [num_levels].
            storage_dict (Dict): Dictionary storing additional arguments (default=None).
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

        # Get key-value features
        kv_feats = self.kv_proj(sample_feats)

        # Get sampled key-value features
        kv_size = kv_feats.shape[-1]
        kv_feats = kv_feats.view(batch_size, -1, self.num_heads, kv_size // self.num_heads)
        kv_feats = kv_feats.transpose(1, 2).reshape(batch_size * self.num_heads, -1, kv_size // self.num_heads)

        sample_map_shapes = sample_map_shapes.fliplr()
        sample_locations = sample_locations.transpose(1, 2).reshape(batch_size * self.num_heads, -1, 2)

        sample_map_ids = torch.arange(self.num_levels, device=sample_locations.device)
        sample_map_ids = sample_map_ids[None, None, :, None]
        sample_map_ids = sample_map_ids.expand(batch_size * self.num_heads, num_in_feats, -1, self.num_points)
        sample_map_ids = sample_map_ids.reshape(batch_size * self.num_heads, -1)

        sample_args = (kv_feats, sample_map_shapes, sample_map_start_ids, sample_locations, sample_map_ids)
        sampled_feats = pytorch_maps_sample_2d(*sample_args)

        sampled_feats = sampled_feats.view(batch_size, self.num_heads, num_in_feats, -1, kv_size // self.num_heads)
        sampled_feats = sampled_feats.transpose(1, 2)

        # Get query and sampled key features if needed
        if self.compute_out_feats:

            # Get query features
            query_feats = self.query_proj(in_feats).view(*common_shape, 1, -1)

            # Get sampled key features
            head_qk_size = query_feats.shape[-1]
            sampled_key_feats = sampled_feats[:, :, :, :, :head_qk_size]
            sampled_key_feats = sampled_key_feats + self.point_encs

        # Get sampled value features
        sampled_val_feats = sampled_feats[:, :, :, :, head_qk_size:] if self.compute_out_feats else sampled_feats

        # Get and add position encodings to sampled value features if needed
        if hasattr(self, 'val_pos_encs'):
            sample_xy = 0.5 * sample_offsets / self.num_points
            sample_z = sample_map_ids.view(batch_size, self.num_heads, num_in_feats, -1, self.num_points, 1)
            sample_z = sample_z.transpose(1, 2) / (self.num_levels-1)

            sample_xyz = torch.cat([sample_xy, sample_z], dim=5).flatten(3, 4)
            sampled_val_feats = sampled_val_feats + self.val_pos_encs(sample_xyz)

        # Perform sample insertion if needed
        if self.sample_insert:
            head_val_size = sampled_val_feats.shape[4]
            insert_feats = sampled_val_feats.permute(2, 0, 1, 3, 4)
            insert_feats = insert_feats.view(self.num_heads, -1, head_val_size)
            insert_feats = torch.bmm(insert_feats, self.insert_weight) + self.insert_bias

            insert_size = insert_feats.shape[2]
            insert_feats = insert_feats.view(self.num_heads, batch_size*num_in_feats, -1, insert_size)
            insert_feats = insert_feats.transpose(0, 1).reshape(batch_size*num_in_feats, -1, insert_size)

            insert_xy = sample_locations.view(batch_size, self.num_heads, num_in_feats, -1, 2)
            insert_xy = insert_xy.transpose(1, 2).reshape(batch_size*num_in_feats, -1, 2)

            insert_map_ids = sample_map_ids.view(batch_size, self.num_heads, num_in_feats, -1)
            insert_map_ids = insert_map_ids.transpose(1, 2).reshape(batch_size*num_in_feats, -1)

            insert_args = (storage_dict['map_feats'], sample_map_shapes, sample_map_start_ids)
            insert_args = (*insert_args, insert_feats, insert_xy, insert_map_ids)

            map_feats = pytorch_maps_insert_2d(*insert_args)
            storage_dict['map_feats'] = map_feats

        # Get output features
        if self.compute_out_feats:

            # Get attention weights
            query_feats = query_feats / math.sqrt(head_qk_size)
            attn_weights = torch.matmul(query_feats, sampled_key_feats.transpose(3, 4)).squeeze(dim=3)
            attn_weights = F.softmax(attn_weights, dim=3)

            # Get weighted value features
            weighted_feats = attn_weights[:, :, :, :, None] * sampled_val_feats
            weighted_feats = weighted_feats.sum(dim=3).view(batch_size, num_in_feats, -1)

            # Get non-zero output features
            out_feats = self.out_proj(weighted_feats)

        else:
            out_feats = torch.zeros_like(in_feats)

        return out_feats


class MSDAv4(nn.Module):
    """
    Class implementing the MSDAv4 module.

    Attributes:
        num_heads (int): Integer containing the number of attention heads.
        num_levels (int): Integer containing the number of map levels to sample from.
        num_points (int): Integer containing the number of sampling points per head and level.

        sampling_offsets (nn.Linear): Module computing the sampling offsets from the input features.
        val_proj (nn.Linear): Module computing value features from sample features.
        val_pos_encs (nn.Linear): Optional module computing the value position encodings.
        norm_z (float): Factor normalizing the sample offsets in the Z-direction.

        sample_insert (bool): Boolean indicating whether to insert sample info in a maps structure.

        If sample_insert is True:
            insert_weight (nn.Parameter): Parameter containing the weight matrix used during sample insertion.
            insert_bias (nn.Parameter): Parameter containing the bias vector used during sample insertion.

        compute_out_feats (bool): Boolean indicating whether output features should be computed.

        If compute_out_feats is True:
            attn_weights (nn.Linear): Module computing the attention weights from the input features.
            out_proj (nn.Linear): Module computing output features from weighted value features.
    """

    def __init__(self, in_size, sample_size, out_size=-1, num_heads=8, num_levels=5, num_points=4, val_size=-1,
                 val_with_pos=False, norm_z=1.0, sample_insert=False, insert_size=1):
        """
        Initializes the MSDAv4 module.

        Args:
            in_size (int): Size of input features.
            sample_size (int): Size of sample features.
            out_size (int): Size of output features (default=-1).
            num_heads (int): Integer containing the number of attention heads (default=8).
            num_levels (int): Integer containing the number of map levels to sample from (default=5).
            num_points (int): Integer containing the number of sampling points per head and level (default=4).
            val_size (int): Size of value features (default=-1).
            val_with_pos (bool): Boolean indicating whether position info is added to value features (default=False).
            norm_z (float): Factor normalizing the sample offsets in the Z-direction (default=1.0).
            sample_insert (bool): Boolean indicating whether to insert sample info in a maps structure (default=False).
            insert_size (int): Integer containing size of features to be inserted during sample insertion (default=1).

        Raises:
            ValueError: Error when the input feature size does not divide the number of heads.
            ValueError: Error when the value feature size does not divide the number of heads.
        """

        # Initialization of default nn.Module
        super().__init__()

        # Set attributes related to the number of heads, levels and points
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points

        # Check divisibility input size by number of heads
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

        # Get and check size of value features
        if val_size == -1:
            val_size = in_size

        elif val_size % num_heads != 0:
            error_msg = f"The value feature size ({val_size}) must divide the number of heads ({num_heads})."
            raise ValueError(error_msg)

        # Initialize module computing the value features
        self.val_proj = nn.Linear(sample_size, val_size)
        nn.init.xavier_uniform_(self.val_proj.weight)
        nn.init.zeros_(self.val_proj.bias)

        # Initialize module computing the value position encodings if requested
        if val_with_pos:
            self.val_pos_encs = nn.Linear(3, val_size // num_heads)

        # Set attribute with Z-normalizer
        self.norm_z = norm_z

        # Set attributes related to sample insertion
        self.sample_insert = sample_insert

        if sample_insert:
            self.insert_weight = nn.Parameter(torch.zeros(num_heads, val_size // num_heads, insert_size))
            self.insert_bias = nn.Parameter(torch.zeros(num_heads, 1, insert_size))

        # Set attribute determining whether output features should be computed
        self.compute_out_feats = True

        # Initialize module computing the unnormalized attention weights
        self.attn_weights = nn.Linear(in_size, num_heads * num_levels * num_points)
        nn.init.zeros_(self.attn_weights.weight)
        nn.init.zeros_(self.attn_weights.bias)

        # Initialize module computing the output features
        out_size = in_size if out_size == -1 else out_size
        self.out_proj = nn.Linear(val_size, out_size)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def no_out_feats_computation(self):
        """
        Method changing the module to not compute output features.

        Raises:
            RuntimeError: Error when sample insertion is False.
        """

        # Check whether sample insertion is True
        if self.sample_insert:

            # Change attribute that no output features should be computed
            self.compute_out_feats = False

            # Delete all attributes related to the computation of output features
            delattr(self, 'attn_weights')
            delattr(self, 'out_proj')

        else:
            error_msg = "Sample insertion should be True when not computing output features."
            raise RuntimeError(error_msg)

    def forward(self, in_feats, sample_priors, sample_feats, sample_map_shapes, sample_map_start_ids, storage_dict=None,
                **kwargs):
        """
        Forward method of the MSDAv4 module.

        Args:
            in_feats (FloatTensor): Input features of shape [batch_size, num_in_feats, in_size].
            sample_priors (FloatTensor): Sample priors of shape [batch_size, num_in_feats, num_levels, {2, 4}].
            sample_feats (FloatTensor): Sample features of shape [batch_size, num_sample_feats, sample_size].
            sample_map_shapes (LongTensor): Map shapes corresponding to samples of shape [num_levels, 2].
            sample_map_start_ids (LongTensor): Start indices of sample maps of shape [num_levels].
            storage_dict (Dict): Dictionary storing additional arguments (default=None).
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
            sample_offsets[:, :, :, :, :, 2] = sample_offsets[:, :, :, :, :, 2] / self.norm_z

            sample_locations = torch.cat([sample_priors[:, :, None, :, None, :], sample_z], dim=5)
            sample_locations = sample_locations + sample_offsets

        elif sample_priors.shape[-1] == 4:
            offset_factors = 0.5 * sample_priors[:, :, None, :, None, 2:] / self.num_points
            sample_offsets[:, :, :, :, :, :2] = sample_offsets[:, :, :, :, :, :2] * offset_factors
            sample_offsets[:, :, :, :, :, 2] = sample_offsets[:, :, :, :, :, 2] / self.norm_z

            sample_locations = torch.cat([sample_priors[:, :, None, :, None, :2], sample_z], dim=5)
            sample_locations = sample_locations + sample_offsets

        else:
            error_msg = f"Last dimension of 'sample_priors' must be 2 or 4, but got {sample_priors.shape[-1]}."
            raise ValueError(error_msg)

        # Get value features
        val_feats = self.val_proj(sample_feats)

        # Get sampled value features
        val_size = val_feats.shape[-1]
        val_feats = val_feats.view(batch_size, -1, self.num_heads, val_size // self.num_heads)
        val_feats = val_feats.transpose(1, 2).reshape(batch_size * self.num_heads, -1, val_size // self.num_heads)

        sample_map_shapes = sample_map_shapes.fliplr()
        sample_locations = sample_locations.transpose(1, 2).reshape(batch_size * self.num_heads, -1, 3)

        sampled_feats = pytorch_maps_sample_3d(val_feats, sample_map_shapes, sample_map_start_ids, sample_locations)
        sampled_feats = sampled_feats.view(batch_size, self.num_heads, num_in_feats, -1, val_size // self.num_heads)
        sampled_feats = sampled_feats.transpose(1, 2)

        # Get and add position encodings to sampled value features if needed
        if hasattr(self, 'val_pos_encs'):
            sample_xy = 0.5 * sample_offsets[:, :, :, :, :, :2] / self.num_points
            sample_z = sample_locations[:, :, 2]
            sample_z = sample_z.view(batch_size, self.num_heads, num_in_feats, self.num_levels, self.num_points, 1)
            sample_z = sample_z.transpose(1, 2)

            sample_xyz = torch.cat([sample_xy, sample_z], dim=5).flatten(3, 4)
            sampled_feats = sampled_feats + self.val_pos_encs(sample_xyz)

        # Perform sample insertion if needed
        if self.sample_insert:
            insert_feats = sampled_feats.permute(2, 0, 1, 3, 4)
            insert_feats = insert_feats.view(self.num_heads, -1, val_size // self.num_heads)
            insert_feats = torch.bmm(insert_feats, self.insert_weight) + self.insert_bias

            insert_size = insert_feats.shape[2]
            insert_feats = insert_feats.view(self.num_heads, batch_size*num_in_feats, -1, insert_size)
            insert_feats = insert_feats.transpose(0, 1).reshape(batch_size*num_in_feats, -1, insert_size)

            insert_xyz = sample_locations.view(batch_size, self.num_heads, num_in_feats, -1, 3)
            insert_xyz = insert_xyz.transpose(1, 2).reshape(batch_size*num_in_feats, -1, 3)

            insert_args = (storage_dict['map_feats'], sample_map_shapes, sample_map_start_ids)
            insert_args = (*insert_args, insert_feats, insert_xyz)

            map_feats = pytorch_maps_insert_3d(*insert_args)
            storage_dict['map_feats'] = map_feats

        # Get output features
        if self.compute_out_feats:

            # Get attention weights
            attn_weights = self.attn_weights(in_feats).view(*common_shape, self.num_levels * self.num_points)
            attn_weights = F.softmax(attn_weights, dim=3)

            # Get weighted value features
            weighted_feats = attn_weights[:, :, :, :, None] * sampled_feats
            weighted_feats = weighted_feats.sum(dim=3).view(batch_size, num_in_feats, val_size)

            # Get non-zero output features
            out_feats = self.out_proj(weighted_feats)

        else:
            out_feats = torch.zeros_like(in_feats)

        return out_feats


class MSDAv5(nn.Module):
    """
    Class implementing the MSDAv5 module.

    Attributes:
        num_heads (int): Integer containing the number of attention heads.
        num_levels (int): Integer containing the number of map levels to sample from.
        num_points (int): Integer containing the number of sampling points per head and level.
        rad_pts (int): Integer containing the number of radial sampling points per head and level.

        sampling_offsets (nn.Linear): Module computing the sampling offsets from the input features.
        val_proj (nn.Linear): Module computing value features from sample features.
        val_pos_encs (nn.Linear): Optional module computing the value position encodings.
        norm_z (float): Factor normalizing the sample offsets in the Z-direction.

        sample_insert (bool): Boolean indicating whether to insert sample info in a maps structure.

        If sample_insert is True:
            insert_weight (nn.Parameter): Parameter containing the weight matrix used during sample insertion.
            insert_bias (nn.Parameter): Parameter containing the bias vector used during sample insertion.

        compute_out_feats (bool): Boolean indicating whether output features should be computed.

        If compute_out_feats is True:
            attn_weights (nn.Linear): Module computing the attention weights from the input features.
            out_proj (nn.Linear): Module computing output features from weighted value features.
    """

    def __init__(self, in_size, sample_size, out_size=-1, num_heads=8, num_levels=5, rad_pts=4, ang_pts=1, dup_pts=1,
                 val_size=-1, val_with_pos=False, norm_z=1.0, sample_insert=False, insert_size=1):
        """
        Initializes the MSDAv5 module.

        Args:
            in_size (int): Size of input features.
            sample_size (int): Size of sample features.
            out_size (int): Size of output features (default=-1).
            num_heads (int): Integer containing the number of attention heads (default=8).
            num_levels (int): Integer containing the number of map levels to sample from (default=5).
            rad_pts (int): Integer containing the number of radial sampling points per head and level (default=4).
            ang_pts (int): Integer containing the number of angular sampling points per head and level (default=1).
            dup_pts (int): Integer containing the number of duplicate sampling points per head and level (default=1).
            val_size (int): Size of value features (default=-1).
            val_with_pos (bool): Boolean indicating whether position info is added to value features (default=False).
            norm_z (float): Factor normalizing the sample offsets in the Z-direction (default=1.0).
            sample_insert (bool): Boolean indicating whether to insert sample info in a maps structure (default=False).
            insert_size (int): Integer containing size of features to be inserted during sample insertion (default=1).

        Raises:
            ValueError: Error when the input feature size does not divide the number of heads.
            ValueError: Error when the value feature size does not divide the number of heads.
        """

        # Initialization of default nn.Module
        super().__init__()

        # Get number of sampling points per head and level
        num_points = rad_pts * ang_pts * dup_pts

        # Set attributes related to the number of heads, levels and points
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.rad_pts = rad_pts

        # Check divisibility input size by number of heads
        if in_size % num_heads != 0:
            error_msg = f"The input feature size ({in_size}) must divide the number of heads ({num_heads})."
            raise ValueError(error_msg)

        # Initialize module computing the sample offsets
        self.sampling_offsets = nn.Linear(in_size, num_heads * num_levels * num_points * 3)
        nn.init.zeros_(self.sampling_offsets.weight)

        thetas = torch.arange(num_heads * ang_pts, dtype=torch.float) - 0.5*(ang_pts-1)
        thetas = thetas * (2.0 * math.pi / (num_heads * ang_pts))

        grid_init = torch.stack([thetas.cos(), thetas.sin(), torch.zeros_like(thetas)], dim=1)
        grid_init = grid_init / grid_init.abs().max(dim=1, keepdim=True)[0]
        grid_init = grid_init.view(num_heads, 1, 1, ang_pts, 1, 3).repeat(1, num_levels, rad_pts, 1, dup_pts, 1)

        sizes = torch.arange(1, rad_pts+1, dtype=torch.float).view(1, 1, rad_pts, 1, 1, 1)
        grid_init = sizes * grid_init
        self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))

        # Get and check size of value features
        if val_size == -1:
            val_size = in_size

        elif val_size % num_heads != 0:
            error_msg = f"The value feature size ({val_size}) must divide the number of heads ({num_heads})."
            raise ValueError(error_msg)

        # Initialize module computing the value features
        self.val_proj = nn.Linear(sample_size, val_size)
        nn.init.xavier_uniform_(self.val_proj.weight)
        nn.init.zeros_(self.val_proj.bias)

        # Initialize module computing the value position encodings if requested
        if val_with_pos:
            self.val_pos_encs = nn.Linear(3, val_size // num_heads)

        # Set attribute with Z-normalizer
        self.norm_z = norm_z

        # Set attributes related to sample insertion
        self.sample_insert = sample_insert

        if sample_insert:
            self.insert_weight = nn.Parameter(torch.zeros(num_heads, val_size // num_heads, insert_size))
            self.insert_bias = nn.Parameter(torch.zeros(num_heads, 1, insert_size))

        # Set attribute determining whether output features should be computed
        self.compute_out_feats = True

        # Initialize module computing the unnormalized attention weights
        self.attn_weights = nn.Linear(in_size, num_heads * num_levels * num_points)
        nn.init.zeros_(self.attn_weights.weight)

        if dup_pts == 1:
            nn.init.zeros_(self.attn_weights.bias)

        else:
            attn_bias = torch.arange(dup_pts) / (dup_pts-1) - 0.5
            attn_bias = attn_bias[None, :].expand(num_heads * num_levels * rad_pts * ang_pts, -1)
            self.attn_weights.bias = nn.Parameter(attn_bias.reshape(-1))

        # Initialize module computing the output features
        out_size = in_size if out_size == -1 else out_size
        self.out_proj = nn.Linear(val_size, out_size)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def no_out_feats_computation(self):
        """
        Method changing the module to not compute output features.

        Raises:
            RuntimeError: Error when sample insertion is False.
        """

        # Check whether sample insertion is True
        if self.sample_insert:

            # Change attribute that no output features should be computed
            self.compute_out_feats = False

            # Delete all attributes related to the computation of output features
            delattr(self, 'attn_weights')
            delattr(self, 'out_proj')

        else:
            error_msg = "Sample insertion should be True when not computing output features."
            raise RuntimeError(error_msg)

    def forward(self, in_feats, sample_priors, sample_feats, sample_map_shapes, sample_map_start_ids, storage_dict=None,
                **kwargs):
        """
        Forward method of the MSDAv5 module.

        Args:
            in_feats (FloatTensor): Input features of shape [batch_size, num_in_feats, in_size].
            sample_priors (FloatTensor): Sample priors of shape [batch_size, num_in_feats, num_levels, {2, 4}].
            sample_feats (FloatTensor): Sample features of shape [batch_size, num_sample_feats, sample_size].
            sample_map_shapes (LongTensor): Map shapes corresponding to samples of shape [num_levels, 2].
            sample_map_start_ids (LongTensor): Start indices of sample maps of shape [num_levels].
            storage_dict (Dict): Dictionary storing additional arguments (default=None).
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
            sample_offsets[:, :, :, :, :, 2] = sample_offsets[:, :, :, :, :, 2] / self.norm_z

            sample_locations = torch.cat([sample_priors[:, :, None, :, None, :], sample_z], dim=5)
            sample_locations = sample_locations + sample_offsets

        elif sample_priors.shape[-1] == 4:
            offset_factors = 0.5 * sample_priors[:, :, None, :, None, 2:] / self.rad_pts
            sample_offsets[:, :, :, :, :, :2] = sample_offsets[:, :, :, :, :, :2] * offset_factors
            sample_offsets[:, :, :, :, :, 2] = sample_offsets[:, :, :, :, :, 2] / self.norm_z

            sample_locations = torch.cat([sample_priors[:, :, None, :, None, :2], sample_z], dim=5)
            sample_locations = sample_locations + sample_offsets

        else:
            error_msg = f"Last dimension of 'sample_priors' must be 2 or 4, but got {sample_priors.shape[-1]}."
            raise ValueError(error_msg)

        # Get value features
        val_feats = self.val_proj(sample_feats)

        # Get sampled value features
        val_size = val_feats.shape[-1]
        val_feats = val_feats.view(batch_size, -1, self.num_heads, val_size // self.num_heads)
        val_feats = val_feats.transpose(1, 2).reshape(batch_size * self.num_heads, -1, val_size // self.num_heads)

        sample_map_shapes = sample_map_shapes.fliplr()
        sample_locations = sample_locations.transpose(1, 2).reshape(batch_size * self.num_heads, -1, 3)

        sampled_feats = pytorch_maps_sample_3d(val_feats, sample_map_shapes, sample_map_start_ids, sample_locations)
        sampled_feats = sampled_feats.view(batch_size, self.num_heads, num_in_feats, -1, val_size // self.num_heads)
        sampled_feats = sampled_feats.transpose(1, 2)

        # Get and add position encodings to sampled value features if needed
        if hasattr(self, 'val_pos_encs'):
            sample_xy = 0.5 * sample_offsets[:, :, :, :, :, :2] / self.rad_pts
            sample_z = sample_locations[:, :, 2]
            sample_z = sample_z.view(batch_size, self.num_heads, num_in_feats, self.num_levels, self.num_points, 1)
            sample_z = sample_z.transpose(1, 2)

            sample_xyz = torch.cat([sample_xy, sample_z], dim=5).flatten(3, 4)
            sampled_feats = sampled_feats + self.val_pos_encs(sample_xyz)

        # Perform sample insertion if needed
        if self.sample_insert:
            insert_feats = sampled_feats.permute(2, 0, 1, 3, 4)
            insert_feats = insert_feats.view(self.num_heads, -1, val_size // self.num_heads)
            insert_feats = torch.bmm(insert_feats, self.insert_weight) + self.insert_bias

            insert_size = insert_feats.shape[2]
            insert_feats = insert_feats.view(self.num_heads, batch_size*num_in_feats, -1, insert_size)
            insert_feats = insert_feats.transpose(0, 1).reshape(batch_size*num_in_feats, -1, insert_size)

            insert_xyz = sample_locations.view(batch_size, self.num_heads, num_in_feats, -1, 3)
            insert_xyz = insert_xyz.transpose(1, 2).reshape(batch_size*num_in_feats, -1, 3)

            insert_args = (storage_dict['map_feats'], sample_map_shapes, sample_map_start_ids)
            insert_args = (*insert_args, insert_feats, insert_xyz)

            map_feats = pytorch_maps_insert_3d(*insert_args)
            storage_dict['map_feats'] = map_feats

        # Get output features
        if self.compute_out_feats:

            # Get attention weights
            attn_weights = self.attn_weights(in_feats).view(*common_shape, self.num_levels * self.num_points)
            attn_weights = F.softmax(attn_weights, dim=3)

            # Get weighted value features
            weighted_feats = attn_weights[:, :, :, :, None] * sampled_feats
            weighted_feats = weighted_feats.sum(dim=3).view(batch_size, num_in_feats, val_size)

            # Get non-zero output features
            out_feats = self.out_proj(weighted_feats)

        else:
            out_feats = torch.zeros_like(in_feats)

        return out_feats


class MSDAv6(nn.Module):
    """
    Class implementing the MSDAv6 module.

    Attributes:
        num_heads (int): Integer containing the number of attention heads.
        num_points (int): Integer containing the number of sampling points per head.
        rad_pts (int): Integer containing the number of radial sampling points per head and level.

        sampling_offsets (nn.Linear): Module computing the sampling offsets from the input features.
        val_proj (nn.Linear): Module computing value features from sample features.
        val_pos_encs (nn.Linear): Optional module computing the value position encodings.
        norm_z (float): Factor normalizing the sample offsets in the Z-direction.

        sample_insert (bool): Boolean indicating whether to insert sample info in a maps structure.

        If sample_insert is True:
            insert_weight (nn.Parameter): Parameter containing the weight matrix used during sample insertion.
            insert_bias (nn.Parameter): Parameter containing the bias vector used during sample insertion.

        compute_out_feats (bool): Boolean indicating whether output features should be computed.

        If compute_out_feats is True:
            attn_weights (nn.Linear): Module computing the attention weights from the input features.
            out_proj (nn.Linear): Module computing output features from weighted value features.
    """

    def __init__(self, in_size, sample_size, out_size=-1, num_heads=8, num_levels=5, rad_pts=4, ang_pts=1, lvl_pts=1,
                 dup_pts=1, val_size=-1, val_with_pos=False, norm_z=1.0, sample_insert=False, insert_size=1):
        """
        Initializes the MSDAv6 module.

        Args:
            in_size (int): Size of input features.
            sample_size (int): Size of sample features.
            out_size (int): Size of output features (default=-1).
            num_heads (int): Integer containing the number of attention heads (default=8).
            num_levels (int): Integer containing the number of map levels to sample from (default=5).
            rad_pts (int): Integer containing the number of radial sampling points per head and level (default=4).
            ang_pts (int): Integer containing the number of angular sampling points per head and level (default=1).
            lvl_pts (int): Integer containing the number of level sampling points per head (default=1).
            dup_pts (int): Integer containing the number of duplicate sampling points per head and level (default=1).
            val_size (int): Size of value features (default=-1).
            val_with_pos (bool): Boolean indicating whether position info is added to value features (default=False).
            norm_z (float): Factor normalizing the sample offsets in the Z-direction (default=1.0).
            sample_insert (bool): Boolean indicating whether to insert sample info in a maps structure (default=False).
            insert_size (int): Integer containing size of features to be inserted during sample insertion (default=1).

        Raises:
            ValueError: Error when the input feature size does not divide the number of heads.
            ValueError: Error when the value feature size does not divide the number of heads.
        """

        # Initialization of default nn.Module
        super().__init__()

        # Get number of sampling points per head
        num_points = rad_pts * ang_pts * lvl_pts * dup_pts

        # Set attributes related to the number of heads and points
        self.num_heads = num_heads
        self.num_points = num_points
        self.rad_pts = rad_pts

        # Check divisibility input size by number of heads
        if in_size % num_heads != 0:
            error_msg = f"The input feature size ({in_size}) must divide the number of heads ({num_heads})."
            raise ValueError(error_msg)

        # Initialize module computing the sample offsets
        self.sampling_offsets = nn.Linear(in_size, num_heads * num_points * 3)
        nn.init.zeros_(self.sampling_offsets.weight)

        thetas = torch.arange(num_heads * ang_pts, dtype=torch.float) - 0.5*(ang_pts-1)
        thetas = thetas * (2.0 * math.pi / (num_heads * ang_pts))

        grid_init = torch.stack([thetas.cos(), thetas.sin(), torch.zeros_like(thetas)], dim=1)
        grid_init = grid_init / grid_init.abs().max(dim=1, keepdim=True)[0]

        grid_init = grid_init[:, None, :].repeat(1, lvl_pts, 1)
        level_offsets = torch.arange(lvl_pts, dtype=torch.float) - 0.5*(lvl_pts-1)
        grid_init[:, :, 2] = grid_init[:, :, 2] + level_offsets[None, :] * norm_z / (num_levels-1)
        grid_init = grid_init.view(num_heads, 1, ang_pts, lvl_pts, 1, 3).repeat(1, rad_pts, 1, 1, dup_pts, 1)

        sizes = torch.arange(1, rad_pts+1, dtype=torch.float).view(1, rad_pts, 1, 1, 1, 1)
        grid_init[:, :, :, :, :, :2] = sizes * grid_init[:, :, :, :, :, :2]
        self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))

        # Get and check size of value features
        if val_size == -1:
            val_size = in_size

        elif val_size % num_heads != 0:
            error_msg = f"The value feature size ({val_size}) must divide the number of heads ({num_heads})."
            raise ValueError(error_msg)

        # Initialize module computing the value features
        self.val_proj = nn.Linear(sample_size, val_size)
        nn.init.xavier_uniform_(self.val_proj.weight)
        nn.init.zeros_(self.val_proj.bias)

        # Initialize module computing the value position encodings if requested
        if val_with_pos:
            self.val_pos_encs = nn.Linear(3, val_size // num_heads)

        # Set attribute with Z-normalizer
        self.norm_z = norm_z

        # Set attributes related to sample insertion
        self.sample_insert = sample_insert

        if sample_insert:
            self.insert_weight = nn.Parameter(torch.zeros(num_heads, val_size // num_heads, insert_size))
            self.insert_bias = nn.Parameter(torch.zeros(num_heads, 1, insert_size))

        # Set attribute determining whether output features should be computed
        self.compute_out_feats = True

        # Initialize module computing the unnormalized attention weights
        self.attn_weights = nn.Linear(in_size, num_heads * num_points)
        nn.init.zeros_(self.attn_weights.weight)

        if dup_pts == 1:
            nn.init.zeros_(self.attn_weights.bias)

        else:
            attn_bias = torch.arange(dup_pts) / (dup_pts-1) - 0.5
            attn_bias = attn_bias[None, :].expand(num_heads * rad_pts * ang_pts * lvl_pts, -1)
            self.attn_weights.bias = nn.Parameter(attn_bias.reshape(-1))

        # Initialize module computing the output features
        out_size = in_size if out_size == -1 else out_size
        self.out_proj = nn.Linear(val_size, out_size)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def no_out_feats_computation(self):
        """
        Method changing the module to not compute output features.

        Raises:
            RuntimeError: Error when sample insertion is False.
        """

        # Check whether sample insertion is True
        if self.sample_insert:

            # Change attribute that no output features should be computed
            self.compute_out_feats = False

            # Delete all attributes related to the computation of output features
            delattr(self, 'attn_weights')
            delattr(self, 'out_proj')

        else:
            error_msg = "Sample insertion should be True when not computing output features."
            raise RuntimeError(error_msg)

    def forward(self, in_feats, sample_priors, sample_feats, sample_map_shapes, sample_map_start_ids, map_ids=None,
                storage_dict=None, **kwargs):
        """
        Forward method of the MSDAv6 module.

        Args:
            in_feats (FloatTensor): Input features of shape [batch_size, num_in_feats, in_size].
            sample_priors (FloatTensor): Sample priors of shape [batch_size, num_in_feats, num_levels, {2, 4}].
            sample_feats (FloatTensor): Sample features of shape [batch_size, num_sample_feats, sample_size].
            sample_map_shapes (LongTensor): Map shapes corresponding to samples of shape [num_levels, 2].
            sample_map_start_ids (LongTensor): Start indices of sample maps of shape [num_levels].
            map_ids (LongTensor): Map indices of input features of shape [batch_size, num_in_feats] (default=None).
            storage_dict (Dict): Dictionary storing additional arguments (default=None).
            kwargs (Dict): Dictionary of keyword arguments not used by this module.

        Returns:
            out_feats (FloatTensor): Output features of shape [batch_size, num_in_feats, out_size].

        Raises:
            ValueError: Error when no map indices are provided.
            ValueError: Error when the last dimension of 'sample_priors' is different from 2 or 4.
        """

        # Get shapes of input tensors
        batch_size, num_in_feats = in_feats.shape[:2]
        common_shape = (batch_size, num_in_feats, self.num_heads)
        num_levels = len(sample_map_shapes)

        # Get sample offsets
        sample_offsets = self.sampling_offsets(in_feats).view(*common_shape, self.num_points, 3)

        # Get sample locations
        sample_priors = sample_priors[:, :, 0, :]

        if map_ids is None:
            error_msg = "Map indices must be provided but are missing."
            raise ValueError(error_msg)

        sample_z = map_ids / (num_levels-1)
        sample_z = sample_z[:, :, None, None, None]

        if sample_priors.shape[-1] == 2:
            offset_normalizers = sample_map_shapes.fliplr()[map_ids, None, None, :]
            sample_offsets[:, :, :, :, :2] = sample_offsets[:, :, :, :, :2] / offset_normalizers
            sample_offsets[:, :, :, :, 2] = sample_offsets[:, :, :, :, 2] / self.norm_z

            sample_locations = torch.cat([sample_priors[:, :, None, None, :], sample_z], dim=4)
            sample_locations = sample_locations + sample_offsets

        elif sample_priors.shape[-1] == 4:
            offset_factors = 0.5 * sample_priors[:, :, None, None, 2:] / self.rad_pts
            sample_offsets[:, :, :, :, :2] = sample_offsets[:, :, :, :, :2] * offset_factors
            sample_offsets[:, :, :, :, 2] = sample_offsets[:, :, :, :, 2] / self.norm_z

            sample_locations = torch.cat([sample_priors[:, :, None, None, :2], sample_z], dim=4)
            sample_locations = sample_locations + sample_offsets

        else:
            error_msg = f"Last dimension of 'sample_priors' must be 2 or 4, but got {sample_priors.shape[-1]}."
            raise ValueError(error_msg)

        # Get value features
        val_feats = self.val_proj(sample_feats)

        # Get sampled value features
        val_size = val_feats.shape[-1]
        val_feats = val_feats.view(batch_size, -1, self.num_heads, val_size // self.num_heads)
        val_feats = val_feats.transpose(1, 2).reshape(batch_size * self.num_heads, -1, val_size // self.num_heads)

        sample_map_shapes = sample_map_shapes.fliplr()
        sample_locations = sample_locations.transpose(1, 2).reshape(batch_size * self.num_heads, -1, 3)

        sampled_feats = pytorch_maps_sample_3d(val_feats, sample_map_shapes, sample_map_start_ids, sample_locations)
        sampled_feats = sampled_feats.view(batch_size, self.num_heads, num_in_feats, -1, val_size // self.num_heads)
        sampled_feats = sampled_feats.transpose(1, 2)

        # Get and add position encodings to sampled value features if needed
        if hasattr(self, 'val_pos_encs'):
            sample_xy = 0.5 * sample_offsets[:, :, :, :, :2] / self.rad_pts
            sample_z = sample_locations[:, :, 2]
            sample_z = sample_z.view(batch_size, self.num_heads, num_in_feats, self.num_points, 1)
            sample_z = sample_z.transpose(1, 2)

            sample_xyz = torch.cat([sample_xy, sample_z], dim=4)
            sampled_feats = sampled_feats + self.val_pos_encs(sample_xyz)

        # Perform sample insertion if needed
        if self.sample_insert:
            insert_feats = sampled_feats.permute(2, 0, 1, 3, 4)
            insert_feats = insert_feats.view(self.num_heads, -1, val_size // self.num_heads)
            insert_feats = torch.bmm(insert_feats, self.insert_weight) + self.insert_bias

            insert_size = insert_feats.shape[2]
            insert_feats = insert_feats.view(self.num_heads, batch_size*num_in_feats, -1, insert_size)
            insert_feats = insert_feats.transpose(0, 1).reshape(batch_size*num_in_feats, -1, insert_size)

            insert_xyz = sample_locations.view(batch_size, self.num_heads, num_in_feats, -1, 3)
            insert_xyz = insert_xyz.transpose(1, 2).reshape(batch_size*num_in_feats, -1, 3)

            insert_args = (storage_dict['map_feats'], sample_map_shapes, sample_map_start_ids)
            insert_args = (*insert_args, insert_feats, insert_xyz)

            map_feats = pytorch_maps_insert_3d(*insert_args)
            storage_dict['map_feats'] = map_feats

        # Get output features
        if self.compute_out_feats:

            # Get attention weights
            attn_weights = self.attn_weights(in_feats).view(*common_shape, self.num_points)
            attn_weights = F.softmax(attn_weights, dim=3)

            # Get weighted value features
            weighted_feats = attn_weights[:, :, :, :, None] * sampled_feats
            weighted_feats = weighted_feats.sum(dim=3).view(batch_size, num_in_feats, val_size)

            # Get non-zero output features
            out_feats = self.out_proj(weighted_feats)

        else:
            out_feats = torch.zeros_like(in_feats)

        return out_feats


@MODELS.register_module()
class ParticleAttn(nn.Module):
    """
    Class implementing the ParticleAttn module.

    Attributes:
        norm (nn.Module): Optional normalization module of the ParticleAttn module.
        act_fn (nn.Module): Optional module with the activation function of the ParticleAttn module.
        pa (nn.Module): Module performing the actual particle attention of the ParticleAttn module.
        skip (bool): Boolean indicating whether skip connection is used or not.
    """

    def __init__(self, in_size, sample_size, out_size=-1, norm='', act_fn='', skip=True, version=1, num_heads=8,
                 num_levels=5, num_points=4, qk_size=-1, val_size=-1, val_with_pos=False, step_size=-1,
                 step_norm_xy='map', step_norm_z=1.0, num_particles=20):
        """
        Initializes the ParticleAttn module.

        Args:
            in_size (int): Size of input features.
            sample_size (int): Size of sample features.
            out_size (int): Size of output features (default=-1).
            norm (str): String containing the type of normalization (default='').
            act_fn (str): String containing the type of activation function (default='').
            skip (bool): Boolean indicating whether skip connection is used or not (default=True).
            version (int): Integer containing the version of the PA module (default=1).
            num_heads (int): Integer containing the number of attention heads (default=8).
            num_levels (int): Integer containing the number of map levels to sample from (default=5).
            num_points (int): Integer containing the number of sampling points per head and level (default=4).
            qk_size (int): Size of query and key features (default=-1).
            val_size (int): Size of value features (default=-1).
            val_with_pos (bool): Boolean indicating whether position info is added to value features (default=False).
            step_size (float): Size of the sample steps relative to the sample step normalization (default=-1).
            step_norm_xy (str): String containing the normalization type of XY-direction sample steps (default=map).
            step_norm_z (float): Value normalizing the sample steps in the Z-direction (default=1.0).
            num_particles (int): Integer containing the number of particles per head (default=20).

        Raises:
            ValueError: Error when unsupported type of normalization is provided.
            ValueError: Error when unsupported type of activation function is provided.
            ValueError: Error when input and output feature sizes are different when skip connection is used.
            ValueError: Error when the output feature size is not specified when no skip connection is used.
            ValueError: Error when invalid PA version number is provided.
        """

        # Initialization of default nn.Module
        super().__init__()

        # Initialization of optional normalization module
        if not norm:
            pass
        elif norm == 'layer':
            self.norm = nn.LayerNorm(in_size)
        else:
            error_msg = f"The ParticleAttn module does not support the '{norm}' normalization type."
            raise ValueError(error_msg)

        # Initialization of optional module with activation function
        if not act_fn:
            pass
        elif act_fn == 'gelu':
            self.act_fn = nn.GELU()
        elif act_fn == 'relu':
            self.act_fn = nn.ReLU(inplace=False) if not norm and skip else nn.ReLU(inplace=True)
        else:
            error_msg = f"The ParticleAttn module does not support the '{act_fn}' activation function."

        # Get and check output feature size
        if skip and out_size == -1:
            out_size = in_size

        elif skip and in_size != out_size:
            error_msg = f"Input ({in_size}) and output ({out_size}) sizes must match when skip connection is used."
            raise ValueError(error_msg)

        elif not skip and out_size == -1:
            error_msg = "The output feature size must be specified when no skip connection is used."
            raise ValueError(error_msg)

        # Initialization of actual particle attention module
        if version == 1:
            self.pa = PAv1(in_size, sample_size, out_size, num_heads, num_levels, num_points, val_size, val_with_pos,
                           step_size, step_norm_xy)

        elif version == 2:
            self.pa = PAv2(in_size, sample_size, out_size, num_heads, num_levels, num_points, val_size, val_with_pos,
                           step_size, step_norm_xy)

        elif version == 3:
            self.pa = PAv3(in_size, sample_size, out_size, num_heads, num_levels, num_points, val_size, val_with_pos,
                           qk_size, step_size, step_norm_xy)

        elif version == 4:
            self.pa = PAv4(in_size, sample_size, out_size, num_heads, num_levels, num_points, val_size, val_with_pos,
                           qk_size, step_size, step_norm_xy)

        elif version == 5:
            self.pa = PAv5(in_size, sample_size, out_size, num_heads, num_levels, num_points, val_size, val_with_pos)

        elif version == 6:
            self.pa = PAv6(in_size, sample_size, out_size, num_heads, num_levels, num_points, val_size, val_with_pos,
                           qk_size, step_size, step_norm_xy)

        elif version == 7:
            self.pa = PAv7(in_size, sample_size, out_size, num_heads, num_levels, num_points, val_size, val_with_pos,
                           qk_size, step_size, step_norm_xy, step_norm_z)

        elif version == 8:
            self.pa = PAv8(in_size, sample_size, out_size, num_heads, val_size, val_with_pos, qk_size, step_size,
                           step_norm_xy, step_norm_z, num_particles)

        elif version == 9:
            self.pa = PAv9(in_size, sample_size, out_size, num_heads, num_levels, num_points, val_size, val_with_pos,
                           qk_size, step_size, step_norm_xy)

        else:
            error_msg = f"Invalid PA version number '{version}'."
            raise ValueError(error_msg)

        # Set skip attribute
        self.skip = skip

    def forward(self, in_feats, sample_priors, sample_feats, sample_map_shapes, sample_map_start_ids, storage_dict,
                add_encs=None, mul_encs=None, map_ids=None, **kwargs):
        """
        Forward method of the ParticleAttn module.

        Args:
            in_feats (FloatTensor): Input features of shape [*, num_in_feats, in_size].
            sample_priors (FloatTensor): Sample priors of shape [*, num_in_feats, {2, 4}].
            sample_feats (FloatTensor): Sample features of shape [*, num_sample_feats, sample_size].
            sample_map_shapes (LongTensor): Map shapes corresponding to samples of shape [num_levels, 2].
            sample_map_start_ids (LongTensor): Start indices of sample maps of shape [num_levels].
            storage_dict (Dict): Dictionary storing additional arguments such as the sample locations.
            add_encs (FloatTensor): Encodings added to queries of shape [*, num_in_feats, in_size] (default=None).
            mul_encs (FloatTensor): Encodings multiplied by queries of shape [*, num_in_feats, in_size] (default=None).
            map_ids (LongTensor): Map indices of input features of shape [*, num_in_feats] (default=None).
            kwargs (Dict): Dictionary of keyword arguments not used by this module.

        Returns:
            out_feats (FloatTensor): Output features of shape [*, num_in_feats, out_size].
        """

        # Apply optional normalization and activation function modules
        delta_feats = in_feats
        delta_feats = self.norm(delta_feats) if hasattr(self, 'norm') else delta_feats
        delta_feats = self.act_fn(delta_feats) if hasattr(self, 'act_fn') else delta_feats

        # Apply actual particle attention module
        orig_shape = delta_feats.shape
        delta_feats = delta_feats.view(-1, *orig_shape[-2:])

        if mul_encs is not None:
            delta_feats = delta_feats * mul_encs.view(-1, *orig_shape[-2:])

        if add_encs is not None:
            delta_feats = delta_feats + add_encs.view(-1, *orig_shape[-2:])

        sample_priors = sample_priors.view(-1, *sample_priors.shape[-2:])
        sample_feats = sample_feats.view(-1, *sample_feats.shape[-2:])

        if map_ids is not None:
            map_ids = map_ids.view(-1, map_ids.shape[-1])

        pa_args = (delta_feats, sample_priors, sample_feats, sample_map_shapes, sample_map_start_ids, storage_dict)
        pa_kwargs = {'map_ids': map_ids}
        delta_feats = self.pa(*pa_args, **pa_kwargs)

        # Get output features
        delta_feats = delta_feats.view(*orig_shape[:-1], -1)
        out_feats = in_feats + delta_feats if self.skip else delta_feats

        return out_feats


class PAv1(nn.Module):
    """
    Class implementing the PAv1 module.

    Attributes:
        num_heads (int): Integer containing the number of attention heads.
        num_levels (int): Integer containing the number of map levels to sample from.
        num_points (int): Integer containing the number of sampling points per head and level.

        val_proj (nn.Linear): Module computing value features from sample features.
        val_pos_encs (nn.Linear): Optional module computing the value position encodings.
        attn_weights (nn.Linear): Module computing the attention weights from the input features.
        out_proj (nn.Linear): Module computing output features from weighted value features.

        update_sample_locations (bool): Boolean indicating whether sample locations should be updated.

        If update_sample_locations is True:
            steps_weight (nn.Parameter): Parameter containing the weight matrix used during sample step computation.
            steps_bias (nn.Parameter): Parameter containing the bias vector used during sample step computation.
            step_size (float): Size of the sample steps relative to the sample step normalization.
            step_norm_xy (str): String containing the normalization type of XY-direction sample steps.
    """

    def __init__(self, in_size, sample_size, out_size=-1, num_heads=8, num_levels=5, num_points=4, val_size=-1,
                 val_with_pos=False, step_size=-1, step_norm_xy='map'):
        """
        Initializes the PAv1 module.

        Args:
            in_size (int): Size of input features.
            sample_size (int): Size of sample features.
            out_size (int): Size of output features (default=-1).
            num_heads (int): Integer containing the number of attention heads (default=8).
            num_levels (int): Integer containing the number of map levels to sample from (default=5).
            num_points (int): Integer containing the number of sampling points per head and level (default=4).
            val_size (int): Size of value features (default=-1).
            val_with_pos (bool): Boolean indicating whether position info is added to value features (default=False).
            step_size (float): Size of the sample steps relative to the sample step normalization (default=-1).
            step_norm_xy (str): String containing the normalization type of XY-direction sample steps (default=map).

        Raises:
            ValueError: Error when the input feature size does not divide the number of heads.
            ValueError: Error when the value feature size does not divide the number of heads.
        """

        # Initialization of default nn.Module
        super().__init__()

        # Set attributes related to the number of heads, levels and points
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points

        # Check divisibility input size by number of heads
        if in_size % num_heads != 0:
            error_msg = f"The input feature size ({in_size}) must divide the number of heads ({num_heads})."
            raise ValueError(error_msg)

        # Get and check size of value features
        if val_size == -1:
            val_size = in_size

        elif val_size % num_heads != 0:
            error_msg = f"The value feature size ({val_size}) must divide the number of heads ({num_heads})."
            raise ValueError(error_msg)

        # Initialize module computing the value features
        self.val_proj = nn.Linear(sample_size, val_size)
        nn.init.xavier_uniform_(self.val_proj.weight)
        nn.init.zeros_(self.val_proj.bias)

        # Initialize module computing the value position encodings if requested
        if val_with_pos:
            self.val_pos_encs = nn.Linear(3, val_size // num_heads)

        # Initialize module computing the unnormalized attention weights
        self.attn_weights = nn.Linear(in_size, num_heads * num_levels * num_points)
        nn.init.zeros_(self.attn_weights.weight)
        nn.init.zeros_(self.attn_weights.bias)

        # Initialize module computing the output features
        out_size = in_size if out_size == -1 else out_size
        self.out_proj = nn.Linear(val_size, out_size)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

        # Set attribute determining whether sample locations should be updated
        self.update_sample_locations = True

        # Initialize modules computing the unnormalized sample steps
        self.steps_weight = nn.Parameter(torch.zeros(num_heads*2, val_size // num_heads, 1))
        self.steps_bias = nn.Parameter(torch.zeros(num_heads*2, 1, 1))

        # Set attributes related to the sizes of the sample steps
        self.step_size = step_size
        self.step_norm_xy = step_norm_xy

    def no_sample_locations_update(self):
        """
        Method changing the module to not update the sample locations.
        """

        # Change attribute to remember that sample locations should not be updated
        self.update_sample_locations = False

        # Delete all attributes related to the update of sample locations
        delattr(self, 'steps_weight')
        delattr(self, 'steps_bias')
        delattr(self, 'step_size')
        delattr(self, 'step_norm_xy')

    def forward(self, in_feats, sample_priors, sample_feats, sample_map_shapes, sample_map_start_ids, storage_dict,
                **kwargs):
        """
        Forward method of the PAv1 module.

        Args:
            in_feats (FloatTensor): Input features of shape [batch_size, num_in_feats, in_size].
            sample_priors (FloatTensor): Sample priors of shape [batch_size, num_in_feats, {2, 4}].
            sample_feats (FloatTensor): Sample features of shape [batch_size, num_sample_feats, sample_size].
            sample_map_shapes (LongTensor): Map shapes corresponding to samples of shape [num_levels, 2].
            sample_map_start_ids (LongTensor): Start indices of sample maps of shape [num_levels].
            storage_dict (Dict): Dictionary storing additional arguments such as the sample locations.
            kwargs (Dict): Dictionary of keyword arguments not used by this module.

        Returns:
            out_feats (FloatTensor): Output features of shape [batch_size, num_in_feats, out_size].

        Raises:
            ValueError: Error when the last dimension of 'sample_priors' is different from 2 or 4.
            ValueError: Error when last dimension of 'sample_priors' is not 4 when using 'anchor' step normalization.
            ValueError: Error when invalid sample step normalization type is provided.
        """

        # Get shapes of input tensors
        batch_size, num_in_feats = in_feats.shape[:2]
        common_shape = (batch_size, num_in_feats, self.num_heads)

        # Flip sample map shapes
        sample_map_shapes = sample_map_shapes.fliplr()

        # Get value features
        val_feats = self.val_proj(sample_feats)
        val_size = val_feats.shape[-1]
        val_feats = val_feats.view(batch_size, -1, self.num_heads, val_size // self.num_heads)
        val_feats = val_feats.transpose(1, 2).view(batch_size * self.num_heads, -1, val_size // self.num_heads)

        # Get sample locations
        sample_locations = storage_dict.pop('sample_locations', None)

        if sample_locations is None:
            thetas = torch.arange(self.num_heads, dtype=torch.float, device=sample_feats.device)
            thetas = thetas * (2.0 * math.pi / self.num_heads)

            sample_offsets = torch.stack([thetas.cos(), thetas.sin()], dim=1)
            sample_offsets = sample_offsets / sample_offsets.abs().max(dim=1, keepdim=True)[0]
            sample_offsets = sample_offsets.view(self.num_heads, 1, 1, 2).repeat(1, self.num_levels, 1, 1)

            sizes = torch.arange(1, self. num_points+1, dtype=torch.float, device=sample_feats.device)
            sample_offsets = sizes[None, None, :, None] * sample_offsets
            sample_offsets = sample_offsets[None, None, :, :, :, :]

            if sample_priors.shape[-1] == 2:
                offset_normalizers = sample_map_shapes[None, None, None, :, None, :]
                sample_locations = sample_priors[:, :, None, None, None, :]
                sample_locations = sample_locations + sample_offsets / offset_normalizers

            elif sample_priors.shape[-1] == 4:
                offset_factors = 0.5 * sample_priors[:, :, None, None, None, 2:] / self.num_points
                sample_locations = sample_priors[:, :, None, None, None, :2]
                sample_locations = sample_locations + sample_offsets * offset_factors

            else:
                error_msg = f"Last dimension of 'sample_priors' must be 2 or 4, but got {sample_priors.shape[-1]}."
                raise ValueError(error_msg)

            sample_locations = sample_locations.transpose(1, 2).reshape(batch_size * self.num_heads, -1, 2)

        # Get sampled value features and corresponding derivatives if needed
        sample_map_ids = torch.arange(self.num_levels, device=sample_feats.device)
        sample_map_ids = sample_map_ids[None, None, :, None]
        sample_map_ids = sample_map_ids.expand(batch_size * self.num_heads, num_in_feats, -1, self.num_points)
        sample_map_ids = sample_map_ids.reshape(batch_size * self.num_heads, -1)
        sample_args = (val_feats, sample_map_shapes, sample_map_start_ids, sample_locations, sample_map_ids)

        if self.update_sample_locations:
            sampled_feats, dx, dy = pytorch_maps_sample_2d(*sample_args, return_derivatives=True)
        else:
            sampled_feats = pytorch_maps_sample_2d(*sample_args, return_derivatives=False)

        sampled_feats = sampled_feats.view(batch_size, self.num_heads, num_in_feats, -1, val_size // self.num_heads)
        sampled_feats = sampled_feats.transpose(1, 2)

        # Get and add position encodings to sampled value features if needed
        if hasattr(self, 'val_pos_encs'):
            sample_xy = sample_locations.view(batch_size, self.num_heads, num_in_feats, -1, 2)
            sample_xy = sample_xy - sample_priors[:, None, :, None, :2]

            if sample_priors.shape[-1] == 4:
                sample_xy = sample_xy / sample_priors[:, None, :, None, 2:]

            sample_z = sample_map_ids.view(batch_size, self.num_heads, num_in_feats, -1, 1)
            sample_z = sample_z / (self.num_levels-1)

            sample_xyz = torch.cat([sample_xy, sample_z], dim=4).transpose(1, 2)
            sampled_feats = sampled_feats + self.val_pos_encs(sample_xyz)

        # Get attention weights
        attn_weights = self.attn_weights(in_feats).view(*common_shape, self.num_levels * self.num_points)
        attn_weights = F.softmax(attn_weights, dim=3)

        # Get weighted value features
        weighted_feats = attn_weights[:, :, :, :, None] * sampled_feats
        weighted_feats = weighted_feats.sum(dim=3).view(batch_size, num_in_feats, val_size)

        # Get output features
        out_feats = self.out_proj(weighted_feats)

        # Update sample locations in storage dictionary if needed
        if self.update_sample_locations:
            derivatives = torch.stack([dx, dy], dim=1)
            derivatives = derivatives.view(batch_size, self.num_heads, 2, -1, val_size // self.num_heads)
            derivatives = derivatives.permute(1, 2, 0, 3, 4).view(self.num_heads*2, -1, val_size // self.num_heads)

            sample_steps = torch.bmm(derivatives, self.steps_weight) + self.steps_bias
            sample_steps = sample_steps.view(self.num_heads, 2, batch_size, -1, self.num_levels, self.num_points)
            sample_steps = sample_steps.permute(2, 0, 3, 4, 5, 1)

            if self.step_size > 0:
                sample_steps = self.step_size * F.normalize(sample_steps, dim=5)

            if self.step_norm_xy == 'map':
                step_normalizers = sample_map_shapes[None, None, None, :, None, :]
                sample_steps = sample_steps / step_normalizers

            elif self.step_norm_xy == 'anchor':
                if sample_priors.shape[-1] == 4:
                    step_factors = 0.5 * sample_priors[:, None, :, None, None, 2:] / self.num_points
                    sample_steps = sample_steps * step_factors

                else:
                    error_msg = "Last dimension of 'sample_priors' must be 4 when using 'anchor' step normalization."
                    raise ValueError(error_msg)

            else:
                error_msg = f"Invalid sample step normalization type '{self.step_norm_xy}'."
                raise ValueError(error_msg)

            sample_steps = sample_steps.view(batch_size * self.num_heads, -1, 2)
            next_sample_locations = sample_locations + sample_steps
            storage_dict['sample_locations'] = next_sample_locations

        return out_feats


class PAv2(nn.Module):
    """
    Class implementing the PAv2 module.

    Attributes:
        num_heads (int): Integer containing the number of attention heads.
        num_levels (int): Integer containing the number of map levels to sample from.
        num_points (int): Integer containing the number of sampling points per head and level.

        val_proj (nn.Linear): Module computing value features from sample features.
        val_pos_encs (nn.Linear): Optional module computing the value position encodings.
        attn_weights (nn.Linear): Module computing the attention weights from the input features.
        out_proj (nn.Linear): Module computing output features from weighted value features.

        update_sample_locations (bool): Boolean indicating whether sample locations should be updated.

        If update_sample_locations is True:
            steps_weight (nn.Parameter): Parameter containing the weight matrix used during sample step computation.
            steps_bias (nn.Parameter): Parameter containing the bias vector used during sample step computation.
            step_size (float): Size of the sample steps relative to the sample step normalization.
            step_norm_xy (str): String containing the normalization type of XY-direction sample steps.
    """

    def __init__(self, in_size, sample_size, out_size=-1, num_heads=8, num_levels=5, num_points=4, val_size=-1,
                 val_with_pos=False, step_size=-1, step_norm_xy='map'):
        """
        Initializes the PAv2 module.

        Args:
            in_size (int): Size of input features.
            sample_size (int): Size of sample features.
            out_size (int): Size of output features (default=-1).
            num_heads (int): Integer containing the number of attention heads (default=8).
            num_levels (int): Integer containing the number of map levels to sample from (default=5).
            num_points (int): Integer containing the number of sampling points per head and level (default=4).
            val_size (int): Size of value features (default=-1).
            val_with_pos (bool): Boolean indicating whether position info is added to value features (default=False).
            step_size (float): Size of the sample steps relative to the sample step normalization (default=-1).
            step_norm_xy (str): String containing the normalization type of XY-direction sample steps (default=map).

        Raises:
            ValueError: Error when the input feature size does not divide the number of heads.
            ValueError: Error when the value feature size does not divide the number of heads.
        """

        # Initialization of default nn.Module
        super().__init__()

        # Set attributes related to the number of heads, levels and points
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points

        # Check divisibility input size by number of heads
        if in_size % num_heads != 0:
            error_msg = f"The input feature size ({in_size}) must divide the number of heads ({num_heads})."
            raise ValueError(error_msg)

        # Get and check size of value features
        if val_size == -1:
            val_size = in_size

        elif val_size % num_heads != 0:
            error_msg = f"The value feature size ({val_size}) must divide the number of heads ({num_heads})."
            raise ValueError(error_msg)

        # Initialize module computing the value features
        self.val_proj = nn.Linear(sample_size, val_size)
        nn.init.xavier_uniform_(self.val_proj.weight)
        nn.init.zeros_(self.val_proj.bias)

        # Initialize module computing the value position encodings if requested
        if val_with_pos:
            self.val_pos_encs = nn.Linear(3, val_size // num_heads)

        # Initialize module computing the unnormalized attention weights
        self.attn_weights = nn.Linear(in_size, num_heads * num_levels * num_points)
        nn.init.zeros_(self.attn_weights.weight)
        nn.init.zeros_(self.attn_weights.bias)

        # Initialize module computing the output features
        out_size = in_size if out_size == -1 else out_size
        self.out_proj = nn.Linear(val_size, out_size)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

        # Set attribute determining whether sample locations should be updated
        self.update_sample_locations = True

        # Initialize modules computing the unnormalized sample steps
        self.steps_weight = nn.Parameter(torch.zeros(num_heads*num_levels*num_points*2, val_size // num_heads, 1))
        self.steps_bias = nn.Parameter(torch.zeros(num_heads*num_levels*num_points*2, 1, 1))

        # Set attributes related to the sizes of the sample steps
        self.step_size = step_size
        self.step_norm_xy = step_norm_xy

    def no_sample_locations_update(self):
        """
        Method changing the module to not update the sample locations.
        """

        # Change attribute to remember that sample locations should not be updated
        self.update_sample_locations = False

        # Delete all attributes related to the update of sample locations
        delattr(self, 'steps_weight')
        delattr(self, 'steps_bias')
        delattr(self, 'step_size')
        delattr(self, 'step_norm_xy')

    def forward(self, in_feats, sample_priors, sample_feats, sample_map_shapes, sample_map_start_ids, storage_dict,
                **kwargs):
        """
        Forward method of the PAv2 module.

        Args:
            in_feats (FloatTensor): Input features of shape [batch_size, num_in_feats, in_size].
            sample_priors (FloatTensor): Sample priors of shape [batch_size, num_in_feats, {2, 4}].
            sample_feats (FloatTensor): Sample features of shape [batch_size, num_sample_feats, sample_size].
            sample_map_shapes (LongTensor): Map shapes corresponding to samples of shape [num_levels, 2].
            sample_map_start_ids (LongTensor): Start indices of sample maps of shape [num_levels].
            storage_dict (Dict): Dictionary storing additional arguments such as the sample locations.
            kwargs (Dict): Dictionary of keyword arguments not used by this module.

        Returns:
            out_feats (FloatTensor): Output features of shape [batch_size, num_in_feats, out_size].

        Raises:
            ValueError: Error when the last dimension of 'sample_priors' is different from 2 or 4.
            ValueError: Error when last dimension of 'sample_priors' is not 4 when using 'anchor' step normalization.
            ValueError: Error when invalid sample step normalization type is provided.
        """

        # Get shapes of input tensors
        batch_size, num_in_feats = in_feats.shape[:2]
        common_shape = (batch_size, num_in_feats, self.num_heads)

        # Flip sample map shapes
        sample_map_shapes = sample_map_shapes.fliplr()

        # Get value features
        val_feats = self.val_proj(sample_feats)
        val_size = val_feats.shape[-1]
        val_feats = val_feats.view(batch_size, -1, self.num_heads, val_size // self.num_heads)
        val_feats = val_feats.transpose(1, 2).view(batch_size * self.num_heads, -1, val_size // self.num_heads)

        # Get sample locations
        sample_locations = storage_dict.pop('sample_locations', None)

        if sample_locations is None:
            thetas = torch.arange(self.num_heads, dtype=torch.float, device=sample_feats.device)
            thetas = thetas * (2.0 * math.pi / self.num_heads)

            sample_offsets = torch.stack([thetas.cos(), thetas.sin()], dim=1)
            sample_offsets = sample_offsets / sample_offsets.abs().max(dim=1, keepdim=True)[0]
            sample_offsets = sample_offsets.view(self.num_heads, 1, 1, 2).repeat(1, self.num_levels, 1, 1)

            sizes = torch.arange(1, self. num_points+1, dtype=torch.float, device=sample_feats.device)
            sample_offsets = sizes[None, None, :, None] * sample_offsets
            sample_offsets = sample_offsets[None, None, :, :, :, :]

            if sample_priors.shape[-1] == 2:
                offset_normalizers = sample_map_shapes[None, None, None, :, None, :]
                sample_locations = sample_priors[:, :, None, None, None, :]
                sample_locations = sample_locations + sample_offsets / offset_normalizers

            elif sample_priors.shape[-1] == 4:
                offset_factors = 0.5 * sample_priors[:, :, None, None, None, 2:] / self.num_points
                sample_locations = sample_priors[:, :, None, None, None, :2]
                sample_locations = sample_locations + sample_offsets * offset_factors

            else:
                error_msg = f"Last dimension of 'sample_priors' must be 2 or 4, but got {sample_priors.shape[-1]}."
                raise ValueError(error_msg)

            sample_locations = sample_locations.transpose(1, 2).reshape(batch_size * self.num_heads, -1, 2)

        # Get sampled value features and corresponding derivatives if needed
        sample_map_ids = torch.arange(self.num_levels, device=sample_feats.device)
        sample_map_ids = sample_map_ids[None, None, :, None]
        sample_map_ids = sample_map_ids.expand(batch_size * self.num_heads, num_in_feats, -1, self.num_points)
        sample_map_ids = sample_map_ids.reshape(batch_size * self.num_heads, -1)
        sample_args = (val_feats, sample_map_shapes, sample_map_start_ids, sample_locations, sample_map_ids)

        if self.update_sample_locations:
            sampled_feats, dx, dy = pytorch_maps_sample_2d(*sample_args, return_derivatives=True)
        else:
            sampled_feats = pytorch_maps_sample_2d(*sample_args, return_derivatives=False)

        sampled_feats = sampled_feats.view(batch_size, self.num_heads, num_in_feats, -1, val_size // self.num_heads)
        sampled_feats = sampled_feats.transpose(1, 2)

        # Get and add position encodings to sampled value features if needed
        if hasattr(self, 'val_pos_encs'):
            sample_xy = sample_locations.view(batch_size, self.num_heads, num_in_feats, -1, 2)
            sample_xy = sample_xy - sample_priors[:, None, :, None, :2]

            if sample_priors.shape[-1] == 4:
                sample_xy = sample_xy / sample_priors[:, None, :, None, 2:]

            sample_z = sample_map_ids.view(batch_size, self.num_heads, num_in_feats, -1, 1)
            sample_z = sample_z / (self.num_levels-1)

            sample_xyz = torch.cat([sample_xy, sample_z], dim=4).transpose(1, 2)
            sampled_feats = sampled_feats + self.val_pos_encs(sample_xyz)

        # Get attention weights
        attn_weights = self.attn_weights(in_feats).view(*common_shape, self.num_levels * self.num_points)
        attn_weights = F.softmax(attn_weights, dim=3)

        # Get weighted value features
        weighted_feats = attn_weights[:, :, :, :, None] * sampled_feats
        weighted_feats = weighted_feats.sum(dim=3).view(batch_size, num_in_feats, val_size)

        # Get output features
        out_feats = self.out_proj(weighted_feats)

        # Update sample locations in storage dictionary if needed
        if self.update_sample_locations:
            derivatives = torch.stack([dx, dy], dim=2)
            derivatives = derivatives.view(batch_size, self.num_heads, num_in_feats, -1, 2, val_size // self.num_heads)
            derivatives = derivatives.permute(1, 3, 4, 0, 2, 5)
            derivatives = derivatives.reshape(-1, batch_size * num_in_feats, val_size // self.num_heads)

            sample_steps = torch.bmm(derivatives, self.steps_weight) + self.steps_bias
            sample_steps = sample_steps.view(self.num_heads, self.num_levels, self.num_points, 2, batch_size, -1)
            sample_steps = sample_steps.permute(4, 0, 5, 1, 2, 3)

            if self.step_size > 0:
                sample_steps = self.step_size * F.normalize(sample_steps, dim=5)

            if self.step_norm_xy == 'map':
                step_normalizers = sample_map_shapes[None, None, None, :, None, :]
                sample_steps = sample_steps / step_normalizers

            elif self.step_norm_xy == 'anchor':
                if sample_priors.shape[-1] == 4:
                    step_factors = 0.5 * sample_priors[:, None, :, None, None, 2:] / self.num_points
                    sample_steps = sample_steps * step_factors

                else:
                    error_msg = "Last dimension of 'sample_priors' must be 4 when using 'anchor' step normalization."
                    raise ValueError(error_msg)

            else:
                error_msg = f"Invalid sample step normalization type '{self.step_norm_xy}'."
                raise ValueError(error_msg)

            sample_steps = sample_steps.reshape(batch_size * self.num_heads, -1, 2)
            next_sample_locations = sample_locations + sample_steps
            storage_dict['sample_locations'] = next_sample_locations

        return out_feats


class PAv3(nn.Module):
    """
    Class implementing the PAv3 module.

    Attributes:
        num_heads (int): Integer containing the number of attention heads.
        num_levels (int): Integer containing the number of map levels to sample from.
        num_points (int): Integer containing the number of sampling points per head and level.

        val_proj (nn.Linear): Module computing value features from sample features.
        val_pos_encs (nn.Linear): Optional module computing the value position encodings.
        attn_weights (nn.Linear): Module computing the attention weights from the input features.
        out_proj (nn.Linear): Module computing output features from weighted value features.

        update_sample_locations (bool): Boolean indicating whether sample locations should be updated.

        If update_sample_locations is True:
            qry_proj (nn.Linear): Module computing query features from input features.
            steps_weight (nn.Parameter): Parameter containing the weight matrix used during sample step computation.
            steps_bias (nn.Parameter): Parameter containing the bias vector used during sample step computation.
            step_size (float): Size of the sample steps relative to the sample step normalization.
            step_norm_xy (str): String containing the normalization type of XY-direction sample steps.
    """

    def __init__(self, in_size, sample_size, out_size=-1, num_heads=8, num_levels=5, num_points=4, val_size=-1,
                 val_with_pos=False, qry_size=-1, step_size=-1, step_norm_xy='map'):
        """
        Initializes the PAv3 module.

        Args:
            in_size (int): Size of input features.
            sample_size (int): Size of sample features.
            out_size (int): Size of output features (default=-1).
            num_heads (int): Integer containing the number of attention heads (default=8).
            num_levels (int): Integer containing the number of map levels to sample from (default=5).
            num_points (int): Integer containing the number of sampling points per head and level (default=4).
            val_size (int): Size of value features (default=-1).
            val_with_pos (bool): Boolean indicating whether position info is added to value features (default=False).
            qry_size (int): Size of query features (default=-1).
            step_size (float): Size of the sample steps relative to the sample step normalization (default=-1).
            step_norm_xy (str): String containing the normalization type of XY-direction sample steps (default=map).

        Raises:
            ValueError: Error when the input feature size does not divide the number of heads.
            ValueError: Error when the value feature size does not divide the number of heads.
            ValueError: Error when the query feature size does not equal the value feature size.
        """

        # Initialization of default nn.Module
        super().__init__()

        # Set attributes related to the number of heads, levels and points
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points

        # Check divisibility input size by number of heads
        if in_size % num_heads != 0:
            error_msg = f"The input feature size ({in_size}) must divide the number of heads ({num_heads})."
            raise ValueError(error_msg)

        # Get and check size of value features
        if val_size == -1:
            val_size = in_size

        elif val_size % num_heads != 0:
            error_msg = f"The value feature size ({val_size}) must divide the number of heads ({num_heads})."
            raise ValueError(error_msg)

        # Initialize module computing the value features
        self.val_proj = nn.Linear(sample_size, val_size)
        nn.init.xavier_uniform_(self.val_proj.weight)
        nn.init.zeros_(self.val_proj.bias)

        # Initialize module computing the value position encodings if requested
        if val_with_pos:
            self.val_pos_encs = nn.Linear(3, val_size // num_heads)

        # Initialize module computing the unnormalized attention weights
        self.attn_weights = nn.Linear(in_size, num_heads * num_levels * num_points)
        nn.init.zeros_(self.attn_weights.weight)
        nn.init.zeros_(self.attn_weights.bias)

        # Initialize module computing the output features
        out_size = in_size if out_size == -1 else out_size
        self.out_proj = nn.Linear(val_size, out_size)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

        # Set attribute determining whether sample locations should be updated
        self.update_sample_locations = True

        # Get and check size of query features
        if qry_size == -1:
            qry_size = val_size

        elif qry_size != val_size:
            error_msg = f"The query feature size ({qry_size}) must equal the value feature size ({val_size})."
            raise ValueError(error_msg)

        # Initialize module computing the query features
        self.qry_proj = nn.Linear(in_size, qry_size)
        nn.init.xavier_uniform_(self.qry_proj.weight)
        nn.init.zeros_(self.qry_proj.bias)

        # Initialize modules computing the unnormalized sample steps
        self.steps_weight = nn.Parameter(torch.zeros(num_heads*num_levels*num_points*2, val_size // num_heads, 1))
        self.steps_bias = nn.Parameter(torch.zeros(num_heads*num_levels*num_points*2, 1, 1))

        # Set attributes related to the sizes of the sample steps
        self.step_size = step_size
        self.step_norm_xy = step_norm_xy

    def no_sample_locations_update(self):
        """
        Method changing the module to not update the sample locations.
        """

        # Change attribute to remember that sample locations should not be updated
        self.update_sample_locations = False

        # Delete all attributes related to the update of sample locations
        delattr(self, 'qry_proj')
        delattr(self, 'steps_weight')
        delattr(self, 'steps_bias')
        delattr(self, 'step_size')
        delattr(self, 'step_norm_xy')

    def forward(self, in_feats, sample_priors, sample_feats, sample_map_shapes, sample_map_start_ids, storage_dict,
                **kwargs):
        """
        Forward method of the PAv3 module.

        Args:
            in_feats (FloatTensor): Input features of shape [batch_size, num_in_feats, in_size].
            sample_priors (FloatTensor): Sample priors of shape [batch_size, num_in_feats, {2, 4}].
            sample_feats (FloatTensor): Sample features of shape [batch_size, num_sample_feats, sample_size].
            sample_map_shapes (LongTensor): Map shapes corresponding to samples of shape [num_levels, 2].
            sample_map_start_ids (LongTensor): Start indices of sample maps of shape [num_levels].
            storage_dict (Dict): Dictionary storing additional arguments such as the sample locations.
            kwargs (Dict): Dictionary of keyword arguments not used by this module.

        Returns:
            out_feats (FloatTensor): Output features of shape [batch_size, num_in_feats, out_size].

        Raises:
            ValueError: Error when the last dimension of 'sample_priors' is different from 2 or 4.
            ValueError: Error when last dimension of 'sample_priors' is not 4 when using 'anchor' step normalization.
            ValueError: Error when invalid sample step normalization type is provided.
        """

        # Get shapes of input tensors
        batch_size, num_in_feats = in_feats.shape[:2]
        common_shape = (batch_size, num_in_feats, self.num_heads)

        # Flip sample map shapes
        sample_map_shapes = sample_map_shapes.fliplr()

        # Get value features
        val_feats = self.val_proj(sample_feats)
        val_size = val_feats.shape[-1]
        val_feats = val_feats.view(batch_size, -1, self.num_heads, val_size // self.num_heads)
        val_feats = val_feats.transpose(1, 2).view(batch_size * self.num_heads, -1, val_size // self.num_heads)

        # Get sample locations
        sample_locations = storage_dict.pop('sample_locations', None)

        if sample_locations is None:
            thetas = torch.arange(self.num_heads, dtype=torch.float, device=sample_feats.device)
            thetas = thetas * (2.0 * math.pi / self.num_heads)

            sample_offsets = torch.stack([thetas.cos(), thetas.sin()], dim=1)
            sample_offsets = sample_offsets / sample_offsets.abs().max(dim=1, keepdim=True)[0]
            sample_offsets = sample_offsets.view(self.num_heads, 1, 1, 2).repeat(1, self.num_levels, 1, 1)

            sizes = torch.arange(1, self. num_points+1, dtype=torch.float, device=sample_feats.device)
            sample_offsets = sizes[None, None, :, None] * sample_offsets
            sample_offsets = sample_offsets[None, None, :, :, :, :]

            if sample_priors.shape[-1] == 2:
                offset_normalizers = sample_map_shapes[None, None, None, :, None, :]
                sample_locations = sample_priors[:, :, None, None, None, :]
                sample_locations = sample_locations + sample_offsets / offset_normalizers

            elif sample_priors.shape[-1] == 4:
                offset_factors = 0.5 * sample_priors[:, :, None, None, None, 2:] / self.num_points
                sample_locations = sample_priors[:, :, None, None, None, :2]
                sample_locations = sample_locations + sample_offsets * offset_factors

            else:
                error_msg = f"Last dimension of 'sample_priors' must be 2 or 4, but got {sample_priors.shape[-1]}."
                raise ValueError(error_msg)

            sample_locations = sample_locations.transpose(1, 2).reshape(batch_size * self.num_heads, -1, 2)

        # Get sampled value features and corresponding derivatives if needed
        sample_map_ids = torch.arange(self.num_levels, device=sample_feats.device)
        sample_map_ids = sample_map_ids[None, None, :, None]
        sample_map_ids = sample_map_ids.expand(batch_size * self.num_heads, num_in_feats, -1, self.num_points)
        sample_map_ids = sample_map_ids.reshape(batch_size * self.num_heads, -1)
        sample_args = (val_feats, sample_map_shapes, sample_map_start_ids, sample_locations, sample_map_ids)

        if self.update_sample_locations:
            sampled_feats, dx, dy = pytorch_maps_sample_2d(*sample_args, return_derivatives=True)
        else:
            sampled_feats = pytorch_maps_sample_2d(*sample_args, return_derivatives=False)

        sampled_feats = sampled_feats.view(batch_size, self.num_heads, num_in_feats, -1, val_size // self.num_heads)
        sampled_feats = sampled_feats.transpose(1, 2)

        # Get and add position encodings to sampled value features if needed
        if hasattr(self, 'val_pos_encs'):
            sample_xy = sample_locations.view(batch_size, self.num_heads, num_in_feats, -1, 2)
            sample_xy = sample_xy - sample_priors[:, None, :, None, :2]

            if sample_priors.shape[-1] == 4:
                sample_xy = sample_xy / sample_priors[:, None, :, None, 2:]

            sample_z = sample_map_ids.view(batch_size, self.num_heads, num_in_feats, -1, 1)
            sample_z = sample_z / (self.num_levels-1)

            sample_xyz = torch.cat([sample_xy, sample_z], dim=4).transpose(1, 2)
            sampled_feats = sampled_feats + self.val_pos_encs(sample_xyz)

        # Get attention weights
        attn_weights = self.attn_weights(in_feats).view(*common_shape, self.num_levels * self.num_points)
        attn_weights = F.softmax(attn_weights, dim=3)

        # Get weighted value features
        weighted_feats = attn_weights[:, :, :, :, None] * sampled_feats
        weighted_feats = weighted_feats.sum(dim=3).view(batch_size, num_in_feats, val_size)

        # Get output features
        out_feats = self.out_proj(weighted_feats)

        # Update sample locations in storage dictionary if needed
        if self.update_sample_locations:
            derivatives = torch.stack([dx, dy], dim=2)
            derivatives = derivatives.view(batch_size, self.num_heads, num_in_feats, -1, 2, val_size // self.num_heads)

            qry_feats = self.qry_proj(in_feats)
            qry_feats = qry_feats.view(*common_shape, -1).transpose(1, 2)
            derivatives = derivatives + qry_feats[:, :, :, None, None, :]

            derivatives = derivatives.permute(1, 3, 4, 0, 2, 5)
            derivatives = derivatives.reshape(-1, batch_size * num_in_feats, val_size // self.num_heads)

            sample_steps = torch.bmm(derivatives, self.steps_weight) + self.steps_bias
            sample_steps = sample_steps.view(self.num_heads, self.num_levels, self.num_points, 2, batch_size, -1)
            sample_steps = sample_steps.permute(4, 0, 5, 1, 2, 3)

            if self.step_size > 0:
                sample_steps = self.step_size * F.normalize(sample_steps, dim=5)

            if self.step_norm_xy == 'map':
                step_normalizers = sample_map_shapes[None, None, None, :, None, :]
                sample_steps = sample_steps / step_normalizers

            elif self.step_norm_xy == 'anchor':
                if sample_priors.shape[-1] == 4:
                    step_factors = 0.5 * sample_priors[:, None, :, None, None, 2:] / self.num_points
                    sample_steps = sample_steps * step_factors

                else:
                    error_msg = "Last dimension of 'sample_priors' must be 4 when using 'anchor' step normalization."
                    raise ValueError(error_msg)

            else:
                error_msg = f"Invalid sample step normalization type '{self.step_norm_xy}'."
                raise ValueError(error_msg)

            sample_steps = sample_steps.reshape(batch_size * self.num_heads, -1, 2)
            next_sample_locations = sample_locations + sample_steps
            storage_dict['sample_locations'] = next_sample_locations

        return out_feats


class PAv4(nn.Module):
    """
    Class implementing the PAv4 module.

    Attributes:
        num_heads (int): Integer containing the number of attention heads.
        num_levels (int): Integer containing the number of map levels to sample from.
        num_points (int): Integer containing the number of sampling points per head and level.

        val_proj (nn.Linear): Module computing value features from sample features.
        val_pos_encs (nn.Linear): Optional module computing the value position encodings.
        attn_weights (nn.Linear): Module computing the attention weights from the input features.
        out_proj (nn.Linear): Module computing output features from weighted value features.

        update_sample_locations (bool): Boolean indicating whether sample locations should be updated.

        If update_sample_locations is True:
            qry_proj (nn.Linear): Module computing query features from input features.
            steps_weight (nn.Parameter): Parameter containing the weight matrix used during sample step computation.
            steps_bias (nn.Parameter): Parameter containing the bias vector used during sample step computation.
            step_size (float): Size of the sample steps relative to the sample step normalization.
            step_norm_xy (str): String containing the normalization type of XY-direction sample steps.
    """

    def __init__(self, in_size, sample_size, out_size=-1, num_heads=8, num_levels=5, num_points=4, val_size=-1,
                 val_with_pos=False, qry_size=-1, step_size=-1, step_norm_xy='map'):
        """
        Initializes the PAv4 module.

        Args:
            in_size (int): Size of input features.
            sample_size (int): Size of sample features.
            out_size (int): Size of output features (default=-1).
            num_heads (int): Integer containing the number of attention heads (default=8).
            num_levels (int): Integer containing the number of map levels to sample from (default=5).
            num_points (int): Integer containing the number of sampling points per head and level (default=4).
            val_size (int): Size of value features (default=-1).
            val_with_pos (bool): Boolean indicating whether position info is added to value features (default=False).
            qry_size (int): Size of query features (default=-1).
            step_size (float): Size of the sample steps relative to the sample step normalization (default=-1).
            step_norm_xy (str): String containing the normalization type of XY-direction sample steps (default=map).

        Raises:
            ValueError: Error when the input feature size does not divide the number of heads.
            ValueError: Error when the value feature size does not divide the number of heads.
            ValueError: Error when the query feature size does not divide the number of heads.
        """

        # Initialization of default nn.Module
        super().__init__()

        # Set attributes related to the number of heads, levels and points
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points

        # Check divisibility input size by number of heads
        if in_size % num_heads != 0:
            error_msg = f"The input feature size ({in_size}) must divide the number of heads ({num_heads})."
            raise ValueError(error_msg)

        # Get and check size of value features
        if val_size == -1:
            val_size = in_size

        elif val_size % num_heads != 0:
            error_msg = f"The value feature size ({val_size}) must divide the number of heads ({num_heads})."
            raise ValueError(error_msg)

        # Initialize module computing the value features
        self.val_proj = nn.Linear(sample_size, val_size)
        nn.init.xavier_uniform_(self.val_proj.weight)
        nn.init.zeros_(self.val_proj.bias)

        # Initialize module computing the value position encodings if requested
        if val_with_pos:
            self.val_pos_encs = nn.Linear(3, val_size // num_heads)

        # Initialize module computing the unnormalized attention weights
        self.attn_weights = nn.Linear(in_size, num_heads * num_levels * num_points)
        nn.init.zeros_(self.attn_weights.weight)
        nn.init.zeros_(self.attn_weights.bias)

        # Initialize module computing the output features
        out_size = in_size if out_size == -1 else out_size
        self.out_proj = nn.Linear(val_size, out_size)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

        # Set attribute determining whether sample locations should be updated
        self.update_sample_locations = True

        # Get and check size of query features
        if qry_size == -1:
            qry_size = in_size

        elif qry_size % num_heads != 0:
            error_msg = f"The query feature size ({qry_size}) must divide the number of heads ({num_heads})."
            raise ValueError(error_msg)

        # Initialize module computing the query features
        self.qry_proj = nn.Linear(in_size, qry_size)
        nn.init.xavier_uniform_(self.qry_proj.weight)
        nn.init.zeros_(self.qry_proj.bias)

        # Initialize modules computing the unnormalized sample steps
        cat_size = (val_size + qry_size) // num_heads
        self.steps_weight = nn.Parameter(torch.zeros(num_heads*num_levels*num_points*2, cat_size, 1))
        self.steps_bias = nn.Parameter(torch.zeros(num_heads*num_levels*num_points*2, 1, 1))

        # Set attributes related to the sizes of the sample steps
        self.step_size = step_size
        self.step_norm_xy = step_norm_xy

    def no_sample_locations_update(self):
        """
        Method changing the module to not update the sample locations.
        """

        # Change attribute to remember that sample locations should not be updated
        self.update_sample_locations = False

        # Delete all attributes related to the update of sample locations
        delattr(self, 'qry_proj')
        delattr(self, 'steps_weight')
        delattr(self, 'steps_bias')
        delattr(self, 'step_size')
        delattr(self, 'step_norm_xy')

    def forward(self, in_feats, sample_priors, sample_feats, sample_map_shapes, sample_map_start_ids, storage_dict,
                **kwargs):
        """
        Forward method of the PAv4 module.

        Args:
            in_feats (FloatTensor): Input features of shape [batch_size, num_in_feats, in_size].
            sample_priors (FloatTensor): Sample priors of shape [batch_size, num_in_feats, {2, 4}].
            sample_feats (FloatTensor): Sample features of shape [batch_size, num_sample_feats, sample_size].
            sample_map_shapes (LongTensor): Map shapes corresponding to samples of shape [num_levels, 2].
            sample_map_start_ids (LongTensor): Start indices of sample maps of shape [num_levels].
            storage_dict (Dict): Dictionary storing additional arguments such as the sample locations.
            kwargs (Dict): Dictionary of keyword arguments not used by this module.

        Returns:
            out_feats (FloatTensor): Output features of shape [batch_size, num_in_feats, out_size].

        Raises:
            ValueError: Error when the last dimension of 'sample_priors' is different from 2 or 4.
            ValueError: Error when last dimension of 'sample_priors' is not 4 when using 'anchor' step normalization.
            ValueError: Error when invalid sample step normalization type is provided.
        """

        # Get shapes of input tensors
        batch_size, num_in_feats = in_feats.shape[:2]
        common_shape = (batch_size, num_in_feats, self.num_heads)

        # Flip sample map shapes
        sample_map_shapes = sample_map_shapes.fliplr()

        # Get value features
        val_feats = self.val_proj(sample_feats)
        val_size = val_feats.shape[-1]
        val_feats = val_feats.view(batch_size, -1, self.num_heads, val_size // self.num_heads)
        val_feats = val_feats.transpose(1, 2).view(batch_size * self.num_heads, -1, val_size // self.num_heads)

        # Get sample locations
        sample_locations = storage_dict.pop('sample_locations', None)

        if sample_locations is None:
            thetas = torch.arange(self.num_heads, dtype=torch.float, device=sample_feats.device)
            thetas = thetas * (2.0 * math.pi / self.num_heads)

            sample_offsets = torch.stack([thetas.cos(), thetas.sin()], dim=1)
            sample_offsets = sample_offsets / sample_offsets.abs().max(dim=1, keepdim=True)[0]
            sample_offsets = sample_offsets.view(self.num_heads, 1, 1, 2).repeat(1, self.num_levels, 1, 1)

            sizes = torch.arange(1, self. num_points+1, dtype=torch.float, device=sample_feats.device)
            sample_offsets = sizes[None, None, :, None] * sample_offsets
            sample_offsets = sample_offsets[None, None, :, :, :, :]

            if sample_priors.shape[-1] == 2:
                offset_normalizers = sample_map_shapes[None, None, None, :, None, :]
                sample_locations = sample_priors[:, :, None, None, None, :]
                sample_locations = sample_locations + sample_offsets / offset_normalizers

            elif sample_priors.shape[-1] == 4:
                offset_factors = 0.5 * sample_priors[:, :, None, None, None, 2:] / self.num_points
                sample_locations = sample_priors[:, :, None, None, None, :2]
                sample_locations = sample_locations + sample_offsets * offset_factors

            else:
                error_msg = f"Last dimension of 'sample_priors' must be 2 or 4, but got {sample_priors.shape[-1]}."
                raise ValueError(error_msg)

            sample_locations = sample_locations.transpose(1, 2).reshape(batch_size * self.num_heads, -1, 2)

        # Get sampled value features and corresponding derivatives if needed
        sample_map_ids = torch.arange(self.num_levels, device=sample_feats.device)
        sample_map_ids = sample_map_ids[None, None, :, None]
        sample_map_ids = sample_map_ids.expand(batch_size * self.num_heads, num_in_feats, -1, self.num_points)
        sample_map_ids = sample_map_ids.reshape(batch_size * self.num_heads, -1)
        sample_args = (val_feats, sample_map_shapes, sample_map_start_ids, sample_locations, sample_map_ids)

        if self.update_sample_locations:
            sampled_feats, dx, dy = pytorch_maps_sample_2d(*sample_args, return_derivatives=True)
        else:
            sampled_feats = pytorch_maps_sample_2d(*sample_args, return_derivatives=False)

        sampled_feats = sampled_feats.view(batch_size, self.num_heads, num_in_feats, -1, val_size // self.num_heads)
        sampled_feats = sampled_feats.transpose(1, 2)

        # Get and add position encodings to sampled value features if needed
        if hasattr(self, 'val_pos_encs'):
            sample_xy = sample_locations.view(batch_size, self.num_heads, num_in_feats, -1, 2)
            sample_xy = sample_xy - sample_priors[:, None, :, None, :2]

            if sample_priors.shape[-1] == 4:
                sample_xy = sample_xy / sample_priors[:, None, :, None, 2:]

            sample_z = sample_map_ids.view(batch_size, self.num_heads, num_in_feats, -1, 1)
            sample_z = sample_z / (self.num_levels-1)

            sample_xyz = torch.cat([sample_xy, sample_z], dim=4).transpose(1, 2)
            sampled_feats = sampled_feats + self.val_pos_encs(sample_xyz)

        # Get attention weights
        attn_weights = self.attn_weights(in_feats).view(*common_shape, self.num_levels * self.num_points)
        attn_weights = F.softmax(attn_weights, dim=3)

        # Get weighted value features
        weighted_feats = attn_weights[:, :, :, :, None] * sampled_feats
        weighted_feats = weighted_feats.sum(dim=3).view(batch_size, num_in_feats, val_size)

        # Get output features
        out_feats = self.out_proj(weighted_feats)

        # Update sample locations in storage dictionary if needed
        if self.update_sample_locations:
            derivatives = torch.stack([dx, dy], dim=2)
            derivatives = derivatives.view(batch_size, self.num_heads, num_in_feats, -1, 2, val_size // self.num_heads)

            qry_feats = self.qry_proj(in_feats)
            qry_feats = qry_feats.view(*common_shape, -1).transpose(1, 2)
            qry_feats = qry_feats[:, :, :, None, None, :].expand_as(derivatives)
            derivatives = torch.cat([derivatives, qry_feats], dim=5)

            derivatives = derivatives.permute(1, 3, 4, 0, 2, 5)
            derivatives = derivatives.reshape(-1, batch_size * num_in_feats, derivatives.shape[-1])

            sample_steps = torch.bmm(derivatives, self.steps_weight) + self.steps_bias
            sample_steps = sample_steps.view(self.num_heads, self.num_levels, self.num_points, 2, batch_size, -1)
            sample_steps = sample_steps.permute(4, 0, 5, 1, 2, 3)

            if self.step_size > 0:
                sample_steps = self.step_size * F.normalize(sample_steps, dim=5)

            if self.step_norm_xy == 'map':
                step_normalizers = sample_map_shapes[None, None, None, :, None, :]
                sample_steps = sample_steps / step_normalizers

            elif self.step_norm_xy == 'anchor':
                if sample_priors.shape[-1] == 4:
                    step_factors = 0.5 * sample_priors[:, None, :, None, None, 2:] / self.num_points
                    sample_steps = sample_steps * step_factors

                else:
                    error_msg = "Last dimension of 'sample_priors' must be 4 when using 'anchor' step normalization."
                    raise ValueError(error_msg)

            else:
                error_msg = f"Invalid sample step normalization type '{self.step_norm_xy}'."
                raise ValueError(error_msg)

            sample_steps = sample_steps.reshape(batch_size * self.num_heads, -1, 2)
            next_sample_locations = sample_locations + sample_steps
            storage_dict['sample_locations'] = next_sample_locations

        return out_feats


class PAv5(nn.Module):
    """
    Class implementing the PAv5 module.

    Attributes:
        num_heads (int): Integer containing the number of attention heads.
        num_levels (int): Integer containing the number of map levels to sample from.
        num_points (int): Integer containing the number of sampling points per head and level.

        sampling_offsets (nn.Linear): Module computing the sampling offsets from the input features.
        val_proj (nn.Linear): Module computing value features from sample features.
        val_pos_encs (nn.Linear): Optional module computing the value position encodings.
        attn_weights (nn.Linear): Module computing the attention weights from the input features.
        out_proj (nn.Linear): Module computing output features from weighted value features.

        update_sample_locations (bool): Boolean indicating whether sample locations should be updated.
    """

    def __init__(self, in_size, sample_size, out_size=-1, num_heads=8, num_levels=5, num_points=4, val_size=-1,
                 val_with_pos=False):
        """
        Initializes the PAv5 module.

        Args:
            in_size (int): Size of input features.
            sample_size (int): Size of sample features.
            out_size (int): Size of output features (default=-1).
            num_heads (int): Integer containing the number of attention heads (default=8).
            num_levels (int): Integer containing the number of map levels to sample from (default=5).
            num_points (int): Integer containing the number of sampling points per head and level (default=4).
            val_size (int): Size of value features (default=-1).
            val_with_pos (bool): Boolean indicating whether position info is added to value features (default=False).

        Raises:
            ValueError: Error when the input feature size does not divide the number of heads.
            ValueError: Error when the value feature size does not divide the number of heads.
        """

        # Initialization of default nn.Module
        super().__init__()

        # Set attributes related to the number of heads, levels and points
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points

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

        # Get and check size of value features
        if val_size == -1:
            val_size = in_size

        elif val_size % num_heads != 0:
            error_msg = f"The value feature size ({val_size}) must divide the number of heads ({num_heads})."
            raise ValueError(error_msg)

        # Initialize module computing the value features
        self.val_proj = nn.Linear(sample_size, val_size)
        nn.init.xavier_uniform_(self.val_proj.weight)
        nn.init.zeros_(self.val_proj.bias)

        # Initialize module computing the value position encodings if requested
        if val_with_pos:
            self.val_pos_encs = nn.Linear(3, val_size // num_heads)

        # Initialize module computing the unnormalized attention weights
        self.attn_weights = nn.Linear(in_size, num_heads * num_levels * num_points)
        nn.init.zeros_(self.attn_weights.weight)
        nn.init.zeros_(self.attn_weights.bias)

        # Initialize module computing the output features
        out_size = in_size if out_size == -1 else out_size
        self.out_proj = nn.Linear(val_size, out_size)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

        # Set attribute determining whether sample locations of storage dictionary should be updated
        self.update_sample_locations = True

    def no_sample_locations_update(self):
        """
        Method changing the module to not update the sample locations.
        """

        # Change attribute to remember that sample locations of storage dictionary should not be updated
        self.update_sample_locations = False

    def forward(self, in_feats, sample_priors, sample_feats, sample_map_shapes, sample_map_start_ids, storage_dict,
                **kwargs):
        """
        Forward method of the PAv5 module.

        Args:
            in_feats (FloatTensor): Input features of shape [batch_size, num_in_feats, in_size].
            sample_priors (FloatTensor): Sample priors of shape [batch_size, num_in_feats, {2, 4}].
            sample_feats (FloatTensor): Sample features of shape [batch_size, num_sample_feats, sample_size].
            sample_map_shapes (LongTensor): Map shapes corresponding to samples of shape [num_levels, 2].
            sample_map_start_ids (LongTensor): Start indices of sample maps of shape [num_levels].
            storage_dict (Dict): Dictionary storing additional arguments such as the sample locations.
            kwargs (Dict): Dictionary of keyword arguments not used by this module.

        Returns:
            out_feats (FloatTensor): Output features of shape [batch_size, num_in_feats, out_size].

        Raises:
            ValueError: Error when the last dimension of 'sample_priors' is different from 2 or 4.
        """

        # Get shapes of input tensors
        batch_size, num_in_feats = in_feats.shape[:2]
        common_shape = (batch_size, num_in_feats, self.num_heads)

        # Flip sample map shapes
        sample_map_shapes = sample_map_shapes.fliplr()

        # Get sample offsets
        sample_offsets = self.sampling_offsets(in_feats).view(*common_shape, self.num_levels, self.num_points, 2)

        if sample_priors.shape[-1] == 2:
            offset_normalizers = sample_map_shapes[None, None, None, :, None, :]
            sample_offsets = sample_offsets / offset_normalizers

        elif sample_priors.shape[-1] == 4:
            offset_factors = 0.5 * sample_priors[:, :, None, None, None, 2:] / self.num_points
            sample_offsets = sample_offsets * offset_factors

        else:
            error_msg = f"Last dimension of 'sample_priors' must be 2 or 4, but got {sample_priors.shape[-1]}."
            raise ValueError(error_msg)

        # Get sample locations
        sample_locations = storage_dict.pop('sample_locations', None)

        if sample_locations is None:
            sample_locations = sample_priors[:, :, None, None, None, :2]
            sample_locations = sample_locations.expand_as(sample_offsets)
            sample_locations = sample_locations.transpose(1, 2).reshape(batch_size * self.num_heads, -1, 2)

        sample_offsets = sample_offsets.transpose(1, 2).reshape_as(sample_locations)
        sample_locations = sample_locations + sample_offsets

        # Get value features
        val_feats = self.val_proj(sample_feats)
        val_size = val_feats.shape[-1]
        val_feats = val_feats.view(batch_size, -1, self.num_heads, val_size // self.num_heads)
        val_feats = val_feats.transpose(1, 2).view(batch_size * self.num_heads, -1, val_size // self.num_heads)

        # Get sampled value features and corresponding derivatives if needed
        sample_map_ids = torch.arange(self.num_levels, device=sample_feats.device)
        sample_map_ids = sample_map_ids[None, None, :, None]
        sample_map_ids = sample_map_ids.expand(batch_size * self.num_heads, num_in_feats, -1, self.num_points)
        sample_map_ids = sample_map_ids.reshape(batch_size * self.num_heads, -1)
        sample_args = (val_feats, sample_map_shapes, sample_map_start_ids, sample_locations, sample_map_ids)

        sampled_feats = pytorch_maps_sample_2d(*sample_args, return_derivatives=False)
        sampled_feats = sampled_feats.view(batch_size, self.num_heads, num_in_feats, -1, val_size // self.num_heads)
        sampled_feats = sampled_feats.transpose(1, 2)

        # Get and add position encodings to sampled value features if needed
        if hasattr(self, 'val_pos_encs'):
            sample_xy = sample_locations.view(batch_size, self.num_heads, num_in_feats, -1, 2)
            sample_xy = sample_xy - sample_priors[:, None, :, None, :2]

            if sample_priors.shape[-1] == 4:
                sample_xy = sample_xy / sample_priors[:, None, :, None, 2:]

            sample_z = sample_map_ids.view(batch_size, self.num_heads, num_in_feats, -1, 1)
            sample_z = sample_z / (self.num_levels-1)

            sample_xyz = torch.cat([sample_xy, sample_z], dim=4).transpose(1, 2)
            sampled_feats = sampled_feats + self.val_pos_encs(sample_xyz)

        # Get attention weights
        attn_weights = self.attn_weights(in_feats).view(*common_shape, self.num_levels * self.num_points)
        attn_weights = F.softmax(attn_weights, dim=3)

        # Get weighted value features
        weighted_feats = attn_weights[:, :, :, :, None] * sampled_feats
        weighted_feats = weighted_feats.sum(dim=3).view(batch_size, num_in_feats, val_size)

        # Get output features
        out_feats = self.out_proj(weighted_feats)

        # Update sample locations in storage dictionary if needed
        if self.update_sample_locations:
            storage_dict['sample_locations'] = sample_locations

        return out_feats


class PAv6(nn.Module):
    """
    Class implementing the PAv6 module.

    Attributes:
        num_heads (int): Integer containing the number of attention heads.
        num_levels (int): Integer containing the number of map levels to sample from.
        num_points (int): Integer containing the number of sampling points per head and level.

        val_proj (nn.Linear): Module computing value features from sample features.
        val_pos_encs (nn.Linear): Optional module computing the value position encodings.
        qry_proj (nn.Linear): Module computing query features from input features.
        attn_weight (nn.Parameter): Parameter containing the weight matrix used during attention weight computation.
        attn_bias (nn.Parameter): Parameter containing the bias vector used during attention weight computation.
        out_proj (nn.Linear): Module computing output features from weighted value features.

        update_sample_locations (bool): Boolean indicating whether sample locations should be updated.

        If update_sample_locations is True:
            steps_weight (nn.Parameter): Parameter containing the weight matrix used during sample step computation.
            steps_bias (nn.Parameter): Parameter containing the bias vector used during sample step computation.
            step_size (float): Size of the sample steps relative to the sample step normalization.
            step_norm_xy (str): String containing the normalization type of XY-direction sample steps.
    """

    def __init__(self, in_size, sample_size, out_size=-1, num_heads=8, num_levels=5, num_points=4, val_size=-1,
                 val_with_pos=False, qry_size=-1, step_size=-1, step_norm_xy='map'):
        """
        Initializes the PAv6 module.

        Args:
            in_size (int): Size of input features.
            sample_size (int): Size of sample features.
            out_size (int): Size of output features (default=-1).
            num_heads (int): Integer containing the number of attention heads (default=8).
            num_levels (int): Integer containing the number of map levels to sample from (default=5).
            num_points (int): Integer containing the number of sampling points per head and level (default=4).
            val_size (int): Size of value features (default=-1).
            val_with_pos (bool): Boolean indicating whether position info is added to value features (default=False).
            qry_size (int): Size of query features (default=-1).
            step_size (float): Size of the sample steps relative to the sample step normalization (default=-1).
            step_norm_xy (str): String containing the normalization type of XY-direction sample steps (default=map).

        Raises:
            ValueError: Error when the input feature size does not divide the number of heads.
            ValueError: Error when the value feature size does not divide the number of heads.
            ValueError: Error when the query feature size does not equal the value feature size.
        """

        # Initialization of default nn.Module
        super().__init__()

        # Set attributes related to the number of heads, levels and points
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points

        # Check divisibility input size by number of heads
        if in_size % num_heads != 0:
            error_msg = f"The input feature size ({in_size}) must divide the number of heads ({num_heads})."
            raise ValueError(error_msg)

        # Get and check size of value features
        if val_size == -1:
            val_size = in_size

        elif val_size % num_heads != 0:
            error_msg = f"The value feature size ({val_size}) must divide the number of heads ({num_heads})."
            raise ValueError(error_msg)

        # Initialize module computing the value features
        self.val_proj = nn.Linear(sample_size, val_size)
        nn.init.xavier_uniform_(self.val_proj.weight)
        nn.init.zeros_(self.val_proj.bias)

        # Initialize module computing the value position encodings if requested
        if val_with_pos:
            self.val_pos_encs = nn.Linear(3, val_size // num_heads)

        # Get and check size of query features
        if qry_size == -1:
            qry_size = val_size

        elif qry_size != val_size:
            error_msg = f"The query feature size ({qry_size}) must equal the value feature size ({val_size})."
            raise ValueError(error_msg)

        # Initialize module computing the query features
        self.qry_proj = nn.Linear(in_size, qry_size)
        nn.init.xavier_uniform_(self.qry_proj.weight)
        nn.init.zeros_(self.qry_proj.bias)

        # Initialize parameters computing the unnormalized attention weights
        self.attn_weight = nn.Parameter(torch.zeros(num_heads*num_levels*num_points, val_size // num_heads, 1))
        self.attn_bias = nn.Parameter(torch.zeros(num_heads*num_levels*num_points, 1, 1))

        # Initialize module computing the output features
        out_size = in_size if out_size == -1 else out_size
        self.out_proj = nn.Linear(val_size, out_size)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

        # Set attribute determining whether sample locations should be updated
        self.update_sample_locations = True

        # Initialize parameters computing the unnormalized sample steps
        self.steps_weight = nn.Parameter(torch.zeros(num_heads*num_levels*num_points*2, val_size // num_heads, 1))
        self.steps_bias = nn.Parameter(torch.zeros(num_heads*num_levels*num_points*2, 1, 1))

        # Set attributes related to the sizes of the sample steps
        self.step_size = step_size
        self.step_norm_xy = step_norm_xy

    def no_sample_locations_update(self):
        """
        Method changing the module to not update the sample locations.
        """

        # Change attribute to remember that sample locations should not be updated
        self.update_sample_locations = False

        # Delete all attributes related to the update of sample locations
        delattr(self, 'steps_weight')
        delattr(self, 'steps_bias')
        delattr(self, 'step_size')
        delattr(self, 'step_norm_xy')

    def forward(self, in_feats, sample_priors, sample_feats, sample_map_shapes, sample_map_start_ids, storage_dict,
                **kwargs):
        """
        Forward method of the PAv6 module.

        Args:
            in_feats (FloatTensor): Input features of shape [batch_size, num_in_feats, in_size].
            sample_priors (FloatTensor): Sample priors of shape [batch_size, num_in_feats, {2, 4}].
            sample_feats (FloatTensor): Sample features of shape [batch_size, num_sample_feats, sample_size].
            sample_map_shapes (LongTensor): Map shapes corresponding to samples of shape [num_levels, 2].
            sample_map_start_ids (LongTensor): Start indices of sample maps of shape [num_levels].
            storage_dict (Dict): Dictionary storing additional arguments such as the sample locations.
            kwargs (Dict): Dictionary of keyword arguments not used by this module.

        Returns:
            out_feats (FloatTensor): Output features of shape [batch_size, num_in_feats, out_size].

        Raises:
            ValueError: Error when the last dimension of 'sample_priors' is different from 2 or 4.
            ValueError: Error when last dimension of 'sample_priors' is not 4 when using 'anchor' step normalization.
            ValueError: Error when invalid sample step normalization type is provided.
        """

        # Get shapes of input tensors
        batch_size, num_in_feats = in_feats.shape[:2]
        common_shape = (batch_size, num_in_feats, self.num_heads)

        # Flip sample map shapes
        sample_map_shapes = sample_map_shapes.fliplr()

        # Get value features
        val_feats = self.val_proj(sample_feats)
        val_size = val_feats.shape[-1]
        val_feats = val_feats.view(batch_size, -1, self.num_heads, val_size // self.num_heads)
        val_feats = val_feats.transpose(1, 2).view(batch_size * self.num_heads, -1, val_size // self.num_heads)

        # Get sample locations
        sample_locations = storage_dict.pop('sample_locations', None)

        if sample_locations is None:
            thetas = torch.arange(self.num_heads, dtype=torch.float, device=sample_feats.device)
            thetas = thetas * (2.0 * math.pi / self.num_heads)

            sample_offsets = torch.stack([thetas.cos(), thetas.sin()], dim=1)
            sample_offsets = sample_offsets / sample_offsets.abs().max(dim=1, keepdim=True)[0]
            sample_offsets = sample_offsets.view(self.num_heads, 1, 1, 2).repeat(1, self.num_levels, 1, 1)

            sizes = torch.arange(1, self. num_points+1, dtype=torch.float, device=sample_feats.device)
            sample_offsets = sizes[None, None, :, None] * sample_offsets
            sample_offsets = sample_offsets[None, None, :, :, :, :]

            if sample_priors.shape[-1] == 2:
                offset_normalizers = sample_map_shapes[None, None, None, :, None, :]
                sample_locations = sample_priors[:, :, None, None, None, :]
                sample_locations = sample_locations + sample_offsets / offset_normalizers

            elif sample_priors.shape[-1] == 4:
                offset_factors = 0.5 * sample_priors[:, :, None, None, None, 2:] / self.num_points
                sample_locations = sample_priors[:, :, None, None, None, :2]
                sample_locations = sample_locations + sample_offsets * offset_factors

            else:
                error_msg = f"Last dimension of 'sample_priors' must be 2 or 4, but got {sample_priors.shape[-1]}."
                raise ValueError(error_msg)

            sample_locations = sample_locations.transpose(1, 2).reshape(batch_size * self.num_heads, -1, 2)

        # Get sampled value features and corresponding derivatives if needed
        sample_map_ids = torch.arange(self.num_levels, device=sample_feats.device)
        sample_map_ids = sample_map_ids[None, None, :, None]
        sample_map_ids = sample_map_ids.expand(batch_size * self.num_heads, num_in_feats, -1, self.num_points)
        sample_map_ids = sample_map_ids.reshape(batch_size * self.num_heads, -1)
        sample_args = (val_feats, sample_map_shapes, sample_map_start_ids, sample_locations, sample_map_ids)

        if self.update_sample_locations:
            sampled_feats, dx, dy = pytorch_maps_sample_2d(*sample_args, return_derivatives=True)
        else:
            sampled_feats = pytorch_maps_sample_2d(*sample_args, return_derivatives=False)

        sampled_feats = sampled_feats.view(batch_size, self.num_heads, num_in_feats, -1, val_size // self.num_heads)
        sampled_feats = sampled_feats.transpose(1, 2)

        # Get and add position encodings to sampled value features if needed
        if hasattr(self, 'val_pos_encs'):
            sample_xy = sample_locations.view(batch_size, self.num_heads, num_in_feats, -1, 2)
            sample_xy = sample_xy - sample_priors[:, None, :, None, :2]

            if sample_priors.shape[-1] == 4:
                sample_xy = sample_xy / sample_priors[:, None, :, None, 2:]

            sample_z = sample_map_ids.view(batch_size, self.num_heads, num_in_feats, -1, 1)
            sample_z = sample_z / (self.num_levels-1)

            sample_xyz = torch.cat([sample_xy, sample_z], dim=4).transpose(1, 2)
            sampled_feats = sampled_feats + self.val_pos_encs(sample_xyz)

        # Get query features
        qry_feats = self.qry_proj(in_feats).view(*common_shape, -1)

        # Get attention weights
        attn_feats = sampled_feats + qry_feats[:, :, :, None, :]
        attn_feats = attn_feats.reshape(batch_size * num_in_feats, -1, val_size // self.num_heads)
        attn_feats = attn_feats.transpose(0, 1)

        attn_weights = torch.bmm(attn_feats, self.attn_weight) + self.attn_bias
        attn_weights = attn_weights.view(self.num_heads, -1, batch_size, num_in_feats)
        attn_weights = attn_weights.permute(2, 3, 0, 1)
        attn_weights = F.softmax(attn_weights, dim=3)

        # Get weighted value features
        weighted_feats = attn_weights[:, :, :, :, None] * sampled_feats
        weighted_feats = weighted_feats.sum(dim=3).view(batch_size, num_in_feats, val_size)

        # Get output features
        out_feats = self.out_proj(weighted_feats)

        # Update sample locations in storage dictionary if needed
        if self.update_sample_locations:
            derivatives = torch.stack([dx, dy], dim=2)
            derivatives = derivatives.view(batch_size, self.num_heads, num_in_feats, -1, 2, val_size // self.num_heads)

            qry_feats = qry_feats.transpose(1, 2)
            derivatives = derivatives + qry_feats[:, :, :, None, None, :]

            derivatives = derivatives.permute(1, 3, 4, 0, 2, 5)
            derivatives = derivatives.reshape(-1, batch_size * num_in_feats, val_size // self.num_heads)

            sample_steps = torch.bmm(derivatives, self.steps_weight) + self.steps_bias
            sample_steps = sample_steps.view(self.num_heads, self.num_levels, self.num_points, 2, batch_size, -1)
            sample_steps = sample_steps.permute(4, 0, 5, 1, 2, 3)

            if self.step_size > 0:
                sample_steps = self.step_size * F.normalize(sample_steps, dim=5)

            if self.step_norm_xy == 'map':
                step_normalizers = sample_map_shapes[None, None, None, :, None, :]
                sample_steps = sample_steps / step_normalizers

            elif self.step_norm_xy == 'anchor':
                if sample_priors.shape[-1] == 4:
                    step_factors = 0.5 * sample_priors[:, None, :, None, None, 2:] / self.num_points
                    sample_steps = sample_steps * step_factors

                else:
                    error_msg = "Last dimension of 'sample_priors' must be 4 when using 'anchor' step normalization."
                    raise ValueError(error_msg)

            else:
                error_msg = f"Invalid sample step normalization type '{self.step_norm_xy}'."
                raise ValueError(error_msg)

            sample_steps = sample_steps.reshape(batch_size * self.num_heads, -1, 2)
            next_sample_locations = sample_locations + sample_steps
            storage_dict['sample_locations'] = next_sample_locations

        return out_feats


class PAv7(nn.Module):
    """
    Class implementing the PAv7 module.

    Attributes:
        num_heads (int): Integer containing the number of attention heads.
        num_levels (int): Integer containing the number of map levels to sample from.
        num_points (int): Integer containing the number of sampling points per head and level.

        val_proj (nn.Linear): Module computing value features from sample features.
        val_pos_encs (nn.Linear): Optional module computing the value position encodings.
        attn_weights (nn.Linear): Module computing the attention weights from the input features.
        out_proj (nn.Linear): Module computing output features from weighted value features.

        update_sample_locations (bool): Boolean indicating whether sample locations should be updated.

        If update_sample_locations is True:
            qry_proj (nn.Linear): Module computing query features from input features.
            steps_weight (nn.Parameter): Parameter containing the weight matrix used during sample step computation.
            steps_bias (nn.Parameter): Parameter containing the bias vector used during sample step computation.
            step_size (float): Size of the sample steps relative to the sample step normalization.
            step_norm_xy (str): String containing the normalization type of XY-direction sample steps.
            step_norm_z (float): Value normalizing the sample steps in the Z-direction.
    """

    def __init__(self, in_size, sample_size, out_size=-1, num_heads=8, num_levels=5, num_points=4, val_size=-1,
                 val_with_pos=False, qry_size=-1, step_size=-1, step_norm_xy='map', step_norm_z=1.0):
        """
        Initializes the PAv7 module.

        Args:
            in_size (int): Size of input features.
            sample_size (int): Size of sample features.
            out_size (int): Size of output features (default=-1).
            num_heads (int): Integer containing the number of attention heads (default=8).
            num_levels (int): Integer containing the number of map levels to sample from (default=5).
            num_points (int): Integer containing the number of sampling points per head and level (default=4).
            val_size (int): Size of value features (default=-1).
            val_with_pos (bool): Boolean indicating whether position info is added to value features (default=False).
            qry_size (int): Size of query features (default=-1).
            step_size (float): Size of the sample steps relative to the sample step normalization (default=-1).
            step_norm_xy (str): String containing the normalization type of XY-direction sample steps (default=map).
            step_norm_z (float): Value normalizing the sample steps in the Z-direction (default=1.0).

        Raises:
            ValueError: Error when the input feature size does not divide the number of heads.
            ValueError: Error when the value feature size does not divide the number of heads.
            ValueError: Error when the query feature size does not equal the value feature size.
        """

        # Initialization of default nn.Module
        super().__init__()

        # Set attributes related to the number of heads, levels and points
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points

        # Check divisibility input size by number of heads
        if in_size % num_heads != 0:
            error_msg = f"The input feature size ({in_size}) must divide the number of heads ({num_heads})."
            raise ValueError(error_msg)

        # Get and check size of value features
        if val_size == -1:
            val_size = in_size

        elif val_size % num_heads != 0:
            error_msg = f"The value feature size ({val_size}) must divide the number of heads ({num_heads})."
            raise ValueError(error_msg)

        # Initialize module computing the value features
        self.val_proj = nn.Linear(sample_size, val_size)
        nn.init.xavier_uniform_(self.val_proj.weight)
        nn.init.zeros_(self.val_proj.bias)

        # Initialize module computing the value position encodings if requested
        if val_with_pos:
            self.val_pos_encs = nn.Linear(3, val_size // num_heads)

        # Initialize module computing the unnormalized attention weights
        self.attn_weights = nn.Linear(in_size, num_heads * num_levels * num_points)
        nn.init.zeros_(self.attn_weights.weight)
        nn.init.zeros_(self.attn_weights.bias)

        # Initialize module computing the output features
        out_size = in_size if out_size == -1 else out_size
        self.out_proj = nn.Linear(val_size, out_size)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

        # Set attribute determining whether sample locations should be updated
        self.update_sample_locations = True

        # Get and check size of query features
        if qry_size == -1:
            qry_size = val_size

        elif qry_size != val_size:
            error_msg = f"The query feature size ({qry_size}) must equal the value feature size ({val_size})."
            raise ValueError(error_msg)

        # Initialize module computing the query features
        self.qry_proj = nn.Linear(in_size, qry_size)
        nn.init.xavier_uniform_(self.qry_proj.weight)
        nn.init.zeros_(self.qry_proj.bias)

        # Initialize modules computing the unnormalized sample steps
        self.steps_weight = nn.Parameter(torch.zeros(num_heads*num_levels*num_points*3, val_size // num_heads, 1))
        self.steps_bias = nn.Parameter(torch.zeros(num_heads*num_levels*num_points*3, 1, 1))

        # Set attributes related to the sizes of the sample steps
        self.step_size = step_size
        self.step_norm_xy = step_norm_xy
        self.step_norm_z = step_norm_z

    def no_sample_locations_update(self):
        """
        Method changing the module to not update the sample locations.
        """

        # Change attribute to remember that sample locations should not be updated
        self.update_sample_locations = False

        # Delete all attributes related to the update of sample locations
        delattr(self, 'qry_proj')
        delattr(self, 'steps_weight')
        delattr(self, 'steps_bias')
        delattr(self, 'step_size')
        delattr(self, 'step_norm_xy')
        delattr(self, 'step_norm_z')

    def forward(self, in_feats, sample_priors, sample_feats, sample_map_shapes, sample_map_start_ids, storage_dict,
                **kwargs):
        """
        Forward method of the PAv7 module.

        Args:
            in_feats (FloatTensor): Input features of shape [batch_size, num_in_feats, in_size].
            sample_priors (FloatTensor): Sample priors of shape [batch_size, num_in_feats, {2, 4}].
            sample_feats (FloatTensor): Sample features of shape [batch_size, num_sample_feats, sample_size].
            sample_map_shapes (LongTensor): Map shapes corresponding to samples of shape [num_levels, 2].
            sample_map_start_ids (LongTensor): Start indices of sample maps of shape [num_levels].
            storage_dict (Dict): Dictionary storing additional arguments such as the sample locations.
            kwargs (Dict): Dictionary of keyword arguments not used by this module.

        Returns:
            out_feats (FloatTensor): Output features of shape [batch_size, num_in_feats, out_size].

        Raises:
            ValueError: Error when the last dimension of 'sample_priors' is different from 2 or 4.
            ValueError: Error when last dimension of 'sample_priors' is not 4 when using 'anchor' step normalization.
            ValueError: Error when invalid sample step normalization type is provided.
        """

        # Get shapes of input tensors
        batch_size, num_in_feats = in_feats.shape[:2]
        common_shape = (batch_size, num_in_feats, self.num_heads)

        # Flip sample map shapes
        sample_map_shapes = sample_map_shapes.fliplr()

        # Get value features
        val_feats = self.val_proj(sample_feats)
        val_size = val_feats.shape[-1]
        val_feats = val_feats.view(batch_size, -1, self.num_heads, val_size // self.num_heads)
        val_feats = val_feats.transpose(1, 2).view(batch_size * self.num_heads, -1, val_size // self.num_heads)

        # Get sample locations
        sample_locations = storage_dict.pop('sample_locations', None)

        if sample_locations is None:
            thetas = torch.arange(self.num_heads, dtype=torch.float, device=sample_feats.device)
            thetas = thetas * (2.0 * math.pi / self.num_heads)

            sample_offsets = torch.stack([thetas.cos(), thetas.sin()], dim=1)
            sample_offsets = sample_offsets / sample_offsets.abs().max(dim=1, keepdim=True)[0]
            sample_offsets = sample_offsets.view(self.num_heads, 1, 1, 2).repeat(1, self.num_levels, 1, 1)

            sizes = torch.arange(1, self. num_points+1, dtype=torch.float, device=sample_feats.device)
            sample_offsets = sizes[None, None, :, None] * sample_offsets
            sample_offsets = sample_offsets[None, None, :, :, :, :]

            if sample_priors.shape[-1] == 2:
                offset_normalizers = sample_map_shapes[None, None, None, :, None, :]
                sample_locations = sample_priors[:, :, None, None, None, :]
                sample_locations = sample_locations + sample_offsets / offset_normalizers

            elif sample_priors.shape[-1] == 4:
                offset_factors = 0.5 * sample_priors[:, :, None, None, None, 2:] / self.num_points
                sample_locations = sample_priors[:, :, None, None, None, :2]
                sample_locations = sample_locations + sample_offsets * offset_factors

            else:
                error_msg = f"Last dimension of 'sample_priors' must be 2 or 4, but got {sample_priors.shape[-1]}."
                raise ValueError(error_msg)

            sample_z = torch.linspace(0, 1, self.num_levels, dtype=sample_priors.dtype, device=sample_priors.device)
            sample_z = sample_z.view(1, 1, 1, self.num_levels, 1, 1).expand(*common_shape, -1, self.num_points, -1)

            sample_locations = torch.cat([sample_locations, sample_z], dim=5)
            sample_locations = sample_locations.transpose(1, 2).reshape(batch_size * self.num_heads, -1, 3)

        # Get sampled value features and corresponding derivatives if needed
        sample_args = (val_feats, sample_map_shapes, sample_map_start_ids, sample_locations)

        if self.update_sample_locations:
            sampled_feats, dx, dy, dz = pytorch_maps_sample_3d(*sample_args, return_derivatives=True)
        else:
            sampled_feats = pytorch_maps_sample_3d(*sample_args, return_derivatives=False)

        sampled_feats = sampled_feats.view(batch_size, self.num_heads, num_in_feats, -1, val_size // self.num_heads)
        sampled_feats = sampled_feats.transpose(1, 2)

        # Get and add position encodings to sampled value features if needed
        if hasattr(self, 'val_pos_encs'):
            sample_xyz = sample_locations.clone()
            sample_xyz = sample_xyz.view(batch_size, self.num_heads, num_in_feats, -1, 3)

            sample_xy = sample_xyz[:, :, :, :, :2]
            sample_xy = sample_xy - sample_priors[:, None, :, None, :2]

            if sample_priors.shape[-1] == 4:
                sample_xy = sample_xy / sample_priors[:, None, :, None, 2:]

            sample_xyz[:, :, :, :, :2] = sample_xy
            sample_xyz = sample_xyz.transpose(1, 2)
            sampled_feats = sampled_feats + self.val_pos_encs(sample_xyz)

        # Get attention weights
        attn_weights = self.attn_weights(in_feats).view(*common_shape, self.num_levels * self.num_points)
        attn_weights = F.softmax(attn_weights, dim=3)

        # Get weighted value features
        weighted_feats = attn_weights[:, :, :, :, None] * sampled_feats
        weighted_feats = weighted_feats.sum(dim=3).view(batch_size, num_in_feats, val_size)

        # Get output features
        out_feats = self.out_proj(weighted_feats)

        # Update sample locations in storage dictionary if needed
        if self.update_sample_locations:
            derivatives = torch.stack([dx, dy, dz], dim=2)
            derivatives = derivatives.view(batch_size, self.num_heads, num_in_feats, -1, 3, val_size // self.num_heads)

            qry_feats = self.qry_proj(in_feats)
            qry_feats = qry_feats.view(*common_shape, -1).transpose(1, 2)
            derivatives = derivatives + qry_feats[:, :, :, None, None, :]

            derivatives = derivatives.permute(1, 3, 4, 0, 2, 5)
            derivatives = derivatives.reshape(-1, batch_size * num_in_feats, val_size // self.num_heads)

            sample_steps = torch.bmm(derivatives, self.steps_weight) + self.steps_bias
            sample_steps = sample_steps.view(self.num_heads, self.num_levels, self.num_points, 3, batch_size, -1)
            sample_steps = sample_steps.permute(4, 0, 5, 1, 2, 3)

            if self.step_size > 0:
                sample_steps = self.step_size * F.normalize(sample_steps, dim=5)

            if self.step_norm_xy == 'map':
                step_normalizers = sample_map_shapes[None, None, None, :, None, :]
                sample_steps[:, :, :, :, :, :2] = sample_steps[:, :, :, :, :, :2] / step_normalizers

            elif self.step_norm_xy == 'anchor':
                if sample_priors.shape[-1] == 4:
                    step_factors = 0.5 * sample_priors[:, None, :, None, None, 2:] / self.num_points
                    sample_steps[:, :, :, :, :, :2] = sample_steps[:, :, :, :, :, :2] * step_factors

                else:
                    error_msg = "Last dimension of 'sample_priors' must be 4 when using 'anchor' step normalization."
                    raise ValueError(error_msg)

            else:
                error_msg = f"Invalid sample step normalization type '{self.step_norm_xy}'."
                raise ValueError(error_msg)

            sample_steps[:, :, :, :, :, 2] = sample_steps[:, :, :, :, :, 2] / self.step_norm_z
            sample_steps = sample_steps.reshape(batch_size * self.num_heads, -1, 3)
            next_sample_locations = sample_locations + sample_steps
            storage_dict['sample_locations'] = next_sample_locations

        return out_feats


class PAv8(nn.Module):
    """
    Class implementing the PAv8 module.

    Attributes:
        num_heads (int): Integer containing the number of attention heads.
        num_particles (int): Integer containing the number of particles per head.

        val_proj (nn.Linear): Module computing value features from sample features.
        val_pos_encs (nn.Linear): Optional module computing the value position encodings.
        attn_weights (nn.Linear): Module computing the attention weights from the input features.
        out_proj (nn.Linear): Module computing output features from weighted value features.

        update_sample_locations (bool): Boolean indicating whether sample locations should be updated.

        If update_sample_locations is True:
            qry_proj (nn.Linear): Module computing query features from input features.
            steps_weight (nn.Parameter): Parameter containing the weight matrix used during sample step computation.
            steps_bias (nn.Parameter): Parameter containing the bias vector used during sample step computation.
            step_size (float): Size of the sample steps relative to the sample step normalization.
            step_norm_xy (str): String containing the normalization type of XY-direction sample steps.
            step_norm_z (float): Value normalizing the sample steps in the Z-direction.
    """

    def __init__(self, in_size, sample_size, out_size=-1, num_heads=8, val_size=-1, val_with_pos=False, qry_size=-1,
                 step_size=-1, step_norm_xy='map', step_norm_z=1.0, num_particles=20):
        """
        Initializes the PAv8 module.

        Args:
            in_size (int): Size of input features.
            sample_size (int): Size of sample features.
            out_size (int): Size of output features (default=-1).
            num_heads (int): Integer containing the number of attention heads (default=8).
            val_size (int): Size of value features (default=-1).
            val_with_pos (bool): Boolean indicating whether position info is added to value features (default=False).
            qry_size (int): Size of query features (default=-1).
            step_size (float): Size of the sample steps relative to the sample step normalization (default=-1).
            step_norm_xy (str): String containing the normalization type of XY-direction sample steps (default=map).
            step_norm_z (float): Value normalizing the sample steps in the Z-direction (default=1.0).
            num_particles (int): Integer containing the number of particles per head (default=20).

        Raises:
            ValueError: Error when the input feature size does not divide the number of heads.
            ValueError: Error when the value feature size does not divide the number of heads.
            ValueError: Error when the query feature size does not equal the value feature size.
        """

        # Initialization of default nn.Module
        super().__init__()

        # Set attributes related to the number of heads and particles
        self.num_heads = num_heads
        self.num_particles = num_particles

        # Check divisibility input size by number of heads
        if in_size % num_heads != 0:
            error_msg = f"The input feature size ({in_size}) must divide the number of heads ({num_heads})."
            raise ValueError(error_msg)

        # Get and check size of value features
        if val_size == -1:
            val_size = in_size

        elif val_size % num_heads != 0:
            error_msg = f"The value feature size ({val_size}) must divide the number of heads ({num_heads})."
            raise ValueError(error_msg)

        # Initialize module computing the value features
        self.val_proj = nn.Linear(sample_size, val_size)
        nn.init.xavier_uniform_(self.val_proj.weight)
        nn.init.zeros_(self.val_proj.bias)

        # Initialize module computing the value position encodings if requested
        if val_with_pos:
            self.val_pos_encs = nn.Linear(3, val_size // num_heads)

        # Initialize module computing the unnormalized attention weights
        self.attn_weights = nn.Linear(in_size, num_heads * num_particles)
        nn.init.zeros_(self.attn_weights.weight)
        nn.init.zeros_(self.attn_weights.bias)

        # Initialize module computing the output features
        out_size = in_size if out_size == -1 else out_size
        self.out_proj = nn.Linear(val_size, out_size)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

        # Set attribute determining whether sample locations should be updated
        self.update_sample_locations = True

        # Get and check size of query features
        if qry_size == -1:
            qry_size = val_size

        elif qry_size != val_size:
            error_msg = f"The query feature size ({qry_size}) must equal the value feature size ({val_size})."
            raise ValueError(error_msg)

        # Initialize module computing the query features
        self.qry_proj = nn.Linear(in_size, qry_size)
        nn.init.xavier_uniform_(self.qry_proj.weight)
        nn.init.zeros_(self.qry_proj.bias)

        # Initialize modules computing the unnormalized sample steps
        self.steps_weight = nn.Parameter(torch.zeros(num_heads*num_particles*3, val_size // num_heads, 1))
        self.steps_bias = nn.Parameter(torch.zeros(num_heads*num_particles*3, 1, 1))

        # Set attributes related to the sizes of the sample steps
        self.step_size = step_size
        self.step_norm_xy = step_norm_xy
        self.step_norm_z = step_norm_z

    def no_sample_locations_update(self):
        """
        Method changing the module to not update the sample locations.
        """

        # Change attribute to remember that sample locations should not be updated
        self.update_sample_locations = False

        # Delete all attributes related to the update of sample locations
        delattr(self, 'qry_proj')
        delattr(self, 'steps_weight')
        delattr(self, 'steps_bias')
        delattr(self, 'step_size')
        delattr(self, 'step_norm_xy')
        delattr(self, 'step_norm_z')

    def forward(self, in_feats, sample_priors, sample_feats, sample_map_shapes, sample_map_start_ids, storage_dict,
                map_ids=None, **kwargs):
        """
        Forward method of the PAv8 module.

        Args:
            in_feats (FloatTensor): Input features of shape [batch_size, num_in_feats, in_size].
            sample_priors (FloatTensor): Sample priors of shape [batch_size, num_in_feats, {2, 4}].
            sample_feats (FloatTensor): Sample features of shape [batch_size, num_sample_feats, sample_size].
            sample_map_shapes (LongTensor): Map shapes corresponding to samples of shape [num_levels, 2].
            sample_map_start_ids (LongTensor): Start indices of sample maps of shape [num_levels].
            storage_dict (Dict): Dictionary storing additional arguments such as the sample locations.
            map_ids (LongTensor): Map indices of input features of shape [batch_size, num_in_feats] (default=None).
            kwargs (Dict): Dictionary of keyword arguments not used by this module.

        Returns:
            out_feats (FloatTensor): Output features of shape [batch_size, num_in_feats, out_size].

        Raises:
            ValueError: Error when no map indices are provided when sample locations are missing.
            ValueError: Error when the last dimension of 'sample_priors' is different from 2 or 4.
            ValueError: Error when last dimension of 'sample_priors' is not 4 when using 'anchor' step normalization.
            ValueError: Error when invalid sample step normalization type is provided.
        """

        # Get shapes of input tensors
        batch_size, num_in_feats = in_feats.shape[:2]
        common_shape = (batch_size, num_in_feats, self.num_heads)

        # Flip sample map shapes
        sample_map_shapes = sample_map_shapes.fliplr()

        # Get value features
        val_feats = self.val_proj(sample_feats)
        val_size = val_feats.shape[-1]
        val_feats = val_feats.view(batch_size, -1, self.num_heads, val_size // self.num_heads)
        val_feats = val_feats.transpose(1, 2).view(batch_size * self.num_heads, -1, val_size // self.num_heads)

        # Get sample locations
        sample_locations = storage_dict.pop('sample_locations', None)

        if sample_locations is None:
            thetas = torch.arange(self.num_heads, dtype=torch.float, device=sample_feats.device)
            thetas = thetas * (2.0 * math.pi / self.num_heads)

            sample_offsets = torch.stack([thetas.cos(), thetas.sin()], dim=1)
            sample_offsets = sample_offsets / sample_offsets.abs().max(dim=1, keepdim=True)[0]
            sample_offsets = sample_offsets.view(self.num_heads, 1, 2)

            sizes = torch.arange(1, self. num_particles+1, dtype=torch.float, device=sample_feats.device)
            sample_offsets = sizes[None, :, None] * sample_offsets
            sample_offsets = sample_offsets[None, None, :, :, :]

            if map_ids is None:
                error_msg = "Map indices must be provided when sample locations are missing."
                raise ValueError(error_msg)

            if sample_priors.shape[-1] == 2:
                offset_normalizers = sample_map_shapes[map_ids, None, None, :]
                sample_locations = sample_priors[:, :, None, None, :]
                sample_locations = sample_locations + sample_offsets / offset_normalizers

            elif sample_priors.shape[-1] == 4:
                offset_factors = 0.5 * sample_priors[:, :, None, None, 2:] / self.num_particles
                sample_locations = sample_priors[:, :, None, None, :2]
                sample_locations = sample_locations + sample_offsets * offset_factors

            else:
                error_msg = f"Last dimension of 'sample_priors' must be 2 or 4, but got {sample_priors.shape[-1]}."
                raise ValueError(error_msg)

            num_levels = len(sample_map_start_ids)
            sample_z = map_ids / (num_levels-1)
            sample_z = sample_z[:, :, None, None, None].expand(-1, -1, self.num_heads, self.num_particles, -1)

            sample_locations = torch.cat([sample_locations, sample_z], dim=4)
            sample_locations = sample_locations.transpose(1, 2).reshape(batch_size * self.num_heads, -1, 3)

        # Get sampled value features and corresponding derivatives if needed
        sample_args = (val_feats, sample_map_shapes, sample_map_start_ids, sample_locations)

        if self.update_sample_locations:
            sampled_feats, dx, dy, dz = pytorch_maps_sample_3d(*sample_args, return_derivatives=True)
        else:
            sampled_feats = pytorch_maps_sample_3d(*sample_args, return_derivatives=False)

        sampled_feats = sampled_feats.view(batch_size, self.num_heads, num_in_feats, -1, val_size // self.num_heads)
        sampled_feats = sampled_feats.transpose(1, 2)

        # Get and add position encodings to sampled value features if needed
        if hasattr(self, 'val_pos_encs'):
            sample_xyz = sample_locations.clone()
            sample_xyz = sample_xyz.view(batch_size, self.num_heads, num_in_feats, -1, 3)

            sample_xy = sample_xyz[:, :, :, :, :2]
            sample_xy = sample_xy - sample_priors[:, None, :, None, :2]

            if sample_priors.shape[-1] == 4:
                sample_xy = sample_xy / sample_priors[:, None, :, None, 2:]

            sample_xyz[:, :, :, :, :2] = sample_xy
            sample_xyz = sample_xyz.transpose(1, 2)
            sampled_feats = sampled_feats + self.val_pos_encs(sample_xyz)

        # Get attention weights
        attn_weights = self.attn_weights(in_feats).view(*common_shape, self.num_particles)
        attn_weights = F.softmax(attn_weights, dim=3)

        # Get weighted value features
        weighted_feats = attn_weights[:, :, :, :, None] * sampled_feats
        weighted_feats = weighted_feats.sum(dim=3).view(batch_size, num_in_feats, val_size)

        # Get output features
        out_feats = self.out_proj(weighted_feats)

        # Update sample locations in storage dictionary if needed
        if self.update_sample_locations:
            derivatives = torch.stack([dx, dy, dz], dim=2)
            derivatives = derivatives.view(batch_size, self.num_heads, num_in_feats, -1, 3, val_size // self.num_heads)

            qry_feats = self.qry_proj(in_feats)
            qry_feats = qry_feats.view(*common_shape, -1).transpose(1, 2)
            derivatives = derivatives + qry_feats[:, :, :, None, None, :]

            derivatives = derivatives.permute(1, 3, 4, 0, 2, 5)
            derivatives = derivatives.reshape(-1, batch_size * num_in_feats, val_size // self.num_heads)

            sample_steps = torch.bmm(derivatives, self.steps_weight) + self.steps_bias
            sample_steps = sample_steps.view(self.num_heads, self.num_particles, 3, batch_size, -1)
            sample_steps = sample_steps.permute(3, 0, 4, 1, 2)

            if self.step_size > 0:
                sample_steps = self.step_size * F.normalize(sample_steps, dim=4)

            if self.step_norm_xy == 'map':
                with torch.no_grad():
                    num_levels = len(sample_map_start_ids)
                    map_ids = sample_locations[:, :, 2] * (num_levels-1)

                    lower_ids = map_ids.floor().to(dtype=torch.int64).clamp_(max=num_levels-2)
                    upper_ids = lower_ids + 1

                    lower_ws = upper_ids - map_ids
                    lower_ws = lower_ws[:, :, None]
                    upper_ws = 1 - lower_ws

                    step_normalizers = lower_ws * sample_map_shapes[lower_ids] + upper_ws * sample_map_shapes[upper_ids]
                    step_normalizers = step_normalizers.view(batch_size, self.num_heads, num_in_feats, -1, 2)

                sample_steps[:, :, :, :, :2] = sample_steps[:, :, :, :, :2] / step_normalizers

            elif self.step_norm_xy == 'anchor':
                if sample_priors.shape[-1] == 4:
                    step_factors = sample_priors[:, None, :, None, 2:] / 8
                    sample_steps[:, :, :, :, :2] = sample_steps[:, :, :, :, :2] * step_factors

                else:
                    error_msg = "Last dimension of 'sample_priors' must be 4 when using 'anchor' step normalization."
                    raise ValueError(error_msg)

            else:
                error_msg = f"Invalid sample step normalization type '{self.step_norm_xy}'."
                raise ValueError(error_msg)

            sample_steps[:, :, :, :, 2] = sample_steps[:, :, :, :, 2] / self.step_norm_z
            sample_steps = sample_steps.reshape(batch_size * self.num_heads, -1, 3)
            next_sample_locations = sample_locations + sample_steps
            storage_dict['sample_locations'] = next_sample_locations

        return out_feats


class PAv9(nn.Module):
    """
    Class implementing the PAv9 module.

    Attributes:
        num_heads (int): Integer containing the number of attention heads.
        num_levels (int): Integer containing the number of map levels to sample from.
        num_points (int): Integer containing the number of sampling points per head and level.

        val_proj (nn.Linear): Module computing value features from sample features.
        val_pos_encs (nn.Linear): Optional module computing the value position encodings.
        attn_weights (nn.Linear): Module computing the attention weights from the input features.
        out_proj (nn.Linear): Module computing output features from weighted value features.

        update_sample_locations (bool): Boolean indicating whether sample locations should be updated.

        If update_sample_locations is True:
            qry_proj (nn.Linear): Module computing query features from input features.
            steps_weight (nn.Parameter): Parameter containing the weight matrix used during sample step computation.
            steps_bias (nn.Parameter): Parameter containing the bias vector used during sample step computation.
            step_size (float): Size of the sample steps relative to the sample step normalization.
            step_norm_xy (str): String containing the normalization type of XY-direction sample steps.
    """

    def __init__(self, in_size, sample_size, out_size=-1, num_heads=8, num_levels=5, num_points=4, val_size=-1,
                 val_with_pos=False, qry_size=-1, step_size=-1, step_norm_xy='map'):
        """
        Initializes the PAv9 module.

        Args:
            in_size (int): Size of input features.
            sample_size (int): Size of sample features.
            out_size (int): Size of output features (default=-1).
            num_heads (int): Integer containing the number of attention heads (default=8).
            num_levels (int): Integer containing the number of map levels to sample from (default=5).
            num_points (int): Integer containing the number of sampling points per head and level (default=4).
            val_size (int): Size of value features (default=-1).
            val_with_pos (bool): Boolean indicating whether position info is added to value features (default=False).
            qry_size (int): Size of query features (default=-1).
            step_size (float): Size of the sample steps relative to the sample step normalization (default=-1).
            step_norm_xy (str): String containing the normalization type of XY-direction sample steps (default=map).

        Raises:
            ValueError: Error when the input feature size does not divide the number of heads.
            ValueError: Error when the value feature size does not divide the number of heads.
            ValueError: Error when the query feature size does not equal the value feature size.
        """

        # Initialization of default nn.Module
        super().__init__()

        # Set attributes related to the number of heads, levels and points
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points

        # Check divisibility input size by number of heads
        if in_size % num_heads != 0:
            error_msg = f"The input feature size ({in_size}) must divide the number of heads ({num_heads})."
            raise ValueError(error_msg)

        # Get and check size of value features
        if val_size == -1:
            val_size = in_size

        elif val_size % num_heads != 0:
            error_msg = f"The value feature size ({val_size}) must divide the number of heads ({num_heads})."
            raise ValueError(error_msg)

        # Initialize module computing the value features
        self.val_proj = nn.Linear(sample_size, val_size)
        nn.init.xavier_uniform_(self.val_proj.weight)
        nn.init.zeros_(self.val_proj.bias)

        # Initialize module computing the value position encodings if requested
        if val_with_pos:
            self.val_pos_encs = nn.Linear(3, val_size // num_heads)

        # Initialize module computing the unnormalized attention weights
        self.attn_weights = nn.Linear(in_size, num_heads * num_levels * num_points)
        nn.init.zeros_(self.attn_weights.weight)
        nn.init.zeros_(self.attn_weights.bias)

        # Initialize module computing the output features
        out_size = in_size if out_size == -1 else out_size
        self.out_proj = nn.Linear(val_size, out_size)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

        # Set attribute determining whether sample locations should be updated
        self.update_sample_locations = True

        # Get and check size of query features
        if qry_size == -1:
            qry_size = val_size

        elif qry_size != val_size:
            error_msg = f"The query feature size ({qry_size}) must equal the value feature size ({val_size})."
            raise ValueError(error_msg)

        # Initialize module computing the query features
        self.qry_proj = nn.Linear(in_size, qry_size)
        nn.init.xavier_uniform_(self.qry_proj.weight)
        nn.init.zeros_(self.qry_proj.bias)

        # Initialize modules computing the unnormalized sample steps
        self.steps_weight = nn.Parameter(torch.zeros(num_heads*num_levels*num_points*2, val_size // num_heads, 1))
        self.steps_bias = nn.Parameter(torch.zeros(num_heads*num_levels*num_points*2, 1, 1))

        # Set attributes related to the sizes of the sample steps
        self.step_size = step_size
        self.step_norm_xy = step_norm_xy

    def no_sample_locations_update(self):
        """
        Method changing the module to not update the sample locations.
        """

        # Change attribute to remember that sample locations should not be updated
        self.update_sample_locations = False

        # Delete all attributes related to the update of sample locations
        delattr(self, 'qry_proj')
        delattr(self, 'steps_weight')
        delattr(self, 'steps_bias')
        delattr(self, 'step_size')
        delattr(self, 'step_norm_xy')

    def forward(self, in_feats, sample_priors, sample_feats, sample_map_shapes, sample_map_start_ids, storage_dict,
                **kwargs):
        """
        Forward method of the PAv9 module.

        Args:
            in_feats (FloatTensor): Input features of shape [batch_size, num_in_feats, in_size].
            sample_priors (FloatTensor): Sample priors of shape [batch_size, num_in_feats, {2, 4}].
            sample_feats (FloatTensor): Sample features of shape [batch_size, num_sample_feats, sample_size].
            sample_map_shapes (LongTensor): Map shapes corresponding to samples of shape [num_levels, 2].
            sample_map_start_ids (LongTensor): Start indices of sample maps of shape [num_levels].
            storage_dict (Dict): Dictionary storing additional arguments such as the sample locations.
            kwargs (Dict): Dictionary of keyword arguments not used by this module.

        Returns:
            out_feats (FloatTensor): Output features of shape [batch_size, num_in_feats, out_size].

        Raises:
            ValueError: Error when the last dimension of 'sample_priors' is different from 2 or 4.
            ValueError: Error when last dimension of 'sample_priors' is not 4 when using 'anchor' step normalization.
            ValueError: Error when invalid sample step normalization type is provided.
        """

        # Get shapes of input tensors
        batch_size, num_in_feats = in_feats.shape[:2]
        common_shape = (batch_size, num_in_feats, self.num_heads)

        # Flip sample map shapes
        sample_map_shapes = sample_map_shapes.fliplr()

        # Get value features
        val_feats = self.val_proj(sample_feats)
        val_size = val_feats.shape[-1]
        val_feats = val_feats.view(batch_size, -1, self.num_heads, val_size // self.num_heads)
        val_feats = val_feats.transpose(1, 2).view(batch_size * self.num_heads, -1, val_size // self.num_heads)

        # Get sample locations
        sample_locations = storage_dict.pop('sample_locations', None)

        if sample_locations is None:
            thetas = torch.arange(self.num_heads, dtype=torch.float, device=sample_feats.device)
            thetas = thetas * (2.0 * math.pi / self.num_heads)

            sample_offsets = torch.stack([thetas.cos(), thetas.sin()], dim=1)
            sample_offsets = sample_offsets / sample_offsets.abs().max(dim=1, keepdim=True)[0]
            sample_offsets = sample_offsets.view(self.num_heads, 1, 1, 2).repeat(1, self.num_levels, 1, 1)

            sizes = torch.arange(1, self. num_points+1, dtype=torch.float, device=sample_feats.device)
            sample_offsets = sizes[None, None, :, None] * sample_offsets
            sample_offsets = sample_offsets[None, None, :, :, :, :]

            if sample_priors.shape[-1] == 2:
                offset_normalizers = sample_map_shapes[None, None, None, :, None, :]
                sample_locations = sample_priors[:, :, None, None, None, :]
                sample_locations = sample_locations + sample_offsets / offset_normalizers

            elif sample_priors.shape[-1] == 4:
                offset_factors = 0.5 * sample_priors[:, :, None, None, None, 2:] / self.num_points
                sample_locations = sample_priors[:, :, None, None, None, :2]
                sample_locations = sample_locations + sample_offsets * offset_factors

            else:
                error_msg = f"Last dimension of 'sample_priors' must be 2 or 4, but got {sample_priors.shape[-1]}."
                raise ValueError(error_msg)

            sample_locations = sample_locations.transpose(1, 2).reshape(batch_size * self.num_heads, -1, 2)

        # Get sampled value features and corresponding derivatives if needed
        sample_map_ids = torch.arange(self.num_levels, device=sample_feats.device)
        sample_map_ids = sample_map_ids[None, None, :, None]
        sample_map_ids = sample_map_ids.expand(batch_size * self.num_heads, num_in_feats, -1, self.num_points)
        sample_map_ids = sample_map_ids.reshape(batch_size * self.num_heads, -1)
        sample_args = (val_feats, sample_map_shapes, sample_map_start_ids, sample_locations, sample_map_ids)

        sampled_feats = pytorch_maps_sample_2d(*sample_args, return_derivatives=False)
        sampled_feats = sampled_feats.view(batch_size, self.num_heads, num_in_feats, -1, val_size // self.num_heads)
        sampled_feats = sampled_feats.transpose(1, 2)

        # Get and add position encodings to sampled value features if needed
        if hasattr(self, 'val_pos_encs'):
            sample_xy = sample_locations.view(batch_size, self.num_heads, num_in_feats, -1, 2)
            sample_xy = sample_xy - sample_priors[:, None, :, None, :2]

            if sample_priors.shape[-1] == 4:
                sample_xy = sample_xy / sample_priors[:, None, :, None, 2:]

            sample_z = sample_map_ids.view(batch_size, self.num_heads, num_in_feats, -1, 1)
            sample_z = sample_z / (self.num_levels-1)

            sample_xyz = torch.cat([sample_xy, sample_z], dim=4).transpose(1, 2)
            sampled_feats = sampled_feats + self.val_pos_encs(sample_xyz)

        # Get attention weights
        attn_weights = self.attn_weights(in_feats).view(*common_shape, self.num_levels * self.num_points)
        attn_weights = F.softmax(attn_weights, dim=3)

        # Get weighted value features
        weighted_feats = attn_weights[:, :, :, :, None] * sampled_feats
        weighted_feats = weighted_feats.sum(dim=3).view(batch_size, num_in_feats, val_size)

        # Get output features
        out_feats = self.out_proj(weighted_feats)

        # Update sample locations in storage dictionary if needed
        if self.update_sample_locations:
            qry_feats = self.qry_proj(in_feats)
            qry_feats = qry_feats.view(*common_shape, -1)
            sampled_feats = sampled_feats + qry_feats[:, :, :, None, :]

            sampled_feats = sampled_feats[:, :, :, :, None, :].expand(-1, -1, -1, -1, 2, -1)
            sampled_feats = sampled_feats.reshape(batch_size * num_in_feats, -1, val_size // self.num_heads)
            sampled_feats = sampled_feats.transpose(0, 1)

            sample_steps = torch.bmm(sampled_feats, self.steps_weight) + self.steps_bias
            sample_steps = sample_steps.view(self.num_heads, self.num_levels, self.num_points, 2, batch_size, -1)
            sample_steps = sample_steps.permute(4, 0, 5, 1, 2, 3)

            if self.step_size > 0:
                sample_steps = self.step_size * F.normalize(sample_steps, dim=5)

            if self.step_norm_xy == 'map':
                step_normalizers = sample_map_shapes[None, None, None, :, None, :]
                sample_steps = sample_steps / step_normalizers

            elif self.step_norm_xy == 'anchor':
                if sample_priors.shape[-1] == 4:
                    step_factors = 0.5 * sample_priors[:, None, :, None, None, 2:] / self.num_points
                    sample_steps = sample_steps * step_factors

                else:
                    error_msg = "Last dimension of 'sample_priors' must be 4 when using 'anchor' step normalization."
                    raise ValueError(error_msg)

            else:
                error_msg = f"Invalid sample step normalization type '{self.step_norm_xy}'."
                raise ValueError(error_msg)

            sample_steps = sample_steps.reshape(batch_size * self.num_heads, -1, 2)
            next_sample_locations = sample_locations + sample_steps
            storage_dict['sample_locations'] = next_sample_locations

        return out_feats


@MODELS.register_module()
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

    def forward(self, in_feats, mul_encs=None, add_encs=None, cum_feats_batch=None, **kwargs):
        """
        Forward method of the SelfAttn1d module.

        Args:
            in_feats (FloatTensor): Input features of shape [num_feats, in_size].
            mul_encs (FloatTensor): Encodings multiplied by queries/keys of shape [num_feats, in_size] (default=None).
            add_encs (FloatTensor): Encodings added to queries/keys of shape [num_feats, in_size] (default=None).
            cum_feats_batch (LongTensor): Cumulative number of features per batch entry [batch_size+1] (default=None).
            kwargs (Dict): Dictionary of keyword arguments not used by this module.

        Returns:
            out_feats (FloatTensor): Output features of shape [num_feats, out_size].
        """

        # Apply optional normalization and activation function modules
        delta_feats = in_feats
        delta_feats = self.norm(delta_feats) if hasattr(self, 'norm') else delta_feats
        delta_feats = self.act_fn(delta_feats) if hasattr(self, 'act_fn') else delta_feats

        # Get query-key and value features
        qk_feats = val_feats = delta_feats[:, None, :]

        if mul_encs is not None:
            qk_feats = qk_feats * mul_encs[:, None, :]

        if add_encs is not None:
            qk_feats = qk_feats + add_encs[:, None, :]

        # Apply multi-head attention module
        num_feats = len(in_feats)
        out_size = self.mha.out_proj.weight.size(dim=0)
        mha_feats = in_feats.new_empty([num_feats, 1, out_size])

        if cum_feats_batch is None:
            cum_feats_batch = torch.tensor([0, num_feats], device=in_feats.device)

        for i0, i1 in zip(cum_feats_batch[:-1], cum_feats_batch[1:]):
            mha_feats[i0:i1] = self.mha(qk_feats[i0:i1], qk_feats[i0:i1], val_feats[i0:i1], need_weights=False)[0]

        # Get output features
        mha_feats = mha_feats[:, 0, :]
        out_feats = in_feats + mha_feats if self.skip else mha_feats

        return out_feats
