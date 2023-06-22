"""
Collection of deprecated attention-based modules.
"""
import math

import torch
from torch import nn
import torch.nn.functional as F

from models.build import MODELS
from models.extensions.deformable.python.sample import pytorch_maps_sample_2d, pytorch_maps_sample_3d


@MODELS.register_module()
class LegacySelfAttn1dV1(nn.Module):
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


@MODELS.register_module()
class LegacySelfAttn1dV2(nn.Module):
    """
    Class implementing the LegacySelfAttn1d module.

    Attributes:
        norm (nn.Module): Optional normalization module of the LegacySelfAttn1d module.
        act_fn (nn.Module): Optional module with the activation function of the LegacySelfAttn1d module.
        mha (nn.MultiheadAttention): Multi-head attention module of the LegacySelfAttn1d module.
        skip (bool): Boolean indicating whether skip connection is used or not.
    """

    def __init__(self, in_size, out_size=-1, norm='', act_fn='', skip=True, num_heads=8):
        """
        Initializes the LegacySelfAttn1d module.

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
            error_msg = f"The LegacySelfAttn1d module does not support the '{norm}' normalization type."
            raise ValueError(error_msg)

        # Initialization of optional module with activation function
        if not act_fn:
            pass
        elif act_fn == 'gelu':
            self.act_fn = nn.GELU()
        elif act_fn == 'relu':
            self.act_fn = nn.ReLU(inplace=False) if not norm and skip else nn.ReLU(inplace=True)
        else:
            error_msg = f"The LegacySelfAttn1d module does not support the '{act_fn}' activation function."

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
        Forward method of the LegacySelfAttn1d module.

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
