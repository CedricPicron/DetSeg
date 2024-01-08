"""
Collection of SPS-based modules.
"""
import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair as pair
from torch.nn.parameter import Parameter

from models.build import build_model, MODELS
from models.functional.convolution import id_conv2d
from models.functional.sparse import id_attn
from models.functional.utils import maps_to_seq, seq_to_maps


@MODELS.register_module()
class IdAttn2d(nn.Module):
    """
    Class implementing the IdAttn2d module.

    Attributes:
        attn_weights (nn.Linear): Module computing the unnormalized attention weights.
        val_proj (nn.Linear): Module computing the value features.
        out_proj (nn.Linear): Module computing the output features.
        num_pts (int): Integer containing the number of attention points.
    """

    def __init__(self, feat_size, num_pts=4):
        """
        Initializes the IdAttn2d module.

        Args:
            feat_size (int): Integer containing the feature size.
            num_pts (int): Integer containing the number of attention points (default=4).

        Raises:
            ValueError: Error when the feature size is not divisible by 8.
        """

        # Check divisibility feature size by 8
        if feat_size % 8 != 0:
            error_msg = f"The feature size ({feat_size}) must be divisible by 8."
            raise ValueError(error_msg)

        # Initialization of default nn.Module
        super().__init__()

        # Initialize module computing the unnormalized attention weights
        self.attn_weights = nn.Linear(feat_size, 8 * num_pts)
        nn.init.zeros_(self.attn_weights.weight)
        nn.init.zeros_(self.attn_weights.bias)

        # Initialize module computing the value features
        self.val_proj = nn.Linear(feat_size, feat_size)
        nn.init.xavier_uniform_(self.val_proj.weight)
        nn.init.zeros_(self.val_proj.bias)

        # Initialize module computing the output features
        self.out_proj = nn.Linear(feat_size, feat_size)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

        # Set remaining attributes
        self.num_pts = num_pts

    def forward(self, in_act_feats, storage_dict, **kwargs):
        """
        Forward method of the IdAttn2d module.

        Args:
            in_act_feats (FloatTensor): Input active features of shape [num_act_feats, feat_size].

            storage_dict (Dict): Storage dictionary containing at least following keys:
                - act_batch_ids (LongTensor): batch indices of active features of shape [num_act_feats];
                - act_map_ids (LongTensor): map indices of active features of shape [num_act_feats];
                - act_xy_ids (LongTensor): (X, Y) location indices of active features of shape [num_act_feats, 2];
                - pas_feats (FloatTensor): passive features of shape [num_pas_feats, feat_size];
                - id_maps (List): list [num_maps] with feature indices of shape [batch_size, 1, fH, fW].

            kwargs (Dict): Dictionary of unused keyword arguments.

        Returns:
            out_act_feats (FloatTensor): Output active features of shape [num_act_feats, feat_size].
        """

        # Retrieve desired items from storage dictionary
        act_batch_ids = storage_dict['act_batch_ids']
        act_map_ids = storage_dict['act_map_ids']
        act_xy_ids = storage_dict['act_xy_ids']
        pas_feats = storage_dict['pas_feats']
        id_maps = storage_dict['id_maps']

        # Get device and number of active features
        device = in_act_feats.device
        num_act_feats = len(in_act_feats)

        # Get attention location indices
        attn_offs = torch.tensor([[-1, -1], [0, -1], [1, -1], [-1, 0], [1, 0], [-1, 1], [0, 1], [1, 1]], device=device)
        attn_offs = torch.arange(1, self.num_pts+1, device=device)[:, None, None] * attn_offs[None, :, :]
        attn_xy_ids = act_xy_ids[:, None, None, :] + attn_offs[None, :, :, :]

        # Get feature indices
        act_batch_ids = act_batch_ids[:, None, None].expand(-1, self.num_pts, 8)
        feat_ids = attn_xy_ids.new_empty([num_act_feats, self.num_pts, 8])

        for map_id, id_map in enumerate(id_maps):
            fH, fW = id_map.size()[-2:]
            map_mask = act_map_ids == map_id

            x_ids = attn_xy_ids[map_mask, :, :, 0].clamp_(min=0, max=fW-1)
            y_ids = attn_xy_ids[map_mask, :, :, 1].clamp_(min=0, max=fH-1)

            batch_ids = act_batch_ids[map_mask]
            feat_ids[map_mask] = id_map[batch_ids, 0, y_ids, x_ids]

        # Get feature weights
        attn_weights = self.attn_weights(in_act_feats).view(num_act_feats, 8, self.num_pts)
        feat_weights = F.softmax(attn_weights, dim=2).transpose(1, 2)

        # Get weighted value features
        weight = self.val_proj.weight
        bias = self.val_proj.bias
        val_feats = id_attn(in_act_feats, pas_feats, feat_ids, feat_weights, weight, bias)

        # Get output active features
        out_act_feats = self.out_proj(val_feats)

        return out_act_feats


@MODELS.register_module()
class IdAvg2d(nn.Module):
    """
    Class implementing the IdAvg2d module.
    """

    def __init__(self):
        """
        Initializes the IdAvg2d module.
        """

        # Initialization of default nn.Module
        super().__init__()

    def forward(self, core_feats, aux_feats, id_map, **kwargs):
        """
        Forward method of the IdAvg2d module.

        Args:
            core_feats (FloatTensor): Core features of shape [num_core_feats, feat_size].
            aux_feats (FloatTensor): Auxiliary features of shape [num_aux_feats, feat_size].
            id_map (LongTensor): Index map with feature indices of shape [num_rois, rH, rW].
            kwargs (Dict): Dictionary with unused keyword arguments.

        Returns:
            avg_feat (FloatTensor): Average feature of shape [1, feat_size].
        """

        # Get average feature
        num_core_feats = len(core_feats)
        num_aux_feats = len(aux_feats)
        num_feats = num_core_feats + num_aux_feats

        counts = torch.bincount(id_map.flatten(), minlength=num_feats)
        counts = counts[:, None]

        avg_feat = (counts[:num_core_feats] * core_feats).sum(dim=0, keepdim=True)
        avg_feat += (counts[num_core_feats:] * aux_feats).sum(dim=0, keepdim=True)
        avg_feat *= 1/id_map.numel()

        return avg_feat


@MODELS.register_module()
class IdBase2d(nn.Module):
    """
    Class implementing the IdBase2d module.

    Attributes:
        act_mask_key (str): String with key to retrieve the active mask from the storage dictionary.
        id (nn.Module): Module performing the 2D index-based processing.
    """

    def __init__(self, act_mask_key, id_cfg):
        """
        Initializes the IdBase2d module.

        Args:
            act_mask_key (str): String with key to retrieve the active mask from the storage dictionary.
            id_cfg (Dict): Configuration dictionary specifying the 2D index-based processing module.
        """

        # Initialization of default nn.Module
        super().__init__()

        # Set active mask key attribute
        self.act_mask_key = act_mask_key

        # Build index-based processing module
        self.id = build_model(id_cfg)

    def forward(self, in_feat_map, storage_dict, **kwargs):
        """
        Forward method of the IdBase2d module.

        Args:
            in_feat_map (FloatTensor): Input feature map of shape [batch_size, feat_size, fH, fW].

            storage_dict (Dict): Storage dictionary containing at least following key:
                - {self.act_mask_key} (BoolTensor): active mask of shape [batch_size, 1, fH, fW].

            kwargs (Dict): Dictionary of unused keyword arguments.

        Returns:
            out_feat_map (FloatTensor): Output feature map of shape [batch_size, feat_size, fH, fW].
        """

        # Get active and passive mask
        act_mask = storage_dict[self.act_mask_key].squeeze(dim=1)
        pas_mask = ~act_mask

        # Get desired items for index-based processing
        batch_ids, y_ids, x_ids = pas_mask.nonzero(as_tuple=True)
        pas_feats = in_feat_map[batch_ids, :, y_ids, x_ids]

        batch_ids, y_ids, x_ids = act_mask.nonzero(as_tuple=True)
        act_feats = in_feat_map[batch_ids, :, y_ids, x_ids]
        pos_ids = torch.stack([x_ids, y_ids], dim=1)

        device = in_feat_map.device
        num_act_feats = len(act_feats)
        num_feats = num_act_feats + len(pas_feats)

        id_map = torch.zeros_like(act_mask, dtype=torch.int64)
        id_map[act_mask] = torch.arange(num_act_feats, device=device)
        id_map[pas_mask] = torch.arange(num_act_feats, num_feats, device=device)

        # Update active features with index-based processing
        id_kwargs = {'aux_feats': pas_feats, 'id_map': id_map, 'roi_ids': batch_ids, 'pos_ids': pos_ids}
        act_feats = self.id(act_feats, **id_kwargs, **kwargs)

        # Get output feature map
        out_feat_map = in_feat_map.clone()
        out_feat_map[batch_ids, :, y_ids, x_ids] = act_feats

        return out_feat_map


@MODELS.register_module()
class IdConv2d(nn.Module):
    """
    Class implementing the IdConv2d module.

    Attributes:
        kernel_size (Tuple): Tuple of size [2] containing the convolution kernel sizes in (H, W) format.
        dilation (Tuple): Tuple of size [2] containing the convolution dilations in (H, W) format.
        weight (Parameter): Parameter with convolution weights of shape [out_channels, kH * kW * in_channels].
        bias (Parameter): Optional parameter with convolution biases of shape [out_channels].
    """

    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, bias=True):
        """
        Initializes the IdConv2d module.

        Args:
            in_channels (int): Integer containing the number of input channels.
            out_channels (int): Integer containing the number of output channels.
            kernel_size (int or Tuple): Integer ot tuple containing the size of the convolving kernel.
            dilation (int or Tuple): Integer or tuple containing the convolution dilation (default=1).
            bias (bool): Boolean indicating whether learnable bias is added to the output (default=True).

        Raises:
            ValueError: Error when an even kernel size is provided in either dimension.
        """

        # Initialization of default nn.Module
        super().__init__()

        # Set convolution attributes
        self.kernel_size = pair(kernel_size)
        self.dilation = pair(dilation)

        # Check kernel size
        if (self.kernel_size[0] % 2 == 0) or (self.kernel_size[1] % 2 == 0):
            error_msg = f"Even kernel sizes are not allowed in either dimension (got {self.kernel_size})."
            raise ValueError(error_msg)

        # Initialize weight parameter
        kH, kW = self.kernel_size
        self.weight = Parameter(torch.empty(out_channels, kH * kW * in_channels))
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        # Initialize optional bias parameter
        if bias:
            self.bias = Parameter(torch.zeros(out_channels))
            fan_in = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)[0]

            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                torch.nn.init.uniform_(self.bias, -bound, bound)

        else:
            self.register_parameter('bias', None)

    def forward(self, in_core_feats, aux_feats, id_map, roi_ids, pos_ids, **kwargs):
        """
        Forward method of the IdConv2d module.

        Args:
            in_core_feats (FloatTensor): Input core features of shape [num_core_feats, in_channels].
            aux_feats (FloatTensor): Auxiliary features of shape [num_aux_feats, in_channels].
            id_map (LongTensor): Index map with feature indices of shape [num_rois, rH, rW].
            roi_ids (LongTensor): RoI indices of core features of shape [num_core_feats].
            pos_ids (LongTensor): RoI-based core position indices in (X, Y) format of shape [num_core_feats, 2].
            kwargs (Dict): Dictionary of unused keyword arguments.

        Returns:
            out_core_feats (FloatTensor): Output core features of shape [num_core_feats, out_channels].
        """

        # Get convolution indices
        kH, kW = self.kernel_size
        dH, dW = self.dilation

        off_y = (kH-1)//2 * dH
        off_x = (kW-1)//2 * dW

        device = aux_feats.device
        offs_y = torch.arange(-off_y, off_y+1, step=dH, device=device)[:, None].expand(-1, kW).flatten()
        offs_x = torch.arange(-off_x, off_x+1, step=dW, device=device)[None, :].expand(kH, -1).flatten()

        pos_y = pos_ids[:, 1, None] + offs_y
        pos_x = pos_ids[:, 0, None] + offs_x

        rH, rW = id_map.size()[1:]
        pad_mask = (pos_y < 0) | (pos_y >= rH) | (pos_x < 0) | (pos_x >= rW)

        roi_ids = roi_ids[:, None].expand(-1, kH*kW)
        pos_y = pos_y.clamp_(min=0, max=rH-1)
        pos_x = pos_x.clamp_(min=0, max=rW-1)

        conv_ids = id_map[roi_ids, pos_y, pos_x]
        conv_ids[pad_mask] = len(in_core_feats) + len(aux_feats)

        # Perform 2D index-based convolution
        out_core_feats = id_conv2d(in_core_feats, aux_feats, conv_ids, self.weight, self.bias)

        return out_core_feats


@MODELS.register_module()
class IdScaleAttn(nn.Module):
    """
    Class implementing the IdScaleAttn module.

    Attributes:
        scale_embed (Parameter): Parameter containing the scale embeddings of shape [num_maps, feat_size].
        attn_weights (nn.Linear): Module computing the unnormalized attention weights.
        val_proj (nn.Linear): Module computing the value features.
        out_proj (nn.Linear): Module computing the output features.
        num_heads (int): Integer containing the number of attention heads.
        num_maps (int): Integer containing the number of feature maps.
    """

    def __init__(self, feat_size, num_maps, num_heads=8):
        """
        Initializes the IdScaleAttn module.

        Args:
            feat_size (int): Integer containing the feature size.
            num_maps (int): Integer containing the number of feature maps.
            num_heads (int): Integer containing the number of attention heads (default=8).

        Raises:
            ValueError: Error when the feature size does not divide the number of attention heads.
        """

        # Check divisibility feature size by number of heads
        if feat_size % num_heads != 0:
            error_msg = f"The feature size ({feat_size}) must divide the number of attention heads ({num_heads})."
            raise ValueError(error_msg)

        # Initialization of default nn.Module
        super().__init__()

        # Initialize scale embedding
        self.scale_embed = Parameter(torch.empty(num_maps, feat_size), requires_grad=True)
        nn.init.zeros_(self.scale_embed)

        # Initialize module computing the unnormalized attention weights
        self.attn_weights = nn.Linear(feat_size, num_heads * num_maps)
        nn.init.zeros_(self.attn_weights.weight)
        nn.init.zeros_(self.attn_weights.bias)

        # Initialize module computing the value features
        self.val_proj = nn.Linear(feat_size, feat_size)
        nn.init.xavier_uniform_(self.val_proj.weight)
        nn.init.zeros_(self.val_proj.bias)

        # Initialize module computing the output features
        self.out_proj = nn.Linear(feat_size, feat_size)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

        # Set remaining attributes
        self.num_heads = num_heads
        self.num_maps = num_maps

    def forward(self, in_act_feats, storage_dict, **kwargs):
        """
        Forward method of the IdScaleAttn module.

        Args:
            in_act_feats (FloatTensor): Input active features of shape [num_act_feats, feat_size].

            storage_dict (Dict): Storage dictionary containing at least following keys:
                - act_batch_ids (LongTensor): batch indices of active features of shape [num_act_feats];
                - act_map_ids (LongTensor): map indices of active features of shape [num_act_feats];
                - act_xy_ids (LongTensor): (X, Y) location indices of active features of shape [num_act_feats, 2];
                - map_shapes (LongTensor): feature map shapes in (H, W) format of shape [num_maps, 2];
                - pas_feats (FloatTensor): passive features of shape [num_pas_feats, feat_size];
                - id_maps (List): list [num_maps] with feature indices of shape [batch_size, 1, fH, fW].

            kwargs (Dict): Dictionary of unused keyword arguments.

        Returns:
            out_act_feats (FloatTensor): Output active features of shape [num_act_feats, feat_size].
        """

        # Retrieve desired items from storage dictionary
        act_batch_ids = storage_dict['act_batch_ids']
        act_map_ids = storage_dict['act_map_ids']
        act_xy_ids = storage_dict['act_xy_ids']
        map_shapes = storage_dict['map_shapes']
        pas_feats = storage_dict['pas_feats']
        id_maps = storage_dict['id_maps']

        # Get device and number of active features
        device = in_act_feats.device
        num_act_feats = len(in_act_feats)

        # Get sample locations
        map_shapes = map_shapes.fliplr()
        sample_xy = (act_xy_ids + 0.5) / map_shapes[act_map_ids]

        sample_xy = sample_xy[:, None, :] * map_shapes[None, :, :]
        sample_xy = sample_xy[:, :, None, :] - 0.5

        # Get sample indices
        sample_offs = torch.tensor([[0, 0], [1, 0], [0, 1], [1, 1]], device=device)
        sample_ids = sample_xy.floor().long() + sample_offs

        # Get sample weights
        sample_weights = 1 - (sample_xy - sample_ids).abs()
        sample_weights = sample_weights.prod(dim=3)

        # Get feature indices
        act_batch_ids = act_batch_ids[:, None].expand(-1, 4)
        feat_ids = sample_ids.new_empty([num_act_feats, self.num_maps, 4])

        for map_id, id_map in enumerate(id_maps):
            fH, fW = id_map.size()[-2:]

            sample_ids_x = sample_ids[:, map_id, :, 0].clamp_(min=0, max=fW-1)
            sample_ids_y = sample_ids[:, map_id, :, 1].clamp_(min=0, max=fH-1)

            feat_ids[:, map_id, :] = id_map[act_batch_ids, 0, sample_ids_y, sample_ids_x]

        feat_ids = feat_ids.flatten(1)

        # Get feature weights
        act_feats = in_act_feats + self.scale_embed[act_map_ids]

        attn_weights = self.attn_weights(act_feats).view(num_act_feats, self.num_heads, self.num_maps)
        attn_weights = F.softmax(attn_weights, dim=2).transpose(1, 2)

        feat_weights = attn_weights[:, :, None, :] * sample_weights[:, :, :, None]
        feat_weights = feat_weights.flatten(1, 2)

        # Get weighted value features
        weight = self.val_proj.weight
        bias = self.val_proj.bias
        val_feats = id_attn(in_act_feats, pas_feats, feat_ids, feat_weights, weight, bias)

        # Get output active features
        out_act_feats = self.out_proj(val_feats)

        return out_act_feats


@MODELS.register_module()
class MapToSps(nn.Module):
    """
    Class implementing the MapToSps module.

    Attributes:
        in_key (str): String with key to retrieve input map from storage dictionary.
        out_act_key (str): String with key to store output active features in storage dictionary.
        out_pas_key (str): String with key to store output passive features in storage dictionary.
        out_id_key (str): String with key to store output index map in storage dictionary.
        out_grp_key (str): String with key to store output group indices in storage dictionary.
        out_pos_key (str): String with key to store output position indices in storage dictionary.
    """

    def __init__(self, in_key, out_act_key, out_pas_key, out_id_key, out_grp_key, out_pos_key):
        """
        Initializes the MapToSps module.

        Args:
            in_key (str): String with key to retrieve input map from storage dictionary.
            out_act_key (str): String with key to store output active features in storage dictionary.
            out_pas_key (str): String with key to store output passive features in storage dictionary.
            out_id_key (str): String with key to store output index map in storage dictionary.
            out_grp_key (str): String with key to store output group indices in storage dictionary.
            out_pos_key (str): String with key to store output position indices in storage dictionary.
        """

        # Initialization of default nn.Module
        super().__init__()

        # Set additional attributes
        self.in_key = in_key
        self.out_act_key = out_act_key
        self.out_pas_key = out_pas_key
        self.out_id_key = out_id_key
        self.out_grp_key = out_grp_key
        self.out_pos_key = out_pos_key

    def forward(self, storage_dict, **kwargs):
        """
        Forward method of the MapToSps module.

        Args:
            storage_dict (Dict): Storage dictionary containing at least following key:
                - {self.in_key} (FloatTensor): input feature map of shape [num_groups, feat_size, fH, fW].

            kwargs (Dict): Dictionary of unused keyword arguments.

        Returns:
            storage_dict (Dict): Storage dictionary containing following additional keys:
                - {self.out_act_key} (FloatTensor): output active features of shape [num_groups*fH*fW, feat_size];
                - {self.out_pas_key} (FloatTensor): output passive features of shape [0, feat_size];
                - {self.out_id_key} (LongTensor): output index map of shape [num_groups, fH, fW];
                - {self.out_grp_key} (LongTensor): output active group indices of shape [num_groups*fH*fW];
                - {self.out_pos_key} (LongTensor): output active position indices of shape [num_groups*fH*fW, 2].
        """

        # Retrieve input map from storage dictionary
        in_map = storage_dict[self.in_key]

        # Get shape and device of input map
        num_groups, feat_size, fH, fW = in_map.size()
        device = in_map.device

        # Get active features
        act_feats = in_map.permute(0, 2, 3, 1).flatten(0, 2)

        # Get passive features
        pas_feats = in_map.new_empty([0, feat_size])

        # Get index map
        id_map = torch.arange(num_groups*fH*fW, device=device).view(num_groups, fH, fW)

        # Get active group indices
        act_grp_ids = torch.arange(num_groups, device=device)[:, None].expand(-1, fH*fW).flatten()

        # Get active position indices
        x_ids = torch.arange(fW, device=device)
        y_ids = torch.arange(fH, device=device)

        act_pos_ids = torch.meshgrid(x_ids, y_ids, indexing='xy')
        act_pos_ids = torch.stack(act_pos_ids, dim=2)
        act_pos_ids = act_pos_ids[None, :, :, :].expand(num_groups, -1, -1, -1).flatten(0, 2)

        # Store outputs in storage dictionary
        storage_dict[self.out_act_key] = act_feats
        storage_dict[self.out_pas_key] = pas_feats
        storage_dict[self.out_id_key] = id_map
        storage_dict[self.out_grp_key] = act_grp_ids
        storage_dict[self.out_pos_key] = act_pos_ids

        return storage_dict


@MODELS.register_module()
class Sparse3d(nn.Module):
    """
    Class implementing the Sparse3d module.

    Attributes:
        seq_feats_key (str): String used to retrieve sequential features from storage dictionary (or None).
        act_map_ids (List): List [num_act_maps] with map indices to get active features from (or None).
        act_mask_key (str): String used to retrieve active mask from storage dictionary (or None).
        pos_feats_key (str): String used to retrieve position features from storage dictionary (or None).
        get_act_batch_ids (bool): Boolean indicating whether to get active batch indices.
        get_act_map_ids (bool): Boolean indicating whether to get active map indices.
        get_act_xy_ids (bool): Boolean indicating whether to get active location indices.
        get_pas_feats (bool): Boolean indicating whether to get passive features.
        get_id_maps (bool): Boolean indicating whether to get list of index maps.
        sparse (nn.Module): Underlying sparse module updating the active features.
    """

    def __init__(self, sparse_cfg, seq_feats_key=None, act_map_ids=None, act_mask_key=None, pos_feats_key=None,
                 get_act_batch_ids=False, get_act_map_ids=False, get_act_xy_ids=False, get_pas_feats=True,
                 get_id_maps=True):
        """
        Initializes the Sparse3d module.

        Args:
            sparse_cfg (Dict): Configuration dictionary specifying the underlying sparse module.
            seq_feats_key (str): String used to retrieve sequential features from storage dictionary (default=None).
            act_map_ids (List): List [num_act_maps] with map indices to get active features from (default=None).
            act_mask_key (str): String used to retrieve active mask from storage dictionary (default=None).
            pos_feats_key (str): String used to retrieve position features from storage dictionary (default=None).
            get_act_batch_ids (bool): Boolean indicating whether to get active batch indices (default=False).
            get_act_map_ids (bool): Boolean indicating whether to get active map indices (default=False).
            get_act_xy_ids (bool): Boolean indicating whether to get active location indices (default=False).
            get_pas_feats (bool): Boolean indicating whether to get passive features (default=True).
            get_id_maps (bool): Boolean indicating whether to get list of index maps (default=True).
        """

        # Initialization of default nn.Module
        super().__init__()

        # Build underlying sparse module
        self.sparse = build_model(sparse_cfg)

        # Set additional attributes
        self.seq_feats_key = seq_feats_key
        self.act_map_ids = act_map_ids
        self.act_mask_key = act_mask_key
        self.pos_feats_key = pos_feats_key
        self.get_act_batch_ids = get_act_batch_ids
        self.get_act_map_ids = get_act_map_ids
        self.get_act_xy_ids = get_act_xy_ids
        self.get_pas_feats = get_pas_feats
        self.get_id_maps = get_id_maps

    def forward(self, in_feat_maps, storage_dict, **kwargs):
        """
        Forward method of the Sparse3d module.

        Args:
            in_feat_maps (List): List [num_maps] of input feature maps of shape [batch_size, feat_size, fH, fW].

            storage_dict (Dict): Storage dictionary (possibly) containing following keys:
                - {self.seq_feats_key} (FloatTensor): sequential features of shape [batch_size, sum(fH*fW), feat_size];
                - {self.act_mask_key} (BoolTensor): mask indicating active features of shape [batch_size, sum(fH*fW)];
                - {self.pos_feats_key} (FloatTensor): position features of shape [batch_size, sum(fH*fW), feat_size];
                - map_shapes (LongTensor): feature map shapes in (H, W) format of shape [num_maps, 2];
                - map_offs (LongTensor): cumulative number of features per feature map of shape [num_maps+1].

            kwargs (Dict): Dictionary of keyword arguments passed to the underlying sparse module.

        Returns:
            out_feat_maps (List): List [num_maps] of output feature maps of shape [batch_size, feat_size, fH, fW].

        Raises:
            ValueError: Error when neither the 'act_map_ids' attribute nor the 'act_mask_key' attribute is provided.
        """

        # Get batch size and device
        batch_size = len(in_feat_maps[0])
        device = in_feat_maps[0].device

        # Add map shapes to storage dictionary if needed
        if 'map_shapes' not in storage_dict:
            map_shapes = [feat_map.shape[-2:] for feat_map in in_feat_maps]
            storage_dict['map_shapes'] = torch.tensor(map_shapes, device=device)

        # Add map offsets to storage dictionary if needed
        if 'map_offs' not in storage_dict:
            map_offs = storage_dict['map_shapes'].prod(dim=1).cumsum(dim=0)
            storage_dict['map_offs'] = torch.cat([map_offs.new_zeros([1]), map_offs], dim=0)

        # Get input sequantial features
        if self.seq_feats_key is None:
            seq_feats = maps_to_seq(in_feat_maps)

        elif self.seq_feats_key not in storage_dict:
            seq_feats = maps_to_seq(in_feat_maps)
            storage_dict[self.seq_feats_key] = seq_feats

        else:
            seq_feats = storage_dict[self.seq_feats_key]

        # Get active mask
        if self.act_map_ids is not None:
            act_masks = []

            for map_id, feat_map in enumerate(in_feat_maps):
                fH, fW = feat_map.size()[-2:]

                if map_id in self.act_map_ids:
                    act_mask = torch.ones(batch_size, fH*fW, dtype=torch.bool, device=device)
                else:
                    act_mask = torch.zeros(batch_size, fH*fW, dtype=torch.bool, device=device)

                act_masks.append(act_mask)

            act_mask = torch.cat(act_masks, dim=1)

        elif self.act_mask_key is not None:
            act_mask = storage_dict[self.act_mask_key]

        else:
            error_msg = "Either the 'act_map_ids' or the 'act_mask_key' attribute must be provided (both are None)."
            raise ValueError(error_msg)

        # Get active features
        act_feats = seq_feats[act_mask]
        storage_dict['act_feats'] = act_feats

        # Get active position features if needed
        if self.pos_feats_key is not None:
            pos_feats = storage_dict[self.pos_feats_key]
            act_pos_feats = pos_feats[act_mask]
            storage_dict['act_pos_feats'] = act_pos_feats

        # Get active batch indices if needed
        if self.get_act_batch_ids:
            act_batch_ids = act_mask.nonzero(as_tuple=True)[0]
            storage_dict['act_batch_ids'] = act_batch_ids

        # Get active map indices if needed
        if self.get_act_map_ids:
            act_feat_ids = act_mask.nonzero(as_tuple=True)[1]
            act_map_ids = act_feat_ids[:, None] - storage_dict['map_offs'][None, 1:-1]

            act_map_ids = (act_map_ids >= 0).sum(dim=1)
            storage_dict['act_map_ids'] = act_map_ids

        # Get active location indices if needed
        if self.get_act_xy_ids:
            act_masks = seq_to_maps(act_mask.unsqueeze(dim=2), storage_dict['map_shapes'])
            act_xy_ids = [act_mask.nonzero()[:, 2:].fliplr() for act_mask in act_masks]

            act_xy_ids = torch.cat(act_xy_ids, dim=0)
            storage_dict['act_xy_ids'] = act_xy_ids

        # Get passive features
        if self.get_pas_feats:
            pas_mask = ~act_mask
            pas_feats = seq_feats[pas_mask]
            storage_dict['pas_feats'] = pas_feats

        # Get index maps
        if self.get_id_maps:
            num_act_feats = len(act_feats)
            num_feats = num_act_feats + len(pas_feats)

            act_ids = torch.arange(num_act_feats, device=device)
            pas_ids = torch.arange(num_act_feats, num_feats, device=device)

            ids = torch.empty_like(act_mask, dtype=torch.int64)
            ids[act_mask] = act_ids
            ids[pas_mask] = pas_ids

            ids = ids.unsqueeze(dim=2)
            id_maps = seq_to_maps(ids, storage_dict['map_shapes'])
            storage_dict['id_maps'] = id_maps

        # Apply underlying sparse module
        act_feats = self.sparse(act_feats, storage_dict=storage_dict, **kwargs)

        # Get output sequential features
        seq_feats[act_mask] = act_feats

        # Add output sequential feature to storage dictionary if needed
        if self.seq_feats_key is not None:
            storage_dict[self.seq_feats_key] = seq_feats

        # Get output feature maps
        out_feat_maps = seq_to_maps(seq_feats, storage_dict['map_shapes'])

        return out_feat_maps


@MODELS.register_module()
class SpsMask(nn.Module):
    """
    Class implementing the SpsMask module.

    Attributes:
        in_act_key (str): String with key to retrieve input active features from storage dictionary.
        in_pas_key (str): String with key to retrieve input passive features from storage dictionary.
        in_id_key (str): String with key to retrieve input index map from storage dictionary.
        in_grp_key (str): String with key to retrieve input group indices from storage dictionary.
        in_pos_key (str): String with key to retrieve input position indices from storage dictionary.
        mask_key (str): String with key to retrieve active mask from storage dictionary.
        out_act_key (str): String with key to store output active features in storage dictionary.
        out_pas_key (str): String with key to store output passive features in storage dictionary.
        out_id_key (str): String with key to store output index map in storage dictionary.
        out_grp_key (str): String with key to store output group indices in storage dictionary.
        out_pos_key (str): String with key to store output position indices in storage dictionary.
    """

    def __init__(self, in_act_key, in_pas_key, in_id_key, in_grp_key, in_pos_key, mask_key, out_act_key, out_pas_key,
                 out_id_key, out_grp_key, out_pos_key):
        """
        Initializes the SpsMask module.

        Args:
            in_act_key (str): String with key to retrieve input active features from storage dictionary.
            in_pas_key (str): String with key to retrieve input passive features from storage dictionary.
            in_id_key (str): String with key to retrieve input index map from storage dictionary.
            in_grp_key (str): String with key to retrieve input group indices from storage dictionary.
            in_pos_key (str): String with key to retrieve input position indices from storage dictionary.
            mask_key (str): String with key to retrieve active mask from storage dictionary.
            out_act_key (str): String with key to store output active features in storage dictionary.
            out_pas_key (str): String with key to store output passive features in storage dictionary.
            out_id_key (str): String with key to store output index map in storage dictionary.
            out_grp_key (str): String with key to store output group indices in storage dictionary.
            out_pos_key (str): String with key to store output position indices in storage dictionary.
        """

        # Initialization of default nn.Module
        super().__init__()

        # Set additional attributes
        self.in_act_key = in_act_key
        self.in_pas_key = in_pas_key
        self.in_id_key = in_id_key
        self.in_grp_key = in_grp_key
        self.in_pos_key = in_pos_key
        self.mask_key = mask_key
        self.out_act_key = out_act_key
        self.out_pas_key = out_pas_key
        self.out_id_key = out_id_key
        self.out_grp_key = out_grp_key
        self.out_pos_key = out_pos_key

    def forward(self, storage_dict, **kwargs):
        """
        Forward method of the SpsMask module.

        Args:
            storage_dict (Dict): Storage dictionary containing at least following keys:
                - {self.in_act_key} (FloatTensor): input active features of shape [num_in_act, feat_size];
                - {self.in_pas_key} (FloatTensor): input passive features of shape [num_in_pas, feat_size];
                - {self.in_id_key} (LongTensor): input index map of shape [num_groups, mH, mW];
                - {self.in_grp_key} (LongTensor): input group indices of shape [num_in_act];
                - {self.in_pos_key} (LongTensor): input position indices of shape [num_in_act, 2];
                - {self.mask_key} (BoolTensor): mask selecting the output active features of shape [num_in_act].

            kwargs (Dict): Dictionary of unused keyword arguments.

        Returns:
            storage_dict (Dict): Storage dictionary containing following additional keys:
                - {self.out_act_key} (FloatTensor): output active features of shape [num_out_act, feat_size];
                - {self.out_pas_key} (FloatTensor): output passive features of shape [num_out_pas, feat_size];
                - {self.out_id_key} (LongTensor): output index map of shape [num_groups, mH, mW];
                - {self.out_grp_key} (LongTensor): output group indices of shape [num_out_act];
                - {self.out_pos_key} (LongTensor): output position indices of shape [num_out_act, 2].
        """

        # Retrieve desired items from storage dictionary
        in_act_feats = storage_dict[self.in_act_key]
        in_pas_feats = storage_dict[self.in_pas_key]
        in_id_map = storage_dict[self.in_id_key]
        in_grp_ids = storage_dict[self.in_grp_key]
        in_pos_ids = storage_dict[self.in_pos_key]
        act_mask = storage_dict[self.mask_key]

        # Get output active features
        out_act_feats = in_act_feats[act_mask]

        # Get output passive features
        out_pas_feats = torch.cat([in_pas_feats, in_act_feats[~act_mask]], dim=0)

        # Get output index map
        device = in_id_map.device

        num_in_act = in_act_feats.size(dim=0)
        num_out_act = out_act_feats.size(dim=0)
        num_in_pas = in_pas_feats.size(dim=0)

        new_act_ids = torch.empty(num_in_act, dtype=torch.int64, device=device)
        new_act_ids[act_mask] = torch.arange(num_out_act, device=device)
        new_pas_ids = num_out_act + torch.arange(num_in_pas, device=device)
        new_act_ids[~act_mask] = num_out_act + num_in_pas + torch.arange(num_in_act-num_out_act, device=device)

        new_ids = torch.cat([new_act_ids, new_pas_ids], dim=0)
        out_id_map = new_ids[in_id_map]

        # Get output group indices
        out_grp_ids = in_grp_ids[act_mask]

        # Get output position indices
        out_pos_ids = in_pos_ids[act_mask]

        # Store outputs in storage dictionary
        storage_dict[self.out_act_key] = out_act_feats
        storage_dict[self.out_pas_key] = out_pas_feats
        storage_dict[self.out_id_key] = out_id_map
        storage_dict[self.out_grp_key] = out_grp_ids
        storage_dict[self.out_pos_key] = out_pos_ids

        return storage_dict


@MODELS.register_module()
class SpsUpsample(nn.Module):
    """
    Class implementing the SpsUpsample module.

    Attributes:
        in_act_key (str): String with key to retrieve input active features from storage dictionary.
        in_pas_key (str): String with key to retrieve input passive features from storage dictionary.
        in_id_key (str): String with key to retrieve input index map from storage dictionary.
        in_grp_key (str): String with key to retrieve input group indices from storage dictionary.
        in_pos_key (str): String with key to retrieve input position indices from storage dictionary.
        out_act_key (str): String with key to store output active features in storage dictionary.
        out_pas_key (str): String with key to store output passive features in storage dictionary.
        out_id_key (str): String with key to store output index map in storage dictionary.
        out_grp_key (str): String with key to store output group indices in storage dictionary.
        out_pos_key (str): String with key to store output position indices in storage dictionary.
    """

    def __init__(self, in_act_key, in_pas_key, in_id_key, in_grp_key, in_pos_key, out_act_key, out_pas_key, out_id_key,
                 out_grp_key, out_pos_key):
        """
        Initializes the SpsUpsample module.

        Args:
            in_act_key (str): String with key to retrieve input active features from storage dictionary.
            in_pas_key (str): String with key to retrieve input passive features from storage dictionary.
            in_id_key (str): String with key to retrieve input index map from storage dictionary.
            in_grp_key (str): String with key to retrieve input group indices from storage dictionary.
            in_pos_key (str): String with key to retrieve input position indices from storage dictionary.
            out_act_key (str): String with key to store output active features in storage dictionary.
            out_pas_key (str): String with key to store output passive features in storage dictionary.
            out_id_key (str): String with key to store output index map in storage dictionary.
            out_grp_key (str): String with key to store output group indices in storage dictionary.
            out_pos_key (str): String with key to store output position indices in storage dictionary.
        """

        # Initialization of default nn.Module
        super().__init__()

        # Set additional attributes
        self.in_act_key = in_act_key
        self.in_pas_key = in_pas_key
        self.in_id_key = in_id_key
        self.in_grp_key = in_grp_key
        self.in_pos_key = in_pos_key
        self.out_act_key = out_act_key
        self.out_pas_key = out_pas_key
        self.out_id_key = out_id_key
        self.out_grp_key = out_grp_key
        self.out_pos_key = out_pos_key

    def forward(self, storage_dict, **kwargs):
        """
        Forward method of the SpsUpsample module.

        Args:
            storage_dict (Dict): Storage dictionary containing at least following keys:
                - {self.in_act_key} (FloatTensor): input active features of shape [num_act, feat_size];
                - {self.in_pas_key} (FloatTensor): input passive features of shape [num_pas, feat_size];
                - {self.in_id_key} (LongTensor): input index map of shape [num_groups, mH, mW];
                - {self.in_grp_key} (LongTensor): input group indices of shape [num_act];
                - {self.in_pos_key} (LongTensor): input position indices of shape [num_act, 2].

            kwargs (Dict): Dictionary of unused keyword arguments.

        Returns:
            storage_dict (Dict): Storage dictionary containing following additional keys:
                - {self.out_act_key} (FloatTensor): output active features of shape [4*num_act, feat_size];
                - {self.out_pas_key} (FloatTensor): output passive features of shape [num_pas, feat_size];
                - {self.out_id_key} (LongTensor): output index map of shape [num_groups, 2*mH, 2*mW];
                - {self.out_grp_key} (LongTensor): output group indices of shape [4*num_act];
                - {self.out_pos_key} (LongTensor): output position indices of shape [4*num_act, 2].
        """

        # Retrieve desired items from storage dictionary
        in_act_feats = storage_dict[self.in_act_key]
        in_pas_feats = storage_dict[self.in_pas_key]
        in_id_map = storage_dict[self.in_id_key]
        in_grp_ids = storage_dict[self.in_grp_key]
        in_pos_ids = storage_dict[self.in_pos_key]

        # Get output active features
        out_act_feats = in_act_feats.repeat_interleave(4, dim=0)

        # Get output passive features
        out_pas_feats = in_pas_feats

        # Get output index map
        num_act = in_act_feats.size(dim=0)
        act_id_mask = in_id_map < num_act

        id_map_0 = torch.where(act_id_mask, 4*in_id_map, 3*num_act + in_id_map)
        id_map_1 = torch.where(act_id_mask, id_map_0 + 1, id_map_0)
        id_map_2 = torch.where(act_id_mask, id_map_0 + 2, id_map_0)
        id_map_3 = torch.where(act_id_mask, id_map_0 + 3, id_map_0)

        id_map_01 = torch.stack([id_map_0, id_map_1], dim=3).flatten(2)
        id_map_23 = torch.stack([id_map_2, id_map_3], dim=3).flatten(2)
        out_id_map = torch.stack([id_map_01, id_map_23], dim=2).flatten(1, 2)

        # Get output group indices
        out_grp_ids = in_grp_ids.repeat_interleave(4, dim=0)

        # Get output position indices
        pos_offs = torch.tensor([[0, 0], [1, 0], [0, 1], [1, 1]], device=in_pos_ids.device)

        out_pos_ids = 2*in_pos_ids[:, None, :] + pos_offs
        out_pos_ids = out_pos_ids.flatten(0, 1)

        # Store outputs in storage dictionary
        storage_dict[self.out_act_key] = out_act_feats
        storage_dict[self.out_pas_key] = out_pas_feats
        storage_dict[self.out_id_key] = out_id_map
        storage_dict[self.out_grp_key] = out_grp_ids
        storage_dict[self.out_pos_key] = out_pos_ids

        return storage_dict
