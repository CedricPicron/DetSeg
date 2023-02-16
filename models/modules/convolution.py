"""
Collection of convolution-based modules.
"""
import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair as pair
from torch.nn.parameter import Parameter

from models.build import MODELS
from models.functional.convolution import conv_transpose2d, id_deform_conv2d


@MODELS.register_module()
class BottleneckConv(nn.Module):
    """
    Class implementing the BottleneckConv module.

    Attributes:
        proj (ProjConv): Module projecting feature map to lower dimensional bottleneck space.
        bottle (ProjConv): Module performing convolution in lower dimensional bottleneck space.
        expansion (ProjConv): Module expanding feature map to desired higher dimensional output space.
        skip (bool): Boolean indicating whether skip connection is used or not.
    """

    def __init__(self, in_channels, bottle_channels, out_channels=-1, kernel_size=3, norm='', skip=True, **kwargs):
        """
        Initializes the BottleneckConv module.

        Args:
            in_channels (int): Number of input channels.
            bottle_channels (int): Number of bottleneck channels.
            out_channels (int): Number of output channels (default=-1).
            kernel_size (int or Tuple): Size of the convolution kernel (default=3).

            norm (str): String containing the type of normalization chosen from {'', 'group'} (default='').
            skip (bool): Boolean indicating whether skip connection is used or not (default=True).

        Raises:
            ValueError: When number of input channels does not match output channels when skip connection is used.
            ValueError: When the number of output channels is not specified when no skip connection is used.
        """

        # Initialization of default nn.Module
        super().__init__()

        # Get and check number of output channels
        if skip and out_channels == -1:
            out_channels = in_channels

        elif skip and in_channels != out_channels:
            raise ValueError(f"Number of input channels ({in_channels}) must match number of output channels \
                             ({out_channels}) when skip connection is used.")

        elif not skip and out_channels == -1:
            raise ValueError("The number of output channels must be specified when no skip connection is used.")

        # Initialization of three ProjConv modules
        self.proj = ProjConv(in_channels, bottle_channels, norm=norm, skip=False, **kwargs)
        self.bottle = ProjConv(bottle_channels, bottle_channels, kernel_size, norm=norm, skip=False, **kwargs)
        self.expansion = ProjConv(bottle_channels, out_channels, norm=norm, skip=False, **kwargs)

        # Set skip attribute
        self.skip = skip

    def forward(self, map_input, **kwargs):
        """
        Forward method of the BottleneckConv module.

        Args:
            Two types of map-based inputs are supported:
                map_input (List): list [num_maps] with input feature maps of shape [batch_size, feat_size, fH, fW];
                map_input (FloatTensor): input feature map of shape [batch_size, feat_size, fH, fW].

        Returns:
            If 'map_input' was a list:
                map_output (List): list [num_maps] with output feature maps of shape [batch_size, feat_size, fH, fW].

            If map_output was a tensor:
                map_output (FloatTensor): output feature map of shape [batch_size, feat_size, fH, fW].

        Raises:
            ValueError: Error when 'map_input' has invalid type.
        """

        # Place inputs into common structure
        if isinstance(map_input, list):
            in_feat_maps = map_input
        elif torch.is_tensor(map_input):
            in_feat_maps = [map_input]
        else:
            raise ValueError(f"Got invalid type {type(map_input)} for 'map_input'.")

        # Initialize empty list for output maps
        out_feat_maps = []

        # Get output feature map corresponding to each input feature map
        for in_feat_map in in_feat_maps:
            feat_map = self.proj(in_feat_map)
            feat_map = self.bottle(feat_map)
            feat_map = self.expansion(feat_map)

            out_feat_map = in_feat_map + feat_map if self.skip else feat_map
            out_feat_maps.append(out_feat_map)

        # Get output in desired format depending on input
        if isinstance(map_input, list):
            map_output = out_feat_maps
        elif torch.is_tensor(map_input):
            map_output = out_feat_maps[0]

        return map_output


@MODELS.register_module()
class ConvTranspose2d(nn.ConvTranspose2d):
    """
    Class implementing the ConvTranspose2d module.

    It extends the ConvTranspose2d module from torch.nn by automatically computing the output padding if a base map
    size is given.
    """

    def forward(self, input, base_map_size=None):
        """
        Forward method of the ConvTranspose2d module.

        Args:
            input (FloatTensor): Tensor with input feature map of shape [batch_size, in_channels, iH, iW].
            base_map_size (Tuple): Tuple containing the base map size in (height, width) format (default=None).

        Returns:
            output (FloatTensor): Tensor with output feature map of shape [batch_size, out_channels, oH, oW].
        """

        # Perform 2D transposed convolution
        kwargs = {'bias': self.bias, 'stride': self.stride, 'padding': self.padding}
        kwargs = {**kwargs, 'output_padding': self.output_padding, 'groups': self.groups, 'dilation': self.dilation}
        output = conv_transpose2d(input, self.weight, base_map_size=base_map_size, **kwargs)

        return output


@MODELS.register_module()
class IdDeformConv2d(nn.Module):
    """
    Class implementing the IdDeformConv2d module.

    Attributes:
        conv_offsets (nn.Linear): Linear layer computing the convolution offsets.
        point_weights (nn.Linear): Optional linear layer computing the unnormalized point weights.
        mod_weights (nn.Linear): Optional linear layer computing the unnormalized modulated weights.
        weight (Parameter): Parameter with convolution weights of shape [out_channels, kH * kW * in_channels].
        bias (Parameter): Optional parameter with convolution biases of shape [out_channels].
    """

    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, dilations=None, bias=True, modulated=False):
        """
        Initializes the IdDeformConv2d module.

        Args:
            in_channels (int): Integer containing the number of input channels.
            out_channels (int): Integer containing the number of output channels.
            kernel_size (int or Tuple): Integer ot tuple containing the size of the convolving kernel.
            dilation (int or Tuple): Integer or tuple containing the convolution dilation (default=1).
            dilations (List): List containing one or multiple convolution dilations (default=None).
            bias (bool): Boolean indicating whether learnable bias is added to the output (default=True).
            modulated (bool): Boolean indicating whether to perform modulated convolutions (default=False).

        Raises:
            ValueError: Error when an even kernel size is provided in either dimension.
        """

        # Initialization of default nn.Module
        super().__init__()

        # Get kernel size and dilations in pair format
        kH, kW = pair(kernel_size)
        dilations = [pair(dilation)] if dilations is None else [pair(dilation) for dilation in dilations]

        # Set number of points attribute
        self.num_points = len(dilations)

        # Initialize convolution offsets module
        self.conv_offsets = nn.Linear(in_channels, kH * kW * self.num_points * 2)
        nn.init.zeros_(self.conv_offsets.weight)
        init_offs_list = []

        for dH, dW in dilations:
            init_offs_x = (kW-1)/2 * dW
            init_offs_y = (kH-1)/2 * dH

            init_offs_x = torch.linspace(-init_offs_x, init_offs_x, steps=kW)[None, :].expand(kH, -1)
            init_offs_y = torch.linspace(-init_offs_y, init_offs_y, steps=kH)[:, None].expand(-1, kW)

            init_offs = torch.stack([init_offs_x, init_offs_y], dim=2)
            init_offs_list.append(init_offs)

        init_offs = torch.stack(init_offs_list, dim=2)
        self.conv_offsets.bias = nn.Parameter(init_offs.view(-1))

        # Initialize point weights module if needed
        if self.num_points > 1:
            self.point_weights = nn.Linear(in_channels, kH * kW * self.num_points)
            nn.init.zeros_(self.point_weights.weight)
            nn.init.zeros_(self.point_weights.bias)

        else:
            self.register_parameter('point_weights', None)

        # Initialize modulation weights module if needed
        if modulated:
            self.mod_weights = nn.Linear(in_channels, kH * kW)
            nn.init.zeros_(self.mod_weights.weight)
            nn.init.zeros_(self.mod_weights.bias)

        else:
            self.register_parameter('mod_weights', None)

        # Initialize weight parameter
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
        Forward method of the IdDeformConv2d module.

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

        # Get device and number of core features
        device = in_core_feats.device
        num_core_feats = len(in_core_feats)

        # Get convolution locations
        conv_xy = self.conv_offsets(in_core_feats)
        conv_xy = conv_xy.view(num_core_feats, -1, self.num_points, 2)
        conv_xy = pos_ids[:, None, None, :] + conv_xy

        # Get convolution indices
        floor_xy = conv_xy.floor()

        conv_ids = floor_xy.int()
        grid_ids = torch.tensor([[0, 0], [1, 0], [0, 1], [1, 1]], device=device)
        conv_ids = conv_ids[:, :, :, None, :] + grid_ids

        conv_ids_x = conv_ids[:, :, :, :, 0]
        conv_ids_y = conv_ids[:, :, :, :, 1]

        rH, rW = id_map.size()[1:]
        pad_mask = (conv_ids_x < 0) | (conv_ids_y < 0) | (conv_ids_x >= rW) | (conv_ids_y >= rH)

        conv_ids_x = conv_ids_x.clamp_(min=0, max=rW-1)
        conv_ids_y = conv_ids_y.clamp_(min=0, max=rH-1)

        roi_ids = roi_ids[:, None, None, None].expand_as(conv_ids_x)
        conv_ids = id_map[roi_ids, conv_ids_y, conv_ids_x]
        conv_ids[pad_mask] = len(in_core_feats) + len(aux_feats)

        # Get convolution weights
        delta_xy = conv_xy - floor_xy
        delta_xy = torch.stack([delta_xy, 1-delta_xy], dim=3)

        x_ids = torch.tensor([1, 0, 1, 0], device=device)
        y_ids = torch.tensor([1, 1, 0, 0], device=device)
        conv_weights = delta_xy[:, :, :, 0, x_ids] * delta_xy[:, :, :, 1, y_ids]

        if self.point_weights is not None:
            point_weights = self.point_weights(in_core_feats).view(num_core_feats, -1, self.num_points)
            point_weights = F.softmax(point_weights, dim=2)
            conv_weights = point_weights[:, :, :, None] * conv_weights

        if self.mod_weights is not None:
            mod_weights = self.mod_weights(in_core_feats).sigmoid()
            conv_weights = mod_weights[:, :, None, None] * conv_weights

        # Perform 2D index-based deformable convolution
        out_core_feats = id_deform_conv2d(in_core_feats, aux_feats, conv_ids, conv_weights, self.weight, self.bias)

        return out_core_feats


@MODELS.register_module()
class ProjConv(nn.Module):
    """
    Class implementing the ProjConv module.

    Attributes:
        conv (nn.Conv2d): Module performing the 2D convolution operation.
        norm (nn.Module): Optional attribute containing the normalization module.
        skip (bool): Boolean indicating whether skip connection is used or not.
    """

    def __init__(self, in_channels, out_channels=-1, kernel_size=1, norm='', skip=True, **kwargs):
        """
        Initializes the ProjConv module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels (default=-1).
            kernel_size (int or Tuple): Size of the convolution kernel (default=1).

            norm (str): String containing the type of normalization chosen from {'', 'group'} (default='').
            skip (bool): Boolean indicating whether skip connection is used or not (default=True).

        Raises:
            ValueError: When number of input channels does not match output channels when skip connection is used.
            ValueError: When the number of output channels is not specified when no skip connection is used.
            ValueError: When unsupported normalization type is provided.
        """

        # Initialization of default nn.Module
        super().__init__()

        # Get and check number of output channels
        if skip and out_channels == -1:
            out_channels = in_channels

        elif skip and in_channels != out_channels:
            raise ValueError(f"Number of input channels ({in_channels}) must match number of output channels \
                             ({out_channels}) when skip connection is used.")

        elif not skip and out_channels == -1:
            raise ValueError("The number of output channels must be specified when no skip connection is used.")

        # Get convolution keyword arguments
        conv_keys = ('stride', 'padding', 'dilation', 'groups', 'bias', 'padding_mode')
        conv_kwargs = {k: v for k, v in kwargs.items() if k in conv_keys}

        # Add default padding to convolution keyword arguments if not yet present
        if 'padding' not in conv_kwargs:
            if isinstance(kernel_size, int):
                conv_kwargs['padding'] = kernel_size//2
            else:
                conv_kwargs['padding'] = (kernel_size[0]//2, kernel_size[1]//2)

        # Initialization of convolution module
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, **conv_kwargs)

        # Initialization of normalization module
        if not norm:
            pass

        elif norm == 'group':
            num_groups = kwargs.pop('num_groups', 8)
            group_keys = ('eps', 'affine')
            group_kwargs = {k: v for k, v in kwargs.items() if k in group_keys}
            self.norm = nn.GroupNorm(num_groups, in_channels, **group_kwargs)

        else:
            raise ValueError(f"The ProjConv module does not support the '{norm}' normalization type.")

        # Set skip attribute
        self.skip = skip

    def forward(self, map_input, **kwargs):
        """
        Forward method of the ProjConv module.

        Args:
            Two types of map-based inputs are supported:
                map_input (List): list [num_maps] with input feature maps of shape [batch_size, feat_size, fH, fW];
                map_input (FloatTensor): input feature map of shape [batch_size, feat_size, fH, fW].

        Returns:
            If 'map_input' was a list:
                map_output (List): list [num_maps] with output feature maps of shape [batch_size, feat_size, fH, fW].

            If map_output was a tensor:
                map_output (FloatTensor): output feature map of shape [batch_size, feat_size, fH, fW].

        Raises:
            ValueError: Error when 'map_input' has invalid type.
        """

        # Place inputs into common structure
        if isinstance(map_input, list):
            in_feat_maps = map_input
        elif torch.is_tensor(map_input):
            in_feat_maps = [map_input]
        else:
            raise ValueError(f"Got invalid type {type(map_input)} for 'map_input'.")

        # Initialize empty list for output maps
        out_feat_maps = []

        # Get output feature map corresponding to each input feature map
        for in_feat_map in in_feat_maps:
            feat_map = self.norm(in_feat_map) if hasattr(self, 'norm') else in_feat_map
            feat_map = F.relu(feat_map, inplace=hasattr(self, 'norm'))
            feat_map = self.conv(feat_map)

            out_feat_map = in_feat_map + feat_map if self.skip else feat_map
            out_feat_maps.append(out_feat_map)

        # Get output in desired format depending on input
        if isinstance(map_input, list):
            map_output = out_feat_maps
        elif torch.is_tensor(map_input):
            map_output = out_feat_maps[0]

        return map_output
