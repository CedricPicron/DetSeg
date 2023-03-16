"""
Collection of position-related functions and modules.
"""
import math

import torch
from torch import nn, Tensor


@torch.no_grad()
def sine_pos_encodings(map_input, max_period=1e4, normalize=True, scale=2*math.pi):
    """
    Function providing sine-based position features and ids for the given input maps.

    Args:
        Two types of map-based inputs are supported:
            map_input (List): list of size [num_maps] with feature maps of shape [batch_size, feat_size, fH, fW];
            map_input (FloatTensor): feature map of shape [batch_size, feat_size, fH, fW].

        max_period (float): Maximum (co)sine period (default=1e4).
        normalize (bool): Boolean indicating whether to normalize sample points to given scale (default=True).
        scale (float): Scale used during normalization of the sample points (default=2*pi).

    Returns:
        If 'map_input' was a list:
            pos_feat_output (List): list [num_maps] with position feature maps of shape [feat_size, fH, fW];
            pos_id_output (List): list [num_maps] with (width, height) position id maps of shape [2, fH, fW].

        If 'map_input' was a tensor:
            pos_feat_output (FloatTensor): position feature map of shape [feat_size, fH, fW];
            pos_id_output (FloatTensor): (width, height) position id map of shape [2, fH, fW].

    Raises:
        ValueError: Error when 'map_input' has invalid type.
    """

    # Place inputs into common structure
    if isinstance(map_input, list):
        feat_maps = map_input
    elif torch.is_tensor(map_input):
        feat_maps = [map_input]
    else:
        raise ValueError(f"Got invalid type {type(map_input)} for 'map_input'.")

    # Get tensor keyword arguments
    tensor_kwargs = {'dtype': feat_maps[0].dtype, 'device': feat_maps[0].device}

    # Initialize empty lists for output maps
    pos_feat_maps = []
    pos_id_maps = []

    # Get position features and ids for every input map
    for feat_map in feat_maps:
        feat_size, fH, fW = feat_map.shape[1:]

        center_x = torch.arange(fW, **tensor_kwargs) + 0.5
        center_y = torch.arange(fH, **tensor_kwargs) + 0.5

        if normalize:
            center_x = center_x / fW
            center_y = center_y / fH

        periods = torch.arange(feat_size//4, **tensor_kwargs) / (feat_size//4)
        periods = max_period ** (periods)

        pts_x = scale * center_x[:, None] / periods
        pts_y = scale * center_y[:, None] / periods

        pos_feat_x = torch.cat([pts_x.sin(), pts_x.cos()], dim=1)
        pos_feat_y = torch.cat([pts_y.sin(), pts_y.cos()], dim=1)

        pos_feat_x = pos_feat_x.repeat(fH, 1, 1)
        pos_feat_y = pos_feat_y.repeat(fW, 1, 1).transpose(0, 1)

        pos_feat_map = torch.cat([pos_feat_x, pos_feat_y], dim=2).permute(2, 0, 1)
        pos_feat_maps.append(pos_feat_map)

        center_x = center_x.repeat(fH, 1)
        center_y = center_y.repeat(fW, 1).t()

        pos_id_map = torch.stack([center_x, center_y], dim=0)
        pos_id_maps.append(pos_id_map)

    # Get output in desired format depending on input
    if isinstance(map_input, list):
        pos_feat_output = pos_feat_maps
        pos_id_output = pos_id_maps

    elif torch.is_tensor(map_input):
        pos_feat_output = pos_feat_maps[0]
        pos_id_output = pos_id_maps[0]

    return pos_feat_output, pos_id_output


class SinePositionEncoder(nn.Module):
    """
    Class implementing the SinePositionEncoder module.

    It is a two-dimensional position encoder with sines and cosines, relative to padding mask.

    Attributes:
        temperature (int): Sample points will cover between 1/temperature and 1 of the (co)sine period.
        normalize (bool): Normalize sample points to given scale.
        scale (float): Scale used during normalization (ignored when normalize is False).
    """

    def __init__(self, temperature=10000, normalize=True, scale=2*math.pi):
        """
        Initializes the SinePositionEncoder module.

        Args:
            temperature (int): Sample points will cover between 1/temperature and 1 of the (co)sine period.
            normalize (bool): Normalize sample points to given scale.
            scale (float): Scale used during normalization (ignored when normalize is False).

        Raises:
            ValueError: Raised when normalize is False and scale was passed.
        """

        super().__init__()

        if scale is not None and normalize is False:
            raise ValueError("Normalize should be True if scale is passed.")

        self.temperature = temperature
        self.normalize = normalize
        self.scale = scale

    def forward(self, features: Tensor, feature_masks: Tensor) -> Tensor:
        """
        Forward method of the SinePositionEncoder module.

        Args:
             features (FloatTensor): Features of shape [batch_size, feat_dim, fH, fW].
             feature_masks (BoolTensor): Boolean masks encoding active pixels of shape [batch_size, fH, fW].

        Returns:
            pos (Tensor): Position encodings of shape [batch_size, feat_dim, fH, fW].
        """

        y_embed = feature_masks.cumsum(1, dtype=torch.float32)
        x_embed = feature_masks.cumsum(2, dtype=torch.float32)

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        feature_size_per_dim = features.shape[1] // 2
        dim_t = torch.arange(feature_size_per_dim, dtype=torch.float32, device=features.device)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode='floor') / feature_size_per_dim)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)

        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        pos = pos.to(features.dtype)

        return pos


def build_position_encoder(args):
    """
    Build position encoder from command-line arguments.

    Args:
        args (argsparse.Namespace): Command-line arguments.

    Returns:
        position_encoder (nn.Module): The specified position encoder module.

    Raises:
        ValueError: Raised when unknown args.position_encoding is provided.
    """

    if args.position_encoding == 'sine':
        position_encoder = SinePositionEncoder()
    else:
        raise ValueError(f'Unknown position encoding {args.position_encoding} was provided.')

    return position_encoder
