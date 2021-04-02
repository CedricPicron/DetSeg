"""
Collection of functions providing position encodings.
"""
import math

import torch


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
