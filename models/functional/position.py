"""
Collection of functions providing position encodings.
"""
import math

import torch


@torch.no_grad()
def sine_pos_encodings(input, input_type, max_period=1e4, normalize=True, scale=2*math.pi, eps=1e-6):
    """
    Function providing sine-based position encondings for the given input.

    Args:
        If input_type is 'pyramid':
            input (Tuple): Input tuple containing:
                - feat_maps (List): list [num_maps] with feature maps of shape [batch_size, feat_size, fH, fW];
                - feat_masks (List): list [num_maps] with masks of active features of shape [batch_size, fH, fW].

        input_type (str): String containing the input type chosen from {'pyramid'}.
        max_period (float): Maximum (co)sine period (default=1e4).
        normalize (bool): Boolean indicating whether to normalize sample points to given scale (default=True).
        scale (float): Scale used during normalization of the sample points (default=2*pi).
        eps (float): Value added to the normalization denominator for numerical stability (default=1e-6).

    Returns:
        If input_type is 'pyramid':
            pos_maps (List): List [num_maps] with position encoding maps of shape [batch_size, feat_size, fH, fW].
    """

    # Get position encodings based on input type
    if input_type == 'pyramid':
        feat_maps, feat_masks = input
        feat_size = feat_maps[0].shape[1]
        pos_maps = []

        for feat_map, feat_mask in zip(feat_maps, feat_masks):
            h_embed = feat_mask.cumsum(dim=1, dtype=torch.float32)
            w_embed = feat_mask.cumsum(dim=2, dtype=torch.float32)

            if normalize:
                h_embed = h_embed / (h_embed[:, -1:, :] + eps) * scale
                w_embed = w_embed / (w_embed[:, :, -1:] + eps) * scale

            periods = torch.arange(feat_size//2, dtype=torch.float32, device=feat_map.device)
            periods = max_period ** (2 * (periods//2) / (feat_size//2))

            pos_h = h_embed[:, :, :, None] / periods
            pos_w = w_embed[:, :, :, None] / periods

            pos_h = torch.stack((pos_h[:, :, :, 0::2].sin(), pos_h[:, :, :, 1::2].cos()), dim=4).flatten(3)
            pos_w = torch.stack((pos_w[:, :, :, 0::2].sin(), pos_w[:, :, :, 1::2].cos()), dim=4).flatten(3)

            pos_map = torch.cat((pos_h, pos_w), dim=3).permute(0, 3, 1, 2).to(feat_map.dtype)
            pos_maps.append(pos_map)

        return pos_maps
