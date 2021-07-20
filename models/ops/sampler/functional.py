"""
Collection of sampler functions.
"""

import torch


def naive_maps_sampler_2d(feats, feat_map_wh, feat_map_offs, sample_xy, sample_map_ids):
    """
    Function sampling 2D locations from collection of 2D feature maps using PyTorch functions.

    Args:
        feats (FloatTensor): Features to sample from of shape [batch_size, num_feats, feat_size].
        feat_map_wh (LongTensor): Feature map sizes in (W, H) format of shape [num_maps, 2].
        feat_map_offs (LongTensor): Feature map offsets of shape [num_maps].
        sample_xy (FloatTensor): Zero-one normalized (X, Y) sample locations of shape [batch_size, num_samples, 2].
        sample_map_ids (LongTensor): Feature map indices to sample from of shape [batch_size, num_samples].

    Returns:
        sample_feats (FloatTensor): Sample features of shape [batch_size, num_samples, feat_size].
    """

    # Get some sizes
    batch_size, _, feat_size = feats.size()
    _, num_samples, _ = sample_xy.size()

    # Get unnormalized sample locations
    sample_wh = feat_map_wh[sample_map_ids]
    sample_w = sample_wh[:, :, 0]
    sample_offs = feat_map_offs[sample_map_ids]
    sample_xy = sample_xy * (sample_wh - 1)

    # Get unweighted sample features
    sample_ij = sample_xy.floor().to(dtype=torch.int64).clamp_(min=0)
    sample_ij = torch.min(torch.stack([sample_ij, sample_wh-2], dim=3), dim=3)[0]

    top_left_ids = sample_offs + sample_w * sample_ij[:, :, 1] + sample_ij[:, :, 0]
    top_right_ids = top_left_ids + 1
    bot_left_ids = top_left_ids + sample_w
    bot_right_ids = bot_left_ids + 1

    sample_ids = torch.cat([top_left_ids, top_right_ids, bot_left_ids, bot_right_ids], dim=1)
    sample_feats = torch.gather(feats, dim=1, index=sample_ids[:, :, None].expand(-1, -1, feat_size))

    # Get weighted sample features
    right_weights, bot_weights = (sample_xy - sample_ij).split(1, dim=2)
    left_weights, top_weights = (1-right_weights, 1-bot_weights)

    top_left_ws = left_weights * top_weights
    top_right_ws = right_weights * top_weights
    bot_left_ws = left_weights * bot_weights
    bot_right_ws = right_weights * bot_weights

    sample_weights = torch.cat([top_left_ws, top_right_ws, bot_left_ws, bot_right_ws], dim=1)
    sample_feats = sample_weights * sample_feats
    sample_feats = sample_feats.view(batch_size, 4, num_samples, feat_size).sum(dim=1)

    return sample_feats


def naive_maps_sampler_3d(feats, feat_map_wh, feat_map_offs, sample_xyz):
    """
    Function sampling 3D locations from collection of 2D feature maps using PyTorch functions.

    Args:
        feats (FloatTensor): Features to sample from of shape [batch_size, num_feats, feat_size].
        feat_map_wh (LongTensor): Feature map sizes in (W, H) format of shape [num_maps, 2].
        feat_map_offs (LongTensor): Feature map offsets of shape [num_maps].
        sample_xyz (FloatTensor): Zero-one normalized (X, Y, Z) sample locations of shape [batch_size, num_samples, 3].

    Returns:
        sample_feats (FloatTensor): Sample features of shape [batch_size, num_samples, feat_size].
    """

    # Get some sizes
    batch_size, _, feat_size = feats.size()
    num_maps, _ = feat_map_wh.size()
    _, num_samples, _ = sample_xyz.size()

    # Get unnormalized sample locations
    sample_z = sample_xyz[:, :, 2] * (num_maps - 1)
    sample_k = sample_z.floor().to(dtype=torch.int64).clamp_(min=0, max=num_maps-2)
    sample_map_ids = torch.stack([sample_k, sample_k+1], dim=2)

    sample_wh = feat_map_wh[sample_map_ids]
    sample_w = sample_wh[:, :, :, 0]
    sample_offs = feat_map_offs[sample_map_ids]
    sample_xy = sample_xyz[:, :, None, :2] * (sample_wh - 1)

    # Get unweighted sample features
    sample_ij = sample_xy.floor().to(dtype=torch.int64).clamp_(min=0)
    sample_ij = torch.min(torch.stack([sample_ij, sample_wh-2], dim=4), dim=4)[0]

    top_left_ids = sample_offs + sample_w * sample_ij[:, :, :, 1] + sample_ij[:, :, :, 0]
    top_right_ids = top_left_ids + 1
    bot_left_ids = top_left_ids + sample_w
    bot_right_ids = bot_left_ids + 1

    sample_ids = torch.cat([top_left_ids, top_right_ids, bot_left_ids, bot_right_ids], dim=2).flatten(1)
    sample_feats = torch.gather(feats, dim=1, index=sample_ids[:, :, None].expand(-1, -1, feat_size))

    # Get weighted sample features
    right_weights, bot_weights = (sample_xy - sample_ij).split(1, dim=3)
    left_weights, top_weights = (1-right_weights, 1-bot_weights)

    back_weights = sample_z - sample_k
    front_weights = 1-back_weights
    front_back_weights = torch.stack([front_weights, back_weights], dim=2)
    front_back_weights = front_back_weights[:, :, :, None]

    top_left_ws = left_weights * top_weights * front_back_weights
    top_right_ws = right_weights * top_weights * front_back_weights
    bot_left_ws = left_weights * bot_weights * front_back_weights
    bot_right_ws = right_weights * bot_weights * front_back_weights

    sample_weights = torch.cat([top_left_ws, top_right_ws, bot_left_ws, bot_right_ws], dim=2).flatten(1, 2)
    sample_feats = sample_weights * sample_feats
    sample_feats = sample_feats.view(batch_size, 8, num_samples, feat_size).sum(dim=1)

    return sample_feats
