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
    sample_offs = feat_map_offs[sample_map_ids]

    eps = 1e-10
    sample_xy = sample_xy.clamp(min=0.0, max=1.0-eps)
    sample_xy = sample_xy * (sample_wh - 1)

    # Get unweighted sample features
    sample_ij = sample_xy.floor().to(dtype=torch.int64)
    sample_w = sample_wh[:, :, 0]

    top_left_ids = sample_offs + sample_w * sample_ij[:, :, 1] + sample_ij[:, :, 0]
    top_right_ids = top_left_ids + 1
    bot_left_ids = top_left_ids + sample_w
    bot_right_ids = bot_left_ids + 1

    sample_ids = torch.cat([top_left_ids, top_right_ids, bot_left_ids, bot_right_ids], dim=1)
    sample_feats = torch.gather(feats, dim=1, index=sample_ids[:, :, None].expand(-1, -1, feat_size))

    # Get weighted sample features
    left_dists, top_dists = (sample_xy - sample_ij).split(1, dim=2)
    right_dists, bot_dists = (1-left_dists, 1-top_dists)

    top_left_ws = left_dists * top_dists
    top_right_ws = right_dists * top_dists
    bot_left_ws = left_dists * bot_dists
    bot_right_ws = right_dists * bot_dists

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
    eps = 1e-10
    sample_xyz = sample_xyz.clamp(min=0.0, max=1.0-eps)
    sample_z = sample_xyz[:, :, 2] * (num_maps - 1)

    sample_k = sample_z.floor().to(dtype=torch.int64)
    sample_map_ids = torch.stack([sample_k, sample_k+1], dim=2)

    sample_wh = feat_map_wh[sample_map_ids]
    sample_offs = feat_map_offs[sample_map_ids]
    sample_xy = sample_xyz[:, :, None, :2] * (sample_wh - 1)

    # Get unweighted sample features
    sample_ij = sample_xy.floor().to(dtype=torch.int64)
    sample_w = sample_wh[:, :, :, 0]

    top_left_ids = sample_offs + sample_w * sample_ij[:, :, :, 1] + sample_ij[:, :, :, 0]
    top_right_ids = top_left_ids + 1
    bot_left_ids = top_left_ids + sample_w
    bot_right_ids = bot_left_ids + 1

    sample_ids = torch.cat([top_left_ids, top_right_ids, bot_left_ids, bot_right_ids], dim=2).flatten(1)
    sample_feats = torch.gather(feats, dim=1, index=sample_ids[:, :, None].expand(-1, -1, feat_size))

    # Get weighted sample features
    left_dists, top_dists = (sample_xy - sample_ij).split(1, dim=3)
    right_dists, bot_dists = (1-left_dists, 1-top_dists)

    front_dists = sample_z - sample_k
    back_dists = 1-front_dists
    front_back_dists = torch.stack([front_dists, back_dists], dim=2)
    front_back_dists = front_back_dists[:, :, :, None]

    top_left_ws = left_dists * top_dists * front_back_dists
    top_right_ws = right_dists * top_dists * front_back_dists
    bot_left_ws = left_dists * bot_dists * front_back_dists
    bot_right_ws = right_dists * bot_dists * front_back_dists

    sample_weights = torch.cat([top_left_ws, top_right_ws, bot_left_ws, bot_right_ws], dim=2).flatten(1, 2)
    sample_feats = sample_weights * sample_feats
    sample_feats = sample_feats.view(batch_size, 8, num_samples, feat_size).sum(dim=1)

    return sample_feats
