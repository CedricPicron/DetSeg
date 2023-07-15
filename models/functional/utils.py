"""
Collection of utility functions.
"""

import torch


def maps_to_seq(feat_maps):
    """
    Function rearranging features from 2D feature maps into a sequential format.

    Args:
        feat_maps (List): List of size [num_maps] with feature maps of shape [batch_size, feat_size, fH, fW].

    Returns:
        seq_feats (FloatTensor): Sequential features of shape [batch_size, sum(fH*fW), feat_size].
    """

    # Get sequential features
    seq_feats = torch.cat([feat_map.flatten(2).permute(0, 2, 1) for feat_map in feat_maps], dim=1)

    return seq_feats


def seq_to_maps(seq_feats, map_shapes):
    """
    Function rearranging sequential features into a format consisting of 2D maps.

    Args:
        seq_feats (FloatTensor): Sequential features of shape [batch_size, sum(fH*fW), feat_size].
        map_shapes (LongTensor): Feature map shapes in (H, W) format of shape [num_maps, 2].

    Returns:
        feat_maps (List): List of size [num_maps] with feature maps of shape [batch_size, feat_size, fH, fW].
    """

    # Get feature maps
    map_shapes = map_shapes.tolist()
    seq_feats = seq_feats.permute(0, 2, 1)
    batch_size, feat_size = seq_feats.size()[:2]

    end_id = 0
    feat_maps = []

    for (fH, fW) in map_shapes:
        start_id = end_id
        end_id += fH*fW

        feat_map = seq_feats[:, :, start_id:end_id].view(batch_size, feat_size, fH, fW)
        feat_maps.append(feat_map)

    return feat_maps
