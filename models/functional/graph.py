"""
Collection of functions related to graphs.
"""

import torch


def MapToGraph(feat_map):
    """
    Function transforming features from map structure to graph structure.

    Args:
        feat_map (FloatTensor): Features in map structure of shape [batch_size, feat_size, fH, fW].

    Returns:
        node_feats (FloatTensor): Graph node features of shape [num_nodes, feat_size].
        node_xy (FloatTensor): Node locations in normalized (x, y) format of shape [num_nodes, 2].
        node_adj_ids (List): List of size [num_nodes] with lists of adjacent node indices (including itself).

    Raises:
        NotImplementedError: Error when batch size is larger than 1.
    """

    # Get shape of input feature map
    batch_size, feat_size, fH, fW = feat_map.size()

    # Check whether batch size is larger than 1
    if batch_size > 1:
        error_msg = f"We currently only support a batch size of 1, but got {batch_size}."
        raise NotImplementedError(error_msg)

    # Get node features
    node_feats = feat_map.permute(0, 2, 3, 1).view(fH * fW, feat_size)

    # Get node indices in (y, x) format
    node_y_ids = torch.arange(fH, dtype=torch.int64)
    node_x_ids = torch.arange(fW, dtype=torch.int64)
    node_yx_ids = torch.meshgrid([node_y_ids, node_x_ids], indexing='ij')
    node_yx_ids = torch.stack(node_yx_ids, dim=2).view(fH * fW, 2)

    # Get node locations in normalized (x, y) format
    node_xy = node_yx_ids.fliplr()
    node_xy = (node_xy + 0.5) / torch.tensor([fW, fH])

    # Get node adjacency indices
    node_offsets = torch.arange(-1, 2, dtype=torch.int64)
    node_offsets = torch.meshgrid([node_offsets, node_offsets], indexing='ij')
    node_offsets = torch.stack(node_offsets, dim=2).view(9, 2)

    node_adj_ids = node_yx_ids[:, None, :] + node_offsets

    masks_y = (node_adj_ids[:, :, 0] >= 0) & (node_adj_ids[:, :, 0] <= fH-1)
    masks_x = (node_adj_ids[:, :, 1] >= 0) & (node_adj_ids[:, :, 1] <= fW-1)
    masks = masks_y & masks_x

    node_adj_ids = node_adj_ids[:, :, 0] * fW + node_adj_ids[:, :, 1]
    node_adj_ids = [node_adj_ids[i][masks[i]].tolist() for i in range(fH * fW)]

    return node_feats, node_xy, node_adj_ids
