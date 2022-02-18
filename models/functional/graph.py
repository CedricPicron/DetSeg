"""
Collection of functions related to graphs.
"""

import torch
from torch_geometric.utils import grid

from models.functional.autograd import NodeToEdgePyCustom


def map_to_graph(feat_map):
    """
    Function transforming features from map structure to graph structure.

    Args:
        feat_map (FloatTensor): Features in map structure of shape [batch_size, feat_size, fH, fW].

    Returns:
        graph (Dict): Graph dictionary containing following keys:
            node_feats (FloatTensor): node features of shape [num_nodes, feat_size];
            edge_ids (LongTensor): node indices for each (directed) edge of shape [2, num_edges];
            node_xy (FloatTensor): node locations in normalized (x, y) format of shape [num_nodes, 2];
            node_batch_ids (LongTensor): node batch indices of shape [num_nodes].
    """

    # Get shape of input feature map
    batch_size, feat_size, fH, fW = feat_map.size()

    # Get node features
    node_feats = feat_map.permute(0, 2, 3, 1).reshape(batch_size * fH * fW, feat_size)
    graph = {'node_feats': node_feats}

    # Get edge indices and normalized node locations
    edge_ids, node_xy = grid(fH, fW, device=node_feats.device)
    edge_ids = torch.cat([edge_ids + i*fH*fW for i in range(batch_size)], dim=1)
    graph['edge_ids'] = edge_ids

    node_xy[:, 1] = node_xy[:, 1].flip(dims=[0])
    node_xy = (node_xy + 0.5) / torch.tensor([fW, fH], device=node_feats.device)
    node_xy = node_xy.repeat(batch_size, 1)
    graph['node_xy'] = node_xy

    # Get node batch indices
    node_batch_ids = torch.arange(batch_size, device=node_feats.device).repeat_interleave(fH * fW)
    graph['node_batch_ids'] = node_batch_ids

    return graph


def node_to_edge(node_src_feats, node_tgt_feats, edge_ids, off_edge_src=None, off_edge_tgt=None, reduction='mul',
                 num_groups=1, implementation='pytorch-custom'):
    """
    Function computing edge features from node source and target features using the specified reduction operation.

    Args:
        node_src_feats (FloatTensor): Tensor containing the node source features of shape [num_nodes, src_feat_size].
        node_tgt_feats (FloatTensor): Tensor containing the node target features of shape [num_nodes, tgt_feat_size].
        edge_ids (LongTensor): Tensor containing the node indices for each (directed) edge of shape [2, num_edges].
        off_edge_src (FloatTensor): Offset edge source features of shape [num_edges, src_feat_size] (default=None).
        off_edge_tgt (FloatTensor): Offset edge target features of shape [num_edges, tgt_feat_size] (default=None).
        reduction (str): String containing the reduction operation (default='mul').
        num_groups (int): Integer containing the number of groups during 'mul-sum' reduction (default=1).
        implementation (str): String containing the type of implementation (default='pytorch-custom').

    Returns:
        edge_feats (FloatTensor): Tensor containing the edge features of shape [num_edges, edge_feat_size].

    Raises:
        ValueError: Error when the source and target feature size are not the same.
        ValueError: Error when the number of groups does not divide the feature size during 'mul-sum' reduction.
        ValueError: Error when an invalid reduction string is provided.
        ValueError: Error when an invalid implementation string is provided.
    """

    # Check source and target feature sizes
    src_feat_size = node_src_feats.size(dim=1)
    tgt_feat_size = node_tgt_feats.size(dim=1)

    if src_feat_size != tgt_feat_size:
        error_msg = f"The source and target feature size must be equal (got {src_feat_size} and {tgt_feat_size})."
        raise ValueError(error_msg)

    # Check number of mul-sum groups if needed
    if reduction == 'mul-sum' and src_feat_size % num_groups != 0:
        error_msg = f"The number of groups ({num_groups}) must divide the feature size ({src_feat_size}) during "
        error_msg += "'mul-sum' reduction."
        raise ValueError(error_msg)

    # Compute edge features
    if implementation == 'pytorch-custom':
        apply_args = (node_src_feats, node_tgt_feats, edge_ids, off_edge_src, off_edge_tgt, reduction, num_groups)
        edge_feats = NodeToEdgePyCustom.apply(*apply_args)

    elif implementation == 'pytorch-naive':
        edge_src_feats = node_src_feats[edge_ids[0]]
        edge_tgt_feats = node_tgt_feats[edge_ids[1]]

        if off_edge_src is not None:
            edge_src_feats = edge_src_feats + off_edge_src

        if off_edge_tgt is not None:
            edge_tgt_feats = edge_tgt_feats + off_edge_tgt

        if reduction == 'dot':
            edge_feats = torch.bmm(edge_src_feats[:, None, :], edge_tgt_feats[:, :, None]).view(-1, 1)

        elif reduction == 'mul':
            edge_feats = edge_src_feats * edge_tgt_feats

        elif reduction == 'mul-sum':
            edge_feats = edge_src_feats * edge_tgt_feats
            edge_feats = edge_feats.view(-1, num_groups, src_feat_size // num_groups).sum(dim=2)

        elif reduction == 'sum':
            edge_feats = edge_src_feats + edge_tgt_feats

        else:
            error_msg = f"Invalid reduction string for the node_to_edge function (got '{reduction}')."
            raise ValueError(error_msg)

    else:
        error_msg = f"Invalid implementation string for the node_to_edge function (got '{implementation}')."
        raise ValueError(error_msg)

    return edge_feats
