"""
Collection of functions related to graphs.
"""

import torch
from torch.autograd import Function
from torch.autograd.function import once_differentiable


class EdgeDotProductPyCustom(Function):
    """
    Class implementing the EdgeDotProductPyCustom autograd function.

    This autograd function recomputes the edge source and target features in the backward pass, avoiding the need to
    keep the potentially large edge source and target features in memory.
    """

    @staticmethod
    def forward(ctx, node_src_feats, node_tgt_feats, edge_ids):
        """
        Forward method of the EdgeDotProductPyCustom autograd function.

        Args:
            ctx (Object): Context object storing additional data.
            node_src_feats (FloatTensor): Tensor with the node source features of shape [num_nodes, src_feat_size].
            node_tgt_feats (FloatTensor): Tensor with the node target features of shape [num_nodes, tgt_feat_size].
            edge_ids (LongTensor): Tensor with the node indices for each (directed) edge of shape [2, num_edges].

        Returns:
            edge_dot_prods (FloatTensor): Tensor containing the edge dot products of shape [num_edges].
        """

        # Compute edge dot products
        edge_src_feats = node_src_feats[edge_ids[0]]
        edge_tgt_feats = node_tgt_feats[edge_ids[1]]
        edge_dot_prods = torch.bmm(edge_src_feats[:, None, :], edge_tgt_feats[:, :, None]).view(-1)

        # Store desired tensors for the backward pass
        ctx.save_for_backward(node_src_feats, node_tgt_feats, edge_ids)

        return edge_dot_prods

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_edge_dot_prods):
        """
        Backward method of the EdgeDotProductPyCustom autograd function.

        Args:
            ctx (Object): Context object storing additional data.
            grad_edge_dot_prods (FloatTensor): Tensor containing the gradient edge dot products of shape [num_edges].

        Returns:
            grad_node_src_feats (FloatTensor): Gradients of node source features of shape [num_nodes, src_feat_size].
            grad_node_tgt_feats (FloatTensor): Gradients of node target features of shape [num_nodes, tgt_feat_size].
            grad_edge_ids (None): None.
        """

        # Recover stored tensors
        node_src_feats, node_tgt_feats, edge_ids = ctx.saved_tensors

        # Recompute edge features
        edge_src_feats = node_src_feats[edge_ids[0]]
        edge_tgt_feats = node_tgt_feats[edge_ids[1]]

        # Get gradient tensors
        grad_edge_src_feats = torch.bmm(grad_edge_dot_prods[:, None, None], edge_tgt_feats[:, None, :]).squeeze(1)
        grad_edge_tgt_feats = torch.bmm(edge_src_feats[:, :, None], grad_edge_dot_prods[:, None, None]).squeeze(2)

        grad_node_src_feats = torch.zeros_like(node_src_feats).index_add_(0, edge_ids[0], grad_edge_src_feats)
        grad_node_tgt_feats = torch.zeros_like(node_tgt_feats).index_add_(0, edge_ids[1], grad_edge_tgt_feats)
        grad_edge_ids = None

        return grad_node_src_feats, grad_node_tgt_feats, grad_edge_ids


def edge_dot_product(node_src_feats, node_tgt_feats, edge_ids, implementation='pytorch-custom'):
    """
    Function computing dot products between source and target features of edges.

    Args:
        node_src_feats (FloatTensor): Tensor containing the node source features of shape [num_nodes, src_feat_size].
        node_tgt_feats (FloatTensor): Tensor containing the node target features of shape [num_nodes, tgt_feat_size].
        edge_ids (LongTensor): Tensor containing the node indices for each (directed) edge of shape [2, num_edges].
        implementation (str): String containing the type of implementation (default='pytorch-custom').

    Returns:
        edge_dot_prods (FloatTensor): Tensor containing the edge dot products of shape [num_edges].

    Raises:
        ValueError: Error when the source and target feature size are not the same.
        ValueError: Error when an invalid implementation string was provided.
    """

    # Check source and target feature sizes
    src_feat_size = node_src_feats.size(dim=1)
    tgt_feat_size = node_tgt_feats.size(dim=1)

    if src_feat_size != tgt_feat_size:
        error_msg = f"The source and target feature size must be equal (got {src_feat_size} and {tgt_feat_size})."
        raise ValueError(error_msg)

    # Compute edge dot products
    if implementation == 'pytorch-custom':
        edge_dot_prods = EdgeDotProductPyCustom.apply(node_src_feats, node_tgt_feats, edge_ids)

    elif implementation == 'pytorch-naive':
        edge_src_feats = node_src_feats[edge_ids[0]]
        edge_tgt_feats = node_tgt_feats[edge_ids[1]]
        edge_dot_prods = torch.bmm(edge_src_feats[:, None, :], edge_tgt_feats[:, :, None]).view(-1)

    else:
        error_msg = f"Invalid implementation string for the edge_dot_product function (got '{implementation}')."
        raise ValueError(error_msg)

    return edge_dot_prods


def map_to_graph(feat_map):
    """
    Function transforming features from map structure to graph structure.

    Args:
        feat_map (FloatTensor): Features in map structure of shape [batch_size, feat_size, fH, fW].

    Returns:
        node_feats (FloatTensor): Graph node features of shape [num_nodes, feat_size].
        node_xy (FloatTensor): Node locations in normalized (x, y) format of shape [num_nodes, 2].
        node_adj_ids (List): List of size [num_nodes] with lists of adjacent node indices (including itself).
        edge_ids (LongTensor): Tensor containing the node indices for each (directed) edge of shape [2, num_edges].

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

    # Get edge indices
    edge_ids = [[j, i] for i, adj_i in enumerate(node_adj_ids) for j in adj_i]
    edge_ids = torch.tensor(edge_ids, dtype=torch.int64, device=node_feats.device).t().contiguous()

    return node_feats, node_xy, node_adj_ids, edge_ids
