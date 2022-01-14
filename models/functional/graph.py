"""
Collection of functions related to graphs.
"""

from torch_geometric.utils import grid
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
