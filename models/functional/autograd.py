"""
Collection of custom autograd functions.
"""

import torch
from torch.autograd import Function
from torch.autograd.function import once_differentiable


class NodeToEdgePyCustom(Function):
    """
    Class implementing the NodeToEdgePyCustom autograd function.

    This custom autograd function reconstructs the edge source and target features in the backward pass, avoiding the
    need to keep the potentially large edge source and target features in memory.
    """

    @staticmethod
    def forward(ctx, node_src_feats, node_tgt_feats, edge_ids, reduction='mul'):
        """
        Forward method of the NodeToEdgePyCustom autograd function.

        Args:
            ctx (FunctionCtx): Context object storing additional data.
            node_src_feats (FloatTensor): Tensor with the node source features of shape [num_nodes, src_feat_size].
            node_tgt_feats (FloatTensor): Tensor with the node target features of shape [num_nodes, tgt_feat_size].
            edge_ids (LongTensor): Tensor with the node indices for each (directed) edge of shape [2, num_edges].
            reduction (str): String containing the reduction operation (default='mul').

        Returns:
            edge_feats (FloatTensor): Tensor containing the edge features of shape [num_edges, edge_feat_size].

        Raises:
            ValueError: Error when an invalid reduction string is provided.
        """

        # Compute edge features
        edge_src_feats = node_src_feats[edge_ids[0]]
        edge_tgt_feats = node_tgt_feats[edge_ids[1]]

        if reduction == 'dot':
            edge_feats = torch.bmm(edge_src_feats[:, None, :], edge_tgt_feats[:, :, None]).squeeze(dim=2)
        elif reduction == 'mul':
            edge_feats = edge_src_feats * edge_tgt_feats
        elif reduction == 'sum':
            edge_feats = edge_src_feats + edge_tgt_feats
        else:
            error_msg = f"Invalid reduction string for the NodeToEdgePyCustom autograd function (got '{reduction}')."
            raise ValueError(error_msg)

        # Store input tensors and reduction string for the backward pass
        ctx.save_for_backward(node_src_feats, node_tgt_feats, edge_ids)
        ctx.reduction = reduction

        return edge_feats

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_edge_feats):
        """
        Backward method of the NodeToEdgePyCustom autograd function.

        Args:
            ctx (FunctionCtx): Context object storing additional data.
            grad_edge_feats (FloatTensor): Gradients of the edge features of shape [num_edges, edge_feat_size].

        Returns:
            grad_node_src_feats (FloatTensor): Gradients of node source features of shape [num_nodes, src_feat_size].
            grad_node_tgt_feats (FloatTensor): Gradients of node target features of shape [num_nodes, tgt_feat_size].
            grad_edge_ids (None): None.
            grad_reduction (None): None.
        """

        # Recover stored tensors and reduction string from forward pass
        node_src_feats, node_tgt_feats, edge_ids = ctx.saved_tensors
        reduction = ctx.reduction

        # Recompute edge features
        edge_src_feats = node_src_feats[edge_ids[0]]
        edge_tgt_feats = node_tgt_feats[edge_ids[1]]

        # Get gradient tensors
        if reduction == 'dot':
            grad_edge_src_feats = torch.bmm(grad_edge_feats[:, :, None], edge_tgt_feats[:, None, :]).squeeze(dim=1)
            grad_edge_tgt_feats = torch.bmm(edge_src_feats[:, :, None], grad_edge_feats[:, :, None]).squeeze(dim=2)

        elif reduction == 'mul':
            grad_edge_src_feats = grad_edge_feats * edge_tgt_feats
            grad_edge_tgt_feats = grad_edge_feats * edge_src_feats

        elif reduction == 'sum':
            grad_edge_src_feats = grad_edge_feats
            grad_edge_tgt_feats = grad_edge_feats

        grad_node_src_feats = torch.zeros_like(node_src_feats).index_add_(0, edge_ids[0], grad_edge_src_feats)
        grad_node_tgt_feats = torch.zeros_like(node_tgt_feats).index_add_(0, edge_ids[1], grad_edge_tgt_feats)
        grad_edge_ids = None
        grad_reduction = None

        return grad_node_src_feats, grad_node_tgt_feats, grad_edge_ids, grad_reduction
