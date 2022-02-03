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
    def forward(ctx, node_src_feats, node_tgt_feats, edge_ids, reduction='mul', num_groups=1):
        """
        Forward method of the NodeToEdgePyCustom autograd function.

        Args:
            ctx (FunctionCtx): Context object storing additional data.
            node_src_feats (FloatTensor): Tensor with the node source features of shape [num_nodes, src_feat_size].
            node_tgt_feats (FloatTensor): Tensor with the node target features of shape [num_nodes, tgt_feat_size].
            edge_ids (LongTensor): Tensor with the node indices for each (directed) edge of shape [2, num_edges].
            reduction (str): String containing the reduction operation (default='mul').
            num_groups (int): Integer containing the number of groups during 'mul-sum' reduction (default=1).

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

        elif reduction == 'mul-sum':
            feat_size = edge_src_feats.size(dim=1)
            edge_feats = edge_src_feats * edge_tgt_feats
            edge_feats = edge_feats.view(-1, num_groups, feat_size // num_groups).sum(dim=2)

        elif reduction == 'sum':
            edge_feats = edge_src_feats + edge_tgt_feats

        else:
            error_msg = f"Invalid reduction string for the NodeToEdgePyCustom autograd function (got '{reduction}')."
            raise ValueError(error_msg)

        # Store input tensors and reduction string for backward pass
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
            grad_num_groups (None): None.
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

        elif reduction == 'mul-sum':
            feat_size = edge_src_feats.size(dim=1)
            num_groups = grad_edge_feats.size(dim=1)

            grad_edge_feats = grad_edge_feats[:, :, None].expand(-1, -1, feat_size // num_groups)
            grad_edge_feats = grad_edge_feats.reshape(-1, feat_size)

            grad_edge_src_feats = grad_edge_feats * edge_tgt_feats
            grad_edge_tgt_feats = grad_edge_feats * edge_src_feats

        elif reduction == 'sum':
            grad_edge_src_feats = grad_edge_feats
            grad_edge_tgt_feats = grad_edge_feats

        grad_node_src_feats = torch.zeros_like(node_src_feats).index_add_(0, edge_ids[0], grad_edge_src_feats)
        grad_node_tgt_feats = torch.zeros_like(node_tgt_feats).index_add_(0, edge_ids[1], grad_edge_tgt_feats)

        grad_edge_ids = None
        grad_reduction = None
        grad_num_groups = None

        return grad_node_src_feats, grad_node_tgt_feats, grad_edge_ids, grad_reduction, grad_num_groups


class SparseDenseMmPyCustom(Function):
    """
    Class implementing the SparseDenseMmPyCustom autograd function.
    """

    @staticmethod
    def forward(ctx, sparse_ids, sparse_vals, sparse_size, dense_matrix):
        """
        Forward method of the SparseDenseMmPyCustom autograd function.

        Args:
            ctx (FunctionCtx): Context object storing additional data.
            sparse_ids (LongTensor): Tensor containing the indices of the sparse input matrix of shape [2, num_elems].
            sparse_vals (FloatTensor): Values of the sparse input matrix of shape [num_elems, num_heads].
            sparse_size (Tuple): Tuple containing the size of the sparse input matrix [M, N].
            dense_matrix (FloatTensor): Dense input matrix of shape [N, P].

        Returns:
            out_matrix (FloatTensor): Dense output matrix of shape [M, P].
        """

        # Perform sparse-dense matrix multiplication
        num_elems, num_heads = sparse_vals.size()
        inter_matrix = dense_matrix[sparse_ids[1]].view(num_elems, num_heads, -1)
        inter_matrix = sparse_vals[:, :, None] * inter_matrix
        inter_matrix = inter_matrix.view(num_elems, -1)

        out_size = (sparse_size[0], dense_matrix.size(dim=1))
        out_matrix = torch.zeros(out_size, dtype=dense_matrix.dtype, device=dense_matrix.device)

        src_ids = sparse_ids[0][:, None].expand_as(inter_matrix)
        out_matrix.scatter_add_(dim=0, index=src_ids, src=inter_matrix)

        # Store input tensors and source indices for backward pass
        ctx.save_for_backward(sparse_ids, sparse_vals, dense_matrix)
        ctx.src_ids = src_ids

        return out_matrix

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_out_matrix):
        """
        Backward method of the SparseDenseMmPyCustom autograd function.

        Args:
            ctx (FunctionCtx): Context object storing additional data.
            grad_out_matrix (FloatTensor): Gradients w.r.t. the dense output matrix of shape [M, P].

        Returns:
            grad_sparse_ids (None): None.
            grad_sparse_vals (FloatTensor): Gradients w.r.t. values of the sparse input matrix of shape [num_elems].
            grad_sparse_size (None): None
            grad_dense_matrix (FloatTensor): Gradients w.r.t. the dense input matrix of shape [N, P].
        """

        # Recover stored tensors and source indices from forward pass
        sparse_ids, sparse_vals, dense_matrix = ctx.saved_tensors
        src_ids = ctx.src_ids

        # Get gradient tensors
        num_elems, num_heads = sparse_vals.size()
        grad_inter_matrix = grad_out_matrix.gather(dim=0, index=src_ids)
        grad_inter_matrix = grad_inter_matrix.view(num_elems, num_heads, -1)

        grad_sparse_vals = grad_inter_matrix * dense_matrix[sparse_ids[1]].view(num_elems, num_heads, -1)
        grad_sparse_vals = grad_sparse_vals.sum(dim=2)

        grad_dense_matrix = grad_inter_matrix * sparse_vals[:, :, None]
        grad_dense_matrix = grad_dense_matrix.view(num_elems, -1)
        grad_dense_matrix = torch.zeros_like(dense_matrix).index_add_(0, sparse_ids[1], grad_dense_matrix)

        grad_sparse_ids = None
        grad_sparse_size = None

        return grad_sparse_ids, grad_sparse_vals, grad_sparse_size, grad_dense_matrix
