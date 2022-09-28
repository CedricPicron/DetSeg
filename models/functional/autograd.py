"""
Collection of custom autograd functions.
"""

import torch
from torch.autograd import Function
from torch.autograd.function import once_differentiable
import torch.nn.functional as F


class AdjConv2d(Function):
    """
    Class implementing the AdjConv2d autograd function.

    This custom autograd function alters the backward pass by not keeping the intermediate data structure in memory.
    """

    @staticmethod
    def forward(ctx, in_feats, weight, bias, adj_ids):
        """
        Forward method of the AdjConv2d autograd function.

        Args:
            ctx (FunctionCtx): Context object storing additional data.
            in_feats (FloatTensor): Input features of shape [num_feats, in_channels].
            weight (FloatTensor): Tensor with convolution weights of shape [out_channels, kH * kW * in_channels].
            bias (FloatTensor): Tensor with the convolution biases of shape [out_channels].
            adj_ids (LongTensor): Adjacency indices of convolution features of shape [num_conv_feats, kH * kW].

        Returns:
            out_feats (FloatTensor): Output convolution features of shape [num_conv_feats, out_channels].
        """

        # Get intermediate features
        inter_feats = in_feats[adj_ids].flatten(1)

        # Get output features
        out_feats = torch.mm(inter_feats, weight.t()) + bias

        # Save desired input tensors for backward pass
        ctx.save_for_backward(in_feats, weight, adj_ids)

        return out_feats

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_out_feats):
        """
        Backward method of the AdjConv2d autograd function.

        Args:
            ctx (FunctionCtx): Context object storing additional data.
            grad_out_feats (FloatTensor): Gradient w.r.t. the output features of shape [num_conv_feats, out_channels].

        Returns:
            grad_in_feats (FloatTensor): Gradient w.r.t. the input features of shape [num_feats, in_channels].
            grad_weight (FloatTensor): Gradient w.r.t. the conv weights of shape [out_channels, kH * kW * in_channels].
            grad_bias (FloatTensor): Gradient w.r.t. the conv biases of shape [out_channels].
            grad_adj_ids (None): None.
        """

        # Recover desired input tensors from forward method
        in_feats, weight, adj_ids = ctx.saved_tensors

        # Recompute intermediate features
        inter_feats = in_feats[adj_ids].flatten(1)

        # Get gradient tensors
        grad_inter_feats = torch.mm(grad_out_feats, weight)
        grad_weight = torch.mm(grad_out_feats.t(), inter_feats)
        grad_bias = grad_out_feats.sum(dim=0)

        in_channels = in_feats.size(dim=1)
        grad_inter_feats = grad_inter_feats.view(-1, in_channels)
        grad_in_feats = torch.zeros_like(in_feats).index_add_(0, adj_ids.flatten(), grad_inter_feats)
        grad_adj_ids = None

        return grad_in_feats, grad_weight, grad_bias, grad_adj_ids


class CustomClamp(Function):
    """
    Class implementing the CustomClamp autograd function.

    This custom autograd function alters the backward pass by computing gradients as if no clamping occurred in the
    forward pass.
    """

    @staticmethod
    def forward(ctx, in_tensor, min=None, max=None):
        """
        Forward method of the CustomClamp autograd function.

        Args:
            ctx (FunctionCtx): Context object storing additional data.
            in_tensor (FloatTensor): Input tensor of arbitrary shape.
            min (float): Minimum value of clamped tensor (default=None).
            max (float): Maximum value of clamped tensor (default=None).

        Returns:
            out_tensor (FloatTensor): Clamped output tensor of same shape as input tensor.
        """

        # Get clamped output tensor
        out_tensor = torch.clamp(in_tensor, min=min, max=max)

        return out_tensor

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_out_tensor):
        """
        Backward method of the CustomClamp autograd function.

        Args:
            ctx (FunctionCtx): Context object storing additional data.
            grad_out_tensor (FloatTensor): Gradient w.r.t. the output tensor.

        Returns:
            grad_in_tensor (FloatTensor): Gradient w.r.t. the input tensor.
            grad_min (None): None.
            grad_max (None): None.
        """

        # Get gradient tensors
        grad_in_tensor = grad_out_tensor
        grad_min = None
        grad_max = None

        return grad_in_tensor, grad_min, grad_max


class CustomDotProduct(Function):
    """
    Class implementing the CustomDotProduct autograd function.

    Dot product function between query and key features, where the queries and keys forming the query-key pairs are
    selected by the given query and key indices. If the query/key indices are missing, the given query/key features
    are used instead without selection.

    This custom dot product function does not keep the tensors with selected queries and keys in memory.
    """

    @staticmethod
    def forward(ctx, qry_feats, key_feats, qry_ids=None, key_ids=None):
        """
        Forward method of the CustomDotProduct autograd function.

        Args:
            ctx (FunctionCtx): Context object storing additional data.
            qry_feats (FloatTensor): Query features of shape [num_qry_feats, feat_size].
            key_feats (FloatTensor): Key features of shape [num_key_feats, feat_size].
            qry_ids (LongTensor): Indices selecting query pair features of shape [num_pairs] (default=None).
            key_ids (LongTensor): Indices selecting key pair features of shape [num_pairs] (default=None).

        Returns:
            dot_prods (FloatTensor): Query-key dot products of shape [num_pairs].
        """

        # Get pairs of query and key features
        qry_pair_feats = qry_feats[qry_ids] if qry_ids is not None else qry_feats
        key_pair_feats = key_feats[key_ids] if key_ids is not None else key_feats

        # Get query-key dot products
        dot_prods = (qry_pair_feats * key_pair_feats).sum(dim=1)

        # Save input tensors for backward pass
        ctx.save_for_backward(qry_feats, key_feats, qry_ids, key_ids)

        return dot_prods

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_dot_prods):
        """
        Backward method of the CustomDotProduct autograd function.

        Args:
            ctx (FunctionCtx): Context object storing additional data.
            grad_dot_prods (FloatTensor): Gradient w.r.t. the query-key dot products of shape [num_pairs].

        Returns:
            grad_qry_feats (FloatTensor): Gradient w.r.t. the query features of shape [num_qry_feats, feat_size].
            grad_key_feats (FloatTensor): Gradient w.r.t. the key features of shape [num_key_feats, feat_size].
            grad_qry_ids (None): None.
            grad_key_ids (None): None.
        """

        # Recover input tensors of forward method
        qry_feats, key_feats, qry_ids, key_ids = ctx.saved_tensors

        # Recompute pairs of query and key features
        qry_pair_feats = qry_feats[qry_ids] if qry_ids is not None else qry_feats
        key_pair_feats = key_feats[key_ids] if key_ids is not None else key_feats

        # Get gradient tensors
        grad_qry_pair_feats = grad_dot_prods[:, None] * key_pair_feats
        grad_key_pair_feats = grad_dot_prods[:, None] * qry_pair_feats

        if qry_ids is not None:
            grad_qry_feats = torch.zeros_like(qry_feats).index_add_(0, qry_ids, grad_qry_pair_feats)
        else:
            grad_qry_feats = grad_qry_pair_feats

        if key_ids is not None:
            grad_key_feats = torch.zeros_like(key_feats).index_add_(0, key_ids, grad_key_pair_feats)
        else:
            grad_key_feats = grad_key_pair_feats

        grad_qry_ids = None
        grad_key_ids = None

        return grad_qry_feats, grad_key_feats, grad_qry_ids, grad_key_ids


class CustomOnes(Function):
    """
    Class implementing the CustomOnes autograd function.

    This custom autograd function alters the backward pass by computing gradients as if the ones operation in the
    forward pass did not occur.
    """

    @staticmethod
    def forward(ctx, in_tensor):
        """
        Forward method of the CustomOnes autograd function.

        Args:
            ctx (FunctionCtx): Context object storing additional data.
            in_tensor (FloatTensor): Input tensor of arbitrary shape.

        Returns:
            out_tensor (FloatTensor): Output tensor filled with ones of same shape as input tensor.
        """

        # Get output tensor filled with ones
        out_tensor = torch.ones_like(in_tensor)

        return out_tensor

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_out_tensor):
        """
        Backward method of the CustomOnes autograd function.

        Args:
            ctx (FunctionCtx): Context object storing additional data.
            grad_out_tensor (FloatTensor): Gradient w.r.t. the output tensor.

        Returns:
            grad_in_tensor (FloatTensor): Gradient w.r.t. the input tensor.
        """

        # Get gradient tensor
        grad_in_tensor = grad_out_tensor

        return grad_in_tensor


class CustomReLU(Function):
    """
    Class implementing the CustomReLU autograd function.

    This custom autograd function alters the backward pass by only setting an input gradient to zero if the input is
    lower than the zero gradient threshold and higher input values would result in a higher loss.
    """

    @staticmethod
    def forward(ctx, in_tensor, zero_grad_thr=0.0):
        """
        Forward method of the CustomReLU autograd function.

        Args:
            ctx (FunctionCtx): Context object storing additional data.
            in_tensor (FloatTensor): Input tensor of arbitrary shape.
            zero_grad_thr (float): Value containing the zero gradient threshold (default=0.0).

        Returns:
            out_tensor (FloatTensor): Output tensor of same shape as input tensor.
        """

        # Get output tensor
        out_tensor = F.relu(in_tensor, inplace=False)

        # Save input tensor and zero gradient threshold for backward pass
        ctx.save_for_backward(in_tensor)
        ctx.zero_grad_thr = zero_grad_thr

        return out_tensor

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_out_tensor):
        """
        Backward method of the CustomReLU autograd function.

        Args:
            ctx (FunctionCtx): Context object storing additional data.
            grad_out_tensor (FloatTensor): Gradient w.r.t. the output tensor.

        Returns:
            grad_in_tensor (FloatTensor): Gradient w.r.t. the input tensor.
            grad_zero_grad_thr (None): None.
        """

        # Recover stored input tensor and zero gradient threshold
        in_tensor = ctx.saved_tensors[0]
        zero_grad_thr = ctx.zero_grad_thr

        # Get gradient tensors
        zero_mask = (in_tensor < zero_grad_thr) & (grad_out_tensor > 0)
        zero_tensor = torch.zeros_like(grad_out_tensor)

        grad_in_tensor = torch.where(zero_mask, zero_tensor, grad_out_tensor)
        grad_zero_grad_thr = None

        return grad_in_tensor, grad_zero_grad_thr


class CustomStep(Function):
    """
    Class implementing the CustomStep autograd function.

    This custom autograd function alters the backward pass by only setting an input gradient to zero if the input is
    lower than the left zero gradient threshold and higher input values would result in a higher loss, or if the input
    is higher than the right zero gradient threshold and lower input values would result in a higher loss.
    """

    @staticmethod
    def forward(ctx, in_tensor, left_zero_grad_thr=0.0, right_zero_grad_thr=0.0):
        """
        Forward method of the CustomStep autograd function.

        Args:
            ctx (FunctionCtx): Context object storing additional data.
            in_tensor (FloatTensor): Input tensor of arbitrary shape.
            left_zero_grad_thr (float): Value containing the left zero gradient threshold (default=0.0).
            right_zero_grad_thr (float): Value containing the right zero gradient threshold (default=0.0).

        Returns:
            out_tensor (FloatTensor): Output tensor of same shape as input tensor.
        """

        # Get output tensor
        out_tensor = torch.where(in_tensor >= 0, 1.0, 0.0)

        # Save input tensor and zero gradient thresholds for backward pass
        ctx.save_for_backward(in_tensor)
        ctx.left_zero_grad_thr = left_zero_grad_thr
        ctx.right_zero_grad_thr = right_zero_grad_thr

        return out_tensor

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_out_tensor):
        """
        Backward method of the CustomStep autograd function.

        Args:
            ctx (FunctionCtx): Context object storing additional data.
            grad_out_tensor (FloatTensor): Gradient w.r.t. the output tensor.

        Returns:
            grad_in_tensor (FloatTensor): Gradient w.r.t. the input tensor.
            grad_left_zero_grad_thr (None): None.
            grad_right_zero_grad_thr (None): None.
        """

        # Recover stored input tensor and zero gradient thresholds
        in_tensor = ctx.saved_tensors[0]
        left_zero_grad_thr = ctx.left_zero_grad_thr
        right_zero_grad_thr = ctx.right_zero_grad_thr

        # Get gradient tensors
        left_zero_mask = (in_tensor < left_zero_grad_thr) & (grad_out_tensor > 0)
        right_zero_mask = (in_tensor > right_zero_grad_thr) & (grad_out_tensor < 0)

        zero_mask = left_zero_mask | right_zero_mask
        zero_tensor = torch.zeros_like(grad_out_tensor)

        grad_in_tensor = torch.where(zero_mask, zero_tensor, grad_out_tensor)
        grad_left_zero_grad_thr = None
        grad_right_zero_grad_thr = None

        return grad_in_tensor, grad_left_zero_grad_thr, grad_right_zero_grad_thr


class NodeToEdgePyCustom(Function):
    """
    Class implementing the NodeToEdgePyCustom autograd function.

    This custom autograd function reconstructs the edge source and target features in the backward pass, avoiding the
    need to keep the potentially large edge source and target features in memory.
    """

    @staticmethod
    def forward(ctx, node_src_feats, node_tgt_feats, edge_ids, off_edge_src=None, off_edge_tgt=None, reduction='mul',
                num_groups=1):
        """
        Forward method of the NodeToEdgePyCustom autograd function.

        Args:
            ctx (FunctionCtx): Context object storing additional data.
            node_src_feats (FloatTensor): Tensor with the node source features of shape [num_nodes, feat_size].
            node_tgt_feats (FloatTensor): Tensor with the node target features of shape [num_nodes, feat_size].
            edge_ids (LongTensor): Tensor with the node indices for each (directed) edge of shape [2, num_edges].
            off_edge_src (FloatTensor): Offset edge source features of shape [num_edges, feat_size] (default=None).
            off_edge_tgt (FloatTensor): Offset edge target features of shape [num_edges, feat_size] (default=None).
            reduction (str): String containing the reduction operation (default='mul').
            num_groups (int): Integer containing the number of groups during 'mul-sum' reduction (default=1).

        Returns:
            edge_feats (FloatTensor): Tensor containing the edge features of shape [num_edges, out_feat_size].

        Raises:
            ValueError: Error when an invalid reduction string is provided.
        """

        # Compute edge features
        edge_src_feats = node_src_feats[edge_ids[0]]
        edge_tgt_feats = node_tgt_feats[edge_ids[1]]

        if off_edge_src is not None:
            edge_src_feats = edge_src_feats + off_edge_src

        if off_edge_tgt is not None:
            edge_tgt_feats = edge_tgt_feats + off_edge_tgt

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
        ctx.save_for_backward(node_src_feats, node_tgt_feats, edge_ids, off_edge_src, off_edge_tgt)
        ctx.reduction = reduction

        return edge_feats

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_edge_feats):
        """
        Backward method of the NodeToEdgePyCustom autograd function.

        Args:
            ctx (FunctionCtx): Context object storing additional data.
            grad_edge_feats (FloatTensor): Gradients of the edge features of shape [num_edges, out_feat_size].

        Returns:
            grad_node_src_feats (FloatTensor): Gradients of node source features of shape [num_nodes, feat_size].
            grad_node_tgt_feats (FloatTensor): Gradients of node target features of shape [num_nodes, feat_size].
            grad_edge_ids (None): None.
            grad_off_edge_src (FloatTensor): Gradient of offset edge source features of shape [num_edges, feat_size].
            grad_off_edge_tgt (FloatTensor): Gradient of offset edge target features of shape [num_edges, feat_size].
            grad_reduction (None): None.
            grad_num_groups (None): None.
        """

        # Recover stored tensors and reduction string from forward pass
        node_src_feats, node_tgt_feats, edge_ids, off_edge_src, off_edge_tgt = ctx.saved_tensors
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

        grad_off_edge_src = grad_edge_src_feats if off_edge_src is not None else None
        grad_off_edge_tgt = grad_edge_tgt_feats if off_edge_tgt is not None else None

        grad_node_src_feats = torch.zeros_like(node_src_feats).index_add_(0, edge_ids[0], grad_edge_src_feats)
        grad_node_tgt_feats = torch.zeros_like(node_tgt_feats).index_add_(0, edge_ids[1], grad_edge_tgt_feats)

        return grad_node_src_feats, grad_node_tgt_feats, None, grad_off_edge_src, grad_off_edge_tgt, None, None


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
