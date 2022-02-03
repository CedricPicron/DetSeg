"""
Collection of functions related to sparse data structures.
"""

from torch_scatter import scatter

from models.functional.autograd import SparseDenseMmPyCustom


def sparse_dense_mm(sparse_ids, sparse_vals, sparse_size, dense_matrix, implementation='pytorch-custom'):
    """
    Function performing sparse-dense matrix multiplication.

    Args:
        sparse_ids (LongTensor): Indices of the sparse input matrix of shape [2, num_elems].
        sparse_vals (FloatTensor): Values of the sparse input matrix of shape [num_elems] or [num_elems, num_heads].
        sparse_size (Tuple): Tuple containing the size of the sparse input matrix [M, N].
        dense_matrix (FloatTensor): Dense input matrix of shape [N, P].
        implementation (str): String containing the type of implementation (default='pytorch-custom').

    Returns:
        out_matrix (FloatTensor): Dense output matrix of shape [M, P].

    Raises:
        ValueError: Error when number of columns of sparse matrix and number of rows of dense matrix do not match.
        ValueError: Error when the number of heads does not divide the number of columns of the dense matrix.
        ValueError: Error when an invalid implementation string is provided.
    """

    # Get and check sizes of input matrices
    if sparse_vals.dim() == 1:
        sparse_vals = sparse_vals[:, None]

    num_elems, num_heads = sparse_vals.size()
    M, N1 = sparse_size
    N2, P = dense_matrix.size()

    if N1 != N2:
        error_msg = f"Number of columns of sparse matrix ({N1}) does not match number of rows of dense matrix ({N2})."
        raise ValueError(error_msg)

    if P % num_heads != 0:
        error_msg = f"The number of heads ({num_heads}) must divide the number of columns of the dense matrix ({P})."
        raise ValueError(error_msg)

    # Perform sparse-dense matrix multiplication
    if implementation == 'pytorch-custom':
        out_matrix = SparseDenseMmPyCustom.apply(sparse_ids, sparse_vals, sparse_size, dense_matrix)

    elif implementation == 'pytorch-naive':
        inter_matrix = dense_matrix[sparse_ids[1]].view(num_elems, num_heads, -1)
        inter_matrix = sparse_vals[:, :, None] * inter_matrix
        inter_matrix = inter_matrix.view(num_elems, -1)
        out_matrix = scatter(inter_matrix, sparse_ids[0], dim=0, dim_size=M, reduce='sum')

    else:
        error_msg = f"Invalid implementation string for the sparse_dense_mm function (got '{implementation}')."
        raise ValueError(error_msg)

    return out_matrix
