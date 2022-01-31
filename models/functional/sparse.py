"""
Collection of functions related to sparse data structures.
"""

from torch_scatter import scatter

from models.functional.autograd import SparseDenseMmPyCustom


def sparse_dense_mm(sparse_ids, sparse_vals, sparse_size, dense_matrix, implementation='pytorch-custom'):
    """
    Function performing sparse-dense matrix multiplication.

    Args:
        sparse_ids (LongTensor): Tensor containing the indices of the sparse input matrix of shape [2, num_elems].
        sparse_vals (FloatTensor): Tensor containing the values of the sparse input matrix of shape [num_elems].
        sparse_size (Tuple): Tuple containing the size of the sparse input matrix [M, N].
        dense_matrix (FloatTensor): Dense input matrix of shape [N, P].
        implementation (str): String containing the type of implementation (default='pytorch-custom').

    Returns:
        out_matrix (FloatTensor): Dense output matrix of shape [M, P].

    Raises:
        ValueError: Error when number of columns of sparse matrix and number of rows of dense matrix do not match.
        ValueError: Error when an invalid implementation string is provided.
    """

    # Get and check sizes of input matrices
    M, N1 = sparse_size
    N2 = dense_matrix.size(dim=0)

    if N1 != N2:
        error_msg = f"Number of columns of sparse matrix ({N1}) does not match number of rows of dense matrix {(N2)}."
        raise ValueError(error_msg)

    # Perform sparse-dense matrix multiplication
    if implementation == 'pytorch-custom':
        out_matrix = SparseDenseMmPyCustom.apply(sparse_ids, sparse_vals, sparse_size, dense_matrix)

    elif implementation == 'pytorch-naive':
        out_matrix = sparse_vals[:, None] * dense_matrix[sparse_ids[1]]
        out_matrix = scatter(out_matrix, sparse_ids[0], dim=0, dim_size=M, reduce='add')

    else:
        error_msg = f"Invalid implementation string for the sparse_dense_mm function (got '{implementation}')."
        raise ValueError(error_msg)

    return out_matrix
