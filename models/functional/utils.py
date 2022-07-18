"""
Collection of utility functions.
"""

from models.functional.autograd import CustomClamp, CustomDotProduct, CustomOnes


def custom_clamp(in_tensor, min=None, max=None):
    """
    Custom clamp function altering the backward pass by computing gradients as if no clamping occurred in the forward
    pass.

    Args:
        in_tensor (FloatTensor): Input tensor of arbitrary shape.
        min (float): Minimum value of clamped tensor (default=None).
        max (float): Maximum value of clamped tensor (default=None).

    Returns:
        out_tensor (FloatTensor): Clamped output tensor of same shape as input tensor.
    """

    # Apply custom clamp function
    out_tensor = CustomClamp.apply(in_tensor, min, max)

    return out_tensor


def custom_dot_product(qry_feats, key_feats, qry_ids=None, key_ids=None):
    """
    Dot product function between query and key features, where the queries and keys forming the query-key pairs are
    selected by the given query and key indices. If the query/key indices are missing, the given query/key features
    are used instead without selection.

    This custom dot product function does not keep the tensors with selected queries and keys in memory.

    Args:
        qry_feats (FloatTensor): Query features of shape [num_qry_feats, feat_size].
        key_feats (FloatTensor): Key features of shape [num_key_feats, feat_size].
        qry_ids (LongTensor): Indices selecting query features for query-key pairs of shape [num_pairs] (default=None).
        key_ids (LongTensor): Indices selecting key features for query-key pairs of shape [num_pairs] (default=None).

    Returns:
        dot_prods (FloatTensor): Query-key dot products of shape [num_pairs].
    """

    # Apply custom dot product function
    dot_prods = CustomDotProduct.apply(qry_feats, key_feats, qry_ids, key_ids)

    return dot_prods


def custom_ones(in_tensor):
    """
    Custom ones function altering the backward pass by computing gradients as if the ones operation in the forward pass
    did not occur.

    Args:
        in_tensor (FloatTensor): Input tensor of arbitrary shape.

    Returns:
        out_tensor (FloatTensor): Output tensor filled with ones of same shape as input tensor.
    """

    # Apply custom ones function
    out_tensor = CustomOnes.apply(in_tensor)

    return out_tensor
