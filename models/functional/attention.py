"""
Collection of functions related to attention.
"""

from models.functional.autograd import IdDeformAttn2d


def id_deform_attn2d(in_core_feats, aux_feats, sample_ids, sample_weights, weight, bias):
    """
    Function implementing the 2D index-based deformable attention operation.

    This custom implementation does not keep intermediate data structures in memory.

    Args:
        in_core_feats (FloatTensor): Input core features of shape [num_core_feats, in_size].
        aux_feats (FloatTensor): Auxiliary features of shape [num_aux_feats, in_size].
        sample_ids (LongTensor): Sample indices of shape [num_core_feats, num_heads, num_points, 4].
        sample_weights (FloatTensor): Sample weights of shape [num_core_feats, num_heads, num_points, 4].
        weight (FloatTensor): Weight parameters of shape [val_size, in_size].
        bias (FloatTensor): Bias parameters of shape [val_size].

    Returns:
        val_core_feats (FloatTensor): Value core features of shape [num_core_feats, val_size].
    """

    # Apply custom IdDeformAttn2d autograd function
    val_core_feats = IdDeformAttn2d.apply(in_core_feats, aux_feats, sample_ids, sample_weights, weight, bias)

    return val_core_feats
