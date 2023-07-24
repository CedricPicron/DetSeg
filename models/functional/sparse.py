"""
Collection of sparsity-based functions.
"""

from models.functional.autograd import IdScaleAttn


def id_scale_attn(in_act_feats, pas_feats, feat_ids, feat_weights, weight, bias=None):
    """
    Function implementing the index-based scale attention operation.

    This custom implementation does not keep intermediate data structures in memory.

    Args:
        in_act_feats (FloatTensor): Input active features of shape [num_act_feats, feat_size].
        pas_feats (FloatTensor): Passive features of shape [num_pas_feats, feat_size].
        feat_ids (LongTensor): Feature indices of shape [num_act_feats, num_maps, 4].
        feat_weights (FloatTensor): Feature weights of shape [num_act_feats, num_heads, num_maps, 4].
        weight (FloatTensor): Value projection weights of shape [feat_size, feat_size].
        bias (FloatTensor): Value projection biases of shape [feat_size] (default=None).

    Returns:
        out_act_feats (FloatTensor): Output active features of shape [num_act_feats, feat_size].
    """

    # Apply custom IdScaleAttn autograd function
    out_act_feats = IdScaleAttn.apply(in_act_feats, pas_feats, feat_ids, feat_weights, weight, bias)

    return out_act_feats
