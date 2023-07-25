"""
Collection of sparsity-based functions.
"""

from models.functional.autograd import IdAttn


def id_attn(in_act_feats, pas_feats, feat_ids, feat_weights, weight, bias=None):
    """
    Function implementing the index-based attention operation.

    This custom implementation does not keep intermediate data structures in memory.

    Args:
        in_act_feats (FloatTensor): Input active features of shape [num_act_feats, feat_size].
        pas_feats (FloatTensor): Passive features of shape [num_pas_feats, feat_size].
        feat_ids (LongTensor): Feature indices of shape [num_act_feats, num_pts[, num_heads]].
        feat_weights (FloatTensor): Feature weights of shape [num_act_feats, num_pts, num_heads].
        weight (FloatTensor): Value projection weights of shape [feat_size, feat_size].
        bias (FloatTensor): Value projection biases of shape [feat_size] (default=None).

    Returns:
        out_act_feats (FloatTensor): Output active features of shape [num_act_feats, feat_size].
    """

    # Apply custom IdAttn autograd function
    out_act_feats = IdAttn.apply(in_act_feats, pas_feats, feat_ids, feat_weights, weight, bias)

    return out_act_feats
