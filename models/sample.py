"""
* Initialization:
    slots = uniform_sample(H, W) -> shape = [num_init_slots, batch_size, feat_dim]
    slots = slots.view(1, num_init_slots*batch_size, feat_dim)

    out: slots -> shape = [1, num_slots_total, feat_dim]

* Cross attention:
    in: features -> shape = [H*W, batch_size, feat_dim]
    in: slots -> shape = [1, num_slots_total, feat_dim]
    in: batch_idx -> shape = [num_slots_total]
    in: seg_map -> shape = [num_slots_total, H, W]
    in: curio_map -> shape = [num_slots_total, H, W]

    ** Sample features:
        1) Over-generate: uniform sampling over (H, W) used for all batch images
        2) Importance sampling: use per slot curiosity map (one sample can be used by mutiple slots)
        3) Per slot coverage

    ** Attention:
        queries = slots -> shape = [1, num_slots_total, feat_dim]
        keys = sampled_features -> shape = [num_feat_samples, num_slots_total, feat_dim]
        values = sampled_features -> shape = [num_feat_samples, num_slots_total, feat_dim]

        Softmax only over own sampled features:
            + Efficient (if implementation possible)
            + Allows simple per slot spatial confidence based on attention
            + Number of negatives can be tuned by coverage amount
            - No slot competition (competition not ideal as multiple slots might attend at object boundaries)

    ** Segmentation map

    ** Curiosity map

    out: slots -> shape = [1, num_slots_total, feat_dim]
    out: seg_map -> shape = [num_slots_total, H, W]
    out: curio_map -> shape = [num_slots_total, H, W]

* Self-attention:
    in: slots, shape = [1, num_slots_total, feat_dim]
    in: batch_idx, shape = [num_slots_total]

    slots = slots.transpose(0, 1) -> shape = [num_slots_total, 1, feat_dim]
    attn_mask = get_mask(batch_idx) -> shape = [num_slots_total, num_slots_total]

    queries = keys = values = slots
    slots = MHA(queries, keys, values, attn_mask=attn_mask)
    out: slots -> [num_slots_total, 1, feat_dim]

* Feedforward network (FFN):
    in: slots -> [num_slots_total, 1, feat_dim]
    slots = FFN(slots).transpose(0, 1)
    out: slots -> [1, num_slots_total, feat_dim]
"""
