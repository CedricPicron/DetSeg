"""
* Initialization:
    in: input_masks -> shape = [batch_size, H, W]
    in: num_init_slots -> int
    in: mask_fill -> float

    batch_size, H, W = input_masks.shape
    num_slots_total = num_init_slots * batch_size
    max_mask_entries = max(sum(input_masks.view(batch_size, H*W), dim=1)).item()

    # Uniform sampling of initial slots:
        pos_idx = cat([randint(H, (num_slots_total,))[:, None], randint(H, (num_slots_total,))[:, None]], dim=1)
        slots = PosEmbedding(pos_idx) -> shape = [num_slots_total, feat_dim]

    # Initialize segmentation maps:
        batch_idx = repeat_interleave(arange(batch_size), num_init_slots)
        masks = input_masks[batch_idx]
        seg_map = zeros(num_slots_total, H, W)
        seg_map[masks] = mask_fill

    # Initialize curiosity maps:
        xy_grid = stack(meshgrid(arange(-H+1, H), arange(-W+1, W)), dim=-1)
        gauss_pdf = scipy.stats.multivariate_normal([0, 0]).pdf
        gauss_grid = from_numpy(gauss_pdf(xy_grid))

        xy_grid = xy_grid[-H:, -W:][None, :].expand(num_slots_total, -1, -1, -1)
        xy_grid = (xy_grid - pos_idx[:, None, None, :]).permute(3, 0, 1, 2)
        curio_map = gauss_grid[xy_grid[0], xy_grid[1]]
        curio_map[masks] = mask_fill

    out: slots -> shape = [1, num_slots_total, feat_dim]
    out: batch_idx -> shape = [num_slots_total]
    out: seg_map -> shape = [num_slots_total, H, W]
    out: curio_map -> shape = [num_slots_total, H, W]
    out: max_mask_entries -> int

* Cross attention:
    in: features -> shape = [H*W, batch_size, feat_dim]
    in: slots -> shape = [1, num_slots_total, feat_dim]
    in: batch_idx -> shape = [num_slots_total]
    in: seg_map -> shape = [num_slots_total, H, W]
    in: curio_map -> shape = [num_slots_total, H, W]

    in: max_mask_entries -> int
    in: samples_per_slot -> int
    in: cov_ratio -> float

    ** Sample features (at integer positions):
        in: features -> shape = [H*W, batch_size, feat_dim]
        in: batch_idx -> shape = [num_slots_total]
        in: curio_map -> shape = [num_slots_total, H, W]
        in: samples_per_slot -> int
        in: cov_ratio -> float

        cov_samples = int(cov_ratio*samples_per_slot)
        imp_samples = samples_per_slot - cov_samples

        # Importance sampling:
            num_slots_total, H, W = curio_map.shape
            _, sorted_idx = sort(curio_map.view(num_slots_total, H*W), dim=1, descending=True)
            imp_idx = sorted_idx[:, :imp_samples]

        # Coverage:
            cov_idx = sorted_idx[:, imp_samples:-max_mask_entries]
            cov_idx = cov_idx[:, random.sample(range(cov_idx.shape[1]), k=cov_samples)]

        feat_idx = cat([imp_idx, cov_idx], dim=1).t() -> shape = [samples_per_slot, num_slots_total]
        sampled_features = features[feat_idx, batch_idx, :]

        out: sampled_features -> shape = [samples_per_slot, num_slots_total, feat_dim]

    ** Attention:
        in: slots -> shape = [1, num_slots_total, feat_dim]
        in: sampled_features -> shape = [samples_per_slot, num_slots_total, feat_dim]

        queries = slots -> shape = [1, num_slots_total, feat_dim]
        keys = values = sampled_features -> shape = [samples_per_slot, num_slots_total, feat_dim]
        slots = cross_mha(queries, keys, values, need_weights=False)

        out: slots -> shape = [1, num_slots_total, feat_dim]

    ** Segmentation map:

    ** Curiosity map:

    out: slots -> shape = [1, num_slots_total, feat_dim]
    out: seg_map -> shape = [num_slots_total, H, W]
    out: curio_map -> shape = [num_slots_total, H, W]

* Self-attention:
    in: slots, shape = [1, num_slots_total, feat_dim]
    in: batch_idx, shape = [num_slots_total]

    queries = keys = values = slots.transpose(0, 1) -> shape = [num_slots_total, 1, feat_dim]
    attn_mask = get_mask(batch_idx) -> shape = [num_slots_total, num_slots_total]
    slots = self_mha(queries, keys, values, need_weights=False, attn_mask=attn_mask)

    out: slots -> [num_slots_total, 1, feat_dim]

* Feedforward network (FFN):
    in: slots -> shape = [num_slots_total, 1, feat_dim]

    slots = ffn(slots).transpose(0, 1)

    out: slots -> shape = [1, num_slots_total, feat_dim]
"""
