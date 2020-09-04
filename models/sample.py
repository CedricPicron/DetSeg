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
        seg_map = zeros(num_slots_total, 3, H, W)
        seg_map[masks[:, None, :, :]] = mask_fill
        seg_map = seg_map.view(num_slots_total, 3, H*W)

    # Initialize curiosity maps:
        xy_grid = stack(meshgrid(arange(-H+1, H), arange(-W+1, W)), dim=-1)
        gauss_pdf = scipy.stats.multivariate_normal([0, 0]).pdf
        gauss_grid = from_numpy(gauss_pdf(xy_grid)).to(torch.float32)

        xy_grid = xy_grid[-H:, -W:][None, :].expand(num_slots_total, -1, -1, -1)
        xy_grid = (xy_grid - pos_idx[:, None, None, :]).permute(3, 0, 1, 2)
        curio_map = gauss_grid[xy_grid[0], xy_grid[1]]
        curio_map[masks] = mask_fill

    # Save default grid for later:
        def_xy_grid = stack(meshgrid(arange(H), arange(W)), dim=-1)

    out: max_mask_entries -> int
    out: slots -> shape = [1, num_slots_total, feat_dim]
    out: batch_idx -> shape = [num_slots_total]
    out: seg_map -> shape = [num_slots_total, 3, H*W]
    out: gauss_grid -> shape = [2*H-1, 2*W-1]
    out: curio_map -> shape = [num_slots_total, H, W]
    out: def_xy_grid -> shape = [H, W, 2]

* Cross attention:
    in: features -> shape = [H*W, batch_size, feat_dim]
    in: slots -> shape = [1, num_slots_total, feat_dim]
    in: batch_idx -> shape = [num_slots_total]
    in: seg_map -> shape = [num_slots_total, 3, H*W]
    in: curio_map -> shape = [num_slots_total, H, W]

    in: max_mask_entries -> int
    in: samples_per_slot -> int
    in: cov_ratio -> float
    in: curio_weights -> shape = [3]
    in: memory_weight -> float

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
        sampled_curio_map = curio_map.view(num_slots_total, -1)[arange(num_slots_total), feat_idx]

        out: feat_idx -> shape = [samples_per_slot, num_slots_total]
        out: sampled_features -> shape = [samples_per_slot, num_slots_total, feat_dim]
        out: sampled_curio_map -> shape = [samples_per_slot, num_slots_total]

    ** HardWeightGate:
        - Forward:
            in: soft_value_weights -> shape = [samples_per_slot, num_slots_total]
            out: hard_value_weights = ones_like(soft_value_weights) -> shape = [samples_per_slot, num_slots_total]

        - Backward:
            in: grad_hard_value_weights -> shape = [samples_per_slot, num_slots_total]
            out: grad_soft_value_weights = grad_hard_value_weights -> shape = [samples_per_slot, num_slots_total]

    ** Attention:
        in: slots -> shape = [1, num_slots_total, feat_dim]
        in: sampled_features -> shape = [samples_per_slot, num_slots_total, feat_dim]
        in: sampled_curio_map -> shape = [samples_per_slot, num_slots_total]

        queries = slots -> shape = [1, num_slots_total, feat_dim]
        keys = sampled_features -> shape = [samples_per_slot, num_slots_total, feat_dim]

        soft_value_weights = softmax(sampled_curio_map, dim=0)
        hard_value_weights = HardWeightGate(soft_value_weights)

        values = hard_value_weights[:, :, None] * sampled_features
        slots = cross_mha(queries, keys, values, need_weights=False)

        out: slots -> shape = [1, num_slots_total, feat_dim]

    ** Segmentation map:
        in: slots -> shape = [1, num_slots_total, feat_dim]
        in: sampled_features -> shape = [samples_per_slot, num_slots_total, feat_dim]
        in: seg_map -> shape = [num_slots_total, 3, H*W]
        in: feat_idx -> shape = [samples_per_slot, num_slots_total]

        proj_slots = slot_proj(slots).expand_as(sampled_features)
        proj_feats = feat_proj(sampled_features)

        proj_slotfeats = cat([proj_slots, proj_feats], dim=-1)
        seg_probs = softmax(seg_classfier(proj_slotfeats), dim=-1) -> shape = [samples_per_slot, num_slots_total, 3]
        seg_map[arange(num_slots_total), :, feat_idx] = seg_probs

        out: seg_map -> shape = [num_slots_total, 3, H*W]
        out: seg_probs -> shape = [samples_per_slot, num_slots_total, 3]

    ** Curiosity map:
        in: feat_idx -> shape = [samples_per_slot, num_slots_total]
        in: seg_probs -> shape = [samples_per_slot, num_slots_total, 3]
        in: def_xy_grid -> shape = [H, W, 2]
        in: gauss_grid -> shape = [2*H-1, 2*W-1]
        in: curio_map -> shape = [num_slots_total, H, W]

        in: curio_weights -> shape = [3] (e.g. [1, 2, -1])
        in: memory_weight -> float

        samples_per_slot = feat_idx.shape[0]
        num_slots_total, H, W = curio_map.shape

        feat_idx = stack([feat_idx//W, feat_idx%W], dim=-1)
        xy_grid = def_xy_grid[:, :, None, None, :].expand(H, W, samples_per_slot, num_slots_total, 2)
        xy_grid = (xy_grid - feat_idx).permute(4, 0, 1, 2, 3)

        gauss_weights = sum(curio_weights*seg_probs, dim=-1) -> shape = [samples_per_slot, num_slots_total]
        gauss_pdfs = gauss_weights*gauss_grid[xy_grid[0], xy_grid[1]]
        curio_delta, _ = max(gauss_pdfs.permute(2, 3, 0, 1), dim=0)
        curio_map = memory_weight*curio_map + (1-memory_weight)*curio_delta

        feat_idx = feat_idx.permute(2, 0, 1)
        sampled_curiosities = 1-max(seg_probs, dim=-1)[0]
        curio_map[arange(num_slots_total), feat_idx[0], feat_idx[1]] = sampled_curiosities

        out: curio_map -> shape = [num_slots_total, H, W]

    out: slots -> shape = [1, num_slots_total, feat_dim]
    out: seg_map -> shape = [num_slots_total, 3, H*W]
    out: curio_map -> shape = [num_slots_total, H, W]

* Self-attention:
    in: slots -> shape = [1, num_slots_total, feat_dim]
    in: batch_idx -> shape = [num_slots_total]

    queries = keys = values = slots.transpose(0, 1) -> shape = [num_slots_total, 1, feat_dim]
    attn_mask = (batch_idx[:, None]-batch_idx[None, :]) != 0 -> shape = [num_slots_total, num_slots_total]
    slots = self_mha(queries, keys, values, need_weights=False, attn_mask=attn_mask)

    out: slots -> [num_slots_total, 1, feat_dim]

* Feedforward network (FFN):
    in: slots -> shape = [num_slots_total, 1, feat_dim]

    slots = ffn(slots).transpose(0, 1)

    out: slots -> shape = [1, num_slots_total, feat_dim]
"""
