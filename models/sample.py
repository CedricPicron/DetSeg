"""
* Initialization:
    in: input_masks -> shape = [batch_size, H, W]
    in: num_init_slots -> int
    in: mask_fill -> float

    batch_size, H, W = input_masks.shape
    num_slots_total = num_init_slots * batch_size
    max_mask_entries = max(sum(input_masks.view(batch_size, H*W), dim=1)).item()

    # Uniform sampling of initial slots:
        pos_idx = cat([randint(H, (num_slots_total,))[:, None], randint(W, (num_slots_total,))[:, None]], dim=1)
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

import copy

import torch
from torch import nn
from torch.autograd.function import Function
import scipy.stats


def get_pos_embedding(pos_idx, device=None, feat_dim=256):
    """
    Used for test purposes only.
    """
    num_init_slots = pos_idx.shape[0]
    pos_embedding = torch.randn(num_init_slots, feat_dim, device=device)

    return pos_embedding


class SampleDecoder(nn.Module):
    """
    Class implementing the SampleDecoder module.

    Attributes:
        layers (nn.ModuleList): List of decoder layers being concatenated.
        num_init_slots (int): Number of initial slots per image.
        num_layers (int): Number of concatenated decoder layers.
    """

    def __init__(self, decoder_layer, num_init_slots, num_layers):
        """
        Initializes the SamplerDecoder module.

        Args:
            decoder_layer (nn.Module): Decoder layer module to be concatenated.
            num_init_slots (int): Number of initial slots per image.
            num_layers (int): Number of concatenated decoder layers.
        """

        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        self.num_init_slots = num_init_slots
        self.num_layers = num_layers

    def forward_init(self, feature_masks, mask_fill=-1e6):
        """
        Initializes slots, segmentation/curiosity maps and other useful stuff.

        Args:
            feature_masks (BoolTensor): Feature masks of shape [HxW, batch_size, feat_dim].

        Returns:
            slots (FloatTensor): Initial slots of shape [1, num_slots_total, feat_dim].
            batch_idx (IntTensor): Batch indices corresponding to the slots of shape [num_slots_total].
            seg_maps (FloatTensor): Initial segmentation maps of shape [num_slots_total, 3, H*W].
            curio_maps (FloatTensor): Initial curiosity maps of shape [num_slots_total, H, W].

            def_xy_grid (IntTensor): Tensor containing the (H,W)-grid coordinates of shape [H, W, 2].
            gauss_grid (FloatTensor): Tensor containing the Gaussian grid of shape [2*H-1, 2*W-1].
            max_mask_entries (int): Maximum masked entries of mask from feature_masks.
        """

        batch_size, H, W = feature_masks.shape
        num_slots_total = self.num_init_slots * batch_size
        device = feature_masks.device

        # Uniform sampling of initial slots
        pos_idx = torch.stack([torch.randint(H, (num_slots_total,)), torch.randint(W, (num_slots_total,))], dim=1)
        slots = get_pos_embedding(pos_idx, device=device)

        # Initialize segmentation maps
        batch_idx = torch.repeat_interleave(torch.arange(batch_size), num_init_slots)
        masks = feature_masks[batch_idx]
        seg_maps = torch.zeros(num_slots_total, 3, H, W, device=device)
        seg_maps[masks[:, None, :, :].expand(-1, 3, -1, -1)] = mask_fill
        seg_maps = seg_maps.view(num_slots_total, 3, H*W)

        # Initialize curiosity maps
        xy_grid = torch.stack(torch.meshgrid(torch.arange(-H+1, H), torch.arange(-W+1, W)), dim=-1)
        gauss_pdf = scipy.stats.multivariate_normal([0, 0]).pdf
        gauss_grid = torch.from_numpy(gauss_pdf(xy_grid)).to(device=device, dtype=torch.float32)

        xy_grid = xy_grid[-H:, -W:][None, :].expand(num_slots_total, -1, -1, -1)
        xy_grid = (xy_grid - pos_idx[:, None, None, :]).permute(3, 0, 1, 2)
        curio_maps = gauss_grid[xy_grid[0], xy_grid[1]]
        curio_maps[masks] = mask_fill

        # Compute some useful items for later
        def_xy_grid = torch.stack(torch.meshgrid(torch.arange(H), torch.arange(W)), dim=-1)
        max_mask_entries = torch.max(torch.sum(feature_masks.view(batch_size, H*W), dim=1)).item()

        return slots, batch_idx, seg_maps, curio_maps, def_xy_grid, gauss_grid, max_mask_entries

    def forward(self, features, feature_masks):
        """
        Forward method of the SampleDecoder module.

        Args:
            features: Feature maps of shape [H*W, batch_size, feat_dim].
            feature_masks (BoolTensor): Feature masks of shape [HxW, batch_size, feat_dim].

        Returns:
            slots (FloatTensor): Object slots of shape [1, num_slots_total, feat_dim].
            batch_idx (IntTensor): Batch indices corresponding to the slots of shape [num_slots_total].
            seg_maps (FloatTensor): Segmentation maps of shape [num_slots_total, 3, H*W].
        """

        # Initialize slots, segmentation/curiosity maps and other useful stuff
        slots, batch_idx, seg_maps, curio_maps, *args = self.forward_init(feature_masks)

        # Loop over different decoder layers
        for layer in self.layers:
            slots, seg_maps, curio_maps = layer(features, slots, batch_idx, seg_maps, curio_maps, *args)

        return slots, batch_idx, seg_maps


class SampleDecoderLayer(nn.Module):
    """
    Class implementing the SampleDecoderLayer module.

    Attributes:
        cross_attention (nn.Module): The cross-attention module.
        self_attention (nn.Module): The self-attention module.
        ffn (nn.Module): The FFN (feedforward network) module.
    """

    def __init__(self, cross_attention, self_attention, ffn):
        """
        Initializes the SampleDecoderLayer module.

        Args:
            cross_attention (nn.Module): The cross-attention module.
            self_attention (nn.Module): The self-attention module.
            ffn (nn.Module): The FFN (feedforward network) module.
        """

        super().__init_()
        self.cross_attention = cross_attention
        self.self_attention = self_attention
        self.ffn = ffn

    def forward(self, features, slots, batch_idx, seg_maps, curio_maps, *args):
        """
        Forward method of the SampleDecoderLayer module.

        Args:
            features: Feature maps of shape [H*W, batch_size, feat_dim].
            slots (FloatTensor): Object slots of shape [1, num_slots_total, feat_dim].
            batch_idx (IntTensor): Batch indices corresponding to the slots of shape [num_slots_total].
            seg_maps (FloatTensor): Segmentation maps of shape [num_slots_total, 3, H*W].
            curio_maps (FloatTensor): Curiosity maps of shape [num_slots_total, H, W].

            args (Tuple): Tuple containing following fixed items:
                def_xy_grid (IntTensor): Tensor containing the (H,W)-grid coordinates of shape [H, W, 2].
                gauss_grid (FloatTensor): Tensor containing the Gaussian grid of shape [2*H-1, 2*W-1].
                max_mask_entries (int): Maximum masked entries of mask from feature_masks.

        Returns:
            slots (FloatTensor): Object slots of shape [1, num_slots_total, feat_dim].
            seg_maps (FloatTensor): Segmentation maps of shape [num_slots_total, 3, H*W].
            curio_maps (FloatTensor): Curiosity maps of shape [num_slots_total, H, W].
        """

        slots, seg_maps, curio_maps = self.cross_attention(features, slots, batch_idx, seg_maps, curio_maps, *args)
        slots = self.self_attention(slots, batch_idx)
        slots = self.ffn(slots)

        return slots, seg_maps, curio_maps


class HardWeightGate(Function):
    @staticmethod
    def forward(ctx, soft_weights):
        return torch.ones_like(soft_weights)

    @staticmethod
    def backward(ctx, grad_hard_weights):
        return grad_hard_weights


class WeightedCrossAttention(nn.Module):
    """
    Class implementing the WeightedCrossAttention module.
    """

    def __init__(self, coverage_ratio, samples_per_slot, feat_dim, num_heads, dropout=0.1, hard_weights=True):
        super().__init__()

        self.coverage_ratio = coverage_ratio
        self.samples_per_slot = samples_per_slot

        self.mha = nn.MultiheadAttention(feat_dim, num_heads, dropout=dropout)
        self.hard_weights = hard_weights

    def sample(self, batch_idx, curio_maps, max_mask_entries):
        cov_samples = int(self.coverage_ratio*self.samples_per_slot)
        imp_samples = self.samples_per_slot - cov_samples

        # Get sample positions based importance sampling
        num_slots_total, H, W = curio_maps.shape
        _, sorted_idx = torch.sort(curio_maps.view(num_slots_total, H*W), dim=1, descending=True)
        imp_idx = sorted_idx[:, :imp_samples]

        # Get sample positions based on coverage
        cov_idx = sorted_idx[:, imp_samples:-max_mask_entries]
        cov_idx = cov_idx[:, torch.random.sample(range(cov_idx.shape[1]), k=cov_samples)]

        # Get sampled features and curiosities
        feat_idx = torch.cat([imp_idx, cov_idx], dim=1).t()
        sampled_features = features[feat_idx, batch_idx, :]
        sampled_curiosities = curio_maps.view(num_slots_total, -1)[torch.arange(num_slots_total), feat_idx]

        return feat_idx, sampled_features, sampled_curiosities

    def weighted_attention(self, slots, sampled_features, sampled_curiosities):
        queries = slots
        keys = sampled_features

        value_weights = torch.softmax(sampled_curiosities, dim=0)
        if self.hard_weights:
            value_weights = HardWeightGate(value_weights)

        values = value_weights[:, :, None] * sampled_features
        slots = self.mha(queries, keys, values, need_weights=False)

        return slots

    def update_seg_maps(self):
        return

    def update_curio_maps(self):
        return

    def forward(self, features, slots, batch_idx, seg_maps, curio_maps, def_xy_grid, gauss_grid, max_mask_entries):
        feat_idx, sampled_features, sampled_curiosities = self.sample(batch_idx, curio_maps, max_mask_entries)
        slots = self.weighted_attention(slots, sampled_features, sampled_curiosities)

        return slots, seg_maps, curio_maps


class SelfAttention(nn.Module):
    """
    Class implementing the SelfAttention module.
    """

    def __init__(self):
        super().__init__()


class FFN(nn.Module):
    """
    Class implementing the FFN module.
    """

    def __init__(self):
        super().__init__()


if __name__ == "__main__":
    import time

    H = W = 32
    batch_size = 5
    feat_dim = 256
    num_init_slots = 7
    num_layers = 1

    device = torch.device("cuda")
    features = torch.randn(H*W, batch_size, feat_dim, device=device, requires_grad=True)
    feature_masks = torch.randn(batch_size, H, W, device=device) > -1.0
    sample_decoder = SampleDecoder(num_init_slots, num_layers)

    start_time = time.time()
    sample_decoder(features, feature_masks)
    init_time = time.time()

    print(f"Initialization time: {(init_time-start_time)*1e3: .1f} ms")

    print(f"Memory usage (after intialization): {torch.cuda.max_memory_allocated()/(1024*1024): .0f} MB")
