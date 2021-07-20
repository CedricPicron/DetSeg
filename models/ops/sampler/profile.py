"""
Script profiling the sampler functions.
"""

import torch
import torch.autograd.profiler as profiler

from functional import naive_maps_sampler_2d, naive_maps_sampler_3d


# Profile 2D naive maps sampler
batch_size = 16
feat_size = 32
init_size = 100
num_maps = 5
num_samples = 6000

feat_map_wh = torch.tensor([[init_size//2**i, init_size//2**i] for i in range(num_maps)], device='cuda')
feat_map_offs = torch.prod(feat_map_wh, dim=1).cumsum(dim=0)
num_feats = feat_map_offs[-1]
feat_map_offs = torch.cat([feat_map_offs.new_zeros((1,)), feat_map_offs[:-1]], dim=0)

feats = torch.randn(batch_size, num_feats, feat_size, device='cuda')
sample_xy = torch.rand(batch_size, num_samples, 2, device='cuda')
sample_map_ids = torch.randint(high=num_maps, size=(batch_size, num_samples), device='cuda')

with profiler.profile(use_cuda=True) as prof:
    function = naive_maps_sampler_2d
    exec("function(feats, feat_map_wh, feat_map_offs, sample_xy, sample_map_ids)")

print(prof.table(sort_by='cuda_time', row_limit=100))
print(f"Max GPU memory: {torch.cuda.max_memory_allocated()/(1024**2): .2f} MB")

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

# Profile 3D naive maps sampler
batch_size = 16
feat_size = 32
init_size = 100
num_maps = 5
num_samples = 6000

feat_map_wh = torch.tensor([[init_size//2**i, init_size//2**i] for i in range(num_maps)], device='cuda')
feat_map_offs = torch.prod(feat_map_wh, dim=1).cumsum(dim=0)
num_feats = feat_map_offs[-1]
feat_map_offs = torch.cat([feat_map_offs.new_zeros((1,)), feat_map_offs[:-1]], dim=0)

feats = torch.randn(batch_size, num_feats, feat_size, device='cuda')
sample_xy = torch.rand(batch_size, num_samples, 3, device='cuda')

with profiler.profile(use_cuda=True) as prof:
    function = naive_maps_sampler_3d
    exec("function(feats, feat_map_wh, feat_map_offs, sample_xy)")

print(prof.table(sort_by='cuda_time', row_limit=100))
print(f"Max GPU memory: {torch.cuda.max_memory_allocated()/(1024**2): .2f} MB")
