"""
Script testing the sampler functions.
"""

import torch

from functional import naive_maps_sampler_2d, naive_maps_sampler_3d


# Test naive 2D sampler
print('\n-------------------')
print('|     Naive 2D    |')
print('-------------------\n')

batch_size = 1
feat_size = 1
init_size = 2
num_maps = 1
num_samples = 1

feat_map_wh = torch.tensor([[init_size//2**i, init_size//2**i] for i in range(num_maps)], device='cuda')
feat_map_offs = torch.prod(feat_map_wh, dim=1).cumsum(dim=0)
num_feats = feat_map_offs[-1]
feat_map_offs = torch.cat([feat_map_offs.new_zeros((1,)), feat_map_offs[:-1]], dim=0)

sample_xy = torch.rand(batch_size, num_samples, 2, device='cuda')
sample_map_ids = torch.randint(high=num_maps, size=(batch_size, num_samples), device='cuda')

# Test top left
feats = torch.tensor([1, 0, 0, 0], dtype=torch.float, device='cuda').view(1, num_feats, 1)
pred = naive_maps_sampler_2d(feats, feat_map_wh, feat_map_offs, sample_xy, sample_map_ids).item()
tgt = ((1-sample_xy[0, 0, 0])*(1-sample_xy[0, 0, 1])).item()
print(f'Error top left: {abs(pred - tgt)}')

# Test top right
feats = torch.tensor([0, 1, 0, 0], dtype=torch.float, device='cuda').view(1, num_feats, 1)
pred = naive_maps_sampler_2d(feats, feat_map_wh, feat_map_offs, sample_xy, sample_map_ids).item()
tgt = (sample_xy[0, 0, 0]*(1-sample_xy[0, 0, 1])).item()
print(f'Error top right: {abs(pred - tgt)}')

# Test bottom left
feats = torch.tensor([0, 0, 1, 0], dtype=torch.float, device='cuda').view(1, num_feats, 1)
pred = naive_maps_sampler_2d(feats, feat_map_wh, feat_map_offs, sample_xy, sample_map_ids).item()
tgt = ((1-sample_xy[0, 0, 0])*sample_xy[0, 0, 1]).item()
print(f'Error bottom left: {abs(pred - tgt)}')

# Test bottom right
feats = torch.tensor([0, 0, 0, 1], dtype=torch.float, device='cuda').view(1, num_feats, 1)
pred = naive_maps_sampler_2d(feats, feat_map_wh, feat_map_offs, sample_xy, sample_map_ids).item()
tgt = (sample_xy[0, 0, 0]*sample_xy[0, 0, 1]).item()
print(f'Error bottom right: {abs(pred - tgt)}')

# Test naive 3D sampler
print('\n-------------------')
print('|     Naive 3D    |')
print('-------------------\n')

batch_size = 1
feat_size = 1
init_size = 2
num_maps = 2
num_samples = 1

feat_map_wh = torch.tensor([[init_size, init_size] for _ in range(num_maps)], device='cuda')
feat_map_offs = torch.prod(feat_map_wh, dim=1).cumsum(dim=0)
num_feats = feat_map_offs[-1]
feat_map_offs = torch.cat([feat_map_offs.new_zeros((1,)), feat_map_offs[:-1]], dim=0)

sample_xy = torch.rand(batch_size, num_samples, 3, device='cuda')
sample_map_ids = torch.randint(high=num_maps, size=(batch_size, num_samples), device='cuda')

# Test front top left
feats = torch.tensor([1, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float, device='cuda').view(1, num_feats, 1)
pred = naive_maps_sampler_3d(feats, feat_map_wh, feat_map_offs, sample_xy).item()
tgt = ((1-sample_xy[0, 0, 0])*(1-sample_xy[0, 0, 1])*(1-sample_xy[0, 0, 2])).item()
print(f'Error front top left: {abs(pred - tgt)}')

# Test front top right
feats = torch.tensor([0, 1, 0, 0, 0, 0, 0, 0], dtype=torch.float, device='cuda').view(1, num_feats, 1)
pred = naive_maps_sampler_3d(feats, feat_map_wh, feat_map_offs, sample_xy).item()
tgt = (sample_xy[0, 0, 0]*(1-sample_xy[0, 0, 1])*(1-sample_xy[0, 0, 2])).item()
print(f'Error front top right: {abs(pred - tgt)}')

# Test front bottom left
feats = torch.tensor([0, 0, 1, 0, 0, 0, 0, 0], dtype=torch.float, device='cuda').view(1, num_feats, 1)
pred = naive_maps_sampler_3d(feats, feat_map_wh, feat_map_offs, sample_xy).item()
tgt = ((1-sample_xy[0, 0, 0])*sample_xy[0, 0, 1]*(1-sample_xy[0, 0, 2])).item()
print(f'Error front bottom left: {abs(pred - tgt)}')

# Test front bottom right
feats = torch.tensor([0, 0, 0, 1, 0, 0, 0, 0], dtype=torch.float, device='cuda').view(1, num_feats, 1)
pred = naive_maps_sampler_3d(feats, feat_map_wh, feat_map_offs, sample_xy).item()
tgt = (sample_xy[0, 0, 0]*sample_xy[0, 0, 1]*(1-sample_xy[0, 0, 2])).item()
print(f'Error front bottom right: {abs(pred - tgt)}')

# Test back top left
feats = torch.tensor([0, 0, 0, 0, 1, 0, 0, 0], dtype=torch.float, device='cuda').view(1, num_feats, 1)
pred = naive_maps_sampler_3d(feats, feat_map_wh, feat_map_offs, sample_xy).item()
tgt = ((1-sample_xy[0, 0, 0])*(1-sample_xy[0, 0, 1])*sample_xy[0, 0, 2]).item()
print(f'Error back top left: {abs(pred - tgt)}')

# Test back top right
feats = torch.tensor([0, 0, 0, 0, 0, 1, 0, 0], dtype=torch.float, device='cuda').view(1, num_feats, 1)
pred = naive_maps_sampler_3d(feats, feat_map_wh, feat_map_offs, sample_xy).item()
tgt = (sample_xy[0, 0, 0]*(1-sample_xy[0, 0, 1])*sample_xy[0, 0, 2]).item()
print(f'Error back top right: {abs(pred - tgt)}')

# Test back bottom left
feats = torch.tensor([0, 0, 0, 0, 0, 0, 1, 0], dtype=torch.float, device='cuda').view(1, num_feats, 1)
pred = naive_maps_sampler_3d(feats, feat_map_wh, feat_map_offs, sample_xy).item()
tgt = ((1-sample_xy[0, 0, 0])*sample_xy[0, 0, 1]*sample_xy[0, 0, 2]).item()
print(f'Error back bottom left: {abs(pred - tgt)}')

# Test back bottom right
feats = torch.tensor([0, 0, 0, 0, 0, 0, 0, 1], dtype=torch.float, device='cuda').view(1, num_feats, 1)
pred = naive_maps_sampler_3d(feats, feat_map_wh, feat_map_offs, sample_xy).item()
tgt = (sample_xy[0, 0, 0]*sample_xy[0, 0, 1]*sample_xy[0, 0, 2]).item()
print(f'Error back bottom right: {abs(pred - tgt)}\n')
