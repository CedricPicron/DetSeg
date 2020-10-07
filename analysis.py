"""
Analysis of trained sample decoder
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, RandomSampler

from datasets.build import build_dataset
from main import get_parser
from models.detr import build_detr
from utils.data import val_collate_fn


# Analysis hook function
def analysis_hook(module, input, output):
    sample_id = module.sample_id
    layer_id = module.layer_id
    image_dir = Path(f"./analysis/{sample_id}/{layer_id}")
    image_dir.mkdir(parents=True, exist_ok=True)

    _, seg_maps, curio_maps = output
    num_slots, H, W = curio_maps.shape
    seg_maps = seg_maps.view(num_slots, 3, H, W)
    seg_maps = seg_maps.permute(0, 2, 3, 1)

    for slot_id in range(1, num_slots):
        seg_map = seg_maps[slot_id].cpu().numpy()
        curio_map = curio_maps[slot_id].cpu().numpy()

        plt.imsave(image_dir / f"{slot_id}a.eps", seg_map)
        plt.imsave(image_dir / f"{slot_id}b.eps", curio_map)


# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', default='', type=str, help='checkpoint path with sample decoder to be analyzed')
parser.add_argument('--num_samples', default=10, type=int, help='number of random validation samples for analysis')
parser.add_argument('--add_args', default=[], nargs=argparse.REMAINDER, help='additional args for main.py argparser')
analysis_args = parser.parse_args()
main_args = get_parser().parse_args(args=analysis_args.add_args)

# Use batch size one for simplicty
main_args.batch_size = 1
main_args.num_workers = 1

# Get validation dataset and sampler
_, val_dataset, _ = build_dataset(main_args)
sampler = RandomSampler(val_dataset)

# Get dataloader
dataloader_kwargs = {'collate_fn': val_collate_fn, 'num_workers': main_args.num_workers, 'pin_memory': True}
dataloader = DataLoader(val_dataset, main_args.batch_size, sampler=sampler, **dataloader_kwargs)

# Get model with default parameters and put it on correct device
device = torch.device(main_args.device)
model = build_detr(main_args).to(device)

# Load model parameters from checkpoint
checkpoint = torch.load(Path(analysis_args.checkpoint), map_location='cpu')
model.load_state_dict(checkpoint['model'])

# Set model in evaluation mode
model.eval()

# Register analysis hooks and add layer ids
for layer_id, layer in enumerate(model.decoder.layers, 1):
    layer.register_forward_hook(analysis_hook)
    layer.layer_id = layer_id

# Perform analysis on random validation samples
with torch.no_grad():
    for sample_id, (images, tgt_dict, eval_dict) in enumerate(dataloader, 1):
        [setattr(layer, 'sample_id', sample_id) for layer in model.decoder.layers]
        model(images.to(device))

        if sample_id == analysis_args.num_samples:
            break
