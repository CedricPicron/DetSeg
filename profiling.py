import argparse

import torch
import torch.autograd.profiler as profiler
from torch.utils._benchmark import Timer

from main import get_parser
from models.backbone import build_backbone
from models.decoder import build_decoder
from models.detr import build_detr
from models.encoder import build_encoder
from utils.data import nested_tensor_from_tensor_list


# Lists of model and sort choices
model_choices = ['backbone', 'detr', 'encoder', 'global_decoder', 'sample_decoder']
sort_choices = ['cpu_time', 'cuda_time', 'cuda_memory_usage', 'self_cuda_memory_usage']

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--memory', action='store_true', help='whether to report memory usage')
parser.add_argument('--model', default='sample_decoder', choices=model_choices, help='model type to be profiled')
parser.add_argument('--sort_by', default='cuda_time', choices=sort_choices, help='metric to sort profiler table')
profiling_args = parser.parse_args()
main_args = get_parser().parse_args(args=[])

# Building the model to be profiled
if profiling_args.model == 'backbone':
    images = torch.randn(2, 3, 1024, 1024)
    images = nested_tensor_from_tensor_list(images).to('cuda')
    inputs = [images]

    main_args.lr_backbone = 1e-5
    model = build_backbone(main_args).to('cuda')

elif profiling_args.model == 'detr':
    images = torch.randn(2, 3, 1024, 1024)
    images = nested_tensor_from_tensor_list(images).to('cuda')
    inputs = [images]

    main_args.lr_backbone = 1e-5
    main_args.lr_decoder = 1e-4
    main_args.lr_encoder = 1e-4
    main_args.num_classes = 91
    main_args.num_encoder_layers = 1
    main_args.num_decoder_layers = 1
    model = build_detr(main_args).to('cuda')

elif profiling_args.model == 'encoder':
    features = torch.randn(1024, 2, 256).to('cuda')
    feature_masks = (torch.randn(2, 32, 32) > 0).to('cuda')
    pos_encodings = torch.randn(1024, 2, 256).to('cuda')
    inputs = [features, feature_masks, pos_encodings]

    main_args.lr_encoder = 1e-4
    main_args.num_encoder_layers = 1
    model = build_encoder(main_args).to('cuda')

elif profiling_args.model == 'global_decoder':
    features = torch.randn(1024, 2, 256).to('cuda')
    feature_masks = (torch.randn(2, 32, 32) > 0).to('cuda')
    pos_encodings = torch.randn(1024, 2, 256).to('cuda')
    inputs = [features, feature_masks, pos_encodings]

    main_args.decoder_type = 'global'
    main_args.lr_decoder = 1e-4
    main_args.num_decoder_layers = 1
    model = build_decoder(main_args).to('cuda')

elif profiling_args.model == 'sample_decoder':
    features = torch.randn(1024, 2, 256).to('cuda')
    feature_masks = (torch.randn(2, 32, 32) > 0).to('cuda')
    pos_encodings = torch.randn(1024, 2, 256).to('cuda')
    inputs = [features, feature_masks, pos_encodings]

    main_args.decoder_type = 'sample'
    main_args.lr_decoder = 1e-4
    main_args.num_decoder_layers = 1
    model = build_decoder(main_args).to('cuda')

# Warm-up and timing with torch.utils._benchmark.Timer
globals_dict = {'inputs': inputs, 'model': model}
timer = Timer(stmt='model(*inputs)', globals=globals_dict)
t = timer.timeit(number=10)._median*1e3

# Profiling with torch.autograd.profiler.profile(use_cuda=True)
with profiler.profile(use_cuda=True, profile_memory=profiling_args.memory) as prof:
    model(*inputs)

# Print profiling table and median timeit time
print(prof.table(sort_by=profiling_args.sort_by, row_limit=100))
print(f"Recorded timeit time: {t: .4f} ms")
