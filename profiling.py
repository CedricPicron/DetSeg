import argparse

import torch
import torch.autograd.profiler as profiler
from torch.utils._benchmark import Timer

from main import get_parser
from models.backbone import build_backbone
from models.criterion import build_criterion
from models.decoder import build_decoder
from models.detr import build_detr
from models.encoder import build_encoder
from utils.data import nested_tensor_from_tensor_list


# Lists of model and sort choices
model_choices = ['backbone', 'criterion', 'detr', 'detr_criterion', 'encoder', 'global_decoder', 'sample_decoder']
sort_choices = ['cpu_time', 'cuda_time', 'cuda_memory_usage', 'self_cuda_memory_usage']

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--forward', action='store_true', help='whether to profile forward pass only')
parser.add_argument('--memory', action='store_true', help='whether to report memory usage')
parser.add_argument('--model', default='sample_decoder', choices=model_choices, help='model type to be profiled')
parser.add_argument('--sort_by', default='cuda_time', choices=sort_choices, help='metric to sort profiler table')
profiling_args = parser.parse_args()
main_args = get_parser().parse_args(args=[])

# Building the model to be profiled
if profiling_args.model == 'backbone':
    main_args.lr_backbone = 1e-5
    model = build_backbone(main_args).to('cuda')

    images = torch.randn(2, 3, 1024, 1024)
    images = nested_tensor_from_tensor_list(images).to('cuda')

    inputs = [images]
    globals_dict = {'model': model, 'inputs': inputs}

    forward_stmt = 'model(*inputs)'
    backward_stmt = 'model(*inputs)[-1].tensors.sum().backward()'

elif profiling_args.model == 'criterion':
    main_args.num_classes = 91
    criterion = build_criterion(main_args).to('cuda')

    def generate_pred_list():
        num_slots_total = main_args.batch_size * main_args.num_init_slots
        logits = torch.randn(num_slots_total, main_args.num_classes+1, device='cuda', requires_grad=True)
        boxes = torch.abs(torch.randn(num_slots_total, 4, device='cuda', requires_grad=True))
        batch_idx, _ = torch.randint(main_args.batch_size, (num_slots_total,), device='cuda').sort()
        pred_list = [{'logits': logits, 'boxes': boxes, 'batch_idx': batch_idx, 'layer_id': 0}]

        return pred_list

    num_target_boxes_total = 20
    labels = torch.randint(main_args.num_classes, (num_target_boxes_total,), device='cuda')
    boxes = torch.abs(torch.randn(num_target_boxes_total, 4, device='cuda'))
    sizes, _ = torch.randint(num_target_boxes_total, (main_args.batch_size+1,), device='cuda').sort()
    tgt_dict = {'labels': labels, 'boxes': boxes, 'sizes': sizes}

    globals_dict = {'criterion': criterion, 'generate_pred_list': generate_pred_list, 'tgt_dict': tgt_dict}
    forward_stmt = 'criterion(generate_pred_list(), tgt_dict)'
    backward_stmt = 'torch.stack([v for v in criterion(generate_pred_list(), tgt_dict)[0].values()]).sum().backward()'

elif profiling_args.model == 'detr':
    main_args.lr_backbone = 1e-5
    main_args.lr_decoder = 1e-4
    main_args.lr_encoder = 1e-4
    main_args.num_classes = 91

    detr = build_detr(main_args).to('cuda')
    images = torch.randn(2, 3, 1024, 1024)
    images = nested_tensor_from_tensor_list(images).to('cuda')

    keys = ['logits', 'boxes']
    globals_dict = {'detr': detr, 'images': images, 'keys': keys}

    forward_stmt = 'detr(images)'
    backward_stmt = 'torch.cat([v for k, v in detr(images)[0].items() if k in keys], dim=1).sum().backward()'

elif profiling_args.model == 'detr_criterion':
    main_args.lr_backbone = 1e-5
    main_args.lr_decoder = 1e-4
    main_args.lr_encoder = 1e-4
    main_args.num_classes = 91

    detr = build_detr(main_args).to('cuda')
    criterion = build_criterion(main_args).to('cuda')

    images = torch.randn(2, 3, 1024, 1024)
    images = nested_tensor_from_tensor_list(images).to('cuda')

    num_target_boxes_total = 20
    labels = torch.randint(main_args.num_classes, (num_target_boxes_total,), device='cuda')
    boxes = torch.abs(torch.randn(num_target_boxes_total, 4, device='cuda'))
    sizes, _ = torch.randint(num_target_boxes_total, (main_args.batch_size+1,), device='cuda').sort()
    tgt_dict = {'labels': labels, 'boxes': boxes, 'sizes': sizes}

    globals_dict = {'criterion': criterion, 'detr': detr, 'images': images, 'tgt_dict': tgt_dict}
    forward_stmt = 'criterion(detr(images), tgt_dict)'
    backward_stmt = 'torch.stack([v for v in criterion(detr(images), tgt_dict)[0].values()]).sum().backward()'

elif profiling_args.model == 'encoder':
    main_args.lr_encoder = 1e-4
    main_args.num_encoder_layers = 6
    model = build_encoder(main_args).to('cuda')

    features = torch.randn(1024, 2, 256).to('cuda')
    feature_masks = (torch.randn(2, 32, 32) > 0).to('cuda')
    pos_encodings = torch.randn(1024, 2, 256).to('cuda')

    inputs = [features, feature_masks, pos_encodings]
    globals_dict = {'model': model, 'inputs': inputs}

    forward_stmt = 'model(*inputs)'
    backward_stmt = 'model(*inputs).sum().backward()'

elif profiling_args.model == 'global_decoder':
    main_args.decoder_type = 'global'
    main_args.lr_decoder = 1e-4
    main_args.num_decoder_layers = 6
    model = build_decoder(main_args).to('cuda')

    features = torch.randn(1024, 2, 256).to('cuda')
    feature_masks = (torch.randn(2, 32, 32) > 0).to('cuda')
    pos_encodings = torch.randn(1024, 2, 256).to('cuda')

    inputs = [features, feature_masks, pos_encodings]
    globals_dict = {'model': model, 'inputs': inputs}

    forward_stmt = 'model(*inputs)'
    backward_stmt = 'model(*inputs)[0].sum().backward()'

elif profiling_args.model == 'sample_decoder':
    main_args.decoder_type = 'sample'
    main_args.lr_decoder = 1e-4
    main_args.num_decoder_layers = 6
    model = build_decoder(main_args).to('cuda')

    features = torch.randn(1024, 2, 256).to('cuda')
    feature_masks = (torch.randn(2, 32, 32) > 0).to('cuda')
    pos_encodings = torch.randn(1024, 2, 256).to('cuda')

    inputs = [features, feature_masks, pos_encodings]
    globals_dict = {'model': model, 'inputs': inputs}

    forward_stmt = 'model(*inputs)'
    backward_stmt = 'model(*inputs)[0].sum().backward()'

# Select forward or backward statement
stmt = forward_stmt if profiling_args.forward else backward_stmt

# Warm-up and timing with torch.utils._benchmark.Timer
timer = Timer(stmt=stmt, globals=globals_dict)
t = timer.timeit(number=10)._median*1e3

# Profiling with torch.autograd.profiler.profile(use_cuda=True)
with profiler.profile(use_cuda=True, profile_memory=profiling_args.memory) as prof:
    exec(stmt)

# Print profiling table and median timeit time
print(prof.table(sort_by=profiling_args.sort_by, row_limit=100))
print(f"Recorded timeit time: {t: .4f} ms")
