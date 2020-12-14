"""
Profiling script.
"""

import argparse

from detectron2.data import MetadataCatalog
import torch
import torch.autograd.profiler as profiler
from torch.utils.benchmark import Timer

from main import get_parser
from models.backbone import build_backbone
from models.bicore import build_bicore
from models.bivinet import build_bivinet
from models.criterion import build_criterion
from models.decoder import build_decoder
from models.detr import build_detr
from models.encoder import build_encoder
from models.heads.detection import build_det_heads
from models.heads.segmentation import build_seg_heads
from utils.data import nested_tensor_from_image_list


# Lists of model and sort choices
model_choices = ['backbone', 'bicore', 'bin_seg_head', 'bivinet_bin_seg', 'bivinet_ret', 'bivinet_sem_seg']
model_choices = [*model_choices, 'criterion', 'detr', 'encoder', 'global_decoder', 'ret_head', 'sample_decoder']
model_choices = [*model_choices, 'sem_seg_head']
sort_choices = ['cpu_time', 'cuda_time', 'cuda_memory_usage', 'self_cuda_memory_usage']

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--forward', action='store_true', help='whether to profile forward pass only')
parser.add_argument('--main_args', nargs='*', help='comma-separted string of main arguments')
parser.add_argument('--memory', action='store_true', help='whether to report memory usage')
parser.add_argument('--model', default='sample_decoder', choices=model_choices, help='model type to be profiled')
parser.add_argument('--sort_by', default='cuda_time', choices=sort_choices, help='metric to sort profiler table')
profiling_args = parser.parse_args()
profiling_args.main_args = profiling_args.main_args[0].split(',')[1:] if profiling_args.main_args is not None else []
main_args = get_parser().parse_args(args=[*profiling_args.main_args])

# Building the model to be profiled
if profiling_args.model == 'backbone':
    main_args.lr_backbone = 1e-5
    model = build_backbone(main_args).to('cuda')

    images = torch.randn(2, 3, 1024, 1024)
    images = nested_tensor_from_image_list(images).to('cuda')
    inputs = [images]

    globals_dict = {'model': model, 'inputs': inputs}
    forward_stmt = "model(*inputs)"
    backward_stmt = "model(*inputs)[0][-1].sum().backward()"

elif profiling_args.model == 'bicore':
    main_args.min_resolution_id = 3
    model = build_bicore(main_args).to('cuda')

    feat_map0 = torch.randn(2, 1024, 1024, 8).to('cuda')
    feat_map1 = torch.randn(2, 512, 512, 16).to('cuda')
    feat_map2 = torch.randn(2, 256, 256, 32).to('cuda')
    feat_map3 = torch.randn(2, 128, 128, 64).to('cuda')
    feat_map4 = torch.randn(2, 64, 64, 128).to('cuda')
    feat_map5 = torch.randn(2, 32, 32, 256).to('cuda')
    feat_map6 = torch.randn(2, 16, 16, 512).to('cuda')
    feat_maps = [feat_map3, feat_map4, feat_map5, feat_map6]
    inputs = [feat_maps]

    globals_dict = {'model': model, 'inputs': inputs}
    forward_stmt = "model(*inputs)"
    backward_stmt = "torch.cat([map.sum()[None] for map in model(*inputs)]).sum().backward()"

elif profiling_args.model == 'bin_seg_head':
    main_args.min_resolution_id = 3
    main_args.seg_heads = ['binary']
    main_args.disputed_loss = True
    model = build_seg_heads(main_args)[0].to('cuda')

    feat_map0 = torch.randn(2, 1024, 1024, 8).to('cuda')
    feat_map1 = torch.randn(2, 512, 512, 16).to('cuda')
    feat_map2 = torch.randn(2, 256, 256, 32).to('cuda')
    feat_map3 = torch.randn(2, 128, 128, 64).to('cuda')
    feat_map4 = torch.randn(2, 64, 64, 128).to('cuda')
    feat_map5 = torch.randn(2, 32, 32, 256).to('cuda')
    feat_map6 = torch.randn(2, 16, 16, 512).to('cuda')
    feat_maps = [feat_map3, feat_map4, feat_map5, feat_map6]

    tgt_map0 = torch.randn(2, 1024, 1024).to('cuda').clamp(-0.5, 0.5) + 0.5
    tgt_map1 = torch.randn(2, 512, 512).to('cuda').clamp(-0.5, 0.5) + 0.5
    tgt_map2 = torch.randn(2, 256, 256).to('cuda').clamp(-0.5, 0.5) + 0.5
    tgt_map3 = torch.randn(2, 128, 128).to('cuda').clamp(-0.5, 0.5) + 0.5
    tgt_map4 = torch.randn(2, 64, 64).to('cuda').clamp(-0.5, 0.5) + 0.5
    tgt_map5 = torch.randn(2, 32, 32).to('cuda').clamp(-0.5, 0.5) + 0.5
    tgt_map6 = torch.randn(2, 16, 16).to('cuda').clamp(-0.5, 0.5) + 0.5
    tgt_maps = [tgt_map3, tgt_map4, tgt_map5, tgt_map6]
    tgt_dict = {'binary_maps': tgt_maps}

    inputs = {'feat_maps': feat_maps, 'tgt_dict': tgt_dict}
    globals_dict = {'model': model, 'inputs': inputs}
    forward_stmt = "model(**inputs)"
    backward_stmt = "model(**inputs)[0]['bin_seg_loss'].backward()"

elif profiling_args.model == 'bivinet_bin_seg':
    main_args.meta_arch = 'BiViNet'
    main_args.min_resolution_id = 3
    main_args.num_core_layers = 4
    main_args.seg_heads = ['binary']
    main_args.disputed_loss = True
    model = build_bivinet(main_args).to('cuda')

    images = torch.randn(2, 3, 1024, 1024)
    images = nested_tensor_from_image_list(images).to('cuda')

    num_targets_total = 20
    sizes = torch.tensor([0, num_targets_total//2, num_targets_total]).to('cuda')
    masks = (torch.randn(num_targets_total, 1024, 1024) > 1.0).to('cuda')
    tgt_dict = {'sizes': sizes, 'masks': masks}

    optimizer = optimizer = torch.optim.AdamW(model.parameters())
    globals_dict = {'model': model, 'images': images, 'tgt_dict': tgt_dict, 'optimizer': optimizer}
    forward_stmt = "model(*[images, tgt_dict.copy()])"
    backward_stmt = "model(*[images, tgt_dict.copy(), optimizer])"

elif profiling_args.model == 'bivinet_ret':
    main_args.det_heads = ['retina']
    main_args.meta_arch = 'BiViNet'
    main_args.min_resolution_id = 3
    main_args.num_core_layers = 4
    main_args.num_classes = 91
    model = build_bivinet(main_args).to('cuda')

    images = torch.randn(2, 3, 1024, 1024)
    images = nested_tensor_from_image_list(images).to('cuda')

    num_targets_total = 20
    labels = torch.randint(main_args.num_classes, (num_targets_total,), device='cuda')
    boxes = torch.abs(torch.randn(num_targets_total, 4, device='cuda'))
    sizes = torch.tensor([0, num_targets_total//2, num_targets_total]).to('cuda')
    tgt_dict = {'labels': labels, 'boxes': boxes, 'sizes': sizes}

    optimizer = optimizer = torch.optim.AdamW(model.parameters())
    globals_dict = {'model': model, 'images': images, 'tgt_dict': tgt_dict, 'optimizer': optimizer}
    forward_stmt = "model(*[images, tgt_dict.copy()])"
    backward_stmt = "model(*[images, tgt_dict.copy(), optimizer])"

elif profiling_args.model == 'bivinet_sem_seg':
    main_args.meta_arch = 'BiViNet'
    main_args.min_resolution_id = 3
    main_args.num_core_layers = 4
    main_args.num_classes = 91
    main_args.seg_heads = ['semantic']
    main_args.val_metadata = MetadataCatalog.get('coco_2017_val')
    model = build_bivinet(main_args).to('cuda')

    images = torch.randn(2, 3, 1024, 1024)
    images = nested_tensor_from_image_list(images).to('cuda')

    num_targets_total = 20
    labels = torch.randint(main_args.num_classes, size=(num_targets_total,)).to('cuda')
    sizes = torch.tensor([0, num_targets_total//2, num_targets_total]).to('cuda')
    masks = (torch.randn(num_targets_total, 1024, 1024) > 1.0).to('cuda')
    tgt_dict = {'labels': labels, 'sizes': sizes, 'masks': masks}

    optimizer = optimizer = torch.optim.AdamW(model.parameters())
    globals_dict = {'model': model, 'images': images, 'tgt_dict': tgt_dict, 'optimizer': optimizer}
    forward_stmt = "model(*[images, tgt_dict.copy()])"
    backward_stmt = "model(*[images, tgt_dict.copy(), optimizer])"

elif profiling_args.model == 'criterion':
    main_args.num_classes = 91
    criterion = build_criterion(main_args).to('cuda')

    def generate_pred_list():
        num_slots_total = main_args.batch_size * main_args.num_init_slots
        logits = torch.randn(num_slots_total, main_args.num_classes+1, device='cuda', requires_grad=True)
        boxes = torch.abs(torch.randn(num_slots_total, 4, device='cuda', requires_grad=True))
        sizes = torch.tensor([i*main_args.num_init_slots for i in range(main_args.batch_size+1)], device='cuda')
        pred_list = [{'logits': logits, 'boxes': boxes, 'sizes': sizes, 'layer_id': 6, 'iter_id': 1}]

        return pred_list

    num_targets_total = 20
    labels = torch.randint(main_args.num_classes, (num_targets_total,), device='cuda')
    boxes = torch.abs(torch.randn(num_targets_total, 4, device='cuda'))
    sizes = torch.tensor([0, num_targets_total//2, num_targets_total]).to('cuda')
    tgt_dict = {'labels': labels, 'boxes': boxes, 'sizes': sizes}

    globals_dict = {'criterion': criterion, 'generate_pred_list': generate_pred_list, 'tgt_dict': tgt_dict}
    forward_stmt = "criterion(generate_pred_list(), tgt_dict)"
    backward_stmt = "torch.stack([v for v in criterion(generate_pred_list(), tgt_dict)[0].values()]).sum().backward()"

elif profiling_args.model == 'detr':
    main_args.meta_arch = 'DETR'
    main_args.lr_backbone = 1e-5
    main_args.lr_encoder = 1e-4
    main_args.lr_decoder = 1e-4
    main_args.num_classes = 91
    model = build_detr(main_args).to('cuda')

    images = torch.randn(2, 3, 800, 800)
    images = nested_tensor_from_image_list(images).to('cuda')

    num_targets_total = 20
    labels = torch.randint(main_args.num_classes, (num_targets_total,), device='cuda')
    boxes = torch.abs(torch.randn(num_targets_total, 4, device='cuda'))
    sizes = torch.tensor([0, num_targets_total//2, num_targets_total]).to('cuda')
    tgt_dict = {'labels': labels, 'boxes': boxes, 'sizes': sizes}

    optimizer = torch.optim.AdamW(model.parameters())
    globals_dict = {'model': model, 'images': images, 'tgt_dict': tgt_dict, 'optimizer': optimizer}
    forward_stmt = "model(*[images])"
    backward_stmt = "model(*[images, tgt_dict.copy(), optimizer])"

elif profiling_args.model == 'encoder':
    main_args.lr_encoder = 1e-4
    main_args.num_encoder_layers = 6
    model = build_encoder(main_args).to('cuda')

    features = torch.randn(1024, 2, 256).to('cuda')
    feature_masks = (torch.randn(2, 32, 32) > 0).to('cuda')
    pos_encodings = torch.randn(1024, 2, 256).to('cuda')

    inputs = [features, feature_masks, pos_encodings]
    globals_dict = {'model': model, 'inputs': inputs}

    forward_stmt = "model(*inputs)"
    backward_stmt = "model(*inputs).sum().backward()"

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

    forward_stmt = "model(*inputs)"
    backward_stmt = "model(*inputs)['slots'].sum().backward()"

elif profiling_args.model == 'ret_head':
    main_args.det_heads = ['retina']
    main_args.min_resolution_id = 3
    main_args.num_classes = 91
    model = build_det_heads(main_args)[0].to('cuda')

    feat_map0 = torch.randn(2, 1024, 1024, 8).to('cuda')
    feat_map1 = torch.randn(2, 512, 512, 16).to('cuda')
    feat_map2 = torch.randn(2, 256, 256, 32).to('cuda')
    feat_map3 = torch.randn(2, 128, 128, 64).to('cuda')
    feat_map4 = torch.randn(2, 64, 64, 128).to('cuda')
    feat_map5 = torch.randn(2, 32, 32, 256).to('cuda')
    feat_map6 = torch.randn(2, 16, 16, 512).to('cuda')
    feat_maps = [feat_map3, feat_map4, feat_map5, feat_map6]

    num_anchors_total = sum(9 * 4**(10-i) for i in range(main_args.min_resolution_id, main_args.max_resolution_id+1))
    anchor_labels = torch.randint(model.num_classes, size=(2, num_anchors_total)).to('cuda')
    anchor_deltas = torch.abs(torch.randn(2, num_anchors_total, 4)).to('cuda')
    tgt_dict = {'anchor_labels': anchor_labels, 'anchor_deltas': anchor_deltas}

    inputs = {'feat_maps': feat_maps, 'tgt_dict': tgt_dict}
    globals_dict = {'model': model, 'inputs': inputs}
    forward_stmt = "model(**inputs)"
    backward_stmt = "sum(v[None] for v in model(**inputs)[0].values()).backward()"

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

    forward_stmt = "model(*inputs)"
    backward_stmt = "model(*inputs)['slots'].sum().backward()"

elif profiling_args.model == 'sem_seg_head':
    main_args.min_resolution_id = 3
    main_args.num_classes = 91
    main_args.seg_heads = ['semantic']
    main_args.val_metadata = MetadataCatalog.get('coco_2017_val')
    model = build_seg_heads(main_args)[0].to('cuda')

    feat_map0 = torch.randn(2, 1024, 1024, 8).to('cuda')
    feat_map1 = torch.randn(2, 512, 512, 16).to('cuda')
    feat_map2 = torch.randn(2, 256, 256, 32).to('cuda')
    feat_map3 = torch.randn(2, 128, 128, 64).to('cuda')
    feat_map4 = torch.randn(2, 64, 64, 128).to('cuda')
    feat_map5 = torch.randn(2, 32, 32, 256).to('cuda')
    feat_map6 = torch.randn(2, 16, 16, 512).to('cuda')
    feat_maps = [feat_map3, feat_map4, feat_map5, feat_map6]

    tgt_map0 = torch.randint(model.num_classes+1, size=(2, 1024, 1024)).to('cuda')
    tgt_map1 = torch.randint(model.num_classes+1, size=(2, 512, 512)).to('cuda')
    tgt_map2 = torch.randint(model.num_classes+1, size=(2, 256, 256)).to('cuda')
    tgt_map3 = torch.randint(model.num_classes+1, size=(2, 128, 128)).to('cuda')
    tgt_map4 = torch.randint(model.num_classes+1, size=(2, 64, 64)).to('cuda')
    tgt_map5 = torch.randint(model.num_classes+1, size=(2, 32, 32)).to('cuda')
    tgt_map6 = torch.randint(model.num_classes+1, size=(2, 16, 16)).to('cuda')
    tgt_maps = [tgt_map3, tgt_map4, tgt_map5, tgt_map6]
    tgt_dict = {'semantic_maps': tgt_maps}

    inputs = {'feat_maps': feat_maps, 'tgt_dict': tgt_dict}
    globals_dict = {'model': model, 'inputs': inputs}
    forward_stmt = "model(**inputs)"
    backward_stmt = "model(**inputs)[0]['sem_seg_loss'].backward()"

# Select forward or backward statement
stmt = forward_stmt if profiling_args.forward else backward_stmt

# Do not build computation graph with forward statement
if profiling_args.forward:
    for parameter in model.parameters():
        parameter.requires_grad = False

# Warm-up and timing with torch.utils.benchmark.Timer
timer = Timer(stmt=stmt, globals=globals_dict)
t = timer.timeit(number=10).median*1e3

# Profiling with torch.autograd.profiler.profile(use_cuda=True)
with profiler.profile(use_cuda=True, profile_memory=profiling_args.memory) as prof:
    exec(stmt)

# Print profiling table and median timeit time
print(prof.table(sort_by=profiling_args.sort_by, row_limit=100))
print(f"Recorded timeit time: {t: .4f} ms")

# Print max GPU memory usage
print(f"Max GPU memory usage: {torch.cuda.max_memory_allocated()/(1024*1024): .0f} MB")
