"""
Profiling script.
"""

import argparse

from detectron2.data import MetadataCatalog
from detectron2.modeling.anchor_generator import DefaultAnchorGenerator
import torch
import torch.autograd.profiler as profiler
from torch.utils.benchmark import Timer

from main import get_parser
from models.backbone import build_backbone
from models.bivinet import build_bivinet
from models.cores.bla import build_bla
from models.cores.fpn import build_fpn
from models.cores.gc import build_gc
from models.criterion import build_criterion
from models.decoder import build_decoder
from models.detr import build_detr
from models.encoder import build_encoder
from models.heads.detection.build import build_det_heads
from models.heads.segmentation.build import build_seg_heads
from structures.boxes import Boxes
from structures.images import Images


# Lists of model and sort choices
model_choices = ['backbone', 'bla_init', 'bla_update', 'bin_seg_head', 'bivinet_bin_seg', 'bivinet_det']
model_choices = [*model_choices, 'bivinet_sem_seg', 'brd_head', 'criterion', 'detr', 'dod_pred', 'dod_self']
model_choices = [*model_choices, 'dod_train', 'dfd_head', 'encoder', 'fpn', 'gc', 'global_decoder', 'ret_head_bla']
model_choices = [*model_choices, 'ret_head_fpn', 'sample_decoder', 'sem_seg_head']
sort_choices = ['cpu_time', 'cuda_time', 'cuda_memory_usage', 'self_cuda_memory_usage']

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--forward', action='store_true', help='whether to profile forward pass only')
parser.add_argument('--inference', action='store_true', help='whether to profile model in inference mode')
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

    images = Images(torch.randn(2, 3, 1024, 1024)).to('cuda')
    inputs = [images]

    globals_dict = {'model': model, 'inputs': inputs}
    forward_stmt = "model(*inputs)"
    backward_stmt = "model(*inputs)[0][-1].sum().backward()"

elif profiling_args.model == 'bla_init':
    main_args.backbone_feat_sizes = [512, 1024, 2048]
    model = build_bla(main_args).to('cuda')

    feat_map2 = torch.randn(2, 256, 256, 256).to('cuda')
    feat_map3 = torch.randn(2, 512, 128, 128).to('cuda')
    feat_map4 = torch.randn(2, 1024, 64, 64).to('cuda')
    feat_map5 = torch.randn(2, 2048, 32, 32).to('cuda')
    feat_maps = [feat_map3, feat_map4, feat_map5]
    inputs = [feat_maps, 0]

    globals_dict = {'model': model, 'inputs': inputs}
    forward_stmt = "model(*inputs)"
    backward_stmt = "torch.cat([map.sum()[None] for map in model(*inputs)]).sum().backward()"

elif profiling_args.model == 'bla_update':
    main_args.backbone_feat_sizes = [512, 1024, 2048]
    main_args.bla_version = 'main'
    main_args.bla_disable_self = False
    main_args.bla_disable_td = False
    main_args.bla_disable_bu = False
    main_args.bla_disable_ffn = False
    model = build_bla(main_args).to('cuda')

    feat_map0 = torch.randn(2, 8, 1024, 1024).to('cuda')
    feat_map1 = torch.randn(2, 16, 512, 512).to('cuda')
    feat_map2 = torch.randn(2, 32, 256, 256).to('cuda')
    feat_map3 = torch.randn(2, 64, 128, 128).to('cuda')
    feat_map4 = torch.randn(2, 128, 64, 64).to('cuda')
    feat_map5 = torch.randn(2, 256, 32, 32).to('cuda')
    feat_map6 = torch.randn(2, 512, 16, 16).to('cuda')
    feat_maps = [feat_map3, feat_map4, feat_map5, feat_map6]
    inputs = [feat_maps, 1]

    globals_dict = {'model': model, 'inputs': inputs}
    forward_stmt = "model(*inputs)"
    backward_stmt = "torch.cat([map.sum()[None] for map in model(*inputs)]).sum().backward()"

elif profiling_args.model == 'bin_seg_head':
    main_args.seg_heads = ['binary']
    main_args.core_feat_sizes = [64, 128, 256, 512]
    main_args.disputed_loss = True
    model = build_seg_heads(main_args)[0].to('cuda')

    feat_map0 = torch.randn(2, 8, 1024, 1024).to('cuda')
    feat_map1 = torch.randn(2, 16, 512, 512).to('cuda')
    feat_map2 = torch.randn(2, 32, 256, 256).to('cuda')
    feat_map3 = torch.randn(2, 64, 128, 128).to('cuda')
    feat_map4 = torch.randn(2, 128, 64, 64).to('cuda')
    feat_map5 = torch.randn(2, 256, 32, 32).to('cuda')
    feat_map6 = torch.randn(2, 512, 16, 16).to('cuda')
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
    main_args.bvn_step_mode = 'multi'
    main_args.core_type = 'BLA'
    main_args.bla_num_layers = 4
    main_args.seg_heads = ['binary']
    main_args.disputed_loss = True
    model = build_bivinet(main_args).to('cuda')

    images = Images(torch.randn(2, 3, 1024, 1024)).to('cuda')

    num_targets_total = 20
    sizes = torch.tensor([0, num_targets_total//2, num_targets_total]).to('cuda')
    masks = (torch.randn(num_targets_total, 1024, 1024) > 1.0).to('cuda')
    tgt_dict = {'sizes': sizes, 'masks': masks}

    optimizer = optimizer = torch.optim.AdamW(model.parameters())
    globals_dict = {'model': model, 'images': images, 'tgt_dict': tgt_dict, 'optimizer': optimizer}
    forward_stmt = "model(*[images, tgt_dict.copy()])"
    backward_stmt = "model(*[images, tgt_dict.copy(), optimizer])"

elif profiling_args.model == 'bivinet_det':
    main_args.num_classes = 80
    main_args.meta_arch = 'BiViNet'
    main_args.bvn_max_downsampling = 7
    main_args.bvn_step_mode = 'single'
    main_args.bvn_sync_heads = False
    main_args.core_type = 'GC'
    main_args.gc_yaml = './configs/gc/tpn_37_ceae_2b3_gn.yaml'
    main_args.det_heads = ['retina']
    main_args.ret_num_convs = 1
    main_args.ret_pred_type = 'conv1'
    main_args.val_metadata = MetadataCatalog.get('coco_2017_val')
    model = build_bivinet(main_args).to('cuda')

    batch_size = main_args.batch_size
    images = Images(torch.randn(2, 3, 800, 800)).to('cuda')

    num_targets_total = 20
    labels = torch.randint(main_args.num_classes, (num_targets_total,), device='cuda')
    boxes = torch.abs(torch.randn(num_targets_total, 4, device='cuda'))
    boxes = Boxes(boxes, 'cxcywh', 'false', [main_args.num_init_slots] * batch_size)
    sizes = torch.tensor([0, num_targets_total//2, num_targets_total]).to('cuda')
    tgt_dict = {'labels': labels, 'boxes': boxes, 'sizes': sizes}

    optimizer = optimizer = torch.optim.AdamW(model.parameters())
    globals_dict = {'model': model, 'images': images, 'tgt_dict': tgt_dict, 'optimizer': optimizer}
    forward_stmt = "model(*[images, tgt_dict.copy()])"
    backward_stmt = "model(*[images, tgt_dict.copy(), optimizer])"

elif profiling_args.model == 'bivinet_sem_seg':
    main_args.num_classes = 80
    main_args.meta_arch = 'BiViNet'
    main_args.bvn_step_mode = 'multi'
    main_args.core_type = 'BLA'
    main_args.bla_num_layers = 4
    main_args.seg_heads = ['binary']
    main_args.val_metadata = MetadataCatalog.get('coco_2017_val')
    model = build_bivinet(main_args).to('cuda')

    images = Images(torch.randn(2, 3, 1024, 1024)).to('cuda')

    num_targets_total = 20
    labels = torch.randint(main_args.num_classes, size=(num_targets_total,)).to('cuda')
    sizes = torch.tensor([0, num_targets_total//2, num_targets_total]).to('cuda')
    masks = (torch.randn(num_targets_total, 1024, 1024) > 1.0).to('cuda')
    tgt_dict = {'labels': labels, 'sizes': sizes, 'masks': masks}

    optimizer = optimizer = torch.optim.AdamW(model.parameters())
    globals_dict = {'model': model, 'images': images, 'tgt_dict': tgt_dict, 'optimizer': optimizer}
    forward_stmt = "model(*[images, tgt_dict.copy()])"
    backward_stmt = "model(*[images, tgt_dict.copy(), optimizer])"

elif profiling_args.model == 'brd_head':
    main_args.num_classes = 80
    main_args.bvn_max_downsampling = 7
    main_args.core_type = 'FPN'
    main_args.core_feat_sizes = [256, 256, 256, 256, 256]
    main_args.det_heads = ['brd']
    main_args.val_metadata = MetadataCatalog.get('coco_2017_val')
    model = build_det_heads(main_args)[0].to('cuda')

    feat_map3 = torch.randn(2, 256, 128, 128).to('cuda')
    feat_map4 = torch.randn(2, 256, 64, 64).to('cuda')
    feat_map5 = torch.randn(2, 256, 32, 32).to('cuda')
    feat_map6 = torch.randn(2, 256, 16, 16).to('cuda')
    feat_map7 = torch.randn(2, 256, 8, 8).to('cuda')
    feat_maps = [feat_map3, feat_map4, feat_map5, feat_map6, feat_map7]

    batch_size = 2
    num_targets = 10
    labels = [torch.randint(main_args.num_classes, (num_targets,), device='cuda') for _ in range(batch_size)]
    boxes = [torch.abs(torch.randn(num_targets, 2, device='cuda')) for _ in range(batch_size)]
    boxes = [torch.cat([boxes_i, boxes_i+1], dim=1) for boxes_i in boxes]
    boxes = [Boxes(boxes_i/boxes_i.max(), 'xyxy', 'img_with_padding') for boxes_i in boxes]
    tgt_dict = {'labels': labels, 'boxes': boxes}

    inputs = {'feat_maps': feat_maps, 'tgt_dict': tgt_dict}
    globals_dict = {'model': model, 'inputs': inputs}
    forward_stmt = "model(**inputs)"
    backward_stmt = "sum(v[None] for v in model(**inputs)[0].values()).backward()"

elif profiling_args.model == 'criterion':
    batch_size = main_args.batch_size
    main_args.num_classes = 80
    model = build_criterion(main_args).to('cuda')

    def generate_out_list():
        num_slots_total = batch_size * main_args.num_init_slots
        logits = torch.randn(num_slots_total, main_args.num_classes+1, device='cuda', requires_grad=True)
        boxes = torch.abs(torch.randn(num_slots_total, 4, device='cuda', requires_grad=True))
        boxes = Boxes(boxes/boxes.max(), 'cxcywh', 'img_without_padding', [main_args.num_init_slots] * batch_size)
        sizes = torch.tensor([i*main_args.num_init_slots for i in range(batch_size+1)], device='cuda')
        out_list = [{'logits': logits, 'boxes': boxes, 'sizes': sizes, 'layer_id': 6, 'iter_id': 1}]

        return out_list

    num_targets_total = 20
    labels = torch.randint(main_args.num_classes, (num_targets_total,), device='cuda')
    boxes = torch.abs(torch.randn(num_targets_total, 4, device='cuda'))
    boxes = Boxes(boxes/boxes.max(), 'cxcywh', 'img_without_padding', [main_args.num_init_slots] * batch_size)
    sizes = torch.tensor([i*(num_targets_total//batch_size) for i in range(batch_size+1)], device='cuda')
    tgt_dict = {'labels': labels, 'boxes': boxes, 'sizes': sizes}

    globals_dict = {'model': model, 'generate_out_list': generate_out_list, 'tgt_dict': tgt_dict}
    forward_stmt = "model(generate_out_list(), tgt_dict)"
    backward_stmt = "torch.stack([v for v in model(generate_out_list(), tgt_dict)[0].values()]).sum().backward()"

elif profiling_args.model == 'detr':
    main_args.meta_arch = 'DETR'
    main_args.lr_backbone = 1e-5
    main_args.lr_encoder = 1e-4
    main_args.lr_decoder = 1e-4
    main_args.num_classes = 80
    main_args.train_metadata = MetadataCatalog.get('coco_2017_train')
    model = build_detr(main_args).to('cuda')

    batch_size = main_args.batch_size
    images = Images(torch.randn(batch_size, 3, 800, 800)).to('cuda')

    num_targets_total = 20
    labels = torch.randint(main_args.num_classes, (num_targets_total,), device='cuda')
    boxes = torch.abs(torch.randn(num_targets_total, 4, device='cuda'))
    boxes = Boxes(boxes/boxes.max(), 'cxcywh', 'img_without_padding', [main_args.num_init_slots] * batch_size)
    sizes = torch.tensor([i*(num_targets_total//batch_size) for i in range(batch_size+1)], device='cuda')
    tgt_dict = {'labels': labels, 'boxes': boxes, 'sizes': sizes}

    optimizer = torch.optim.AdamW(model.parameters())
    globals_dict = {'model': model, 'images': images, 'tgt_dict': tgt_dict, 'optimizer': optimizer}
    forward_stmt = "model(*[images])"
    backward_stmt = "model(*[images, tgt_dict.copy(), optimizer])"

elif profiling_args.model == 'dfd_head':
    main_args.num_classes = 80
    main_args.bvn_max_downsampling = 7
    main_args.core_type = 'FPN'
    main_args.core_feat_sizes = [256, 256, 256, 256, 256]
    main_args.det_heads = ['dfd']
    main_args.val_metadata = MetadataCatalog.get('coco_2017_val')
    model = build_det_heads(main_args)[0].to('cuda')

    feat_map3 = torch.randn(2, 256, 128, 128).to('cuda')
    feat_map4 = torch.randn(2, 256, 64, 64).to('cuda')
    feat_map5 = torch.randn(2, 256, 32, 32).to('cuda')
    feat_map6 = torch.randn(2, 256, 16, 16).to('cuda')
    feat_map7 = torch.randn(2, 256, 8, 8).to('cuda')
    feat_maps = [feat_map3, feat_map4, feat_map5, feat_map6, feat_map7]

    batch_size = 2
    num_targets = 10
    labels = [torch.randint(main_args.num_classes, (num_targets,), device='cuda') for _ in range(batch_size)]
    boxes = [torch.abs(torch.randn(num_targets, 2, device='cuda')) for _ in range(batch_size)]
    boxes = [torch.cat([boxes_i, boxes_i+1], dim=1) for boxes_i in boxes]
    boxes = [Boxes(boxes_i/boxes_i.max(), 'xyxy', 'img_with_padding') for boxes_i in boxes]
    tgt_dict = {'labels': labels, 'boxes': boxes}

    inputs = {'feat_maps': feat_maps, 'tgt_dict': tgt_dict}
    globals_dict = {'model': model, 'inputs': inputs}
    forward_stmt = "model(**inputs)"
    backward_stmt = "sum(v[None] for v in model(**inputs)[0].values()).backward()"

elif profiling_args.model == 'dod_pred':
    main_args.num_classes = 80
    main_args.bvn_max_downsampling = 7
    main_args.core_type = 'FPN'
    main_args.core_feat_sizes = [256, 256, 256, 256, 256]
    main_args.det_heads = ['dod']
    main_args.dod_rel_preds = False
    main_args.val_metadata = MetadataCatalog.get('coco_2017_val')
    model = build_det_heads(main_args)[0].to('cuda')

    feat_map3 = torch.randn(2, 256, 128, 128).to('cuda')
    feat_map4 = torch.randn(2, 256, 64, 64).to('cuda')
    feat_map5 = torch.randn(2, 256, 32, 32).to('cuda')
    feat_map6 = torch.randn(2, 256, 16, 16).to('cuda')
    feat_map7 = torch.randn(2, 256, 8, 8).to('cuda')
    feat_maps = [feat_map3, feat_map4, feat_map5, feat_map6, feat_map7]

    inputs = {'feat_maps': feat_maps, 'mode': 'pred'}
    globals_dict = {'model': model, 'inputs': inputs}
    forward_stmt = "model(**inputs)"
    backward_stmt = "model(**inputs)[2].sum().backward()"

elif profiling_args.model == 'dod_self':
    main_args.num_classes = 80
    main_args.bvn_max_downsampling = 7
    main_args.core_type = 'FPN'
    main_args.core_feat_sizes = [256, 256, 256, 256, 256]
    main_args.det_heads = ['dod']
    main_args.dod_rel_preds = False
    main_args.dod_ftm_decision = 'rel'
    main_args.val_metadata = MetadataCatalog.get('coco_2017_val')
    model = build_det_heads(main_args)[0].to('cuda')

    feat_map3 = torch.randn(2, 256, 128, 128).to('cuda')
    feat_map4 = torch.randn(2, 256, 64, 64).to('cuda')
    feat_map5 = torch.randn(2, 256, 32, 32).to('cuda')
    feat_map6 = torch.randn(2, 256, 16, 16).to('cuda')
    feat_map7 = torch.randn(2, 256, 8, 8).to('cuda')
    feat_maps = [feat_map3, feat_map4, feat_map5, feat_map6, feat_map7]

    batch_size = 2
    num_targets = 10
    labels = [torch.randint(main_args.num_classes, (num_targets,), device='cuda') for _ in range(batch_size)]
    boxes = [torch.abs(torch.randn(num_targets, 2, device='cuda')) for _ in range(batch_size)]
    boxes = [torch.cat([boxes_i, boxes_i+1], dim=1) for boxes_i in boxes]
    boxes = [Boxes(boxes_i/boxes_i.max(), 'xyxy', 'img_with_padding') for boxes_i in boxes]
    tgt_dict = {'labels': labels, 'boxes': boxes}

    inputs = {'feat_maps': feat_maps, 'tgt_dict': tgt_dict, 'mode': 'self'}
    globals_dict = {'model': model, 'inputs': inputs}
    forward_stmt = "model(**inputs)"
    backward_stmt = "sum(v[None] for v in model(**inputs)[0].values()).backward()"

elif profiling_args.model == 'dod_train':
    main_args.num_classes = 80
    main_args.bvn_max_downsampling = 7
    main_args.core_type = 'FPN'
    main_args.core_feat_sizes = [256, 256, 256, 256, 256]
    main_args.det_heads = ['dod']
    main_args.dod_ftm_decision = 'rel'
    main_args.val_metadata = MetadataCatalog.get('coco_2017_val')
    model = build_det_heads(main_args)[0].to('cuda')

    feat_map3 = torch.randn(2, 256, 128, 128).to('cuda')
    feat_map4 = torch.randn(2, 256, 64, 64).to('cuda')
    feat_map5 = torch.randn(2, 256, 32, 32).to('cuda')
    feat_map6 = torch.randn(2, 256, 16, 16).to('cuda')
    feat_map7 = torch.randn(2, 256, 8, 8).to('cuda')
    feat_maps = [feat_map3, feat_map4, feat_map5, feat_map6, feat_map7]

    batch_size = 2
    num_targets = 10
    labels = [torch.randint(main_args.num_classes, (num_targets,), device='cuda') for _ in range(batch_size)]
    boxes = [torch.abs(torch.randn(num_targets, 2, device='cuda')) for _ in range(batch_size)]
    boxes = [torch.cat([boxes_i, boxes_i+1], dim=1) for boxes_i in boxes]
    boxes = [Boxes(boxes_i/boxes_i.max(), 'xyxy', 'img_with_padding') for boxes_i in boxes]
    tgt_dict = {'labels': labels, 'boxes': boxes}

    num_feats = sum(feat_map.flatten(2).shape[2] for feat_map in feat_maps)
    tensor_kwargs = {'device': 'cuda', 'dtype': torch.bool}

    train_dict = {}
    train_dict['logits'] = torch.randn(batch_size, num_feats, 1, device='cuda', requires_grad=True)
    train_dict['pos_preds'] = torch.randint(2, size=(batch_size, num_feats), **tensor_kwargs)
    train_dict['pos_useful'] = torch.randint(2, size=(batch_size, num_feats), **tensor_kwargs)
    train_dict['tgt_found'] = [torch.randint(2, size=(num_targets,), **tensor_kwargs) for _ in range(batch_size)]

    inputs = {'feat_maps': feat_maps, 'tgt_dict': tgt_dict, 'mode': 'train', 'train_dict': train_dict}
    globals_dict = {'model': model, 'inputs': inputs}
    forward_stmt = "model(**inputs)"
    backward_stmt = "sum(v[None] for v in model(**inputs)[0].values()).backward()"

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

elif profiling_args.model == 'fpn':
    main_args.backbone_feat_sizes = [512, 1024, 2048]
    model = build_fpn(main_args).to('cuda')

    feat_map2 = torch.randn(2, 256, 256, 256).to('cuda')
    feat_map3 = torch.randn(2, 512, 128, 128).to('cuda')
    feat_map4 = torch.randn(2, 1024, 64, 64).to('cuda')
    feat_map5 = torch.randn(2, 2048, 32, 32).to('cuda')
    feat_maps = [feat_map3, feat_map4, feat_map5]
    inputs = [feat_maps]

    globals_dict = {'model': model, 'inputs': inputs}
    forward_stmt = "model(*inputs)"
    backward_stmt = "torch.cat([map.sum()[None] for map in model(*inputs)]).sum().backward()"

elif profiling_args.model == 'gc':
    main_args.backbone_feat_sizes = [512, 1024, 2048]
    main_args.gc_yaml = './configs/gc/tpn_37_ceae_5b1_gn.yaml'
    model = build_gc(main_args).to('cuda')

    feat_map3 = torch.randn(2, 512, 128, 128).to('cuda')
    feat_map4 = torch.randn(2, 1024, 64, 64).to('cuda')
    feat_map5 = torch.randn(2, 2048, 32, 32).to('cuda')
    feat_maps = [feat_map3, feat_map4, feat_map5]
    inputs = [feat_maps]

    globals_dict = {'model': model, 'inputs': inputs}
    forward_stmt = "model(*inputs)"
    backward_stmt = "torch.cat([map.sum()[None] for map in model(*inputs)]).sum().backward()"

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

elif profiling_args.model == 'ret_head_bla':
    main_args.num_classes = 80
    main_args.core_type = 'BLA'
    main_args.core_feat_sizes = [64, 128, 256, 512]
    main_args.det_heads = ['retina']
    main_args.ret_num_convs = 4
    main_args.ret_pred_type = 'conv1'
    main_args.val_metadata = MetadataCatalog.get('coco_2017_val')
    model = build_det_heads(main_args)[0].to('cuda')

    feat_map0 = torch.randn(2, 8, 1024, 1024).to('cuda')
    feat_map1 = torch.randn(2, 16, 512, 512).to('cuda')
    feat_map2 = torch.randn(2, 32, 256, 256).to('cuda')
    feat_map3 = torch.randn(2, 64, 128, 128).to('cuda')
    feat_map4 = torch.randn(2, 128, 64, 64).to('cuda')
    feat_map5 = torch.randn(2, 256, 32, 32).to('cuda')
    feat_map6 = torch.randn(2, 512, 16, 16).to('cuda')
    feat_maps = [feat_map3, feat_map4, feat_map5, feat_map6]

    anchor_generator = DefaultAnchorGenerator(sizes=[128.0], aspect_ratios=[1.0], strides=[8])
    anchors = anchor_generator(feat_maps)
    model.anchors = [Boxes(map_anchors.tensor.to('cuda'), format='xyxy') for map_anchors in anchors]

    min_id, max_id = (main_args.bvn_min_downsampling, main_args.bvn_max_downsampling+1)
    num_anchors_total = sum(9 * 4**(10-i) for i in range(min_id, max_id))
    anchor_labels = torch.randint(model.num_classes, size=(2, num_anchors_total)).to('cuda')
    anchor_deltas = torch.abs(torch.randn(2, num_anchors_total, 4)).to('cuda')
    tgt_dict = {'anchor_labels': anchor_labels, 'anchor_deltas': anchor_deltas}

    inputs = {'feat_maps': feat_maps, 'tgt_dict': tgt_dict}
    globals_dict = {'model': model, 'inputs': inputs}
    forward_stmt = "model(**inputs)"
    backward_stmt = "sum(v[None] for v in model(**inputs)[0].values()).backward()"

elif profiling_args.model == 'ret_head_fpn':
    main_args.num_classes = 80
    main_args.bvn_max_downsampling = 7
    main_args.core_type = 'FPN'
    main_args.core_feat_sizes = [256, 256, 256, 256, 256]
    main_args.det_heads = ['retina']
    main_args.ret_num_convs = 4
    main_args.ret_pred_type = 'conv3'
    main_args.val_metadata = MetadataCatalog.get('coco_2017_val')
    model = build_det_heads(main_args)[0].to('cuda')

    feat_map2 = torch.randn(2, 256, 1024, 1024).to('cuda')
    feat_map2 = torch.randn(2, 256, 512, 512).to('cuda')
    feat_map2 = torch.randn(2, 256, 256, 256).to('cuda')
    feat_map3 = torch.randn(2, 256, 128, 128).to('cuda')
    feat_map4 = torch.randn(2, 256, 64, 64).to('cuda')
    feat_map5 = torch.randn(2, 256, 32, 32).to('cuda')
    feat_map6 = torch.randn(2, 256, 16, 16).to('cuda')
    feat_map7 = torch.randn(2, 256, 8, 8).to('cuda')
    feat_maps = [feat_map3, feat_map4, feat_map5, feat_map6, feat_map7]

    anchor_generator = DefaultAnchorGenerator(sizes=[128.0], aspect_ratios=[1.0], strides=[8])
    anchors = anchor_generator(feat_maps)
    model.anchors = [Boxes(map_anchors.tensor.to('cuda'), format='xyxy') for map_anchors in anchors]

    min_id, max_id = (main_args.bvn_min_downsampling, main_args.bvn_max_downsampling+1)
    num_anchors_total = sum(9 * 4**(10-i) for i in range(min_id, max_id))
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
    main_args.num_classes = 80
    main_args.core_feat_sizes = [64, 128, 256, 512]
    main_args.seg_heads = ['semantic']
    main_args.val_metadata = MetadataCatalog.get('coco_2017_val')
    model = build_seg_heads(main_args)[0].to('cuda')

    feat_map0 = torch.randn(2, 8, 1024, 1024).to('cuda')
    feat_map1 = torch.randn(2, 16, 512, 512).to('cuda')
    feat_map2 = torch.randn(2, 32, 256, 256).to('cuda')
    feat_map3 = torch.randn(2, 64, 128, 128).to('cuda')
    feat_map4 = torch.randn(2, 128, 64, 64).to('cuda')
    feat_map5 = torch.randn(2, 256, 32, 32).to('cuda')
    feat_map6 = torch.randn(2, 512, 16, 16).to('cuda')
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
stmt = forward_stmt if profiling_args.forward or profiling_args.inference else backward_stmt

# Check whether computation graph needs to be built
if profiling_args.forward or profiling_args.inference:
    for parameter in model.parameters():
        parameter.requires_grad = False

# Put model in inference mode if desired
if profiling_args.inference:
    model.eval()
    globals_dict['inputs'].pop('tgt_dict', None)

# Warm-up and timing with torch.utils.benchmark.Timer
timer = Timer(stmt=stmt, globals=globals_dict)
t = timer.timeit(number=10).median*1e3

# Profiling with torch.autograd.profiler.profile(use_cuda=True)
with profiler.profile(use_cuda=True, profile_memory=profiling_args.memory) as prof:
    exec(stmt)

# Print profiling table
print(prof.table(sort_by=profiling_args.sort_by, row_limit=100))

# Print number of parameters if desired
if not profiling_args.forward and not profiling_args.inference:
    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters: {num_parameters/10**6: .1f} M")

# Print number of frame per second and max GPU memory
print(f"Number of FPS: {1000/t: .1f} FPS")
print(f"Max GPU memory: {torch.cuda.max_memory_allocated()/(1024**3): .2f} GB")
