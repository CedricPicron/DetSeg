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
from models.archs.build import build_arch
from models.backbones.build import build_backbone
from models.cores.build import build_core
from models.heads.build import build_heads
from structures.boxes import Boxes
from structures.images import Images


# Lists of model and sort choices
model_choices = ['bch_dod', 'bch_gvd', 'bch_sbd', 'bifpn', 'bin', 'brd', 'bvn_bin', 'bvn_ret', 'bvn_sem', 'dc', 'dfd']
model_choices = [*model_choices, 'dod', 'fpn', 'gc', 'gct', 'gvd', 'mbd', 'mmdet_arch', 'mmdet_backbone', 'mmdet_core']
model_choices = [*model_choices, 'resnet', 'ret', 'sbd', 'sem']
sort_choices = ['cpu_time', 'cuda_time', 'cuda_memory_usage', 'self_cuda_memory_usage']

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--main_args', nargs='*', help='comma-separated string of main arguments')
parser.add_argument('--memory', action='store_true', help='whether to report memory usage')
parser.add_argument('--mode', default='train', choices=['train', 'val', 'test'], help='mode of model to be profiled')
parser.add_argument('--model', choices=model_choices, help='type of model to be profiled')
parser.add_argument('--sort_by', default='cuda_time', choices=sort_choices, help='metric to sort profiler table')
profiling_args = parser.parse_args()
profiling_args.main_args = profiling_args.main_args[0].split(',')[1:] if profiling_args.main_args is not None else []
main_args = get_parser().parse_args(args=[*profiling_args.main_args])

# Building the model to be profiled
if profiling_args.model == 'bch_dod':
    main_args.num_classes = 80
    main_args.arch_type = 'bch'
    main_args.heads = ['dod']
    model = build_arch(main_args).to('cuda')

    images = Images(torch.randn(2, 3, 800, 800)).to('cuda')

    num_targets_total = 20
    labels = torch.randint(main_args.num_classes, (num_targets_total,), device='cuda')
    boxes = torch.abs(torch.randn(num_targets_total, 4, device='cuda'))
    boxes = Boxes(boxes, 'cxcywh', 'false', [num_targets_total//2] * 2)
    sizes = torch.tensor([0, num_targets_total//2, num_targets_total]).to('cuda')
    tgt_dict = {'labels': labels, 'boxes': boxes, 'sizes': sizes}

    optimizer = optimizer = torch.optim.AdamW(model.parameters())
    inputs = {'images': images, 'tgt_dict': tgt_dict, 'optimizer': optimizer}
    globals_dict = {'model': model, 'inputs': inputs}
    forward_stmt = "model(**inputs)"
    backward_stmt = "model(**inputs)"

elif profiling_args.model == 'bch_gvd':
    main_args.metadata = MetadataCatalog.get('coco_2017_val')
    main_args.num_classes = 80
    main_args.arch_type = 'bch'
    main_args.backbone_type = 'resnet'
    main_args.resnet_out_ids = [3, 4, 5]
    main_args.core_type = 'gc'
    main_args.core_ids = [3, 4, 5, 6, 7]
    main_args.gc_yaml = './configs/gc/tpn_37_eeec_3b2_gn.yaml'
    main_args.heads = ['gvd']
    main_args.gvd_cfg_path = './configs/gvd/sel_v29.py'
    model = build_arch(main_args).to('cuda')

    images = Images(torch.randn(2, 3, 800, 800)).to('cuda')

    num_targets_total = 20
    labels = torch.randint(main_args.num_classes, (num_targets_total,), device='cuda')
    boxes = torch.abs(torch.randn(num_targets_total, 4, device='cuda'))
    boxes = Boxes(boxes, 'cxcywh', 'false', [num_targets_total//2] * 2)
    sizes = torch.tensor([0, num_targets_total//2, num_targets_total]).to('cuda')
    masks = torch.rand(num_targets_total, 800, 800, device='cuda') > 0.5
    tgt_dict = {'labels': labels, 'boxes': boxes, 'sizes': sizes, 'masks': masks}

    optimizer = optimizer = torch.optim.AdamW(model.parameters())
    inputs = {'images': images, 'tgt_dict': tgt_dict, 'optimizer': optimizer}
    globals_dict = {'model': model, 'inputs': inputs}
    forward_stmt = "model(**inputs)"
    backward_stmt = "model(**inputs)"

elif profiling_args.model == 'bch_sbd':
    main_args.metadata = MetadataCatalog.get('coco_2017_val')
    main_args.num_classes = 80
    main_args.arch_type = 'bch'
    main_args.backbone_type = 'resnet'
    main_args.mmdet_backbone_cfg_path = './configs/mmdet/backbones/resnext_101_dcn_v1.py'
    main_args.core_type = 'gc'
    main_args.core_ids = [3, 4, 5, 6, 7]
    main_args.dc_num_layers = 6
    main_args.dc_da_num_points = 4
    main_args.gc_yaml = './configs/gc/tpn_37_eeec_3b2_gn.yaml'
    main_args.heads = ['sbd']
    main_args.dod_anchor_num_sizes = 3
    main_args.dod_anchor_asp_ratios = [0.5, 1.0, 2.0]
    main_args.dod_tgt_rel_pos = 5
    main_args.dod_tgt_rel_neg = 10
    main_args.sbd_match_rel_pos = 15
    main_args.sbd_match_rel_neg = 15
    main_args.sbd_update_types = ['ca', 'sa', 'ffn']
    main_args.sbd_update_layers = 6
    main_args.sbd_ca_version = 0
    main_args.sbd_ca_num_points = 1
    model = build_arch(main_args).to('cuda')

    images = Images(torch.randn(2, 3, 800, 800)).to('cuda')

    num_targets_total = 20
    labels = torch.randint(main_args.num_classes, (num_targets_total,), device='cuda')
    boxes = torch.abs(torch.randn(num_targets_total, 4, device='cuda'))
    boxes = Boxes(boxes, 'cxcywh', 'false', [num_targets_total//2] * 2)
    sizes = torch.tensor([0, num_targets_total//2, num_targets_total]).to('cuda')
    tgt_dict = {'labels': labels, 'boxes': boxes, 'sizes': sizes}

    optimizer = optimizer = torch.optim.AdamW(model.parameters())
    inputs = {'images': images, 'tgt_dict': tgt_dict, 'optimizer': optimizer}
    globals_dict = {'model': model, 'inputs': inputs}
    forward_stmt = "model(**inputs)"
    backward_stmt = "model(**inputs)"

elif profiling_args.model == 'bifpn':
    main_args.backbone_out_ids = [3, 4, 5]
    main_args.backbone_out_sizes = [512, 1024, 2048]
    main_args.core_type = 'bifpn'
    main_args.core_ids = [3, 4, 5, 6, 7]
    main_args.bifpn_num_layers = 7
    main_args.bifpn_norm_type = 'group'
    main_args.bifpn_separable_conv = True
    model = build_core(main_args).to('cuda')

    feat_map3 = torch.randn(2, 512, 128, 128).to('cuda')
    feat_map4 = torch.randn(2, 1024, 64, 64).to('cuda')
    feat_map5 = torch.randn(2, 2048, 32, 32).to('cuda')
    feat_maps = [feat_map3, feat_map4, feat_map5]

    inputs = {'in_feat_maps': feat_maps}
    globals_dict = {'model': model, 'inputs': inputs}
    forward_stmt = "model(**inputs)"
    backward_stmt = "torch.cat([map.sum()[None] for map in model(**inputs)]).sum().backward()"

elif profiling_args.model == 'bin':
    main_args.heads = ['bin']
    main_args.core_out_sizes = [256, 256, 256, 256, 256]
    main_args.disputed_loss = True
    model = build_heads(main_args)['bin'].to('cuda')

    feat_map3 = torch.randn(2, 256, 128, 128).to('cuda')
    feat_map4 = torch.randn(2, 256, 64, 64).to('cuda')
    feat_map5 = torch.randn(2, 256, 32, 32).to('cuda')
    feat_map6 = torch.randn(2, 256, 16, 16).to('cuda')
    feat_map7 = torch.randn(2, 256, 8, 8).to('cuda')
    feat_maps = [feat_map3, feat_map4, feat_map5, feat_map6, feat_map7]

    tgt_map3 = torch.randn(2, 128, 128).to('cuda').clamp(-0.5, 0.5) + 0.5
    tgt_map4 = torch.randn(2, 64, 64).to('cuda').clamp(-0.5, 0.5) + 0.5
    tgt_map5 = torch.randn(2, 32, 32).to('cuda').clamp(-0.5, 0.5) + 0.5
    tgt_map6 = torch.randn(2, 16, 16).to('cuda').clamp(-0.5, 0.5) + 0.5
    tgt_map7 = torch.randn(2, 8, 8).to('cuda').clamp(-0.5, 0.5) + 0.5
    tgt_maps = [tgt_map3, tgt_map4, tgt_map5, tgt_map6, tgt_map7]
    tgt_dict = {'binary_maps': tgt_maps}

    inputs = {'feat_maps': feat_maps, 'tgt_dict': tgt_dict}
    globals_dict = {'model': model, 'inputs': inputs}
    forward_stmt = "model(**inputs)"
    backward_stmt = "model(**inputs)[0]['bin_seg_loss'].backward()"

elif profiling_args.model == 'brd':
    main_args.metadata = MetadataCatalog.get('coco_2017_val')
    main_args.num_classes = 80
    main_args.core_out_sizes = [256, 256, 256, 256, 256]
    main_args.heads = ['brd']
    model = build_heads(main_args)['brd'].to('cuda')

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

elif profiling_args.model == 'bvn_bin':
    main_args.arch_type = 'bvn'
    main_args.bvn_step_mode = 'single'
    main_args.bvn_sync_heads = False
    main_args.heads = ['bin']
    main_args.disputed_loss = True
    model = build_arch(main_args).to('cuda')

    images = Images(torch.randn(2, 3, 800, 800)).to('cuda')

    num_targets_total = 20
    sizes = torch.tensor([0, num_targets_total//2, num_targets_total]).to('cuda')
    masks = torch.rand(num_targets_total, 800, 800, device='cuda') > 0.5
    tgt_dict = {'sizes': sizes, 'masks': masks}

    optimizer = optimizer = torch.optim.AdamW(model.parameters())
    inputs = {'images': images, 'tgt_dict': tgt_dict, 'optimizer': optimizer}
    globals_dict = {'model': model, 'inputs': inputs}
    forward_stmt = "model(**inputs)"
    backward_stmt = "model(**inputs)"

elif profiling_args.model == 'bvn_ret':
    main_args.metadata = MetadataCatalog.get('coco_2017_val')
    main_args.num_classes = 80
    main_args.arch_type = 'bvn'
    main_args.bvn_step_mode = 'single'
    main_args.bvn_sync_heads = False
    main_args.backbone_type = 'resnet'
    main_args.mmdet_backbone_cfg_path = './configs/mmdet/backbones/resnet_50_v0.py'
    main_args.resnet_name = 'resnet50'
    main_args.core_type = 'gc'
    main_args.gc_yaml = './configs/gc/tpn_37_eeec_3b2_gn.yaml'
    main_args.heads = ['ret']
    main_args.ret_num_convs = 1
    main_args.ret_pred_type = 'conv1'
    model = build_arch(main_args).to('cuda')

    images = Images(torch.randn(2, 3, 800, 800)).to('cuda')

    num_targets_total = 20
    labels = torch.randint(main_args.num_classes, (num_targets_total,), device='cuda')
    boxes = torch.abs(torch.randn(num_targets_total, 4, device='cuda'))
    boxes = Boxes(boxes, 'cxcywh', 'false', [num_targets_total//2] * 2)
    sizes = torch.tensor([0, num_targets_total//2, num_targets_total]).to('cuda')
    tgt_dict = {'labels': labels, 'boxes': boxes, 'sizes': sizes}

    optimizer = optimizer = torch.optim.AdamW(model.parameters())
    inputs = {'images': images, 'tgt_dict': tgt_dict, 'optimizer': optimizer}
    globals_dict = {'model': model, 'inputs': inputs}
    forward_stmt = "model(**inputs)"
    backward_stmt = "model(**inputs)"

elif profiling_args.model == 'bvn_sem':
    main_args.metadata = MetadataCatalog.get('coco_2017_val')
    main_args.num_classes = 80
    main_args.arch_type = 'bvn'
    main_args.bvn_step_mode = 'single'
    main_args.bvn_sync_heads = False
    main_args.heads = ['sem']
    main_args.disputed_loss = True
    model = build_arch(main_args).to('cuda')

    images = Images(torch.randn(2, 3, 800, 800)).to('cuda')

    num_targets_total = 20
    labels = torch.randint(main_args.num_classes, size=(num_targets_total,)).to('cuda')
    sizes = torch.tensor([0, num_targets_total//2, num_targets_total]).to('cuda')
    masks = torch.rand(num_targets_total, 800, 800, device='cuda') > 0.5
    tgt_dict = {'labels': labels, 'sizes': sizes, 'masks': masks}

    optimizer = optimizer = torch.optim.AdamW(model.parameters())
    inputs = {'images': images, 'tgt_dict': tgt_dict, 'optimizer': optimizer}
    globals_dict = {'model': model, 'inputs': inputs}
    forward_stmt = "model(**inputs)"
    backward_stmt = "model(**inputs)"

elif profiling_args.model == 'dc':
    main_args.backbone_out_ids = [3, 4, 5]
    main_args.backbone_out_sizes = [512, 1024, 2048]
    main_args.core_type = 'dc'
    main_args.core_ids = [3, 4, 5, 6, 7]
    main_args.dc_num_layers = 6
    main_args.dc_da_version = 0
    main_args.dc_da_rad_pts = 4
    main_args.dc_da_lvl_pts = 1
    main_args.dc_prior_type = 'location'
    main_args.dc_scale_encs = False
    main_args.dc_scale_invariant = False
    main_args.dc_no_ffn = False
    main_args.dc_ffn_hidden_size = 1024
    model = build_core(main_args).to('cuda')

    feat_map3 = torch.randn(2, 512, 128, 128).to('cuda')
    feat_map4 = torch.randn(2, 1024, 64, 64).to('cuda')
    feat_map5 = torch.randn(2, 2048, 32, 32).to('cuda')
    feat_maps = [feat_map3, feat_map4, feat_map5]

    images = Images(torch.randn(2, 3, 800, 800)).to('cuda')

    inputs = {'in_feat_maps': feat_maps, 'images': images}
    globals_dict = {'model': model, 'inputs': inputs}
    forward_stmt = "model(**inputs)"
    backward_stmt = "torch.cat([map.sum()[None] for map in model(**inputs)]).sum().backward()"

elif profiling_args.model == 'dfd':
    main_args.metadata = MetadataCatalog.get('coco_2017_val')
    main_args.num_classes = 80
    main_args.core_out_sizes = [256, 256, 256, 256, 256]
    main_args.heads = ['dfd']
    model = build_heads(main_args)['dfd'].to('cuda')

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

elif profiling_args.model == 'dod':
    main_args.num_classes = 80
    main_args.core_out_ids = [3, 4, 5, 6, 7]
    main_args.core_out_sizes = [256, 256, 256, 256, 256]
    main_args.heads = ['dod']
    main_args.dod_rel_preds = False
    main_args.dod_anchor_num_sizes = 3
    main_args.dod_anchor_asp_ratios = [0.5, 1.0, 2.0]
    main_args.dod_sel_mode = 'rel'
    main_args.dod_tgt_decision = 'rel'
    main_args.dod_tgt_mode = 'static'
    model = build_heads(main_args)['dod'].to('cuda')

    feat_map3 = torch.randn(2, 256, 128, 128).to('cuda')
    feat_map4 = torch.randn(2, 256, 64, 64).to('cuda')
    feat_map5 = torch.randn(2, 256, 32, 32).to('cuda')
    feat_map6 = torch.randn(2, 256, 16, 16).to('cuda')
    feat_map7 = torch.randn(2, 256, 8, 8).to('cuda')
    feat_maps = [feat_map3, feat_map4, feat_map5, feat_map6, feat_map7]

    num_targets_total = 20
    labels = torch.randint(main_args.num_classes, (num_targets_total,), device='cuda')
    boxes = torch.abs(torch.randn(num_targets_total, 4, device='cuda'))
    boxes = Boxes(boxes, 'cxcywh', 'false', [num_targets_total//2] * 2)
    sizes = torch.tensor([0, num_targets_total//2, num_targets_total]).to('cuda')
    tgt_dict = {'labels': labels, 'boxes': boxes, 'sizes': sizes}

    inputs = {'feat_maps': feat_maps, 'tgt_dict': tgt_dict}
    globals_dict = {'model': model, 'inputs': inputs}
    forward_stmt = "model(**inputs)"
    backward_stmt = "sum(v[None] for v in model(**inputs)[0].values()).backward()"

elif profiling_args.model == 'fpn':
    main_args.backbone_out_ids = [3, 4, 5]
    main_args.backbone_out_sizes = [512, 1024, 2048]
    main_args.core_type = 'fpn'
    main_args.core_ids = [3, 4, 5, 6, 7]
    model = build_core(main_args).to('cuda')

    feat_map3 = torch.randn(2, 512, 128, 128).to('cuda')
    feat_map4 = torch.randn(2, 1024, 64, 64).to('cuda')
    feat_map5 = torch.randn(2, 2048, 32, 32).to('cuda')
    feat_maps = [feat_map3, feat_map4, feat_map5]

    inputs = {'in_feat_maps': feat_maps}
    globals_dict = {'model': model, 'inputs': inputs}
    forward_stmt = "model(**inputs)"
    backward_stmt = "torch.cat([map.sum()[None] for map in model(**inputs)]).sum().backward()"

elif profiling_args.model == 'gc':
    main_args.backbone_out_ids = [3, 4, 5]
    main_args.backbone_out_sizes = [512, 1024, 2048]
    main_args.core_type = 'gc'
    main_args.core_ids = [3, 4, 5, 6, 7]
    main_args.gc_yaml = './configs/gc/tpn_37_eeec_3b2_gn.yaml'
    model = build_core(main_args).to('cuda')

    feat_map3 = torch.randn(2, 512, 128, 128).to('cuda')
    feat_map4 = torch.randn(2, 1024, 64, 64).to('cuda')
    feat_map5 = torch.randn(2, 2048, 32, 32).to('cuda')
    feat_maps = [feat_map3, feat_map4, feat_map5]

    inputs = {'in_feat_maps': feat_maps}
    globals_dict = {'model': model, 'inputs': inputs}
    forward_stmt = "model(**inputs)"
    backward_stmt = "torch.cat([map.sum()[None] for map in model(**inputs)]).sum().backward()"

elif profiling_args.model == 'gct':
    main_args.num_classes = 80
    main_args.arch_type = 'gct'
    main_args.gct_cfg_path = './configs/gct/r50_c4_v000.py'
    model = build_arch(main_args).to('cuda')

    images = Images(torch.randn(2, 3, 800, 800)).to('cuda')

    num_targets_total = 20
    labels = torch.randint(main_args.num_classes, (num_targets_total,), device='cuda')
    boxes = torch.abs(torch.randn(num_targets_total, 4, device='cuda'))
    boxes = Boxes(boxes, 'cxcywh', 'false', [num_targets_total//2] * 2)
    sizes = torch.tensor([0, num_targets_total//2, num_targets_total]).to('cuda')
    tgt_dict = {'labels': labels, 'boxes': boxes, 'sizes': sizes}

    optimizer = torch.optim.AdamW(model.parameters())
    inputs = {'images': images, 'tgt_dict': tgt_dict, 'optimizer': optimizer}
    globals_dict = {'model': model, 'inputs': inputs}
    forward_stmt = "model(**inputs)"
    backward_stmt = "model(**inputs)"

elif profiling_args.model == 'gvd':
    main_args.metadata = MetadataCatalog.get('coco_2017_val')
    main_args.num_classes = 80
    main_args.heads = ['gvd']
    main_args.gvd_cfg_path = './configs/gvd/sel_v68.py'
    model = build_heads(main_args)['gvd'].to('cuda')

    feat_map3 = torch.randn(2, 256, 128, 128).to('cuda')
    feat_map4 = torch.randn(2, 256, 64, 64).to('cuda')
    feat_map5 = torch.randn(2, 256, 32, 32).to('cuda')
    feat_map6 = torch.randn(2, 256, 16, 16).to('cuda')
    feat_map7 = torch.randn(2, 256, 8, 8).to('cuda')
    feat_maps = [feat_map3, feat_map4, feat_map5, feat_map6, feat_map7]

    num_targets_total = 20
    labels = torch.randint(main_args.num_classes, (num_targets_total,), device='cuda')
    boxes = torch.abs(torch.randn(num_targets_total, 4, device='cuda'))
    boxes = Boxes(boxes, 'cxcywh', 'false', [num_targets_total//2] * 2)
    sizes = torch.tensor([0, num_targets_total//2, num_targets_total]).to('cuda')
    masks = torch.rand(num_targets_total, 800, 800, device='cuda') > 0.5
    tgt_dict = {'labels': labels, 'boxes': boxes, 'sizes': sizes, 'masks': masks}

    images = Images(torch.randn(2, 3, 1024, 1024)).to('cuda')

    inputs = {'feat_maps': feat_maps, 'tgt_dict': tgt_dict, 'images': images}
    globals_dict = {'model': model, 'inputs': inputs}
    forward_stmt = "model(**inputs)"
    backward_stmt = "sum(model(**inputs)[0].values()).backward()"

elif profiling_args.model == 'mbd':
    main_args.metadata = MetadataCatalog.get('coco_2017_val')
    main_args.num_classes = 80
    main_args.core_out_ids = [3, 4, 5, 6, 7]
    main_args.core_out_sizes = [256, 256, 256, 256, 256]
    main_args.heads = ['mbd']
    main_args.dod_anchor_num_sizes = 3
    main_args.dod_anchor_asp_ratios = [0.5, 1.0, 2.0]
    main_args.dod_sel_mode = 'rel'
    main_args.dod_tgt_decision = 'rel'
    main_args.sbd_state_type = 'rel_static'
    main_args.sbd_osi_type = 'one_step_mlp'
    main_args.sbd_se = False
    main_args.sbd_match_mode = 'static'
    main_args.sbd_match_static_mode = 'rel'
    main_args.sbd_loss_apply_freq = 'last'
    main_args.sbd_loss_freeze_inter = False
    main_args.sbd_loss_box_types = 'smooth_l1'
    main_args.sbd_loss_box_weights = 1.0
    main_args.sbd_pred_dup_removal = 'nms'
    main_args.sbd_update_types = ['ca', 'sa', 'ffn']
    main_args.sbd_update_layers = 6
    main_args.sbd_ca_type = 'deformable_attn'
    main_args.sbd_ca_version = 0
    main_args.sbd_ca_val_with_pos = False
    main_args.sbd_ca_step_size = -1
    main_args.sbd_ca_step_norm_xy = 'map'
    main_args.sbd_ca_step_norm_z = 1.0
    main_args.mbd_ca_type = 'deformable_attn'
    main_args.mbd_ca_layers = 6
    main_args.mbd_ca_version = 2
    main_args.mbd_match_thr = 0.0
    main_args.mbd_loss_gt_seg = True
    main_args.mbd_loss_seg_types = 'sigmoid_focal'
    model = build_heads(main_args)['mbd'].to('cuda')

    feat_map3 = torch.randn(2, 256, 128, 128).to('cuda')
    feat_map4 = torch.randn(2, 256, 64, 64).to('cuda')
    feat_map5 = torch.randn(2, 256, 32, 32).to('cuda')
    feat_map6 = torch.randn(2, 256, 16, 16).to('cuda')
    feat_map7 = torch.randn(2, 256, 8, 8).to('cuda')
    feat_maps = [feat_map3, feat_map4, feat_map5, feat_map6, feat_map7]

    num_targets_total = 20
    labels = torch.randint(main_args.num_classes, (num_targets_total,), device='cuda')
    boxes = torch.abs(torch.randn(num_targets_total, 4, device='cuda'))
    boxes = Boxes(boxes, 'cxcywh', 'false', [num_targets_total//2] * 2)
    sizes = torch.tensor([0, num_targets_total//2, num_targets_total]).to('cuda')
    masks = torch.rand(num_targets_total, 800, 800, device='cuda') > 0.5
    tgt_dict = {'labels': labels, 'boxes': boxes, 'sizes': sizes, 'masks': masks}

    images = Images(torch.randn(2, 3, 800, 800)).to('cuda')

    inputs = {'feat_maps': feat_maps, 'tgt_dict': tgt_dict, 'images': images}
    globals_dict = {'model': model, 'inputs': inputs}
    forward_stmt = "model(**inputs)"
    backward_stmt = "sum(v[None] for v in model(**inputs)[0].values()).backward()"

elif profiling_args.model == 'mmdet_arch':
    main_args.num_classes = 80
    main_args.arch_type = 'mmdet'
    main_args.mmdet_arch_cfg_path = './configs/mmdet/archs/mask_rcnn_v1.py'
    main_args.backbone_type = 'resnet'
    main_args.core_type = 'gc'
    main_args.gc_yaml = './configs/gc/tpn_37_eeec_3b2_gn.yaml'
    model = build_arch(main_args).to('cuda')

    images = Images(torch.randn(2, 3, 800, 800)).to('cuda')

    num_targets_total = 20
    labels = torch.randint(main_args.num_classes, (num_targets_total,), device='cuda')
    boxes = torch.abs(torch.randn(num_targets_total, 4, device='cuda'))
    boxes = Boxes(boxes, 'cxcywh', 'false', [num_targets_total//2] * 2)
    masks = torch.rand(num_targets_total, 800, 800, device='cuda') > 0.5
    sizes = torch.tensor([0, num_targets_total//2, num_targets_total]).to('cuda')
    tgt_dict = {'labels': labels, 'boxes': boxes, 'masks': masks, 'sizes': sizes}

    optimizer = optimizer = torch.optim.AdamW(model.parameters())
    inputs = {'images': images, 'tgt_dict': tgt_dict, 'optimizer': optimizer}
    globals_dict = {'model': model, 'inputs': inputs}
    forward_stmt = "model(**inputs)"
    backward_stmt = "model(**inputs)"

elif profiling_args.model == 'mmdet_backbone':
    main_args.backbone_type = 'mmdet'
    main_args.mmdet_backbone_cfg_path = './configs/mmdet/backbones/resnext_101_dcn_v0.py'
    model = build_backbone(main_args).to('cuda')

    images = Images(torch.randn(2, 3, 800, 800)).to('cuda')

    inputs = {'images': images}
    globals_dict = {'model': model, 'inputs': inputs}
    forward_stmt = "model(**inputs)"
    backward_stmt = "torch.cat([map.sum()[None] for map in model(**inputs)]).sum().backward()"

elif profiling_args.model == 'mmdet_core':
    main_args.core_type = 'mmdet'
    main_args.mmdet_core_cfg_path = './configs/mmdet/cores/rfp_v0.py'
    model = build_core(main_args).to('cuda')

    images = torch.randn(2, 3, 1024, 1024).to('cuda')
    feat_map2 = torch.randn(2, 256, 256, 256).to('cuda')
    feat_map3 = torch.randn(2, 512, 128, 128).to('cuda')
    feat_map4 = torch.randn(2, 1024, 64, 64).to('cuda')
    feat_map5 = torch.randn(2, 2048, 32, 32).to('cuda')
    feat_maps = [images, feat_map2, feat_map3, feat_map4, feat_map5]

    inputs = {'in_feat_maps': feat_maps}
    globals_dict = {'model': model, 'inputs': inputs}
    forward_stmt = "model(**inputs)"
    backward_stmt = "torch.cat([map.sum()[None] for map in model(**inputs)]).sum().backward()"

elif profiling_args.model == 'resnet':
    main_args.backbone_type = 'resnet'
    main_args.resnet_out_ids = [3, 4, 5]
    model = build_backbone(main_args).to('cuda')

    images = Images(torch.randn(2, 3, 800, 800)).to('cuda')

    inputs = {'images': images}
    globals_dict = {'model': model, 'inputs': inputs}
    forward_stmt = "model(**inputs)"
    backward_stmt = "torch.cat([map.sum()[None] for map in model(**inputs)]).sum().backward()"

elif profiling_args.model == 'ret':
    main_args.metadata = MetadataCatalog.get('coco_2017_val')
    main_args.num_classes = 80
    main_args.core_out_ids = [3, 4, 5, 6, 7]
    main_args.core_out_sizes = [256, 256, 256, 256, 256]
    main_args.heads = ['ret']
    main_args.ret_num_convs = 1
    main_args.ret_pred_type = 'conv1'
    model = build_heads(main_args)['ret'].to('cuda')

    feat_map3 = torch.randn(2, 256, 128, 128).to('cuda')
    feat_map4 = torch.randn(2, 256, 64, 64).to('cuda')
    feat_map5 = torch.randn(2, 256, 32, 32).to('cuda')
    feat_map6 = torch.randn(2, 256, 16, 16).to('cuda')
    feat_map7 = torch.randn(2, 256, 8, 8).to('cuda')
    feat_maps = [feat_map3, feat_map4, feat_map5, feat_map6, feat_map7]

    anchor_generator = DefaultAnchorGenerator(sizes=[128.0], aspect_ratios=[1.0], strides=[8])
    anchors = anchor_generator(feat_maps)
    model.anchors = [Boxes(map_anchors.tensor.to('cuda'), format='xyxy') for map_anchors in anchors]

    num_anchors_total = sum(9 * 4**(10-i) for i in main_args.core_out_ids)
    anchor_labels = torch.randint(model.num_classes, size=(2, num_anchors_total)).to('cuda')
    anchor_deltas = torch.abs(torch.randn(2, num_anchors_total, 4)).to('cuda')
    tgt_dict = {'anchor_labels': anchor_labels, 'anchor_deltas': anchor_deltas}

    inputs = {'feat_maps': feat_maps, 'tgt_dict': tgt_dict}
    globals_dict = {'model': model, 'inputs': inputs}
    forward_stmt = "model(**inputs)"
    backward_stmt = "sum(v[None] for v in model(**inputs)[0].values()).backward()"

elif profiling_args.model == 'sbd':
    main_args.metadata = MetadataCatalog.get('coco_2017_val')
    main_args.num_classes = 80
    main_args.core_out_ids = [3, 4, 5, 6, 7]
    main_args.core_out_sizes = [256, 256, 256, 256, 256]
    main_args.heads = ['sbd']
    main_args.dod_anchor_num_sizes = 3
    main_args.dod_anchor_asp_ratios = [0.5, 1.0, 2.0]
    main_args.dod_sel_mode = 'rel'
    main_args.dod_sel_rel_thr = 300
    main_args.dod_tgt_decision = 'rel'
    main_args.sbd_state_size = 256
    main_args.sbd_state_type = 'rel_static'
    main_args.sbd_osi_type = 'one_step_mlp'
    main_args.sbd_se = False
    main_args.sbd_hcls_layers = 1
    main_args.sbd_cls_freeze_inter = False
    main_args.sbd_cls_no_sharing = False
    main_args.sbd_hbox_layers = 1
    main_args.sbd_box_freeze_inter = False
    main_args.sbd_box_no_sharing = False
    main_args.sbd_match_mode = 'static'
    main_args.sbd_match_box_types = ['l1', 'giou']
    main_args.sbd_match_box_weights = [5.0, 2.0]
    main_args.sbd_match_static_mode = 'rel'
    main_args.sbd_loss_apply_freq = 'last'
    main_args.sbd_loss_box_types = 'smooth_l1'
    main_args.sbd_loss_box_weights = 1.0
    main_args.sbd_pred_dup_removal = 'nms'
    main_args.sbd_update_types = ['ca', 'sa', 'ffn']
    main_args.sbd_update_layers = 6
    main_args.sbd_ca_type = 'deformable_attn'
    main_args.sbd_ca_version = 0
    main_args.sbd_ca_num_points = 1
    main_args.sbd_ca_val_with_pos = False
    main_args.sbd_ca_step_size = -1
    main_args.sbd_ca_step_norm_xy = 'map'
    main_args.sbd_ca_step_norm_z = 1.0
    model = build_heads(main_args)['sbd'].to('cuda')

    feat_map3 = torch.randn(2, 256, 128, 128).to('cuda')
    feat_map4 = torch.randn(2, 256, 64, 64).to('cuda')
    feat_map5 = torch.randn(2, 256, 32, 32).to('cuda')
    feat_map6 = torch.randn(2, 256, 16, 16).to('cuda')
    feat_map7 = torch.randn(2, 256, 8, 8).to('cuda')
    feat_maps = [feat_map3, feat_map4, feat_map5, feat_map6, feat_map7]

    num_targets_total = 20
    labels = torch.randint(main_args.num_classes, (num_targets_total,), device='cuda')
    boxes = torch.abs(torch.randn(num_targets_total, 4, device='cuda'))
    boxes = Boxes(boxes, 'cxcywh', 'false', [num_targets_total//2] * 2)
    sizes = torch.tensor([0, num_targets_total//2, num_targets_total]).to('cuda')
    tgt_dict = {'labels': labels, 'boxes': boxes, 'sizes': sizes}

    images = Images(torch.randn(2, 3, 800, 800)).to('cuda')

    inputs = {'feat_maps': feat_maps, 'tgt_dict': tgt_dict, 'images': images}
    globals_dict = {'model': model, 'inputs': inputs}
    forward_stmt = "model(**inputs)"
    backward_stmt = "sum(v[None] for v in model(**inputs)[0].values()).backward()"

elif profiling_args.model == 'sem':
    main_args.metadata = MetadataCatalog.get('coco_2017_val')
    main_args.num_classes = 80
    main_args.core_out_sizes = [256, 256, 256, 256, 256]
    main_args.heads = ['sem']
    model = build_heads(main_args)['sem'].to('cuda')

    feat_map3 = torch.randn(2, 256, 128, 128).to('cuda')
    feat_map4 = torch.randn(2, 256, 64, 64).to('cuda')
    feat_map5 = torch.randn(2, 256, 32, 32).to('cuda')
    feat_map6 = torch.randn(2, 256, 16, 16).to('cuda')
    feat_map7 = torch.randn(2, 256, 8, 8).to('cuda')
    feat_maps = [feat_map3, feat_map4, feat_map5, feat_map6, feat_map7]

    tgt_map3 = torch.randint(model.num_classes+1, size=(2, 128, 128)).to('cuda')
    tgt_map4 = torch.randint(model.num_classes+1, size=(2, 64, 64)).to('cuda')
    tgt_map5 = torch.randint(model.num_classes+1, size=(2, 32, 32)).to('cuda')
    tgt_map6 = torch.randint(model.num_classes+1, size=(2, 16, 16)).to('cuda')
    tgt_map7 = torch.randint(model.num_classes+1, size=(2, 8, 8)).to('cuda')
    tgt_maps = [tgt_map3, tgt_map4, tgt_map5, tgt_map6, tgt_map7]
    tgt_dict = {'semantic_maps': tgt_maps}

    inputs = {'feat_maps': feat_maps, 'tgt_dict': tgt_dict}
    globals_dict = {'model': model, 'inputs': inputs}
    forward_stmt = "model(**inputs)"
    backward_stmt = "model(**inputs)[0]['sem_seg_loss'].backward()"

# Some additional preparation depending on profiling mode
if profiling_args.mode == 'train':
    model.train()
    stmt = backward_stmt

elif profiling_args.mode in ('val', 'test'):
    globals_dict['inputs'].pop('optimizer', None)
    model.eval()
    stmt = forward_stmt

    for parameter in model.parameters():
        parameter.requires_grad = False

    if profiling_args.mode == 'test':
        globals_dict['inputs'].pop('tgt_dict', None)

# Warm-up and timing with torch.utils.benchmark.Timer
timer = Timer(stmt=stmt, globals=globals_dict)
t = timer.timeit(number=10).median*1e3

# Profiling with torch.autograd.profiler.profile(use_cuda=True)
with profiler.profile(use_cuda=True, profile_memory=profiling_args.memory) as prof:
    exec(stmt)

# Print profiling table
print(prof.table(sort_by=profiling_args.sort_by, row_limit=100))

# Print number of parameters if in training mode
if profiling_args.mode == 'train':
    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters: {num_parameters/10**6: .1f} M")

# Print number of frame per second and max GPU memory
print(f"Number of FPS: {1000/t: .1f} FPS")
print(f"Max GPU memory: {torch.cuda.max_memory_allocated()/(1024**3): .2f} GB")
