"""
Main program script.
"""
import argparse
import datetime
import json
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler, SequentialSampler

from datasets.build import build_dataset
from engine import evaluate, save_checkpoint, save_log, train, visualize
from models.bivinet import build_bivinet
from models.detr import build_detr
from utils.data import collate_fn, SubsetSampler
import utils.distributed as distributed
from utils.flops import compute_flops


def get_parser():
    parser = argparse.ArgumentParser(add_help=False)

    # General
    parser.add_argument('--device', default='cuda', type=str, help='device to use training/validation')
    parser.add_argument('--checkpoint_full', default='', type=str, help='path with full checkpoint to resume from')
    parser.add_argument('--checkpoint_model', default='', type=str, help='path with model checkpoint to resume from')
    parser.add_argument('--output_dir', default='', type=str, help='save path during training (no saving when empty)')

    # Evaluation
    parser.add_argument('--eval', action='store_true', help='evaluate model from checkpoint and return')

    # FLOPS computation
    parser.add_argument('--get_flops', action='store_true', help='compute number of FLOPS of model and return')
    parser.add_argument('--flops_samples', default=100, type=int, help='input samples used during FLOPS computation')

    # Visualization
    parser.add_argument('--visualize', action='store_true', help='visualize model from checkpoint and return')
    parser.add_argument('--num_images', default=10, type=int, help='number of images to be visualized')
    parser.add_argument('--image_offset', default=0, type=int, help='image id of first image to be visualized')
    parser.add_argument('--random_offset', action='store_true', help='generate random image offset')

    # Distributed
    parser.add_argument('--dist_url', default='env://', type=str, help='url used to set up distributed training')
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')

    # Dataset
    parser.add_argument('--dataset', default='coco', type=str, help='name of dataset used for training and validation')
    parser.add_argument('--evaluator', default='detection', type=str, help='type of evaluator used during validation')

    # Data loading
    parser.add_argument('--batch_size', default=2, type=int, help='batch size per device')
    parser.add_argument('--num_workers', default=2, type=int, help='number of subprocesses to use for data loading')

    # Meta-architecture
    parser.add_argument('--meta_arch', default='BiViNet', choices=['BiViNet', 'DETR'], help='meta-architecture type')

    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str, help='name of the convolutional backbone to use')
    parser.add_argument('--dilation', action='store_true', help='replace stride with dilation in the last conv. block')

    # BiViNet
    parser.add_argument('--bvn_min_downsampling', default=3, type=int, help='minumum downsampling exponent')
    parser.add_argument('--bvn_max_downsampling', default=6, type=int, help='maximum downsampling exponent')
    parser.add_argument('--bvn_step_mode', default='single', choices=['multi', 'single'], help='BiViNet step mode')
    parser.add_argument('--bvn_sync_heads', action='store_true', help='synchronize heads copies in multi-step mode')

    # * Core
    parser.add_argument('--core_type', default='GC', choices=['BLA', 'FPN', 'GC'], help='type of core module')

    # ** BLA (Bidirectional Local Attention)
    parser.add_argument('--bla_version', default='main', choices=['main', 'v1', 'v2', 'v3'], help='BLA version string')
    parser.add_argument('--bla_num_layers', default=4, type=int, help='number of consecutive BLA core layers')

    parser.add_argument('--bla_base_feat_size', default=8, type=int, help='feature size of highest resolution map')
    parser.add_argument('--bla_base_num_heads', default=1, type=int, help='number of heads of highest resolution map')
    parser.add_argument('--bla_max_feat_size', default=1024, type=int, help='largest allowed feature size per map')
    parser.add_argument('--bla_max_num_heads', default=8, type=int, help='maximum number of attention heads per map')

    parser.add_argument('--bla_disable_self', action='store_true', help='disables BLA self-attention mechanism')
    parser.add_argument('--bla_disable_td', action='store_true', help='disables BLA top-down attention mechanism')
    parser.add_argument('--bla_disable_bu', action='store_true', help='disables BLA bottom-up attention mechanism')
    parser.add_argument('--bla_disable_pos', action='store_true', help='disables BLA position dependent attention')
    parser.add_argument('--bla_attn_dropout', default=0.1, type=float, help='dropout value used with BLA attention')

    parser.add_argument('--bla_disable_ffn', action='store_true', help='disables BLA FFN layers')
    parser.add_argument('--bla_ffn_size_multiplier', default=8, type=int, help='size multiplier used during BLA FFN')
    parser.add_argument('--bla_ffn_dropout', default=0.1, type=float, help='dropout value used during BLA FFN')

    # ** FPN (Feature Pyramid Network)
    parser.add_argument('--fpn_feat_size', default=256, type=int, help='feature size of FPN output maps')
    parser.add_argument('--fpn_fuse_type', default='sum', choices=['avg', 'sum'], help='FPN fusing operation')

    # ** GC (Generalized Core)
    parser.add_argument('--gc_yaml', default='', type=str, help='path to yaml-file with GC specification')

    # * Heads
    # ** Detection heads
    parser.add_argument('--det_heads', nargs='*', default='', type=str, help='names of desired detection heads')

    # *** BRD (Base Reinforced Detector) head
    parser.add_argument('--brd_feat_size', default=256, type=int, help='internal feature size of the BRD head')

    parser.add_argument('--brd_num_groups', default=8, type=int, help='number of group normalization groups')
    parser.add_argument('--brd_prior_prob', default=0.01, type=float, help='prior object probability')
    parser.add_argument('--brd_inference_samples', default=100, type=int, help='number of samples during inference')
    parser.add_argument('--brd_policy_layers', default=1, type=int, help='number of policy hidden layers')

    parser.add_argument('--brd_num_heads', default=8, type=int, help='number of decoder attention heads')
    parser.add_argument('--brd_dec_hidden_size', default=1024, type=int, help='feature size in decoder hidden layer')
    parser.add_argument('--brd_dec_layers', default=2, type=int, help='number of decoder layers')

    parser.add_argument('--brd_head_hidden_size', default=256, type=int, help='feature size in head hidden layer')
    parser.add_argument('--brd_head_layers', default=1, type=int, help='number of head hidden layers')
    parser.add_argument('--brd_head_prior_cls_prob', default=0.01, type=float, help='prior class probability')

    parser.add_argument('--brd_inter_loss', action='store_true', help='apply loss on intermediate layer predictions')
    parser.add_argument('--brd_rel_preds', action='store_true', help='predict boxes relative to previous predictions')
    parser.add_argument('--brd_use_all_preds', action='store_true', help='apply loss on all predictions from layer')
    parser.add_argument('--brd_use_lsa', action='store_true', help='use linear sum assignment during loss matching')

    parser.add_argument('--brd_delta_range_xy', default=1.0, type=float, help='range of object location delta')
    parser.add_argument('--brd_delta_range_wh', default=8.0, type=float, help='range of object size delta')

    parser.add_argument('--brd_focal_alpha', default=0.25, type=float, help='BRD head focal alpha value')
    parser.add_argument('--brd_focal_gamma', default=2.0, type=float, help='BRD head focal gamma value')

    parser.add_argument('--brd_reward_weight', default=1.0, type=float, help='BRD head reward weight factor')
    parser.add_argument('--brd_punish_weight', default=0.1, type=float, help='BRD head punishment weight factor')

    parser.add_argument('--brd_cls_rank_weight', default=1.0, type=float, help='classification ranking weight')
    parser.add_argument('--brd_l1_rank_weight', default=5.0, type=float, help='L1 bounding box ranking weight')
    parser.add_argument('--brd_giou_rank_weight', default=2.0, type=float, help='GIoU bounding box ranking weight')

    parser.add_argument('--brd_cls_loss_weight', default=1.0, type=float, help='classification loss weight')
    parser.add_argument('--brd_l1_loss_weight', default=5.0, type=float, help='L1 bounding box loss weight')
    parser.add_argument('--brd_giou_loss_weight', default=2.0, type=float, help='GIoU bounding box loss weight')

    # *** Duplicate-Free Detector (DFD) head
    parser.add_argument('--dfd_cls_feat_size', default=256, type=int, help='classification hidden feature size')
    parser.add_argument('--dfd_cls_norm', default='group', type=str, help='normalization type of classification head')
    parser.add_argument('--dfd_cls_prior_prob', default=0.01, type=float, help='prior class probability')
    parser.add_argument('--dfd_cls_kernel_size', default=3, type=int, help='classification hidden layer kernel size')
    parser.add_argument('--dfd_cls_bottle_size', default=64, type=int, help='classification bottleneck feature size')
    parser.add_argument('--dfd_cls_hidden_layers', default=1, type=int, help='number of classification hidden layers')

    parser.add_argument('--dfd_cls_focal_alpha', default=0.25, type=float, help='classification focal alpha value')
    parser.add_argument('--dfd_cls_focal_gamma', default=2.0, type=float, help='classification focal gamma value')
    parser.add_argument('--dfd_cls_weight', default=1e0, type=float, help='classification loss weight')

    parser.add_argument('--dfd_obj_feat_size', default=256, type=int, help='objectness hidden feature size')
    parser.add_argument('--dfd_obj_norm', default='group', type=str, help='normalization type of objectness head')
    parser.add_argument('--dfd_obj_prior_prob', default=0.01, type=float, help='prior object probability')
    parser.add_argument('--dfd_obj_kernel_size', default=3, type=int, help='objectness hidden layer kernel size')
    parser.add_argument('--dfd_obj_bottle_size', default=64, type=int, help='objectness bottleneck feature size')
    parser.add_argument('--dfd_obj_hidden_layers', default=1, type=int, help='number of objectness hidden layers')

    parser.add_argument('--dfd_obj_focal_alpha', default=0.25, type=float, help='objectness focal alpha value')
    parser.add_argument('--dfd_obj_focal_gamma', default=2.0, type=float, help='objectness focal gamma value')
    parser.add_argument('--dfd_obj_weight', default=1e1, type=float, help='objectness loss weight')

    parser.add_argument('--dfd_box_feat_size', default=256, type=int, help='bounding box hidden feature size')
    parser.add_argument('--dfd_box_norm', default='group', type=str, help='normalization type of bounding box head')
    parser.add_argument('--dfd_box_kernel_size', default=3, type=int, help='bounding box hidden layer kernel size')
    parser.add_argument('--dfd_box_bottle_size', default=64, type=int, help='bounding box bottleneck feature size')
    parser.add_argument('--dfd_box_hidden_layers', default=1, type=int, help='number of bounding box hidden layers')

    parser.add_argument('--dfd_box_sl1_beta', default=0.0, type=float, help='bounding box smooth L1 beta value')
    parser.add_argument('--dfd_box_weight', default=2e-1, type=float, help='bounding box loss weight')

    parser.add_argument('--dfd_pos_feat_size', default=64, type=int, help='position encoding feature size')
    parser.add_argument('--dfd_pos_norm', default='', type=str, help='normalization type of position head')
    parser.add_argument('--dfd_pos_kernel_size', default=3, type=int, help='kernel size of position head')
    parser.add_argument('--dfd_pos_bottle_size', default=8, type=int, help='bottleneck feature size of position head')
    parser.add_argument('--dfd_pos_hidden_layers', default=2, type=int, help='number of position head hidden layers')

    parser.add_argument('--dfd_ins_feat_size', default=256, type=int, help='instance hidden feature size')
    parser.add_argument('--dfd_ins_norm', default='group', type=str, help='normalization type of instance head')
    parser.add_argument('--dfd_ins_prior_prob', default=0.01, type=float, help='prior instance probability')
    parser.add_argument('--dfd_ins_kernel_size', default=3, type=int, help='instance hidden layer kernel size')
    parser.add_argument('--dfd_ins_bottle_size', default=64, type=int, help='instance bottleneck feature size')
    parser.add_argument('--dfd_ins_hidden_layers', default=1, type=int, help='number of instance hidden layers')
    parser.add_argument('--dfd_ins_out_size', default=256, type=int, help='instance output feature size')

    parser.add_argument('--dfd_ins_focal_alpha', default=0.25, type=float, help='instance focal alpha value')
    parser.add_argument('--dfd_ins_focal_gamma', default=2.0, type=float, help='instance focal gamma value')
    parser.add_argument('--dfd_ins_weight', default=5e0, type=float, help='instance loss weight')

    parser.add_argument('--dfd_inf_nms_candidates', default=1000, type=int, help='max candidates for inference NMS')
    parser.add_argument('--dfd_inf_nms_threshold', default=0.5, type=float, help='IoU threshold during inference NMS')
    parser.add_argument('--dfd_inf_ins_candidates', default=1000, type=int, help='max candidates for instance head')
    parser.add_argument('--dfd_inf_ins_threshold', default=0.5, type=float, help='instance threshold during inference')
    parser.add_argument('--dfd_inf_max_detections', default=100, type=int, help='max number of inference detections')

    # *** Retina head
    parser.add_argument('--ret_feat_size', default=256, type=int, help='internal feature size of the retina head')
    parser.add_argument('--ret_num_convs', default=4, type=int, help='number of retina head convolutions')
    parser.add_argument('--ret_pred_type', default='conv1', choices=['conv1', 'conv3'], help='last prediction module')

    parser.add_argument('--ret_focal_alpha', default=0.25, type=float, help='retina head focal alpha value')
    parser.add_argument('--ret_focal_gamma', default=2.0, type=float, help='retina head focal gamma value')
    parser.add_argument('--ret_smooth_l1_beta', default=0.0, type=float, help='retina head smooth L1 beta value')

    parser.add_argument('--ret_normalizer', default=100.0, type=float, help='initial retina head loss normalizer')
    parser.add_argument('--ret_momentum', default=0.9, type=float, help='momentum factor of retina head loss')

    parser.add_argument('--ret_cls_weight', default=1.0, type=float, help='retina classification weight factor')
    parser.add_argument('--ret_box_weight', default=1.0, type=float, help='retina box regression weight factor')

    parser.add_argument('--ret_score_threshold', default=0.05, type=float, help='retina head test score threshold')
    parser.add_argument('--ret_max_candidates', default=1000, type=int, help='retina head max candidates before NMS')
    parser.add_argument('--ret_nms_threshold', default=0.5, type=float, help='retina head NMS threshold')
    parser.add_argument('--ret_max_detections', default=100, type=int, help='retina head max test detections')

    # ** Segmentation heads
    parser.add_argument('--seg_heads', nargs='*', default='', type=str, help='names of desired segmentation heads')

    # *** Binary segmentation head
    parser.add_argument('--disputed_loss', action='store_true', help='whether to apply loss at disputed positions')
    parser.add_argument('--disputed_beta', default=0.2, type=float, help='threshold used for disputed smooth L1 loss')
    parser.add_argument('--bin_seg_weight', default=1.0, type=float, help='binary segmentation loss weight')

    # *** Semantic segmentation head
    parser.add_argument('--bg_weight', default=0.1, type=float, help='weight scaling losses in background positions')
    parser.add_argument('--sem_seg_weight', default=1.0, type=float, help='semantic segmentation loss weight')

    # DETR
    parser.add_argument('--load_orig_detr', action='store_true', help='load untrained detr parts from original DETR')

    # * Position encoding
    parser.add_argument('--position_encoding', default='sine', type=str, help='type of position encoding')

    # * Transformer
    parser.add_argument('--feat_dim', default=256, type=int, help='feature dimension used in transformer')

    # ** Multi-head attention (MHA)
    parser.add_argument('--mha_dropout', default=0.1, type=float, help='dropout used during multi-head attention')
    parser.add_argument('--num_heads', default=8, type=int, help='number of attention heads')

    # ** Feedforward network (FFN)
    parser.add_argument('--ffn_dropout', default=0.1, type=float, help='dropout used during feedforward network')
    parser.add_argument('--ffn_hidden_dim', default=2048, type=float, help='hidden dimension of feedforward network')

    # ** Encoder
    parser.add_argument('--num_encoder_layers', default=6, type=int, help='number of encoder layers in transformer')

    # ** Decoder
    parser.add_argument('--decoder_type', default='sample', choices=['global', 'sample'], help='decoder type')
    parser.add_argument('--num_decoder_layers', default=6, type=int, help='number of decoder layers in transformer')

    # *** Global decoder
    parser.add_argument('--num_slots', default=100, type=int, help='number of object slots per image')

    # *** Sample decoder
    parser.add_argument('--num_init_slots', default=64, type=int, help='number of initial object slots per image')
    parser.add_argument('--no_curio_sharing', action='store_true', help='do not share curiosity kernel between layers')

    # *** Sample decoder layer
    parser.add_argument('--num_decoder_iterations', default=1, type=int, help='number of decoder iterations per layer')
    parser.add_argument('--iter_type', default='outside', type=str, choices=['inside', 'outside'], help='iter type')
    parser.add_argument('--num_pos_samples', default=16, type=int, help='number of positive features sampled per slot')
    parser.add_argument('--num_neg_samples', default=16, type=int, help='number of negative features sampled per slot')
    parser.add_argument('--sample_type', default='after', type=str, choices=['before', 'after'], help='sample type')
    parser.add_argument('--curio_loss_coef', default=1, type=float, help='coefficient scaling the curiosity loss')
    parser.add_argument('--curio_kernel_size', default=3, type=int, help='kernel size of curiosity convolution')
    parser.add_argument('--curio_dropout', default=0.1, type=float, help='dropout used during curiosity update')

    # * DETR criterion
    parser.add_argument('--aux_loss', action='store_true', help='apply auxiliary losses at intermediate predictions')

    # ** Matcher coefficients
    parser.add_argument('--match_coef_class', default=1, type=float, help='class coefficient in the matching cost')
    parser.add_argument('--match_coef_l1', default=5, type=float, help='L1 box coefficient in the matching cost')
    parser.add_argument('--match_coef_giou', default=2, type=float, help='GIoU box coefficient in the matching cost')

    # ** Loss coefficients
    parser.add_argument('--loss_coef_class', default=1, type=float, help='class coefficient in loss')
    parser.add_argument('--loss_coef_l1', default=5, type=float, help='L1 box coefficient in loss')
    parser.add_argument('--loss_coef_giou', default=2, type=float, help='GIoU box coefficient in loss')
    parser.add_argument('--no_obj_weight', default=0.1, type=float, help='relative weight of the no-object class')

    # Optimizer
    parser.add_argument('--max_grad_norm', default=0.1, type=float, help='maximum gradient norm during training')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='L2 weight decay coefficient')

    # * Learning rates (General)
    parser.add_argument('--lr_backbone', default=1e-5, type=float, help='backbone learning rate')

    # * Learning rates (BiViNet)
    parser.add_argument('--lr_core', default=1e-4, type=float, help='BiVeNet core learning rate')
    parser.add_argument('--lr_heads', default=1e-4, type=float, help='BiViNet heads learning rate')

    # * Learning rates (DETR)
    parser.add_argument('--lr_projector', default=1e-4, type=float, help='DETR projector learning rate')
    parser.add_argument('--lr_encoder', default=1e-4, type=float, help='DETR encoder learning rate')
    parser.add_argument('--lr_decoder', default=1e-4, type=float, help='DETR decoder learning rate')
    parser.add_argument('--lr_class_head', default=1e-4, type=float, help='DETR classification head learning rate')
    parser.add_argument('--lr_bbox_head', default=1e-4, type=float, help='DETR bounding box head learning rate')

    # Scheduler
    parser.add_argument('--epochs', default=36, type=int, help='total number of training epochs')
    parser.add_argument('--lr_drops', nargs='*', default=[27, 33], type=int, help='epochs of learning rate drops')

    return parser


def main(args):
    """
    Function containing the main program script.

    Args:
        args (argparse.Namespace): Command-line arguments.
    """

    # Initialize distributed mode if needed
    distributed.init_distributed_mode(args)
    print(args)

    # Get training/validation datasets and evaluator
    train_dataset, val_dataset, evaluator = build_dataset(args)

    # Get training and validation samplers
    if args.distributed:
        train_sampler = DistributedSampler(train_dataset)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
    else:
        train_sampler = RandomSampler(train_dataset)
        val_sampler = SequentialSampler(val_dataset)

    # Get training and validation dataloaders
    train_batch_sampler = torch.utils.data.BatchSampler(train_sampler, args.batch_size, drop_last=True)
    dataloader_kwargs = {'collate_fn': collate_fn, 'num_workers': args.num_workers, 'pin_memory': True}

    train_dataloader = DataLoader(train_dataset, batch_sampler=train_batch_sampler, **dataloader_kwargs)
    val_dataloader = DataLoader(val_dataset, args.batch_size, sampler=val_sampler, **dataloader_kwargs)

    # Build model and place it on correct device
    device = torch.device(args.device)
    model = build_bivinet(args) if args.meta_arch == 'BiViNet' else build_detr(args)
    model = model.to(device)

    # Load untrained model parts from original DETR if required
    if args.load_orig_detr and args.meta_arch == 'DETR':
        model.load_from_original_detr()

    # Try loading checkpoint
    checkpoint_path = ''

    if args.checkpoint_full:
        try:
            checkpoint = torch.load(args.checkpoint_full, map_location='cpu')
            checkpoint_path = args.checkpoint_full
            load_model_only = False
        except FileNotFoundError:
            pass

    if not checkpoint_path and args.checkpoint_model:
        checkpoint = torch.load(args.checkpoint_model, map_location='cpu')
        checkpoint_path = args.checkpoint_model
        load_model_only = True

    # Load model from checkpoint if present
    if checkpoint_path:
        model.load_state_dict(checkpoint['model'])

    # Wrap model into DistributedDataParallel (DDP) if needed
    if args.distributed:
        model = distributed.DistributedDataParallel(model, device_id=args.gpu)

    # If requested, evaluate model from checkpoint and return
    if args.eval:
        val_stats, evaluators = evaluate(model, val_dataloader, evaluator=evaluator)

        if not checkpoint_path and not args.output_dir:
            return

        if distributed.is_main_process():
            output_dir = Path(args.output_dir) if args.output_dir else Path(checkpoint_path).parent

            with (output_dir / 'eval.txt').open('w') as eval_file:
                eval_file.write(json.dumps(val_stats) + "\n")

            if evaluators is not None:
                for i, evaluator in enumerate(evaluators, 1):
                    for metric in evaluator.metrics:
                        evaluations = evaluator.sub_evaluators[metric].eval
                        torch.save(evaluations, output_dir / f'eval_{i}_{metric}.pth')

        return

    # If requested, compute average number of FLOPS of model and return
    if args.get_flops:
        avg_flops = compute_flops(model, val_dataset, num_samples=args.flops_samples)
        print(f"Average number of FLOPS: {avg_flops: .1f} GFLOPS")
        return

    # If requested, visualize model from checkpoint and return
    if args.visualize:
        if not checkpoint_path and not args.output_dir:
            print("No output directory was given or could be derived from checkpoint. Returning now.")
            return

        if args.distributed:
            print("Distributed mode is not supported for visualization. Returning now.")
            return

        if args.batch_size > 1:
            print("It's recommended to use 'batch_size=1' so that printed loss and analysis dicts are image-specific.")

        output_dir = Path(args.output_dir) if args.output_dir else Path(checkpoint_path).parent / 'visualization'
        output_dir.mkdir(exist_ok=True)

        # Get visualization dataloader
        subset_sampler = SubsetSampler(val_dataset, args.num_images, args.image_offset, args.random_offset)
        dataloader = DataLoader(val_dataset, args.batch_size, sampler=subset_sampler, **dataloader_kwargs)

        # Compute and save annotated images and return
        visualize(model, dataloader, output_dir)
        return

    # Get default optimizer and scheduler
    param_families = model.module.get_param_families() if args.distributed else model.get_param_families()
    param_dicts = {family: {'params': [], 'lr': getattr(args, f'lr_{family}')} for family in param_families}

    for param_name, parameter in model.named_parameters():
        if parameter.requires_grad:
            for family_name in param_families:
                if family_name in param_name:
                    param_dicts[family_name]['params'].append(parameter)
                    break

    optimizer = torch.optim.AdamW(param_dicts.values(), weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.lr_drops)

    # Update default optimizer and/or scheduler based on checkpoint
    if checkpoint_path:
        if load_model_only:
            optimizer.step()
            [scheduler.step() for _ in range(checkpoint['epoch'])]
        else:
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])

    # Get start epoch
    start_epoch = checkpoint['epoch']+1 if checkpoint_path else 1

    # Start training timer
    start_time = time.time()

    # Main training loop
    for epoch in range(start_epoch, args.epochs+1):
        train_sampler.set_epoch(epoch) if args.distributed else None
        train_stats = train(model, train_dataloader, optimizer, args.max_grad_norm, epoch)
        scheduler.step()

        checkpoint_model = model.module if args.distributed else model
        save_checkpoint(args, epoch, checkpoint_model, optimizer, scheduler)

        val_stats, _ = evaluate(model, val_dataloader, evaluator=evaluator, epoch=epoch)
        save_log(args.output_dir, epoch, train_stats, val_stats)

    # End training timer and report total training time
    total_time = time.time() - start_time
    total_time = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training {args.epochs+1 - start_epoch} epochs finished after {total_time}")


if __name__ == '__main__':
    prog = "python main.py"
    description = "SampleDETR training and evaluation script"
    fmt = argparse.MetavarTypeHelpFormatter

    parser = argparse.ArgumentParser(prog=prog, description=description, parents=[get_parser()], formatter_class=fmt)
    main(parser.parse_args())
