"""
Main program script.
"""
import argparse
import datetime
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler, SequentialSampler

from datasets.build import build_dataset
from engine import evaluate, save_checkpoint, save_log, train
from models.bivinet import build_bivinet
from models.criterion import build_criterion
from models.detr import build_detr
from utils.data import train_collate_fn, val_collate_fn
import utils.distributed as distributed


def get_parser():
    parser = argparse.ArgumentParser(add_help=False)

    # General
    parser.add_argument('--checkpoint', default='', type=str, help='path with checkpoint to resume from')
    parser.add_argument('--device', default='cuda', type=str, help='device to use training/validation')
    parser.add_argument('--epochs', default=300, type=int, help='total number of training epochs')
    parser.add_argument('--eval', action='store_true', help='evaluate model from checkpoint and return')
    parser.add_argument('--load_orig_detr', action='store_true', help='load untrained detr parts from original detr')
    parser.add_argument('--output_dir', default='', type=str, help='path where to save (no saving when empty)')

    # Distributed
    parser.add_argument('--dist_url', default='env://', type=str, help='url used to set up distributed training')
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')

    # Dataset
    parser.add_argument('--dataset', default='coco', type=str, help='name of dataset used for training and validation')

    # Data loading
    parser.add_argument('--batch_size', default=2, type=int, help='batch size per device')
    parser.add_argument('--num_workers', default=2, type=int, help='number of subprocesses to use for data loading')

    # DETR
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

    # Meta-architecture
    parser.add_argument('--meta_arch', default='BiViNet', choices=['BiViNet', 'DETR'], help='meta-architecture type')

    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str, help='name of the convolutional backbone to use')
    parser.add_argument('--dilation', action='store_true', help='replace stride with dilation in the last conv. block')

    # BiViNet
    parser.add_argument('--min_resolution_id', default=2, type=int, help='highest resolution downsampling exponent')
    parser.add_argument('--max_resolution_id', default=6, type=int, help='lowest resolution downsampling exponent')

    # * BiCore
    parser.add_argument('--bicore_type', default='BiAttnConv', type=str, help='type of BiCore module')
    parser.add_argument('--base_feat_size', default=8, type=int, help='feature size of highest resolution map')
    parser.add_argument('--base_num_heads', default=1, type=int, help='number of heads of highest resolution map')
    parser.add_argument('--max_feat_size', default=1024, type=int, help='largest allowed feature size per map')
    parser.add_argument('--max_num_heads', default=8, type=int, help='maximum number of attention heads per map')
    parser.add_argument('--bicore_dropout', default=0.1, type=float, help='dropout value used with BiCore modules')
    parser.add_argument('--ffn_size_multiplier', default=8, type=int, help='size multiplier used during BiCore FFN')

    # * Heads
    # ** Objectness head
    parser.add_argument('--obj_head_type', default='default', type=str, help='type of objectness head module')
    parser.add_argument('--obj_head_weight', default=1.0, type=float, help='weight factor scaling the objectness loss')
    parser.add_argument('--disputed_loss', action='store_true', help='whether to apply loss at disputed positions')
    parser.add_argument('--no_map_size_correction', action='store_true', help='whether to disable map size correction')
    parser.add_argument('--obj_head_beta', default=0.2, type=float, help='threshold used for disputed smooth L1 loss')

    # Criterion
    parser.add_argument('--aux_loss', action='store_true', help='apply auxiliary losses at intermediate predictions')

    # * Matcher coefficients
    parser.add_argument('--match_coef_class', default=1, type=float, help='class coefficient in the matching cost')
    parser.add_argument('--match_coef_l1', default=5, type=float, help='L1 box coefficient in the matching cost')
    parser.add_argument('--match_coef_giou', default=2, type=float, help='GIoU box coefficient in the matching cost')

    # * Loss coefficients
    parser.add_argument('--loss_coef_class', default=1, type=float, help='class coefficient in loss')
    parser.add_argument('--loss_coef_l1', default=5, type=float, help='L1 box coefficient in loss')
    parser.add_argument('--loss_coef_giou', default=2, type=float, help='GIoU box coefficient in loss')
    parser.add_argument('--no_obj_weight', default=0.1, type=float, help='relative weight of the no-object class')

    # Optimizer
    parser.add_argument('--max_grad_norm', default=0.1, type=float, help='maximum gradient norm during training')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='L2 weight decay coefficient')

    # * Learning rates (general)
    parser.add_argument('--lr_backbone', default=1e-5, type=float, help='backbone learning rate')

    # * Learning rates (BiViNet)
    parser.add_argument('--lr_projs', default=1e-4, type=float, help='BiViNet projectors learning rate')
    parser.add_argument('--lr_core', default=1e-4, type=float, help='BiVeNet core learning rate')
    parser.add_argument('--lr_heads', default=1e-4, type=float, help='BiViNet heads learning rate')

    # * Learning rates (DETR)
    parser.add_argument('--lr_projector', default=1e-4, type=float, help='DETR projector learning rate')
    parser.add_argument('--lr_encoder', default=1e-4, type=float, help='DETR encoder learning rate')
    parser.add_argument('--lr_decoder', default=1e-4, type=float, help='DETR decoder learning rate')
    parser.add_argument('--lr_class_head', default=1e-4, type=float, help='DETR classification head learning rate')
    parser.add_argument('--lr_bbox_head', default=1e-4, type=float, help='DETR bounding box head learning rate')

    # Scheduler
    parser.add_argument('--lr_drop', default=200, type=int, help='scheduler period of learning rate decay')

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
    train_dataloader_kwargs = {'collate_fn': train_collate_fn, 'num_workers': args.num_workers, 'pin_memory': True}
    train_dataloader = DataLoader(train_dataset, batch_sampler=train_batch_sampler, **train_dataloader_kwargs)

    val_dataloader_kwargs = {'collate_fn': val_collate_fn, 'num_workers': args.num_workers, 'pin_memory': True}
    val_dataloader = DataLoader(val_dataset, args.batch_size, sampler=val_sampler, **val_dataloader_kwargs)

    # Build model and place it on correct device
    device = torch.device(args.device)
    model = build_bivinet(args) if args.meta_arch == 'BiViNet' else build_detr(args)
    model = model.to(device)
    criterion = build_criterion(args).to(device)

    # Load untrained model parts from original DETR if required
    if args.load_orig_detr and args.meta_arch == 'DETR':
        model.load_from_original_detr()

    # Load model from checkpoint if required
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        model.load_state_dict(checkpoint['model'])

    # Wrap model into DDP if needed
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])

    # Evaluate loaded model if required and return
    if args.eval:
        _, evaluator = evaluate(model, criterion, val_dataloader, evaluator)
        output_dir = Path(args.output_dir)

        if args.output_dir and distributed.is_main_process():
            for metric in evaluator.metrics:
                evaluations = evaluator.sub_evaluators[metric].eval
                torch.save(evaluations, output_dir / f'eval_{metric}.pth')

        return

    # Get optimizer, scheduler and start epoch
    param_families = model.get_parameter_families()
    param_dicts = {family: {'params': [], 'lr': getattr(args, f'lr_{family}')} for family in param_families}

    for param_name, parameter in model.named_parameters():
        if parameter.requires_grad:
            for family_name in param_families:
                if family_name in param_name:
                    param_dicts[family_name]['params'].append(parameter)
                    break

    optimizer = torch.optim.AdamW(param_dicts.values(), weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
    start_epoch = 1

    # Load optimizer, scheduler and start epoch from checkpoint if required
    if args.checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch']

    # Start training timer
    start_time = time.time()

    # Main training loop
    for epoch in range(start_epoch, args.epochs+1):
        train_sampler.set_epoch(epoch) if args.distributed else None
        train_stats = train(model, criterion, train_dataloader, optimizer, args.max_grad_norm, epoch)
        scheduler.step()

        checkpoint_model = model.module if args.distributed else model
        save_checkpoint(args, epoch, checkpoint_model, optimizer, scheduler)

        val_stats, _ = evaluate(model, criterion, val_dataloader, evaluator, epoch=epoch)
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
