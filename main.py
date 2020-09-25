import argparse

import torch
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler, SequentialSampler

from datasets.build import build_dataset
from models.criterion import build_criterion
from models.detr import build_detr
from utils.data import collate_fn
import utils.distributed as distributed


def get_parser():
    parser = argparse.ArgumentParser(add_help=False)

    # General
    parser.add_argument('--checkpoint', default='', type=str, help='path with checkpoint to resume from')
    parser.add_argument('--device', default='cuda', type=str, help='device to use training/testing')
    parser.add_argument('--eval', action='store_true', help='evaluate model from checkpoint and return')
    parser.add_argument('--output_dir', default='', type=str, help='path where to save (no saving when empty)')

    # Distributed
    parser.add_argument('--dist_url', default='env://', type=str, help='url used to set up distributed training')
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')

    # Dataset
    parser.add_argument('--dataset', default='coco', type=str, help='name of dataset used for training and validation')

    # Data loading
    parser.add_argument('--batch_size', default=2, type=int, help='batch size per device')
    parser.add_argument('--num_workers', default=2, type=int, help='number of subprocesses to use for data loading')

    # Model
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str, help='name of the convolutional backbone to use')
    parser.add_argument('--dilation', action='store_true', help='replace stride with dilation in the last conv. block')

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
    parser.add_argument('--num_decoder_iterations', default=1, type=int, help='number of decoder iterations per layer')

    # *** Global decoder
    parser.add_argument('--num_slots', default=100, type=int, help='number of object slots per image')

    # *** Sample decoder
    parser.add_argument('--num_init_slots', default=64, type=int, help='number of initial object slots per image')
    parser.add_argument('--samples_per_slot', default=16, type=int, help='number of features sampled per slot')
    parser.add_argument('--coverage_ratio', default=0.1, type=float, help='ratio of coverage samples')
    parser.add_argument('--hard_weights', default=True, type=bool, help='use hard weights during forward method')
    parser.add_argument('--seg_head_dim', default=32, type=int, help='projected dimension in segmentation heads')
    parser.add_argument('--curio_weight_obj', default=1.0, type=float, help='curiosity weight for object features')
    parser.add_argument('--curio_weight_edge', default=2.0, type=float, help='curiosity weight for edge features')
    parser.add_argument('--curio_weight_nobj', default=-1.0, type=float, help='curiosity weight for no-obj. features')
    parser.add_argument('--curio_kernel_size', default=3, type=int, help='kernel size of curiosity convolution')

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
    parser.add_argument('--lr_backbone', default=0.0, type=float, help='backbone learning rate')
    parser.add_argument('--lr_encoder', default=0.0, type=float, help='encoder learning rate')
    parser.add_argument('--lr_decoder', default=1e-4, type=float, help='decoder learning rate')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='L2 weight decay coefficient')

    # Scheduler
    parser.add_argument('--lr_drop', default=200, type=int, help='scheduler period of learning rate decay')

    return parser


def main(args):
    # Initialize distributed mode if needed
    distributed.init_distributed_mode(args)
    print(args)

    # Get training and validation datasets
    train_dataset, val_dataset = build_dataset(args)

    # Get training and validation samplers
    if args.distributed:
        train_sampler = DistributedSampler(train_dataset)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
    else:
        train_sampler = RandomSampler(train_dataset)
        val_sampler = SequentialSampler(val_dataset)

    # Get training and validation dataloaders
    dataloader_kwargs = {'collate_fn': collate_fn, 'num_workers': args.num_workers, 'pin_memory': True}
    train_batch_sampler = torch.utils.data.BatchSampler(train_sampler, args.batch_size, drop_last=True)
    train_dataloader = DataLoader(train_dataset, batch_sampler=train_batch_sampler, **dataloader_kwargs)
    val_dataloader = DataLoader(val_dataset, args.batch_size, sampler=val_sampler, **dataloader_kwargs)

    # Get model/criterion and put them on correct device
    device = torch.device(args.device)
    model = build_detr(args).to(device)
    criterion = build_criterion(args).to(device)

    # Load model from checkpoint if required
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        model.load_state_dict(checkpoint['model'])

    # Wrap model into DDP if needed
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])

    # Get training optimizer, scheduler and start epoch
    params = model.named_parameters()
    backbone_dict = {'params': [p for n, p in params if 'backbone' in n and p.requires_grad], 'lr': args.lr_backbone}
    encoder_dict = {'params': [p for n, p in params if 'encoder' in n and p.requires_grad], 'lr': args.lr_encoder}
    decoder_dict = {'params': [p for n, p in params if 'decoder' in n and p.requires_grad], 'lr': args.lr_decoder}
    param_dicts = [backbone_dict, encoder_dict, decoder_dict]

    optimizer = torch.optim.AdamW(param_dicts, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
    start_epoch = 1

    # Load optimizer, scheduler and start epoch from checkpoint if required
    if args.checkpoint and not args.eval:
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch']

    return model, criterion, train_dataloader, val_dataloader, scheduler, start_epoch


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='python main.py',
                                     description='SampleDETR training and evaluation script',
                                     parents=[get_parser()],
                                     formatter_class=argparse.MetavarTypeHelpFormatter)
    main(parser.parse_args())
