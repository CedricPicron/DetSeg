import argparse

import torch

from models.criterion import build_criterion
from models.detr import build_detr
import utils.distributed as distributed


def get_parser():
    parser = argparse.ArgumentParser(add_help=False)

    # General
    parser.add_argument('--output_dir', default='', type=str, help='path where to save (no saving when empty)')
    parser.add_argument('--device', default='cuda', type=str, help='device to use training/testing')
    parser.add_argument('--batch_size', default=2, type=int, help='batch size per GPU')

    # Distributed
    parser.add_argument('--dist_url', default='env://', type=str, help='url used to set up distributed training')
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')

    # Model
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str, help='name of the convolutional backbone to use')
    parser.add_argument('--dilation', action='store_true', help='replace stride with dilation in the last conv. block')
    parser.add_argument('--lr_backbone', default=0.0, type=float, help='backbone learning rate')

    # * Position encoding
    parser.add_argument('--position_encoding', default='sine', type=str, help='type of position encoding')

    # * Transformer
    parser.add_argument('--feat_dim', default=256, type=int, help='feature dimension used in transformer')
    parser.add_argument('--lr_transformer', default=1e-5, type=float, help='transformer learning rate')
    parser.add_argument('--num_encoder_layers', default=6, type=int, help='number of encoder layers in transformer')
    parser.add_argument('--num_decoder_layers', default=6, type=int, help='number of decoder layers in transformer')

    # ** Multi-head attention (MHA)
    parser.add_argument('--mha_dropout', default=0.1, type=float, help='dropout used during multi-head attention')
    parser.add_argument('--num_heads', default=8, type=int, help='number of attention heads')

    # ** Feedforward network (FFN)
    parser.add_argument('--ffn_dropout', default=0.1, type=float, help='dropout used during feedforward network')
    parser.add_argument('--ffn_hidden_dim', default=2048, type=float, help='hidden dimension of feedforward network')

    # ** Sample decoder
    parser.add_argument('--decoder_iterations', default=1, type=int, help='number of decoder iterations per layer')
    parser.add_argument('--num_init_slots', default=100, type=int, help='number of initial slots per image')
    parser.add_argument('--samples_per_slot', default=100, type=int, help='number of features sampled per slot')
    parser.add_argument('--coverage_ratio', default=0.1, type=float, help='ratio of coverage samples')
    parser.add_argument('--hard_weights', default=True, type=bool, help='use hard weights during forward method')
    parser.add_argument('--curio_weight_obj', default=1.0, type=float, help='curiosity weight for object features')
    parser.add_argument('--curio_weight_edge', default=2.0, type=float, help='curiosity weight for edge features')
    parser.add_argument('--curio_weight_nobj', default=-1.0, type=float, help='curiosity weight for no-obj. features')
    parser.add_argument('--curio_memory', default=0.9, type=float, help='memory weight in non-sampled positions')

    # Criterion
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false', help='disables auxiliary losses')

    # * Matcher coefficients
    parser.add_argument('--match_coef_class', default=1, type=float, help='class coefficient in the matching cost')
    parser.add_argument('--match_coef_bbox', default=5, type=float, help='L1 box coefficient in the matching cost')
    parser.add_argument('--match_coef_giou', default=2, type=float, help='GIoU box coefficient in the matching cost')

    # * Loss coefficients
    parser.add_argument('--loss_coef_class', default=1, type=float, help='class coefficient in loss')
    parser.add_argument('--loss_coef_bbox', default=5, type=float, help='L1 box coefficient in loss')
    parser.add_argument('--loss_coef_giou', default=2, type=float, help='GIoU box coefficient in loss')
    parser.add_argument('--no_obj_weight', default=0.1, type=float, help='relative weight of the no-object class')

    return parser


def main(args):
    distributed.init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)
    model = build_detr(args).to(device)
    criterion = build_criterion(args).to(device)

    return model, criterion


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='python main.py',
                                     description='SampleDETR training and evaluation script',
                                     parents=[get_parser()],
                                     formatter_class=argparse.MetavarTypeHelpFormatter)
    main(parser.parse_args())
