"""
General build function for architecture modules.
"""

from .bch import BCH
from .bvn import BVN
from .detr import DETR
from .mmdet import MMDetArch

from models.backbones.build import build_backbone
from models.cores.build import build_core
from models.heads.build import build_heads

from models.modules.detr.criterion import build_criterion
from models.modules.detr.decoder import build_decoder
from models.modules.detr.encoder import build_encoder
from models.modules.detr.position import build_position_encoder


def build_arch(args):
    """
    Build architecture module from command-line arguments.

    Args:
        args (argparse.Namespace): Command-line arguments.

    Returns:
        arch (nn.Module): The specified architecture module.

    Raises:
        ValueError: Error when unknown architecture type was provided.
    """

    # Build architecture module and backbone map ids to args
    if args.arch_type == 'bch':
        args.backbone_map_ids = list(range(args.core_min_map_id, args.core_max_map_id+1))
        backbone = build_backbone(args)

        core = build_core(args)
        heads = build_heads(args)
        arch = BCH(backbone, core, heads)

    elif args.arch_type == 'bvn':
        args.backbone_map_ids = list(range(args.core_min_map_id, args.core_max_map_id+1))
        backbone = build_backbone(args)

        core = build_core(args)
        heads = build_heads(args)
        arch = BVN(backbone, core, args.bvn_step_mode, heads, args.bvn_sync_heads)

    elif args.arch_type == 'detr':
        args.backbone_map_ids = [5]
        backbone = build_backbone(args)

        position_encoder = build_position_encoder(args)
        encoder = build_encoder(args)
        decoder = build_decoder(args)
        criterion = build_criterion(args)

        train_dict = {'backbone': args.lr_backbone > 0, 'projector': args.lr_projector > 0}
        train_dict = {**train_dict, 'encoder': args.lr_encoder > 0, 'decoder': args.lr_decoder > 0}
        train_dict = {**train_dict, 'class_head': args.lr_class_head > 0, 'bbox_head': args.lr_bbox_head > 0}

        metadata = args.val_metadata
        arch = DETR(backbone, position_encoder, encoder, decoder, criterion, args.num_classes, train_dict, metadata)

    elif args.arch_type == 'mmdet':
        args.backbone_map_ids = list(range(args.core_min_map_id, args.core_max_map_id+1))
        backbone = build_backbone(args)
        core = build_core(args)

        arch = MMDetArch(args.mmdet_arch_cfg_path, backbone, core)
        args.requires_masks = arch.requires_masks

    else:
        raise ValueError(f"Unknown architecture type {args.arch_type} was provided.")

    return arch
