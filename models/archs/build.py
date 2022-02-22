"""
General build function for architecture modules.
"""

from mmcv import Config

from .bch import BCH
from .bvn import BVN
from .mmdet import MMDetArch

from models.build import build_model
from models.backbones.build import build_backbone
from models.cores.build import build_core
from models.heads.build import build_heads


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

    # Build architecture module
    if args.arch_type == 'bch':
        backbone = build_backbone(args)
        core = build_core(args)
        heads = build_heads(args)
        arch = BCH(backbone, core, heads)

    elif args.arch_type == 'bvn':
        backbone = build_backbone(args)
        core = build_core(args)
        heads = build_heads(args)
        arch = BVN(backbone, core, args.bvn_step_mode, heads, args.bvn_sync_heads)

    elif args.arch_type == 'gct':
        arch_cfg = Config.fromfile(args.gct_cfg_path)
        args.requires_masks = arch_cfg.model.pop('requires_masks', False)
        arch = build_model(arch_cfg.model)

    elif args.arch_type == 'mmdet':
        backbone = build_backbone(args)
        core = build_core(args)
        arch = MMDetArch(args.mmdet_arch_cfg_path, backbone, core)
        args.requires_masks = arch.requires_masks

    else:
        raise ValueError(f"Unknown architecture type {args.arch_type} was provided.")

    return arch
