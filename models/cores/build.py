"""
General build function for core modules.
"""

from models.cores.fpn import FPN
from models.cores.gc import GC


def build_core(args):
    """
    Build core module from command-line arguments.

    Args:
        args (argparse.Namespace): Command-line arguments.

    Returns:
        core (nn.Module): The specified core module.

    Raises:
        ValueError: Error when unknown core type was provided.
    """

    # Build core module
    if args.core_type == 'fpn':
        in_feat_sizes = args.backbone_feat_sizes
        out_feat_sizes = [args.fpn_feat_size] * len(in_feat_sizes)

        num_bottom_up_layers = args.core_max_map_id - args.core_min_map_id - len(in_feat_sizes) + 1
        bottom_up_dict = {'feat_sizes': [args.fpn_feat_size] * num_bottom_up_layers}
        core = FPN(in_feat_sizes, out_feat_sizes, args.fpn_fuse_type, bottom_up_dict)

    elif args.core_type == 'gc':
        core = GC(args.gc_yaml)

    else:
        raise ValueError(f"Unknown core type {args.core_type} was provided.")

    # Add core output feature sizes to args
    args.core_feat_sizes = core.feat_sizes

    return core
