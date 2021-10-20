"""
General build function for core modules.
"""

from .bifpn import BiFPN
from .dc import DeformableCore
from .fpn import FPN
from .gc import GC
from .mmdet import MMDetCore


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
    if args.core_type == 'bifpn':
        in_feat_sizes = args.backbone_feat_sizes
        in_bot_layers = args.core_max_map_id - args.core_min_map_id - len(in_feat_sizes) + 1

        feat_size = args.bifpn_feat_size
        num_layers = args.bifpn_num_layers
        separable_conv = args.bifpn_separable_conv
        core = BiFPN(in_feat_sizes, in_bot_layers, feat_size, num_layers, separable_conv)

    elif args.core_type == 'dc':
        in_feat_sizes = args.backbone_feat_sizes
        in_bot_layers = args.core_max_map_id - args.core_min_map_id - len(in_feat_sizes) + 1
        map_ids = list(range(args.core_min_map_id, args.core_max_map_id + 1))

        feat_size = args.dc_feat_size
        num_layers = args.dc_num_layers

        da_dict = {'norm': args.dc_da_norm, 'act_fn': args.dc_da_act_fn, 'skip': True, 'version': args.dc_da_version}
        da_dict = {**da_dict, 'num_heads': args.dc_da_num_heads, 'num_points': args.dc_da_num_points}
        da_dict = {**da_dict, 'rad_pts': args.dc_da_rad_pts, 'ang_pts': args.dc_da_ang_pts}
        da_dict = {**da_dict, 'lvl_pts': args.dc_da_lvl_pts, 'dup_pts': args.dc_da_dup_pts}
        da_dict = {**da_dict, 'qk_size': args.dc_da_qk_size, 'val_size': args.dc_da_val_size}
        da_dict = {**da_dict, 'val_with_pos': args.dc_da_val_with_pos, 'norm_z': args.dc_da_norm_z}

        core_kwargs = {'prior_type': args.dc_prior_type, 'prior_factor': args.dc_prior_factor}
        core_kwargs = {**core_kwargs, 'scale_encs': args.dc_scale_encs}
        core = DeformableCore(in_feat_sizes, in_bot_layers, map_ids, feat_size, num_layers, da_dict, **core_kwargs)

    elif args.core_type == 'fpn':
        in_feat_sizes = args.backbone_feat_sizes
        out_feat_sizes = [args.fpn_feat_size] * len(in_feat_sizes)

        num_bottom_up_layers = args.core_max_map_id - args.core_min_map_id - len(in_feat_sizes) + 1
        bottom_up_dict = {'feat_sizes': [args.fpn_feat_size] * num_bottom_up_layers}
        core = FPN(in_feat_sizes, out_feat_sizes, args.fpn_fuse_type, bottom_up_dict)

    elif args.core_type == 'gc':
        core = GC(args.gc_yaml)

    elif args.core_type == 'mmdet':
        core = MMDetCore(args.mmdet_core_cfg_path)

    else:
        raise ValueError(f"Unknown core type {args.core_type} was provided.")

    # Add core output feature sizes to args
    args.core_feat_sizes = core.feat_sizes

    return core
