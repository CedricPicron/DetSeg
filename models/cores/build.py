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
        in_ids = args.backbone_out_ids
        in_sizes = args.backbone_out_sizes
        core_ids = args.core_ids

        feat_size = args.bifpn_feat_size
        num_layers = args.bifpn_num_layers
        norm_type = args.bifpn_norm_type
        separable_conv = args.bifpn_separable_conv
        core = BiFPN(in_ids, in_sizes, core_ids, feat_size, num_layers, norm_type, separable_conv)

    elif args.core_type == 'dc':
        in_ids = args.backbone_out_ids
        in_sizes = args.backbone_out_sizes
        core_ids = args.core_ids

        feat_size = args.dc_feat_size
        num_layers = args.dc_num_layers

        da_dict = {'norm': args.dc_da_norm, 'act_fn': args.dc_da_act_fn, 'skip': True, 'version': args.dc_da_version}
        da_dict = {**da_dict, 'num_heads': args.dc_da_num_heads, 'num_points': args.dc_da_num_points}
        da_dict = {**da_dict, 'rad_pts': args.dc_da_rad_pts, 'ang_pts': args.dc_da_ang_pts}
        da_dict = {**da_dict, 'lvl_pts': args.dc_da_lvl_pts, 'dup_pts': args.dc_da_dup_pts}
        da_dict = {**da_dict, 'qk_size': args.dc_da_qk_size, 'val_size': args.dc_da_val_size}
        da_dict = {**da_dict, 'val_with_pos': args.dc_da_val_with_pos, 'norm_z': args.dc_da_norm_z}

        core_kwargs = {'prior_type': args.dc_prior_type, 'prior_factor': args.dc_prior_factor}
        core_kwargs = {**core_kwargs, 'scale_encs': args.dc_scale_encs, 'scale_invariant': args.dc_scale_invariant}
        core_kwargs = {**core_kwargs, 'with_ffn': not args.dc_no_ffn, 'ffn_hidden_size': args.dc_ffn_hidden_size}
        core = DeformableCore(in_ids, in_sizes, core_ids, feat_size, num_layers, da_dict, **core_kwargs)

    elif args.core_type == 'fpn':
        in_ids = args.backbone_out_ids
        in_sizes = args.backbone_out_sizes
        core_ids = args.core_ids

        feat_size = args.fpn_feat_size
        fuse_type = args.fpn_fuse_type
        core = FPN(in_ids, in_sizes, core_ids, feat_size, fuse_type)

    elif args.core_type == 'gc':
        in_ids = args.backbone_out_ids
        in_sizes = args.backbone_out_sizes
        core_ids = args.core_ids

        yaml_file = args.gc_yaml
        core = GC(in_ids, in_sizes, core_ids, yaml_file)

    elif args.core_type == 'mmdet':
        core = MMDetCore(args.mmdet_core_cfg_path)

    else:
        raise ValueError(f"Unknown core type {args.core_type} was provided.")

    # Add core output indices and output feature sizes to args
    args.core_out_ids = core.out_ids
    args.core_out_sizes = core.out_sizes

    return core
