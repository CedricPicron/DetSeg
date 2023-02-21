"""
General build function for core modules.
"""
from collections import OrderedDict

from mmcv import Config

from .bifpn import BiFPN
from .dc import DeformableCore
from .fpn import FPN
from .gc import GC
from .mmdet import MMDetCore
from models.build import build_model
from models.modules.container import Sequential


def build_core(args):
    """
    Build core module from command-line arguments.

    Args:
        args (argparse.Namespace): Command-line arguments.

    Returns:
        core (nn.Module): The specified core module.

    Raises:
        ValueError: Error when an unknown core type was provided.
    """

    # Initialize empty dictionary of core modules
    cores = OrderedDict()

    # Get input indices and input sizes
    in_ids = args.backbone_out_ids
    in_sizes = args.backbone_out_sizes

    # Initialize configuration index
    cfg_id = 0

    # Build core modules
    for core_type in args.cores:

        if core_type == 'bifpn':
            core_ids = args.core_ids
            feat_size = args.bifpn_feat_size
            num_layers = args.bifpn_num_layers
            norm_type = args.bifpn_norm_type
            separable_conv = args.bifpn_separable_conv

            core = BiFPN(in_ids, in_sizes, core_ids, feat_size, num_layers, norm_type, separable_conv)
            cores[core_type] = core

        elif core_type == 'cfg':
            core_cfg_path = args.core_cfg_paths[cfg_id]
            core_cfg = Config.fromfile(core_cfg_path)
            cfg_id += 1

            core_name = list(core_cfg.keys())[0]
            core_cfg = getattr(core_cfg, core_name)

            core_out_ids = core_cfg.pop('out_ids')
            core_out_sizes = core_cfg.pop('out_sizes')

            core = build_model(core_cfg)
            cores[core_name] = core

            core.out_ids = core_out_ids
            core.out_sizes = core_out_sizes

        elif core_type == 'dc':
            core_ids = args.core_ids
            feat_size = args.dc_feat_size
            num_layers = args.dc_num_layers

            da_dict = {'norm': args.dc_da_norm, 'act_fn': args.dc_da_act_fn, 'skip': True}
            da_dict = {**da_dict, 'version': args.dc_da_version, 'num_heads': args.dc_da_num_heads}
            da_dict = {**da_dict, 'num_points': args.dc_da_num_points, 'rad_pts': args.dc_da_rad_pts}
            da_dict = {**da_dict, 'ang_pts': args.dc_da_ang_pts, 'lvl_pts': args.dc_da_lvl_pts}
            da_dict = {**da_dict, 'dup_pts': args.dc_da_dup_pts, 'qk_size': args.dc_da_qk_size}
            da_dict = {**da_dict, 'val_size': args.dc_da_val_size, 'val_with_pos': args.dc_da_val_with_pos}
            da_dict = {**da_dict, 'norm_z': args.dc_da_norm_z}

            core_kwargs = {'prior_type': args.dc_prior_type, 'prior_factor': args.dc_prior_factor}
            core_kwargs = {**core_kwargs, 'scale_encs': args.dc_scale_encs, 'scale_invariant': args.dc_scale_invariant}
            core_kwargs = {**core_kwargs, 'with_ffn': not args.dc_no_ffn, 'ffn_hidden_size': args.dc_ffn_hidden_size}

            core = DeformableCore(in_ids, in_sizes, core_ids, feat_size, num_layers, da_dict, **core_kwargs)
            cores[core_type] = core

        elif core_type == 'fpn':
            core_ids = args.core_ids
            feat_size = args.fpn_feat_size
            fuse_type = args.fpn_fuse_type

            core = FPN(in_ids, in_sizes, core_ids, feat_size, fuse_type)
            cores[core_type] = core

        elif core_type == 'gc':
            core_ids = args.core_ids
            yaml_file = args.gc_yaml

            core = GC(in_ids, in_sizes, core_ids, yaml_file)
            cores[core_type] = core

        elif core_type == 'mmdet':
            core = MMDetCore(args.mmdet_core_cfg_path)
            cores[core_type] = core

        else:
            raise ValueError(f"Unknown core type '{core_type}' was provided.")

        # Update input indices and input sizes
        in_ids = core.out_ids
        in_sizes = core.out_sizes

    # Add core output indices and output feature sizes to args
    args.core_out_ids = core.out_ids
    args.core_out_sizes = core.out_sizes

    # Get final core module from dictionary of core modules
    core = list(cores.values())[0] if len(cores) == 1 else Sequential(cores)

    return core
