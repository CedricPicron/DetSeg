"""
General build function for core modules.
"""

from .bla import build_bla
from .fpn import build_fpn
from .gc import build_gc


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
    if args.core_type == 'BLA':
        core = build_bla(args)
    elif args.core_type == 'FPN':
        core = build_fpn(args)
    elif args.core_type == 'GC':
        core = build_gc(args)
    else:
        raise ValueError(f"Unknown core type {args.core_type} was provided.")

    # Add core output feature sizes to args
    args.core_feat_sizes = core.feat_sizes

    return core
