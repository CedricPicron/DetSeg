"""
General build function for core modules.
"""

from .bla import build_bla
from .fpn import build_fpn


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

    if args.core_type == 'BLA':
        core = build_bla(args)
    elif args.core_type == 'FPN':
        core = build_fpn(args)
    else:
        raise ValueError(f"Unknown core type {args.core_type} was provided.")

    return core
