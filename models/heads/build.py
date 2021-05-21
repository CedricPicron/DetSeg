"""
General build function for heads.
"""

from .detection.build import build_det_heads
from .segmentation.build import build_seg_heads


def build_heads(args):
    """
    Build head modules from command-line arguments.

    Args:
        args (argparse.Namespace): Command-line arguments.

    Returns:
        heads (Dict): Dictionary with specified head modules.
    """

    # Build list of head modules
    det_heads = build_det_heads(args)
    seg_heads = build_seg_heads(args)
    heads = {**det_heads, **seg_heads}

    return heads
