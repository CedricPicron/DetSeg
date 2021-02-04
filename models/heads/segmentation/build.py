"""
General build function for segmentation heads.
"""

from .binary import BinarySegHead
from .semantic import SemanticSegHead


def build_seg_heads(args):
    """
    Build segmentation head modules from command-line arguments.

    Args:
        args (argparse.Namespace): Command-line arguments.

    Returns:
        seg_heads (List): List of specified segmentation head modules.

    Raises:
        ValueError: Error when unknown segmentation head type was provided.
    """

    # Initialize empty list of segmentation head modules
    seg_heads = []

    # Build desired segmentation head modules
    for seg_head_type in args.seg_heads:
        if seg_head_type == 'binary':
            head_args = [args.disputed_loss, args.disputed_beta, args.bin_seg_weight]
            binary_seg_head = BinarySegHead(args.core_feat_sizes, *head_args)
            seg_heads.append(binary_seg_head)

        elif seg_head_type == 'semantic':
            head_args = [args.num_classes, args.bg_weight, args.sem_seg_weight, args.val_metadata]
            semantic_seg_head = SemanticSegHead(args.core_feat_sizes, *head_args)
            seg_heads.append(semantic_seg_head)

        else:
            raise ValueError(f"Unknown segmentation head type '{seg_head_type}' was provided.")

    return seg_heads
