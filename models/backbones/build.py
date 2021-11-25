"""
General build function for backbone modules.
"""

from .mmdet import MMDetBackbone
from .resnet import ResNet


def build_backbone(args):
    """
    Build backbone module from command-line arguments.

    Args:
        args (argparse.Namespace): Command-line arguments.

    Returns:
        backbone (nn.Module): The specified backbone module.

    Raises:
        ValueError: Error when unknown backbone type was provided.
    """

    # Build backbone module
    if args.backbone_type == 'mmdet':
        backbone = MMDetBackbone(args.mmdet_backbone_cfg_path)

    elif args.backbone_type == 'resnet':
        backbone = ResNet(args.resnet_name, args.resnet_out_ids, args.resnet_dilation)

    else:
        raise ValueError(f"Unknown backbone type {args.backbone_type} was provided.")

    # Add backbone output indices and output feature sizes to args
    args.backbone_out_ids = backbone.out_ids
    args.backbone_out_sizes = backbone.out_sizes

    return backbone
