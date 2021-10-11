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
        return_layers = {f'layer{i-1}': str(i) for i in args.backbone_map_ids if i >= 2 and i <= 5}
        backbone = ResNet(args.resnet_name, args.resnet_dilation, return_layers)

    else:
        raise ValueError(f"Unknown backbone type {args.backbone_type} was provided.")

    # Add backbone output feature sizes to args
    args.backbone_feat_sizes = backbone.feat_sizes

    return backbone
