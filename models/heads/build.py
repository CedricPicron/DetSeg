"""
General build function for heads.
"""

from mmengine.config import Config

from models.build import build_model


def build_heads(args):
    """
    Build head modules from command-line arguments.

    Args:
        args (argparse.Namespace): Command-line arguments.

    Returns:
        heads (Dict): Dictionary with specified head modules.

    Raises:
        ValueError: Error when an unknown head type was provided.
    """

    # Initialize empty dictionary of head modules
    heads = {}

    # Initialize 'requires_masks' command-line argument
    args.requires_masks = False

    # Build head modules
    for head_type in args.heads:

        if head_type == 'dino':
            dino_head_cfg = Config.fromfile(args.dino_head_cfg_path)
            args.requires_masks = dino_head_cfg.model.pop('requires_masks', True)

            dino_head = build_model(dino_head_cfg.model, metadata=args.metadata)
            heads[head_type] = dino_head

        elif head_type == 'gvd':
            gvd_cfg = Config.fromfile(args.gvd_cfg_path)
            args.requires_masks = gvd_cfg.model.pop('requires_masks', True)

            gvd_head = build_model(gvd_cfg.model, metadata=args.metadata)
            heads[head_type] = gvd_head

        else:
            raise ValueError(f"Unknown head type '{head_type}' was provided.")

    return heads
