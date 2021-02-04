"""
General build function for detection heads.
"""

from .retina import RetinaHead


def build_det_heads(args):
    """
    Build detection head modules from command-line arguments.

    Args:
        args (argparse.Namespace): Command-line arguments.

    Returns:
        det_heads (List): List of specified detection head modules.

    Raises:
        ValueError: Error when unknown detection head type was provided.
    """

    # Get map ids (i.e. map downsampling exponents)
    map_ids = list(range(args.bvn_min_downsampling, args.bvn_min_downsampling+len(args.core_feat_sizes)))

    # Get feature sizes of input maps and decide whether input projection is needed
    in_feat_sizes = args.core_feat_sizes
    in_proj = any(feat_size != in_feat_sizes[0] for feat_size in in_feat_sizes)

    # Initialize empty list of detection head modules
    det_heads = []

    # Build desired detection head modules
    for det_head_type in args.det_heads:
        if det_head_type == 'retina':
            pred_head_dict = {'in_proj': in_proj, 'feat_size': args.ret_feat_size, 'num_convs': args.ret_num_convs}
            pred_head_dict = {**pred_head_dict, 'pred_type': args.ret_pred_type}

            loss_dict = {'focal_alpha': args.ret_focal_alpha, 'focal_gamma': args.ret_focal_gamma}
            loss_dict = {**loss_dict, 'smooth_l1_beta': args.ret_smooth_l1_beta, 'normalizer': args.ret_normalizer}
            loss_dict = {**loss_dict, 'momentum': args.ret_momentum, 'cls_weight': args.ret_cls_weight}
            loss_dict = {**loss_dict, 'box_weight': args.ret_box_weight}

            test_dict = {'score_threshold': args.ret_score_threshold, 'max_candidates': args.ret_max_candidates}
            test_dict = {**test_dict, 'nms_threshold': args.ret_nms_threshold}
            test_dict = {**test_dict, 'max_detections': args.ret_max_detections}

            num_classes = args.num_classes
            metadata = args.val_metadata

            ret_head = RetinaHead(num_classes, map_ids, in_feat_sizes, pred_head_dict, loss_dict, test_dict, metadata)
            det_heads.append(ret_head)

        else:
            raise ValueError(f"Unknown detection head type '{det_head_type}' was provided.")

    return det_heads
