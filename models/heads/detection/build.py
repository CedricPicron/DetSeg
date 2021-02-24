"""
General build function for detection heads.
"""

from .brd import BRD
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

    # Initialize empty list of detection head modules and get metadata
    det_heads = []
    metadata = args.val_metadata

    # Build desired detection head modules
    for det_head_type in args.det_heads:
        if det_head_type == 'brd':
            feat_size = args.brd_feat_size
            assert all(feat_size == core_feat_size for core_feat_size in args.core_feat_sizes)

            policy_dict = {'num_groups': args.brd_num_groups, 'prior_prob': args.brd_prior_prob}
            policy_dict = {**policy_dict, 'inference_samples': args.brd_inference_samples}
            policy_dict = {**policy_dict, 'num_hidden_layers': args.brd_policy_layers}

            decoder_dict = {'num_heads': args.brd_num_heads, 'hidden_size': args.brd_dec_hidden_size}
            decoder_dict = {**decoder_dict, 'num_layers': args.brd_dec_layers}

            head_dict = {'num_classes': args.num_classes, 'hidden_size': args.brd_head_hidden_size}
            head_dict = {**head_dict, 'num_hidden_layers': args.brd_head_layers}
            head_dict = {**head_dict, 'prior_cls_prob': args.brd_head_prior_cls_prob}

            loss_dict = {'focal_alpha': args.brd_focal_alpha, 'focal_gamma': args.brd_focal_gamma}
            loss_dict = {**loss_dict, 'reward_weight': args.brd_reward_weight, 'punish_weight': args.brd_punish_weight}
            loss_dict = {**loss_dict, 'cls_weight': args.brd_cls_weight, 'l1_weight': args.brd_l1_weight}
            loss_dict = {**loss_dict, 'giou_weight': args.brd_giou_weight}

            brd_head = BRD(feat_size, policy_dict, decoder_dict, head_dict, loss_dict, metadata)
            det_heads.append(brd_head)

        elif det_head_type == 'retina':
            num_classes = args.num_classes
            in_feat_sizes = args.core_feat_sizes

            map_ids = list(range(args.bvn_min_downsampling, args.bvn_min_downsampling+len(in_feat_sizes)))
            in_proj = any(feat_size != in_feat_sizes[0] for feat_size in in_feat_sizes)

            pred_head_dict = {'in_proj': in_proj, 'feat_size': args.ret_feat_size, 'num_convs': args.ret_num_convs}
            pred_head_dict = {**pred_head_dict, 'pred_type': args.ret_pred_type}

            loss_dict = {'focal_alpha': args.ret_focal_alpha, 'focal_gamma': args.ret_focal_gamma}
            loss_dict = {**loss_dict, 'smooth_l1_beta': args.ret_smooth_l1_beta, 'normalizer': args.ret_normalizer}
            loss_dict = {**loss_dict, 'momentum': args.ret_momentum, 'cls_weight': args.ret_cls_weight}
            loss_dict = {**loss_dict, 'box_weight': args.ret_box_weight}

            test_dict = {'score_threshold': args.ret_score_threshold, 'max_candidates': args.ret_max_candidates}
            test_dict = {**test_dict, 'nms_threshold': args.ret_nms_threshold}
            test_dict = {**test_dict, 'max_detections': args.ret_max_detections}

            ret_head = RetinaHead(num_classes, map_ids, in_feat_sizes, pred_head_dict, loss_dict, test_dict, metadata)
            det_heads.append(ret_head)

        else:
            raise ValueError(f"Unknown detection head type '{det_head_type}' was provided.")

    return det_heads
