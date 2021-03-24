"""
General build function for detection heads.
"""

from .brd import BRD
from .dfd import DFD
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

            loss_dict = {'inter_loss': args.brd_inter_loss, 'rel_preds': args.brd_rel_preds}
            loss_dict = {**loss_dict, 'use_all_preds': args.brd_use_all_preds, 'use_lsa': args.brd_use_lsa}

            loss_dict = {**loss_dict, 'delta_range_xy': args.brd_delta_range_xy}
            loss_dict = {**loss_dict, 'delta_range_wh': args.brd_delta_range_wh}

            loss_dict = {**loss_dict, 'focal_alpha': args.brd_focal_alpha, 'focal_gamma': args.brd_focal_gamma}
            loss_dict = {**loss_dict, 'reward_weight': args.brd_reward_weight, 'punish_weight': args.brd_punish_weight}

            loss_dict = {**loss_dict, 'cls_rank_weight': args.brd_cls_rank_weight}
            loss_dict = {**loss_dict, 'l1_rank_weight': args.brd_l1_rank_weight}
            loss_dict = {**loss_dict, 'giou_rank_weight': args.brd_giou_rank_weight}

            loss_dict = {**loss_dict, 'cls_loss_weight': args.brd_cls_loss_weight}
            loss_dict = {**loss_dict, 'l1_loss_weight': args.brd_l1_loss_weight}
            loss_dict = {**loss_dict, 'giou_loss_weight': args.brd_giou_loss_weight}

            brd_head = BRD(feat_size, policy_dict, decoder_dict, head_dict, loss_dict, metadata)
            det_heads.append(brd_head)

        elif det_head_type == 'dfd':
            feat_size = args.dfd_feat_size
            num_classes = args.num_classes
            assert all(feat_size == core_feat_size for core_feat_size in args.core_feat_sizes)

            dd_dict = {'hidden_size': args.dfd_dd_hidden_size, 'layers': args.dfd_dd_layers}
            dd_dict = {**dd_dict, 'prior_cls_prob': args.dfd_dd_prior_cls_prob}

            dd_dict = {**dd_dict, 'delta_range_xy': args.dfd_dd_delta_range_xy}
            dd_dict = {**dd_dict, 'delta_range_wh': args.dfd_dd_delta_range_wh}

            dd_dict = {**dd_dict, 'weight_mode': args.dfd_dd_weight_mode}
            dd_dict = {**dd_dict, 'weight_power': args.dfd_dd_weight_power}

            dd_dict = {**dd_dict, 'focal_alpha': args.dfd_dd_focal_alpha}
            dd_dict = {**dd_dict, 'focal_gamma': args.dfd_dd_focal_gamma}
            dd_dict = {**dd_dict, 'cls_weight': args.dfd_dd_cls_weight}

            dd_dict = {**dd_dict, 'box_beta': args.dfd_dd_box_beta}
            dd_dict = {**dd_dict, 'box_weight': args.dfd_dd_box_weight}

            dd_dict = {**dd_dict, 'nms_candidates': args.dfd_dd_nms_candidates}
            dd_dict = {**dd_dict, 'nms_threshold': args.dfd_dd_nms_threshold}
            dd_dict = {**dd_dict, 'max_detections': args.dfd_dd_max_detections}

            reward_dict = {'abs_hidden_size': args.dfd_abs_hidden_size, 'abs_layers': args.dfd_abs_layers}
            reward_dict = {**reward_dict, 'abs_samples': args.dfd_abs_samples, 'abs_beta': args.dfd_abs_beta}
            reward_dict = {**reward_dict, 'abs_weight': args.dfd_abs_weight}

            dfd_head = DFD(feat_size, num_classes, dd_dict, reward_dict, metadata)
            det_heads.append(dfd_head)

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
