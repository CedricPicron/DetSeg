"""
General build function for detection heads.
"""
from copy import deepcopy

from .brd import BRD
from .dfd import DFD
from .dod import DOD
from .retina import RetinaHead
from .sbd import SBD


def build_det_heads(args):
    """
    Build detection head modules from command-line arguments.

    Args:
        args (argparse.Namespace): Command-line arguments.

    Returns:
        det_heads (Dict): Dictionary with specified detection head modules.

    Raises:
        ValueError: Error when unknown detection head type was provided.
    """

    # Initialize empty dictionary of detection head modules and get metadata
    det_heads = {}
    metadata = args.val_metadata

    # Build detection head modules
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
            head_dict = {**head_dict, 'layers': args.brd_head_layers}
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
            det_heads[det_head_type] = brd_head

        elif det_head_type == 'dfd':
            in_feat_size = args.core_feat_sizes[0]
            assert all(in_feat_size == core_feat_size for core_feat_size in args.core_feat_sizes)

            cls_dict = {'feat_size': args.dfd_cls_feat_size, 'norm': args.dfd_cls_norm}
            cls_dict = {**cls_dict, 'num_classes': args.num_classes, 'prior_prob': args.dfd_cls_prior_prob}
            cls_dict = {**cls_dict, 'kernel_size': args.dfd_cls_kernel_size, 'bottle_size': args.dfd_cls_bottle_size}
            cls_dict = {**cls_dict, 'hidden_layers': args.dfd_cls_hidden_layers}
            cls_dict = {**cls_dict, 'focal_alpha': args.dfd_cls_focal_alpha, 'focal_gamma': args.dfd_cls_focal_gamma}
            cls_dict = {**cls_dict, 'weight': args.dfd_cls_weight}

            obj_dict = {'feat_size': args.dfd_obj_feat_size, 'norm': args.dfd_obj_norm}
            obj_dict = {**obj_dict, 'prior_prob': args.dfd_obj_prior_prob, 'kernel_size': args.dfd_obj_kernel_size}
            obj_dict = {**obj_dict, 'bottle_size': args.dfd_obj_bottle_size}
            obj_dict = {**obj_dict, 'hidden_layers': args.dfd_obj_hidden_layers}
            obj_dict = {**obj_dict, 'focal_alpha': args.dfd_obj_focal_alpha, 'focal_gamma': args.dfd_obj_focal_gamma}
            obj_dict = {**obj_dict, 'weight': args.dfd_obj_weight}

            box_dict = {'feat_size': args.dfd_box_feat_size, 'norm': args.dfd_box_norm}
            box_dict = {**box_dict, 'kernel_size': args.dfd_box_kernel_size, 'bottle_size': args.dfd_box_bottle_size}
            box_dict = {**box_dict, 'hidden_layers': args.dfd_box_hidden_layers, 'sl1_beta': args.dfd_box_sl1_beta}
            box_dict = {**box_dict, 'weight': args.dfd_box_weight}

            pos_dict = {'feat_size': args.dfd_pos_feat_size, 'norm': args.dfd_pos_norm}
            pos_dict = {**pos_dict, 'kernel_size': args.dfd_pos_kernel_size, 'bottle_size': args.dfd_pos_bottle_size}
            pos_dict = {**pos_dict, 'hidden_layers': args.dfd_pos_hidden_layers}

            ins_dict = {'feat_size': args.dfd_ins_feat_size, 'norm': args.dfd_ins_norm}
            ins_dict = {**ins_dict, 'prior_prob': args.dfd_ins_prior_prob, 'kernel_size': args.dfd_ins_kernel_size}
            ins_dict = {**ins_dict, 'bottle_size': args.dfd_ins_bottle_size}
            ins_dict = {**ins_dict, 'hidden_layers': args.dfd_ins_hidden_layers, 'out_size': args.dfd_ins_out_size}
            ins_dict = {**ins_dict, 'focal_alpha': args.dfd_ins_focal_alpha, 'focal_gamma': args.dfd_ins_focal_gamma}
            ins_dict = {**ins_dict, 'weight': args.dfd_ins_weight}

            inf_dict = {'nms_candidates': args.dfd_inf_nms_candidates, 'nms_threshold': args.dfd_inf_nms_threshold}
            inf_dict = {**inf_dict, 'ins_candidates': args.dfd_inf_ins_candidates}
            inf_dict = {**inf_dict, 'ins_threshold': args.dfd_inf_ins_threshold}
            inf_dict = {**inf_dict, 'max_detections': args.dfd_inf_max_detections}

            dfd_head = DFD(in_feat_size, cls_dict, obj_dict, box_dict, pos_dict, ins_dict, inf_dict, metadata)
            det_heads[det_head_type] = dfd_head

        elif det_head_type == 'dod':
            in_feat_size = args.core_feat_sizes[0]
            assert all(in_feat_size == core_feat_size for core_feat_size in args.core_feat_sizes)
            map_ids = list(range(args.core_min_map_id, args.core_min_map_id+len(args.core_feat_sizes)))

            if not isinstance(args.dod_anchor_asp_ratios, list):
                args.dod_anchor_asp_ratios = [args.dod_anchor_asp_ratios]

            net_dict = {'feat_size': args.dod_feat_size, 'norm': args.dod_norm, 'kernel_size': args.dod_kernel_size}
            net_dict = {**net_dict, 'bottle_size': args.dod_bottle_size, 'hidden_layers': args.dod_hidden_layers}
            net_dict = {**net_dict, 'rel_preds': args.dod_rel_preds, 'prior_prob': args.dod_prior_prob}

            anchor_dict = {'map_ids': map_ids, 'num_sizes': args.dod_anchor_num_sizes}
            anchor_dict = {**anchor_dict, 'scale_factor': args.dod_anchor_scale_factor}
            anchor_dict = {**anchor_dict, 'aspect_ratios': args.dod_anchor_asp_ratios}

            sel_dict = {'mode': args.dod_sel_mode, 'abs_thr': args.dod_sel_abs_thr, 'rel_thr': args.dod_sel_rel_thr}

            tgt_dict = {'metric': args.dod_tgt_metric, 'decision': args.dod_tgt_decision}
            tgt_dict = {**tgt_dict, 'abs_pos_tgt': args.dod_abs_pos_tgt, 'abs_neg_tgt': args.dod_abs_neg_tgt}
            tgt_dict = {**tgt_dict, 'rel_pos_tgt': args.dod_rel_pos_tgt, 'rel_neg_tgt': args.dod_rel_neg_tgt}
            tgt_dict = {**tgt_dict, 'mode': args.dod_tgt_mode}

            loss_dict = {'type': args.dod_loss_type, 'focal_alpha': args.dod_focal_alpha}
            loss_dict = {**loss_dict, 'focal_gamma': args.dod_focal_gamma, 'pos_weight': args.dod_pos_weight}
            loss_dict = {**loss_dict, 'neg_weight': args.dod_neg_weight}

            pred_dict = {'num_pos': args.dod_pred_num_pos, 'max_dets': args.dod_pred_max_dets}

            dod_head = DOD(in_feat_size, net_dict, anchor_dict, sel_dict, tgt_dict, loss_dict, pred_dict, metadata)
            det_heads[det_head_type] = dod_head

        elif det_head_type == 'ret':
            num_classes = args.num_classes
            in_feat_sizes = args.core_feat_sizes

            map_ids = list(range(args.core_min_map_id, args.core_min_map_id+len(in_feat_sizes)))
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
            det_heads[det_head_type] = ret_head

        elif det_head_type == 'sbd':
            in_feat_size = args.core_feat_sizes[0]
            assert all(in_feat_size == core_feat_size for core_feat_size in args.core_feat_sizes)

            if not isinstance(args.sbd_loss_box_types, list):
                args.sbd_loss_box_types = [args.sbd_loss_box_types]

            if not isinstance(args.sbd_loss_box_weights, list):
                args.sbd_loss_box_weights = [args.sbd_loss_box_weights]

            args_copy = deepcopy(args)
            args_copy.det_heads = ['dod']
            dod = build_det_heads(args_copy)['dod']

            osi_dict = {'type': args.sbd_osi_type, 'layers': args.sbd_osi_layers, 'in_size': in_feat_size}
            osi_dict = {**osi_dict, 'hidden_size': args.sbd_osi_hidden_size, 'out_size': args.sbd_osi_out_size}
            osi_dict = {**osi_dict, 'norm': args.sbd_osi_norm, 'act_fn': args.sbd_osi_act_fn}
            osi_dict = {**osi_dict, 'skip': args.sbd_osi_skip}

            cls_dict = {'type': args.sbd_hcls_type, 'layers': args.sbd_hcls_layers, 'in_size': args.sbd_osi_out_size}
            cls_dict = {**cls_dict, 'hidden_size': args.sbd_hcls_hidden_size, 'out_size': args.sbd_hcls_out_size}
            cls_dict = {**cls_dict, 'norm': args.sbd_hcls_norm, 'act_fn': args.sbd_hcls_act_fn}
            cls_dict = {**cls_dict, 'skip': args.sbd_hcls_skip, 'num_classes': args.num_classes}

            box_dict = {'type': args.sbd_hbox_type, 'layers': args.sbd_hbox_layers, 'in_size': args.sbd_osi_out_size}
            box_dict = {**box_dict, 'hidden_size': args.sbd_hbox_hidden_size, 'out_size': args.sbd_hbox_out_size}
            box_dict = {**box_dict, 'norm': args.sbd_hbox_norm, 'act_fn': args.sbd_hbox_act_fn}
            box_dict = {**box_dict, 'skip': args.sbd_hbox_skip}

            match_dict = {'mode': args.sbd_match_mode, 'rel_thr': args.sbd_match_rel_thr}

            loss_dict = {'with_bg': not args.sbd_loss_no_bg, 'bg_weight': args.sbd_loss_bg_weight}
            loss_dict = {**loss_dict, 'cls_type': args.sbd_loss_cls_type, 'cls_alpha': args.sbd_loss_cls_alpha}
            loss_dict = {**loss_dict, 'cls_gamma': args.sbd_loss_cls_gamma, 'cls_weight': args.sbd_loss_cls_weight}
            loss_dict = {**loss_dict, 'box_types': args.sbd_loss_box_types, 'box_beta': args.sbd_loss_box_beta}
            loss_dict = {**loss_dict, 'box_weights': args.sbd_loss_box_weights}

            pred_dict = {'dup_removal': args.sbd_pred_dup_removal, 'nms_candidates': args.sbd_pred_nms_candidates}
            pred_dict = {**pred_dict, 'nms_thr': args.sbd_pred_nms_thr, 'max_dets': args.sbd_pred_max_dets}

            sbd_head = SBD(dod, osi_dict, cls_dict, box_dict, match_dict, loss_dict, pred_dict, metadata)
            det_heads[det_head_type] = sbd_head

        else:
            raise ValueError(f"Unknown detection head type '{det_head_type}' was provided.")

    return det_heads
