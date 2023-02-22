"""
General build function for heads.
"""
from copy import deepcopy

from mmengine.config import Config

from .dod import DOD
from models.build import build_model
from .retina import RetinaHead
from .sbd import SBD


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

        elif head_type == 'dod':
            in_feat_size = args.core_out_sizes[0]
            assert all(in_feat_size == core_out_size for core_out_size in args.core_out_sizes)
            map_ids = args.core_out_ids

            if not isinstance(args.dod_anchor_asp_ratios, list):
                args.dod_anchor_asp_ratios = [args.dod_anchor_asp_ratios]

            if 'sort_thr' in args:
                sort_thr = args.sort_thr
            else:
                sort_list = [args.dod_tgt_rel_pos, args.dod_tgt_rel_neg, args.dod_pred_num_pos]
                sort_thr = max(sort_list)

            net_dict = {'feat_size': args.dod_feat_size, 'norm': args.dod_norm, 'kernel_size': args.dod_kernel_size}
            net_dict = {**net_dict, 'bottle_size': args.dod_bottle_size, 'hidden_layers': args.dod_hidden_layers}
            net_dict = {**net_dict, 'rel_preds': args.dod_rel_preds, 'prior_prob': args.dod_prior_prob}

            anchor_dict = {'map_ids': map_ids, 'num_sizes': args.dod_anchor_num_sizes}
            anchor_dict = {**anchor_dict, 'scale_factor': args.dod_anchor_scale_factor}
            anchor_dict = {**anchor_dict, 'aspect_ratios': args.dod_anchor_asp_ratios}

            sel_dict = {'mode': args.dod_sel_mode, 'abs_thr': args.dod_sel_abs_thr, 'rel_thr': args.dod_sel_rel_thr}

            tgt_dict = {'metric': args.dod_tgt_metric, 'sort_thr': sort_thr, 'decision': args.dod_tgt_decision}
            tgt_dict = {**tgt_dict, 'abs_pos': args.dod_tgt_abs_pos, 'abs_neg': args.dod_tgt_abs_neg}
            tgt_dict = {**tgt_dict, 'rel_pos': args.dod_tgt_rel_pos, 'rel_neg': args.dod_tgt_rel_neg}
            tgt_dict = {**tgt_dict, 'mode': args.dod_tgt_mode}

            loss_dict = {'type': args.dod_loss_type, 'focal_alpha': args.dod_focal_alpha}
            loss_dict = {**loss_dict, 'focal_gamma': args.dod_focal_gamma, 'pos_weight': args.dod_pos_weight}
            loss_dict = {**loss_dict, 'neg_weight': args.dod_neg_weight}

            pred_dict = {'num_pos': args.dod_pred_num_pos, 'max_dets': args.dod_pred_max_dets}

            dod_head = DOD(in_feat_size, net_dict, anchor_dict, sel_dict, tgt_dict, loss_dict, pred_dict)
            heads[head_type] = dod_head

        elif head_type == 'gvd':
            gvd_cfg = Config.fromfile(args.gvd_cfg_path)
            args.requires_masks = gvd_cfg.model.pop('requires_masks', True)

            gvd_head = build_model(gvd_cfg.model, metadata=args.metadata)
            heads[head_type] = gvd_head

        elif head_type == 'ret':
            num_classes = args.num_classes
            in_feat_sizes = args.core_out_sizes
            metadata = args.metadata

            map_ids = args.core_out_ids
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
            heads[head_type] = ret_head

        elif head_type == 'sbd':
            in_feat_size = args.core_out_sizes[0]
            assert all(in_feat_size == core_out_size for core_out_size in args.core_out_sizes)
            num_levels = len(args.core_out_sizes)

            if not isinstance(args.sbd_match_box_types, list):
                args.sbd_match_box_types = [args.sbd_match_box_types]

            if not isinstance(args.sbd_match_box_weights, list):
                args.sbd_match_box_weights = [args.sbd_match_box_weights]

            if not isinstance(args.sbd_loss_box_types, list):
                args.sbd_loss_box_types = [args.sbd_loss_box_types]

            if not isinstance(args.sbd_loss_box_weights, list):
                args.sbd_loss_box_weights = [args.sbd_loss_box_weights]

            if not isinstance(args.sbd_update_types, list):
                args.sbd_update_types = [args.sbd_update_types]

            sort_list = [args.dod_tgt_rel_pos, args.dod_tgt_rel_neg, args.dod_pred_num_pos, args.sbd_match_rel_pos]
            sort_list = [*sort_list, args.sbd_match_rel_neg]

            args_copy = deepcopy(args)
            args_copy.heads = ['dod']
            args_copy.sort_thr = max(sort_list)
            dod = build_heads(args_copy)['dod']

            state_dict = {'size': args.sbd_state_size, 'type': args.sbd_state_type}

            osi_dict = {'type': args.sbd_osi_type, 'layers': args.sbd_osi_layers, 'in_size': in_feat_size}
            osi_dict = {**osi_dict, 'hidden_size': args.sbd_osi_hidden_size, 'out_size': args.sbd_state_size}
            osi_dict = {**osi_dict, 'norm': args.sbd_osi_norm, 'act_fn': args.sbd_osi_act_fn}
            osi_dict = {**osi_dict, 'skip': args.sbd_osi_skip}

            ae_dict = {'type': args.sbd_hae_type, 'layers': args.sbd_hae_layers, 'in_size': args.sbd_state_size}
            ae_dict = {**ae_dict, 'hidden_size': args.sbd_hae_hidden_size, 'out_size': args.sbd_state_size}
            ae_dict = {**ae_dict, 'norm': args.sbd_hae_norm, 'act_fn': args.sbd_hae_act_fn}
            ae_dict = {**ae_dict, 'skip': not args.sbd_hae_no_skip}

            se_dict = {'needed': args.sbd_se, 'type': args.sbd_hse_type, 'layers': args.sbd_hse_layers}
            se_dict = {**se_dict, 'in_size': args.sbd_state_size, 'hidden_size': args.sbd_hse_hidden_size}
            se_dict = {**se_dict, 'out_size': args.sbd_state_size, 'norm': args.sbd_hse_norm}
            se_dict = {**se_dict, 'act_fn': args.sbd_hse_act_fn, 'skip': not args.sbd_hse_no_skip}

            cls_dict = {'type': args.sbd_hcls_type, 'layers': args.sbd_hcls_layers, 'in_size': args.sbd_state_size}
            cls_dict = {**cls_dict, 'hidden_size': args.sbd_hcls_hidden_size, 'out_size': args.sbd_hcls_out_size}
            cls_dict = {**cls_dict, 'norm': args.sbd_hcls_norm, 'act_fn': args.sbd_hcls_act_fn}
            cls_dict = {**cls_dict, 'skip': args.sbd_hcls_skip, 'num_classes': args.num_classes}
            cls_dict = {**cls_dict, 'freeze_inter': args.sbd_cls_freeze_inter, 'no_sharing': args.sbd_cls_no_sharing}

            box_dict = {'type': args.sbd_hbox_type, 'layers': args.sbd_hbox_layers, 'in_size': args.sbd_state_size}
            box_dict = {**box_dict, 'hidden_size': args.sbd_hbox_hidden_size, 'out_size': args.sbd_hbox_out_size}
            box_dict = {**box_dict, 'norm': args.sbd_hbox_norm, 'act_fn': args.sbd_hbox_act_fn}
            box_dict = {**box_dict, 'skip': args.sbd_hbox_skip, 'freeze_inter': args.sbd_box_freeze_inter}
            box_dict = {**box_dict, 'no_sharing': args.sbd_box_no_sharing}

            match_dict = {'mode': args.sbd_match_mode, 'cls_type': args.sbd_match_cls_type}
            match_dict = {**match_dict, 'cls_alpha': args.sbd_match_cls_alpha, 'cls_gamma': args.sbd_match_cls_gamma}
            match_dict = {**match_dict, 'cls_weight': args.sbd_match_cls_weight, 'box_types': args.sbd_match_box_types}
            match_dict = {**match_dict, 'box_weights': args.sbd_match_box_weights}
            match_dict = {**match_dict, 'static_mode': args.sbd_match_static_mode}
            match_dict = {**match_dict, 'static_metric': args.sbd_match_static_metric}
            match_dict = {**match_dict, 'abs_pos': args.sbd_match_abs_pos, 'abs_neg': args.sbd_match_abs_neg}
            match_dict = {**match_dict, 'rel_pos': args.sbd_match_rel_pos, 'rel_neg': args.sbd_match_rel_neg}

            loss_dict = {'ae_weight': args.sbd_loss_ae_weight, 'apply_freq': args.sbd_loss_apply_freq}
            loss_dict = {**loss_dict, 'bg_weight': args.sbd_loss_bg_weight, 'cls_type': args.sbd_loss_cls_type}
            loss_dict = {**loss_dict, 'cls_alpha': args.sbd_loss_cls_alpha, 'cls_gamma': args.sbd_loss_cls_gamma}
            loss_dict = {**loss_dict, 'cls_weight': args.sbd_loss_cls_weight, 'box_types': args.sbd_loss_box_types}
            loss_dict = {**loss_dict, 'box_beta': args.sbd_loss_box_beta, 'box_weights': args.sbd_loss_box_weights}

            pred_dict = {'dup_removal': args.sbd_pred_dup_removal, 'nms_candidates': args.sbd_pred_nms_candidates}
            pred_dict = {**pred_dict, 'nms_thr': args.sbd_pred_nms_thr, 'max_dets': args.sbd_pred_max_dets}

            update_dict = {'types': args.sbd_update_types, 'layers': args.sbd_update_layers}
            update_dict = {**update_dict, 'iters': args.sbd_update_iters}

            ca_dict = {'type': args.sbd_ca_type, 'layers': args.sbd_ca_layers, 'in_size': args.sbd_state_size}
            ca_dict = {**ca_dict, 'sample_size': in_feat_size, 'out_size': args.sbd_state_size}
            ca_dict = {**ca_dict, 'norm': args.sbd_ca_norm, 'act_fn': args.sbd_ca_act_fn, 'skip': True}
            ca_dict = {**ca_dict, 'version': args.sbd_ca_version, 'num_heads': args.sbd_ca_num_heads}
            ca_dict = {**ca_dict, 'num_levels': num_levels, 'num_points': args.sbd_ca_num_points}
            ca_dict = {**ca_dict, 'rad_pts': args.sbd_ca_rad_pts, 'ang_pts': args.sbd_ca_ang_pts}
            ca_dict = {**ca_dict, 'lvl_pts': args.sbd_ca_lvl_pts, 'dup_pts': args.sbd_ca_dup_pts}
            ca_dict = {**ca_dict, 'qk_size': args.sbd_ca_qk_size, 'val_size': args.sbd_ca_val_size}
            ca_dict = {**ca_dict, 'val_with_pos': args.sbd_ca_val_with_pos, 'norm_z': args.sbd_ca_norm_z}
            ca_dict = {**ca_dict, 'step_size': args.sbd_ca_step_size, 'step_norm_xy': args.sbd_ca_step_norm_xy}
            ca_dict = {**ca_dict, 'step_norm_z': args.sbd_ca_step_norm_z, 'num_particles': args.sbd_ca_num_particles}

            sa_dict = {'type': args.sbd_sa_type, 'layers': args.sbd_sa_layers, 'in_size': args.sbd_state_size}
            sa_dict = {**sa_dict, 'out_size': args.sbd_state_size, 'norm': args.sbd_sa_norm}
            sa_dict = {**sa_dict, 'act_fn': args.sbd_sa_act_fn, 'skip': True, 'num_heads': args.sbd_sa_num_heads}

            ffn_dict = {'type': args.sbd_ffn_type, 'layers': args.sbd_ffn_layers, 'in_size': args.sbd_state_size}
            ffn_dict = {**ffn_dict, 'hidden_size': args.sbd_ffn_hidden_size, 'out_size': args.sbd_state_size}
            ffn_dict = {**ffn_dict, 'norm': args.sbd_ffn_norm, 'act_fn': args.sbd_ffn_act_fn, 'skip': True}

            sbd_args = (dod, state_dict, osi_dict, ae_dict, se_dict, cls_dict, box_dict, match_dict, loss_dict)
            sbd_args = (*sbd_args, pred_dict, update_dict, ca_dict, sa_dict, ffn_dict, args.metadata)

            sbd_head = SBD(*sbd_args)
            heads[head_type] = sbd_head

        else:
            raise ValueError(f"Unknown head type '{head_type}' was provided.")

    return heads
