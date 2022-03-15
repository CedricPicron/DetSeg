"""
Main program script.
"""
import argparse
import datetime
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler, SequentialSampler

from datasets.build import build_dataset
from engine import evaluate, save_checkpoint, save_log, train
from models.archs.build import build_arch
from utils.data import collate_fn, SubsetSampler
import utils.distributed as distributed
from utils.flops import compute_flops


def get_parser():
    parser = argparse.ArgumentParser(add_help=False)

    # General
    parser.add_argument('--device', default='cuda', type=str, help='name of device to use')
    parser.add_argument('--checkpoint_full', default='', type=str, help='path with full checkpoint to resume from')
    parser.add_argument('--checkpoint_model', default='', type=str, help='path with model checkpoint to resume from')
    parser.add_argument('--output_dir', default='', type=str, help='path to output directory')

    # Distributed
    parser.add_argument('--dist_url', default='env://', type=str, help='url used to set up distributed training')
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')

    # Dataset
    parser.add_argument('--dataset', default='coco', type=str, help='name of dataset')
    parser.add_argument('--train_split', default='2017_train', type=str, help='name of the training split')
    parser.add_argument('--eval_split', default='2017_val', type=str, help='name of the evaluation split')
    parser.add_argument('--evaluator', default='detection', type=str, help='type of evaluator used during evaluation')
    parser.add_argument('--eval_nms_thr', default=0.5, type=float, help='IoU threshold during evaluation NMS')

    # Transforms
    parser.add_argument('--train_transforms_type', default='multi_scale', type=str, help='training transforms type')
    parser.add_argument('--eval_transforms_type', default='single_scale', type=str, help='evaluation transforms type')

    # Data loading
    parser.add_argument('--batch_size', default=2, type=int, help='batch size per device')
    parser.add_argument('--num_workers', default=2, type=int, help='number of subprocesses to use for data loading')

    # Evaluation
    parser.add_argument('--eval', action='store_true', help='perform evaluation task instead of training')
    parser.add_argument('--eval_task', default='performance', type=str, help='name of the evaluation task')

    # * FLOPS computation
    parser.add_argument('--flops_samples', default=100, type=int, help='input samples used during FLOPS computation')

    # * Performance
    parser.add_argument('--perf_save_res', action='store_true', help='save results even when having annotations')
    parser.add_argument('--perf_save_tag', default='single_scale', type=str, help='tag used in evaluation file names')
    parser.add_argument('--perf_with_vis', action='store_true', help='also gather visualizations during evaluation')

    # * Visualization
    parser.add_argument('--num_images', default=10, type=int, help='number of images to be visualized')
    parser.add_argument('--image_offset', default=0, type=int, help='image id of first image to be visualized')
    parser.add_argument('--random_offset', action='store_true', help='generate random image offset')
    parser.add_argument('--vis_score_thr', default=0.4, type=float, help='score threshold used during visualization')

    # Architecture
    parser.add_argument('--arch_type', default='bch', type=str, help='type of architecture module')

    # * BVN (Bidirectional Vision Network)
    parser.add_argument('--bvn_step_mode', default='single', choices=['multi', 'single'], help='BVN step mode')
    parser.add_argument('--bvn_sync_heads', action='store_true', help='synchronize heads copies in multi-step mode')

    # * GCT (Graph-Connecting Trees)
    parser.add_argument('--gct_cfg_path', default='', type=str, help='path to GCT architecture config')

    # * MMDetection architecture
    parser.add_argument('--mmdet_arch_cfg_path', default='', type=str, help='path to MMDetection architecture config')

    # Backbone
    parser.add_argument('--backbone_type', default='resnet', type=str, help='type of backbone module')

    # * MMDetection backbone
    parser.add_argument('--mmdet_backbone_cfg_path', default='', type=str, help='path to MMDetection backbone config')

    # * ResNet
    parser.add_argument('--resnet_name', default='resnet50', type=str, help='full name of the desired ResNet model')
    parser.add_argument('--resnet_out_ids', nargs='*', default=[3, 4, 5], type=int, help='ResNet output map indices')
    parser.add_argument('--resnet_dilation', action='store_true', help='whether to use dilation for last ResNet layer')

    # Core
    parser.add_argument('--core_type', default='fpn', type=str, help='type of core module')
    parser.add_argument('--core_ids', nargs='*', default=[3, 4, 5, 6, 7], type=int, help='core feature map indices')

    # * BiFPN (Bidirectional FPN)
    parser.add_argument('--bifpn_feat_size', default=256, type=int, help='feature size of BiFPN output maps')
    parser.add_argument('--bifpn_num_layers', default=7, type=int, help='number of consecutive BiFPN layers')
    parser.add_argument('--bifpn_norm_type', default='batch', type=str, help='type of BiFPN normalization layer')
    parser.add_argument('--bifpn_separable_conv', action='store_true', help='whether to use separable convolutions')

    # * DC (Deformable Core)
    parser.add_argument('--dc_feat_size', default=256, type=int, help='feature size of DC output maps')
    parser.add_argument('--dc_num_layers', default=6, type=int, help='number of consecutive DC layers')

    parser.add_argument('--dc_da_norm', default='layer', type=str, help='normalization type of DA network')
    parser.add_argument('--dc_da_act_fn', default='', type=str, help='activation function of DA network')
    parser.add_argument('--dc_da_version', default=0, type=int, help='version of DA network')
    parser.add_argument('--dc_da_num_heads', default=8, type=int, help='number of DA attention heads')
    parser.add_argument('--dc_da_num_points', default=4, type=int, help='number of DA points')
    parser.add_argument('--dc_da_rad_pts', default=4, type=int, help='number of DA radial points')
    parser.add_argument('--dc_da_ang_pts', default=1, type=int, help='number of DA angular points')
    parser.add_argument('--dc_da_lvl_pts', default=1, type=int, help='number of DA level points')
    parser.add_argument('--dc_da_dup_pts', default=1, type=int, help='number of DA duplicate points')
    parser.add_argument('--dc_da_qk_size', default=256, type=int, help='size of DA query and key features')
    parser.add_argument('--dc_da_val_size', default=256, type=int, help='size of DA value features')
    parser.add_argument('--dc_da_val_with_pos', action='store_true', help='adds position info to DA value features')
    parser.add_argument('--dc_da_norm_z', default=1.0, type=float, help='Z-normalizer of DA sample offsets')

    parser.add_argument('--dc_prior_type', default='location', type=str, help='type of used sample priors')
    parser.add_argument('--dc_prior_factor', default=2.0, type=float, help='factor scaling box-type sample priors')
    parser.add_argument('--dc_scale_encs', action='store_true', help='whether to use scale encodings')
    parser.add_argument('--dc_scale_invariant', action='store_true', help='whether core should be scale invariant')

    # * FPN (Feature Pyramid Network)
    parser.add_argument('--fpn_feat_size', default=256, type=int, help='feature size of FPN output maps')
    parser.add_argument('--fpn_fuse_type', default='sum', choices=['avg', 'sum'], help='FPN fusing operation')

    # * GC (Generalized Core)
    parser.add_argument('--gc_yaml', default='', type=str, help='path to yaml-file with GC specification')

    # * MMDetection core
    parser.add_argument('--mmdet_core_cfg_path', default='', type=str, help='path to MMDetection core config')

    # Heads
    parser.add_argument('--heads', nargs='*', default='', type=str, help='names of desired heads')

    # * Binary segmentation head
    parser.add_argument('--disputed_loss', action='store_true', help='whether to apply loss at disputed positions')
    parser.add_argument('--disputed_beta', default=0.2, type=float, help='threshold used for disputed smooth L1 loss')
    parser.add_argument('--bin_seg_weight', default=1.0, type=float, help='binary segmentation loss weight')

    # * BRD (Base Reinforced Detector) head
    parser.add_argument('--brd_feat_size', default=256, type=int, help='internal feature size of the BRD head')

    parser.add_argument('--brd_num_groups', default=8, type=int, help='number of group normalization groups')
    parser.add_argument('--brd_prior_prob', default=0.01, type=float, help='prior object probability')
    parser.add_argument('--brd_inference_samples', default=100, type=int, help='number of samples during inference')
    parser.add_argument('--brd_policy_layers', default=1, type=int, help='number of policy hidden layers')

    parser.add_argument('--brd_num_heads', default=8, type=int, help='number of decoder attention heads')
    parser.add_argument('--brd_dec_hidden_size', default=1024, type=int, help='feature size in decoder hidden layer')
    parser.add_argument('--brd_dec_layers', default=2, type=int, help='number of decoder layers')

    parser.add_argument('--brd_head_hidden_size', default=256, type=int, help='feature size in head hidden layer')
    parser.add_argument('--brd_head_layers', default=1, type=int, help='number of head hidden layers')
    parser.add_argument('--brd_head_prior_cls_prob', default=0.01, type=float, help='prior class probability')

    parser.add_argument('--brd_inter_loss', action='store_true', help='apply loss on intermediate layer predictions')
    parser.add_argument('--brd_rel_preds', action='store_true', help='predict boxes relative to previous predictions')
    parser.add_argument('--brd_use_all_preds', action='store_true', help='apply loss on all predictions from layer')
    parser.add_argument('--brd_use_lsa', action='store_true', help='use linear sum assignment during loss matching')

    parser.add_argument('--brd_delta_range_xy', default=1.0, type=float, help='range of object location delta')
    parser.add_argument('--brd_delta_range_wh', default=8.0, type=float, help='range of object size delta')

    parser.add_argument('--brd_focal_alpha', default=0.25, type=float, help='BRD head focal alpha value')
    parser.add_argument('--brd_focal_gamma', default=2.0, type=float, help='BRD head focal gamma value')

    parser.add_argument('--brd_reward_weight', default=1.0, type=float, help='BRD head reward weight factor')
    parser.add_argument('--brd_punish_weight', default=0.1, type=float, help='BRD head punishment weight factor')

    parser.add_argument('--brd_cls_rank_weight', default=1.0, type=float, help='classification ranking weight')
    parser.add_argument('--brd_l1_rank_weight', default=5.0, type=float, help='L1 bounding box ranking weight')
    parser.add_argument('--brd_giou_rank_weight', default=2.0, type=float, help='GIoU bounding box ranking weight')

    parser.add_argument('--brd_cls_loss_weight', default=1.0, type=float, help='classification loss weight')
    parser.add_argument('--brd_l1_loss_weight', default=5.0, type=float, help='L1 bounding box loss weight')
    parser.add_argument('--brd_giou_loss_weight', default=2.0, type=float, help='GIoU bounding box loss weight')

    # * Duplicate-Free Detector (DFD) head
    parser.add_argument('--dfd_cls_feat_size', default=256, type=int, help='classification hidden feature size')
    parser.add_argument('--dfd_cls_norm', default='group', type=str, help='normalization type of classification head')
    parser.add_argument('--dfd_cls_prior_prob', default=0.01, type=float, help='prior class probability')
    parser.add_argument('--dfd_cls_kernel_size', default=3, type=int, help='classification hidden layer kernel size')
    parser.add_argument('--dfd_cls_bottle_size', default=64, type=int, help='classification bottleneck feature size')
    parser.add_argument('--dfd_cls_hidden_layers', default=1, type=int, help='number of classification hidden layers')

    parser.add_argument('--dfd_cls_focal_alpha', default=0.25, type=float, help='classification focal alpha value')
    parser.add_argument('--dfd_cls_focal_gamma', default=2.0, type=float, help='classification focal gamma value')
    parser.add_argument('--dfd_cls_weight', default=1e0, type=float, help='classification loss weight')

    parser.add_argument('--dfd_obj_feat_size', default=256, type=int, help='objectness hidden feature size')
    parser.add_argument('--dfd_obj_norm', default='group', type=str, help='normalization type of objectness head')
    parser.add_argument('--dfd_obj_prior_prob', default=0.01, type=float, help='prior object probability')
    parser.add_argument('--dfd_obj_kernel_size', default=3, type=int, help='objectness hidden layer kernel size')
    parser.add_argument('--dfd_obj_bottle_size', default=64, type=int, help='objectness bottleneck feature size')
    parser.add_argument('--dfd_obj_hidden_layers', default=1, type=int, help='number of objectness hidden layers')

    parser.add_argument('--dfd_obj_focal_alpha', default=0.25, type=float, help='objectness focal alpha value')
    parser.add_argument('--dfd_obj_focal_gamma', default=2.0, type=float, help='objectness focal gamma value')
    parser.add_argument('--dfd_obj_weight', default=1e1, type=float, help='objectness loss weight')

    parser.add_argument('--dfd_box_feat_size', default=256, type=int, help='bounding box hidden feature size')
    parser.add_argument('--dfd_box_norm', default='group', type=str, help='normalization type of bounding box head')
    parser.add_argument('--dfd_box_kernel_size', default=3, type=int, help='bounding box hidden layer kernel size')
    parser.add_argument('--dfd_box_bottle_size', default=64, type=int, help='bounding box bottleneck feature size')
    parser.add_argument('--dfd_box_hidden_layers', default=1, type=int, help='number of bounding box hidden layers')

    parser.add_argument('--dfd_box_sl1_beta', default=0.0, type=float, help='bounding box smooth L1 beta value')
    parser.add_argument('--dfd_box_weight', default=2e-1, type=float, help='bounding box loss weight')

    parser.add_argument('--dfd_pos_feat_size', default=64, type=int, help='position encoding feature size')
    parser.add_argument('--dfd_pos_norm', default='', type=str, help='normalization type of position head')
    parser.add_argument('--dfd_pos_kernel_size', default=3, type=int, help='kernel size of position head')
    parser.add_argument('--dfd_pos_bottle_size', default=8, type=int, help='bottleneck feature size of position head')
    parser.add_argument('--dfd_pos_hidden_layers', default=2, type=int, help='number of position head hidden layers')

    parser.add_argument('--dfd_ins_feat_size', default=256, type=int, help='instance hidden feature size')
    parser.add_argument('--dfd_ins_norm', default='group', type=str, help='normalization type of instance head')
    parser.add_argument('--dfd_ins_prior_prob', default=0.01, type=float, help='prior instance probability')
    parser.add_argument('--dfd_ins_kernel_size', default=3, type=int, help='instance hidden layer kernel size')
    parser.add_argument('--dfd_ins_bottle_size', default=64, type=int, help='instance bottleneck feature size')
    parser.add_argument('--dfd_ins_hidden_layers', default=1, type=int, help='number of instance hidden layers')
    parser.add_argument('--dfd_ins_out_size', default=256, type=int, help='instance output feature size')

    parser.add_argument('--dfd_ins_focal_alpha', default=0.25, type=float, help='instance focal alpha value')
    parser.add_argument('--dfd_ins_focal_gamma', default=2.0, type=float, help='instance focal gamma value')
    parser.add_argument('--dfd_ins_weight', default=5e0, type=float, help='instance loss weight')

    parser.add_argument('--dfd_inf_nms_candidates', default=1000, type=int, help='max candidates for inference NMS')
    parser.add_argument('--dfd_inf_nms_threshold', default=0.5, type=float, help='IoU threshold during inference NMS')
    parser.add_argument('--dfd_inf_ins_candidates', default=1000, type=int, help='max candidates for instance head')
    parser.add_argument('--dfd_inf_ins_threshold', default=0.5, type=float, help='instance threshold during inference')
    parser.add_argument('--dfd_inf_max_detections', default=100, type=int, help='max number of inference detections')

    # * Dense Object Discovery (DOD) head
    parser.add_argument('--dod_feat_size', default=256, type=int, help='DOD hidden feature size')
    parser.add_argument('--dod_norm', default='group', type=str, help='normalization type of DOD head')
    parser.add_argument('--dod_kernel_size', default=3, type=int, help='DOD hidden layer kernel size')
    parser.add_argument('--dod_bottle_size', default=64, type=int, help='DOD bottleneck feature size')
    parser.add_argument('--dod_hidden_layers', default=1, type=int, help='number of DOD hidden layers')
    parser.add_argument('--dod_rel_preds', action='store_true', help='make relative DOD predictions')
    parser.add_argument('--dod_prior_prob', default=0.01, type=float, help='prior object probability of DOD network')

    parser.add_argument('--dod_anchor_num_sizes', default=1, type=int, help='number of DOD anchor sizes')
    parser.add_argument('--dod_anchor_scale_factor', default=4.0, type=float, help='DOD anchor scale factor')
    parser.add_argument('--dod_anchor_asp_ratios', nargs='*', default=1.0, type=float, help='DOD anchor aspect ratios')

    parser.add_argument('--dod_sel_mode', default='rel', type=str, help='DOD anchor selection mode')
    parser.add_argument('--dod_sel_abs_thr', default=0.5, type=float, help='DOD absolute anchor threshold')
    parser.add_argument('--dod_sel_rel_thr', default=300, type=int, help='DOD relative anchor threshold')

    parser.add_argument('--dod_tgt_metric', default='iou', type=str, help='DOD anchor-target matching metric')
    parser.add_argument('--dod_tgt_decision', default='rel', type=str, help='DOD target decision maker type')
    parser.add_argument('--dod_tgt_abs_pos', default=0.5, type=float, help='DOD absolute positive target threshold')
    parser.add_argument('--dod_tgt_abs_neg', default=0.3, type=float, help='DOD absolute negative target threshold')
    parser.add_argument('--dod_tgt_rel_pos', default=5, type=int, help='DOD relative positive target threshold')
    parser.add_argument('--dod_tgt_rel_neg', default=10, type=int, help='DOD relative negative target threshold')
    parser.add_argument('--dod_tgt_mode', default='static', type=str, help='DOD target mode')

    parser.add_argument('--dod_loss_type', default='sigmoid_focal', type=str, help='type of loss used by DOD head')
    parser.add_argument('--dod_focal_alpha', default=0.25, type=float, help='DOD focal alpha value')
    parser.add_argument('--dod_focal_gamma', default=2.0, type=float, help='DOD focal gamma value')
    parser.add_argument('--dod_pos_weight', default=1.0, type=float, help='loss term weight for positive DOD targets')
    parser.add_argument('--dod_neg_weight', default=1.0, type=float, help='loss term weight for negative DOD targets')

    parser.add_argument('--dod_pred_num_pos', default=5, type=int, help='number of positive anchors per DOD target')
    parser.add_argument('--dod_pred_max_dets', default=100, type=int, help='maximum number of DOD detections')

    # * General Vision Decoder (GVD) head
    parser.add_argument('--gvd_cfg_path', default='', type=str, help='path to GVD head config')

    # * Map-Based Detector (MBD) head
    parser.add_argument('--mbd_hrae_type', default='one_step_mlp', type=str, help='HRAE network type')
    parser.add_argument('--mbd_hrae_layers', default=1, type=int, help='number of layers of HRAE network')
    parser.add_argument('--mbd_hrae_hidden_size', default=1024, type=int, help='hidden feature size of HRAE network')
    parser.add_argument('--mbd_hrae_norm', default='layer', type=str, help='normalization type of HRAE network')
    parser.add_argument('--mbd_hrae_act_fn', default='relu', type=str, help='activation function of HRAE network')
    parser.add_argument('--mbd_hrae_no_skip', action='store_true', help='remove skip connection of HRAE network')

    parser.add_argument('--mbd_haae_type', default='one_step_mlp', type=str, help='HAAE network type')
    parser.add_argument('--mbd_haae_layers', default=1, type=int, help='number of layers of HAAE network')
    parser.add_argument('--mbd_haae_hidden_size', default=1024, type=int, help='hidden feature size of HAAE network')
    parser.add_argument('--mbd_haae_norm', default='layer', type=str, help='normalization type of HAAE network')
    parser.add_argument('--mbd_haae_act_fn', default='relu', type=str, help='activation function of HAAE network')
    parser.add_argument('--mbd_haae_no_skip', action='store_true', help='remove skip connection of HAAE network')

    parser.add_argument('--mbd_ca_type', default='deformable_attn', type=str, help='CA network type')
    parser.add_argument('--mbd_ca_layers', default=6, type=int, help='number of layers of CA network')
    parser.add_argument('--mbd_ca_norm', default='layer', type=str, help='normalization type of CA network')
    parser.add_argument('--mbd_ca_act_fn', default='', type=str, help='activation function of CA network')
    parser.add_argument('--mbd_ca_version', default=2, type=int, help='version of CA network')
    parser.add_argument('--mbd_ca_num_heads', default=8, type=int, help='number of CA attention heads')
    parser.add_argument('--mbd_ca_num_points', default=4, type=int, help='number of CA points')
    parser.add_argument('--mbd_ca_rad_pts', default=4, type=int, help='number of CA radial points')
    parser.add_argument('--mbd_ca_ang_pts', default=1, type=int, help='number of CA angular points')
    parser.add_argument('--mbd_ca_lvl_pts', default=1, type=int, help='number of CA level points')
    parser.add_argument('--mbd_ca_dup_pts', default=1, type=int, help='number of CA duplicate points')
    parser.add_argument('--mbd_ca_qk_size', default=256, type=int, help='size of CA query and key features')
    parser.add_argument('--mbd_ca_val_size', default=256, type=int, help='size of CA value features')
    parser.add_argument('--mbd_ca_val_with_pos', action='store_true', help='adds position info to CA value features')
    parser.add_argument('--mbd_ca_norm_z', default=1.0, type=float, help='Z-normalizer of CA sample offsets')
    parser.add_argument('--mbd_ca_step_size', default=-1, type=float, help='CA step size relative to normalization')
    parser.add_argument('--mbd_ca_step_norm_xy', default='map', type=str, help='XY-normalizer of CA sample steps')
    parser.add_argument('--mbd_ca_step_norm_z', default=1, type=float, help='Z-normalizer of CA sample steps')
    parser.add_argument('--mbd_ca_num_particles', default=20, type=int, help='number of particles per CA head')

    parser.add_argument('--mbd_match_thr', default=0.5, type=float, help='minimum box IoU for positive matching')

    parser.add_argument('--mbd_loss_gt_seg', action='store_true', help='use ground-truth segmentation during training')
    parser.add_argument('--mbd_loss_seg_types', nargs='*', default='sigmoid_focal', help='segmentation loss types')
    parser.add_argument('--mbd_loss_seg_alpha', default=0.25, type=float, help='segmentation focal alpha value')
    parser.add_argument('--mbd_loss_seg_gamma', default=2.0, type=float, help='segmentation focal gamma value')
    parser.add_argument('--mbd_loss_seg_weights', nargs='*', default=1.0, type=float, help='segmentation loss weights')

    parser.add_argument('--mbd_pred_thr', default=0.6, type=float, help='minimum probability for positive prediction')

    # * Retina head
    parser.add_argument('--ret_feat_size', default=256, type=int, help='internal feature size of the retina head')
    parser.add_argument('--ret_num_convs', default=4, type=int, help='number of retina head convolutions')
    parser.add_argument('--ret_pred_type', default='conv1', choices=['conv1', 'conv3'], help='last prediction module')

    parser.add_argument('--ret_focal_alpha', default=0.25, type=float, help='retina head focal alpha value')
    parser.add_argument('--ret_focal_gamma', default=2.0, type=float, help='retina head focal gamma value')
    parser.add_argument('--ret_smooth_l1_beta', default=0.0, type=float, help='retina head smooth L1 beta value')

    parser.add_argument('--ret_normalizer', default=100.0, type=float, help='initial retina head loss normalizer')
    parser.add_argument('--ret_momentum', default=0.9, type=float, help='momentum factor of retina head loss')

    parser.add_argument('--ret_cls_weight', default=1.0, type=float, help='retina classification weight factor')
    parser.add_argument('--ret_box_weight', default=1.0, type=float, help='retina box regression weight factor')

    parser.add_argument('--ret_score_threshold', default=0.05, type=float, help='retina head test score threshold')
    parser.add_argument('--ret_max_candidates', default=1000, type=int, help='retina head max candidates before NMS')
    parser.add_argument('--ret_nms_threshold', default=0.5, type=float, help='retina head NMS threshold')
    parser.add_argument('--ret_max_detections', default=100, type=int, help='retina head max test detections')

    # * State-Based Detector (SBD) head
    parser.add_argument('--sbd_state_size', default=256, type=int, help='size of SBD object states')
    parser.add_argument('--sbd_state_type', default='rel_static', type=str, help='type of SBD object states')

    parser.add_argument('--sbd_osi_type', default='one_step_mlp', type=str, help='OSI network type')
    parser.add_argument('--sbd_osi_layers', default=1, type=int, help='number of layers of OSI network')
    parser.add_argument('--sbd_osi_hidden_size', default=1024, type=int, help='hidden feature size of OSI network')
    parser.add_argument('--sbd_osi_norm', default='layer', type=str, help='normalization type of OSI network')
    parser.add_argument('--sbd_osi_act_fn', default='relu', type=str, help='activation function of OSI network')
    parser.add_argument('--sbd_osi_skip', action='store_true', help='whether to use skip connection in OSI network')

    parser.add_argument('--sbd_hae_type', default='one_step_mlp', type=str, help='HAE network type')
    parser.add_argument('--sbd_hae_layers', default=1, type=int, help='number of layers of HAE network')
    parser.add_argument('--sbd_hae_hidden_size', default=1024, type=int, help='hidden feature size of HAE network')
    parser.add_argument('--sbd_hae_norm', default='layer', type=str, help='normalization type of HAE network')
    parser.add_argument('--sbd_hae_act_fn', default='relu', type=str, help='activation function of HAE network')
    parser.add_argument('--sbd_hae_no_skip', action='store_true', help='remove skip connection of HAE network')

    parser.add_argument('--sbd_se', action='store_true', help='compute separate scale encoding (SE)')
    parser.add_argument('--sbd_hse_type', default='one_step_mlp', type=str, help='HSE network type')
    parser.add_argument('--sbd_hse_layers', default=1, type=int, help='number of layers of HSE network')
    parser.add_argument('--sbd_hse_hidden_size', default=1024, type=int, help='hidden feature size of HSE network')
    parser.add_argument('--sbd_hse_norm', default='layer', type=str, help='normalization type of HSE network')
    parser.add_argument('--sbd_hse_act_fn', default='relu', type=str, help='activation function of HSE network')
    parser.add_argument('--sbd_hse_no_skip', action='store_true', help='remove skip connection of HSE network')

    parser.add_argument('--sbd_hcls_type', default='one_step_mlp', type=str, help='HCLS network type')
    parser.add_argument('--sbd_hcls_layers', default=1, type=int, help='number of layers of HCLS network')
    parser.add_argument('--sbd_hcls_hidden_size', default=1024, type=int, help='hidden feature size of HCLS network')
    parser.add_argument('--sbd_hcls_out_size', default=256, type=int, help='output feature size of HCLS network')
    parser.add_argument('--sbd_hcls_norm', default='layer', type=str, help='normalization type of HCLS network')
    parser.add_argument('--sbd_hcls_act_fn', default='relu', type=str, help='activation function of HCLS network')
    parser.add_argument('--sbd_hcls_skip', action='store_true', help='whether to use skip connection in HCLS network')
    parser.add_argument('--sbd_cls_freeze_inter', action='store_true', help='freeze intermediate shared CLS network')
    parser.add_argument('--sbd_cls_no_sharing', action='store_true', help='whether to not share CLS network')

    parser.add_argument('--sbd_hbox_type', default='one_step_mlp', type=str, help='HBOX network type')
    parser.add_argument('--sbd_hbox_layers', default=1, type=int, help='number of layers of HBOX network')
    parser.add_argument('--sbd_hbox_hidden_size', default=1024, type=int, help='hidden feature size of HBOX network')
    parser.add_argument('--sbd_hbox_out_size', default=256, type=int, help='output feature size of HBOX network')
    parser.add_argument('--sbd_hbox_norm', default='layer', type=str, help='normalization type of HBOX network')
    parser.add_argument('--sbd_hbox_act_fn', default='relu', type=str, help='activation function of HBOX network')
    parser.add_argument('--sbd_hbox_skip', action='store_true', help='whether to use skip connection in HBOX network')
    parser.add_argument('--sbd_box_freeze_inter', action='store_true', help='freeze intermediate shared BOX network')
    parser.add_argument('--sbd_box_no_sharing', action='store_true', help='whether to not share BOX network')

    parser.add_argument('--sbd_match_mode', default='static', type=str, help='SBD prediction-target matching mode')
    parser.add_argument('--sbd_match_cls_type', default='sigmoid_focal', type=str, help='SBD hungarian cls type')
    parser.add_argument('--sbd_match_cls_alpha', default=0.25, type=float, help='SBD hungarian focal alpha value')
    parser.add_argument('--sbd_match_cls_gamma', default=2.0, type=float, help='SBD hungarian focal gamma value')
    parser.add_argument('--sbd_match_cls_weight', default=2.0, type=float, help='SBD hungarian cls weight')
    parser.add_argument('--sbd_match_box_types', nargs='*', default='iou', help='SBD hungarian box types')
    parser.add_argument('--sbd_match_box_weights', nargs='*', default=1, type=float, help='SBD hungarian box weights')
    parser.add_argument('--sbd_match_static_mode', default='rel', type=str, help='SBD static matching mode')
    parser.add_argument('--sbd_match_static_metric', default='iou', type=str, help='SBD static matching metric')
    parser.add_argument('--sbd_match_abs_pos', default=0.5, type=float, help='SBD absolute positive match threshold')
    parser.add_argument('--sbd_match_abs_neg', default=0.3, type=float, help='SBD absolute negative match threshold')
    parser.add_argument('--sbd_match_rel_pos', default=5, type=int, help='SBD relative positive match threshold')
    parser.add_argument('--sbd_match_rel_neg', default=10, type=int, help='SBD relative negative match threshold')

    parser.add_argument('--sbd_loss_ae_weight', default=1.0, type=float, help='SBD anchor encoding loss weight')
    parser.add_argument('--sbd_loss_apply_freq', default='last', type=str, help='frequency of SBD loss application')
    parser.add_argument('--sbd_loss_bg_weight', default=1.0, type=float, help='SBD classification background weight')
    parser.add_argument('--sbd_loss_cls_type', default='sigmoid_focal', type=str, help='SBD classification loss type')
    parser.add_argument('--sbd_loss_cls_alpha', default=0.25, type=float, help='SBD classification focal alpha value')
    parser.add_argument('--sbd_loss_cls_gamma', default=2.0, type=float, help='SBD classification focal gamma value')
    parser.add_argument('--sbd_loss_cls_weight', default=1.0, type=float, help='SBD classification loss weight')
    parser.add_argument('--sbd_loss_box_types', nargs='*', default='smooth_l1', help='SBD bounding box loss types')
    parser.add_argument('--sbd_loss_box_beta', default=0.0, type=float, help='SBD bounding box smooth L1 beta value')
    parser.add_argument('--sbd_loss_box_weights', nargs='*', default=1.0, type=float, help='SBD box loss weights')

    parser.add_argument('--sbd_pred_dup_removal', default='nms', type=str, help='SBD prediction duplicate removal')
    parser.add_argument('--sbd_pred_nms_candidates', default=1000, type=int, help='SBD NMS candidates')
    parser.add_argument('--sbd_pred_nms_thr', default=0.5, type=float, help='SBD NMS IoU threshold')
    parser.add_argument('--sbd_pred_max_dets', default=100, type=int, help='maximum number of SBD detections')

    parser.add_argument('--sbd_update_types', nargs='*', default='', type=str, help='types of SBD update layers')
    parser.add_argument('--sbd_update_layers', default=6, type=int, help='number of SBD update layers')
    parser.add_argument('--sbd_update_iters', default=1, type=int, help='number of SBD update iterations')

    parser.add_argument('--sbd_ca_type', default='deformable_attn', type=str, help='CA network type')
    parser.add_argument('--sbd_ca_layers', default=1, type=int, help='number of layers of CA network')
    parser.add_argument('--sbd_ca_norm', default='layer', type=str, help='normalization type of CA network')
    parser.add_argument('--sbd_ca_act_fn', default='', type=str, help='activation function of CA network')
    parser.add_argument('--sbd_ca_version', default=0, type=int, help='version of CA network')
    parser.add_argument('--sbd_ca_num_heads', default=8, type=int, help='number of CA attention heads')
    parser.add_argument('--sbd_ca_num_points', default=4, type=int, help='number of CA points')
    parser.add_argument('--sbd_ca_rad_pts', default=4, type=int, help='number of CA radial points')
    parser.add_argument('--sbd_ca_ang_pts', default=1, type=int, help='number of CA angular points')
    parser.add_argument('--sbd_ca_lvl_pts', default=1, type=int, help='number of CA level points')
    parser.add_argument('--sbd_ca_dup_pts', default=1, type=int, help='number of CA duplicate points')
    parser.add_argument('--sbd_ca_qk_size', default=256, type=int, help='size of CA query and key features')
    parser.add_argument('--sbd_ca_val_size', default=256, type=int, help='size of CA value features')
    parser.add_argument('--sbd_ca_val_with_pos', action='store_true', help='adds position info to CA value features')
    parser.add_argument('--sbd_ca_norm_z', default=1.0, type=float, help='Z-normalizer of CA sample offsets')
    parser.add_argument('--sbd_ca_step_size', default=-1, type=float, help='CA step size relative to normalization')
    parser.add_argument('--sbd_ca_step_norm_xy', default='map', type=str, help='XY-normalizer of CA sample steps')
    parser.add_argument('--sbd_ca_step_norm_z', default=1, type=float, help='Z-normalizer of CA sample steps')
    parser.add_argument('--sbd_ca_num_particles', default=20, type=int, help='number of particles per CA head')

    parser.add_argument('--sbd_sa_type', default='self_attn_1d', type=str, help='SA network type')
    parser.add_argument('--sbd_sa_layers', default=1, type=int, help='number of layers of SA network')
    parser.add_argument('--sbd_sa_norm', default='layer', type=str, help='normalization type of SA network')
    parser.add_argument('--sbd_sa_act_fn', default='', type=str, help='activation function of SA network')
    parser.add_argument('--sbd_sa_num_heads', default=8, type=int, help='number of SA attention heads')

    parser.add_argument('--sbd_ffn_type', default='two_step_mlp', type=str, help='FFN network type')
    parser.add_argument('--sbd_ffn_layers', default=1, type=int, help='number of layers of FFN network')
    parser.add_argument('--sbd_ffn_hidden_size', default=1024, type=int, help='hidden feature size of FFN network')
    parser.add_argument('--sbd_ffn_norm', default='layer', type=str, help='normalization type of FFN network')
    parser.add_argument('--sbd_ffn_act_fn', default='relu', type=str, help='activation function of FFN network')

    # * Semantic segmentation head
    parser.add_argument('--bg_weight', default=0.1, type=float, help='weight scaling losses in background positions')
    parser.add_argument('--sem_seg_weight', default=1.0, type=float, help='semantic segmentation loss weight')

    # Optimizer
    parser.add_argument('--max_grad_norm', default=-1, type=float, help='maximum gradient norm during training')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='L2 weight decay coefficient')

    # * Learning rates (General)
    parser.add_argument('--lr_default', default=1e-4, type=float, help='default learning rate')
    parser.add_argument('--lr_backbone', default=1e-5, type=float, help='backbone learning rate')

    # * Learning rates (BCH and BVN)
    parser.add_argument('--lr_core', default=1e-4, type=float, help='core learning rate')

    # * Learning rates (BCH, BVN and GCT)
    parser.add_argument('--lr_heads', default=1e-4, type=float, help='heads learning rate')

    # * Learning rates (Deformable DETR)
    parser.add_argument('--lr_reference_points', default=1e-5, type=float, help='reference points learning rate')

    # * Learning rates (GCT)
    parser.add_argument('--lr_map', default=1e-5, type=float, help='GCT map learning rate')
    parser.add_argument('--lr_graph', default=1e-4, type=float, help='GCT graph learning rate')

    # * Learning rates (MMDetArch)
    parser.add_argument('--lr_neck', default=1e-4, type=float, help='neck learning rate')

    # * Learning rates (MSDA)
    parser.add_argument('--lr_offset', default=1e-5, type=float, help='learning rate of deformable offsets')

    # * Learning rates (PA)
    parser.add_argument('--lr_steps', default=1e-4, type=float, help='PA sample steps learning rate')

    # Scheduler
    parser.add_argument('--epochs', default=36, type=int, help='total number of training epochs')
    parser.add_argument('--lr_drops', nargs='*', default=[27, 33], type=int, help='epochs of learning rate drops')

    return parser


def main(args):
    """
    Function containing the main program script.

    Args:
        args (argparse.Namespace): Command-line arguments.

    Raises:
        ValueError: Error when no output directory could be determined for the 'visualize' evaluation task.
        ValueError: Error when in distributed mode for the 'visualize' evaluation task.
        ValueError: Error when an unknown evaluation task is provided.
    """

    # Initialize distributed mode if needed
    distributed.init_distributed_mode(args)
    print(args)

    # Get datasets and evaluator
    datasets, evaluator = build_dataset(args)

    # Build model and place it on correct device
    device = torch.device(args.device)
    model = build_arch(args)
    model = model.to(device)

    # Try loading checkpoint
    checkpoint_path = ''

    if args.checkpoint_full:
        try:
            checkpoint = torch.load(args.checkpoint_full, map_location='cpu')
            checkpoint_path = args.checkpoint_full
            load_model_only = False
        except FileNotFoundError:
            pass

    if not checkpoint_path and args.checkpoint_model:
        checkpoint = torch.load(args.checkpoint_model, map_location='cpu')
        checkpoint_path = args.checkpoint_model
        load_model_only = True

    # Load model from checkpoint if present
    if checkpoint_path:
        model.load_state_dict(checkpoint['model'])

    # Wrap model into DistributedDataParallel (DDP) if needed
    if args.distributed:
        model = distributed.DistributedDataParallel(model, device_id=args.gpu)

    # Set 'requires_masks' attribute of datasets
    for dataset in datasets.values():
        dataset.requires_masks = args.requires_masks

    # Get shared dataloader keyword arguments
    dataloader_kwargs = {'collate_fn': collate_fn, 'num_workers': args.num_workers, 'pin_memory': True}

    # Get training dataloader if needed
    if not args.eval:
        train_dataset = datasets['train']
        train_sampler = DistributedSampler(train_dataset) if args.distributed else RandomSampler(train_dataset)
        train_batch_sampler = torch.utils.data.BatchSampler(train_sampler, args.batch_size, drop_last=True)
        train_dataloader = DataLoader(train_dataset, batch_sampler=train_batch_sampler, **dataloader_kwargs)

    # Get evaluation dataloader
    eval_dataset = datasets['eval']

    if args.eval and args.eval_task == 'visualize':
        eval_sampler = SubsetSampler(eval_dataset, args.num_images, args.image_offset, args.random_offset)
    elif args.distributed:
        eval_sampler = DistributedSampler(eval_dataset, shuffle=False)
    else:
        eval_sampler = SequentialSampler(eval_dataset)

    eval_dataloader = DataLoader(eval_dataset, args.batch_size, sampler=eval_sampler, **dataloader_kwargs)

    # Get output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    elif checkpoint_path:
        output_dir = Path(checkpoint_path).parent
    else:
        output_dir = None

    # Perform evaluation task if requested
    if args.eval:

        # Compute average number of FLOPS of model and return
        if args.eval_task == 'compute_flops':
            avg_flops = compute_flops(model, eval_dataset, num_samples=args.flops_samples)
            print(f"Average number of FLOPS: {avg_flops: .1f} GFLOPS")
            return

        # Evaluate model performance and return
        elif args.eval_task == 'performance':
            perf_kwargs = {'save_stats': True, 'save_results': args.perf_save_res}
            perf_kwargs = {**perf_kwargs, 'save_tag': f'{args.eval_split}_{args.perf_save_tag}'}
            perf_kwargs = {**perf_kwargs, 'visualize': args.perf_with_vis, 'vis_score_thr': args.vis_score_thr}
            evaluate(model, eval_dataloader, evaluator=evaluator, output_dir=output_dir, **perf_kwargs)
            return

        # Visualize model predictions and return
        elif args.eval_task == 'visualize':
            if output_dir is None:
                error_msg = "The 'visualize' evaluation task requires an output directory, but no output directory "
                error_msg += "was given or could be derived from checkpoint."
                raise ValueError(error_msg)

            if args.distributed:
                error_msg = "Distributed mode is not supported for the 'visualize' evaluation task."
                raise ValueError(error_msg)

            if args.batch_size > 1:
                msg = "It's recommended to use 'batch_size=1' for the 'visualize' evaluation task, so that the printed "
                msg += "loss and analysis dictionaries are image-specific."
                print(msg)

            vis_kwargs = {'print_freq': 1, 'visualize': True, 'vis_score_thr': args.vis_score_thr}
            evaluate(model, eval_dataloader, output_dir=output_dir, **vis_kwargs)
            return

        # Raise error
        else:
            error_msg = f"An unknown evaluation task was provided (got '{args.eval_task}')."
            raise ValueError(error_msg)

    # Get default optimizer and scheduler
    param_families = model.module.get_param_families() if args.distributed else model.get_param_families()
    param_families = ['offset', 'steps', 'reference_points', *param_families, 'default']
    param_dicts = {family: {'params': [], 'lr': getattr(args, f'lr_{family}')} for family in param_families}

    for param_name, parameter in model.named_parameters():
        if parameter.requires_grad:
            for family_name in param_families:
                if family_name in param_name or family_name == 'default':
                    param_dicts[family_name]['params'].append(parameter)
                    break

    for family_name in param_families:
        if len(param_dicts[family_name]['params']) == 0:
            param_dicts.pop(family_name)

        elif param_dicts[family_name]['lr'] <= 0:
            for parameter in param_dicts[family_name]['params']:
                parameter.requires_grad_(False)

            param_dicts.pop(family_name)

    optimizer = torch.optim.AdamW(param_dicts.values(), weight_decay=args.weight_decay)
    optimizer.param_group_names = list(param_dicts.keys())
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.lr_drops)

    # Update default optimizer and/or scheduler based on checkpoint
    if checkpoint_path:
        if load_model_only:
            for group in optimizer.param_groups:
                for p in group['params']:
                    p.grad = torch.zeros_like(p)

            optimizer.step()
            [scheduler.step() for _ in range(checkpoint['epoch'])]

        else:
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])

    # Get start epoch
    start_epoch = checkpoint['epoch']+1 if checkpoint_path else 1

    # Start training timer
    start_time = time.time()

    # Main training loop
    for epoch in range(start_epoch, args.epochs+1):
        train_sampler.set_epoch(epoch) if args.distributed else None
        train_stats = train(model, train_dataloader, optimizer, epoch, max_grad_norm=args.max_grad_norm)
        scheduler.step()

        checkpoint_model = model.module if args.distributed else model
        save_checkpoint(args, epoch, checkpoint_model, optimizer, scheduler)

        eval_stats = evaluate(model, eval_dataloader, evaluator=evaluator, epoch=epoch)
        save_log(args.output_dir, epoch, train_stats, eval_stats)

    # End training timer and report total training time
    total_time = time.time() - start_time
    total_time = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training {args.epochs+1 - start_epoch} epochs finished after {total_time}")


if __name__ == '__main__':
    prog = "python main.py"
    description = "Main training and evaluation script"
    fmt = argparse.MetavarTypeHelpFormatter

    parser = argparse.ArgumentParser(prog=prog, description=description, parents=[get_parser()], formatter_class=fmt)
    main(parser.parse_args())
