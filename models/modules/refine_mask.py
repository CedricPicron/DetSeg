"""
Collection of RefineMask functions and modules copied from https://github.com/zhanggang001/RefineMask.

Only slight modifications were made to fit the code inside the 120 line length limit.
"""

from mmcv.cnn import build_upsample_layer, ConvModule
from mmcv.ops import SimpleRoIAlign
from mmdet.models.losses.cross_entropy_loss import binary_cross_entropy
from mmdet.models.roi_heads.mask_heads.fcn_mask_head import BYTES_PER_FLOAT, _do_paste_mask, GPU_MEM_LIMIT
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

from models.build import build_model, MODELS


def generate_block_target(mask_target, boundary_width=3):
    mask_target = mask_target.float()

    # boundary region
    kernel_size = 2 * boundary_width + 1
    laplacian_kernel = - torch.ones(1, 1, kernel_size, kernel_size).to(
        dtype=torch.float32, device=mask_target.device).requires_grad_(False)
    laplacian_kernel[0, 0, boundary_width, boundary_width] = kernel_size ** 2 - 1

    pad_target = F.pad(mask_target.unsqueeze(1), (boundary_width, boundary_width, boundary_width, boundary_width),
                       "constant", 0)

    # pos_boundary
    pos_boundary_targets = F.conv2d(pad_target, laplacian_kernel, padding=0)
    pos_boundary_targets = pos_boundary_targets.clamp(min=0) / float(kernel_size ** 2)
    pos_boundary_targets[pos_boundary_targets > 0.1] = 1
    pos_boundary_targets[pos_boundary_targets <= 0.1] = 0
    pos_boundary_targets = pos_boundary_targets.squeeze(1)

    # neg_boundary
    neg_boundary_targets = F.conv2d(1 - pad_target, laplacian_kernel, padding=0)
    neg_boundary_targets = neg_boundary_targets.clamp(min=0) / float(kernel_size ** 2)
    neg_boundary_targets[neg_boundary_targets > 0.1] = 1
    neg_boundary_targets[neg_boundary_targets <= 0.1] = 0
    neg_boundary_targets = neg_boundary_targets.squeeze(1)

    # generate block target
    block_target = torch.zeros_like(mask_target).long().requires_grad_(False)
    boundary_inds = (pos_boundary_targets + neg_boundary_targets) > 0
    foreground_inds = (mask_target - pos_boundary_targets) > 0
    block_target[boundary_inds] = 1
    block_target[foreground_inds] = 2
    return block_target


@MODELS.register_module()
class BARCrossEntropyLoss(nn.Module):

    def __init__(self,
                 stage_instance_loss_weight=[1.0, 1.0, 1.0, 1.0],
                 boundary_width=2,
                 start_stage=1):
        super(BARCrossEntropyLoss, self).__init__()

        self.stage_instance_loss_weight = stage_instance_loss_weight
        self.boundary_width = boundary_width
        self.start_stage = start_stage

    def forward(self, stage_instance_preds, stage_instance_targets):
        loss_mask_set = []
        for idx in range(len(stage_instance_preds)):
            instance_pred, instance_target = stage_instance_preds[idx].squeeze(1), stage_instance_targets[idx]
            if idx <= self.start_stage:
                loss_mask = binary_cross_entropy(instance_pred, instance_target)
                loss_mask_set.append(loss_mask)
                pre_pred = instance_pred.sigmoid() >= 0.5

            else:
                pre_boundary = generate_block_target(pre_pred.float(), boundary_width=self.boundary_width) == 1
                boundary_region = pre_boundary.unsqueeze(1)

                target_boundary = generate_block_target(
                    stage_instance_targets[idx - 1].float(), boundary_width=self.boundary_width) == 1
                boundary_region = boundary_region | target_boundary.unsqueeze(1)

                boundary_region = F.interpolate(
                    boundary_region.float(),
                    instance_pred.shape[-2:], mode='bilinear', align_corners=True)
                boundary_region = (boundary_region >= 0.5).squeeze(1)

                loss_mask = F.binary_cross_entropy_with_logits(instance_pred, instance_target, reduction='none')
                loss_mask = loss_mask[boundary_region].sum() / boundary_region.sum().clamp(min=1).float()
                loss_mask_set.append(loss_mask)

                # generate real mask pred, set boundary width as 1, same as inference
                pre_boundary = generate_block_target(pre_pred.float(), boundary_width=1) == 1

                pre_boundary = F.interpolate(
                    pre_boundary.unsqueeze(1).float(),
                    instance_pred.shape[-2:], mode='bilinear', align_corners=True) >= 0.5

                pre_pred = F.interpolate(
                    stage_instance_preds[idx - 1],
                    instance_pred.shape[-2:], mode='bilinear', align_corners=True)

                pre_pred[pre_boundary] = stage_instance_preds[idx][pre_boundary]
                pre_pred = pre_pred.squeeze(1).sigmoid() >= 0.5

        assert len(self.stage_instance_loss_weight) == len(loss_mask_set)
        loss_instance = sum([weight * loss for weight, loss in zip(self.stage_instance_loss_weight, loss_mask_set)])

        return loss_instance


class MultiBranchFusion(nn.Module):

    def __init__(self, feat_dim, dilations=[1, 3, 5]):
        super(MultiBranchFusion, self).__init__()

        for idx, dilation in enumerate(dilations):
            self.add_module(f'dilation_conv_{idx + 1}', ConvModule(
                feat_dim, feat_dim, kernel_size=3, padding=dilation, dilation=dilation))

        self.merge_conv = ConvModule(feat_dim, feat_dim, kernel_size=1, act_cfg=None)

    def forward(self, x):
        feat_1 = self.dilation_conv_1(x)
        feat_2 = self.dilation_conv_2(x)
        feat_3 = self.dilation_conv_3(x)
        out_feat = self.merge_conv(feat_1 + feat_2 + feat_3)
        return out_feat


class MultiBranchFusionAvg(MultiBranchFusion):

    def forward(self, x):
        feat_1 = self.dilation_conv_1(x)
        feat_2 = self.dilation_conv_2(x)
        feat_3 = self.dilation_conv_3(x)
        feat_4 = F.avg_pool2d(x, int(x.shape[-1]))
        out_feat = self.merge_conv(feat_1 + feat_2 + feat_3 + feat_4)
        return out_feat


@MODELS.register_module()
class SimpleSFMStage(nn.Module):

    def __init__(self,
                 semantic_in_channel=256,
                 semantic_out_channel=256,
                 instance_in_channel=256,
                 instance_out_channel=256,
                 fusion_type='MultiBranchFusion',
                 dilations=[1, 3, 5],
                 out_size=14,
                 num_classes=80,
                 semantic_out_stride=4,
                 upsample_cfg=dict(type='bilinear', scale_factor=2)):
        super(SimpleSFMStage, self).__init__()

        self.semantic_out_stride = semantic_out_stride
        self.num_classes = num_classes

        # for extracting instance-wise semantic feats
        self.semantic_transform_in = nn.Conv2d(semantic_in_channel, semantic_out_channel, 1)
        self.semantic_roi_extractor = SimpleRoIAlign(output_size=out_size, spatial_scale=1.0/semantic_out_stride)

        fuse_in_channel = instance_in_channel + semantic_out_channel + 1
        self.fuse_conv = nn.ModuleList([
            nn.Conv2d(fuse_in_channel, instance_in_channel, 1),
            globals()[fusion_type](instance_in_channel, dilations=dilations)])

        self.fuse_transform_out = nn.Conv2d(instance_in_channel, instance_out_channel - 1, 1)
        self.upsample = build_upsample_layer(upsample_cfg.copy())
        self.relu = nn.ReLU(inplace=True)

        self._init_weights()

    def _init_weights(self):
        for m in [self.semantic_transform_in, self.fuse_transform_out]:
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(m.bias, 0)

        for m in self.fuse_conv:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, instance_feats, instance_logits, semantic_feat, rois, upsample=True):
        # instance-wise semantic feats
        semantic_feat = self.relu(self.semantic_transform_in(semantic_feat))
        ins_semantic_feats = self.semantic_roi_extractor(semantic_feat, rois)

        # fuse instance feats & instance masks & semantic feats
        concat_tensors = [instance_feats, ins_semantic_feats, instance_logits.sigmoid()]
        fused_feats = torch.cat(concat_tensors, dim=1)
        for conv in self.fuse_conv:
            fused_feats = self.relu(conv(fused_feats))

        # concat instance masks with fused feats again
        fused_feats = self.relu(self.fuse_transform_out(fused_feats))
        fused_feats = torch.cat([fused_feats, instance_logits.sigmoid()], dim=1)
        fused_feats = self.upsample(fused_feats) if upsample else fused_feats
        return fused_feats


@MODELS.register_module()
class SimpleRefineMaskHead(nn.Module):

    def __init__(self,
                 num_convs_instance=2,
                 num_convs_semantic=4,
                 conv_in_channels_instance=256,
                 conv_in_channels_semantic=256,
                 conv_kernel_size_instance=3,
                 conv_kernel_size_semantic=3,
                 conv_out_channels_instance=256,
                 conv_out_channels_semantic=256,
                 conv_cfg=None,
                 norm_cfg=None,
                 fusion_type='MultiBranchFusionAvg',
                 dilations=[1, 3, 5],
                 semantic_out_stride=4,
                 stage_num_classes=[80, 80, 80, 80],
                 stage_sup_size=[14, 28, 56, 112],
                 pre_upsample_last_stage=False,  # if True, upsample features and then compute logits
                 upsample_cfg=dict(type='bilinear', scale_factor=2),
                 loss_cfg=dict(
                    type='BARCrossEntropyLoss',
                    stage_instance_loss_weight=[0.25, 0.5, 0.75, 1.0],
                    boundary_width=2,
                    start_stage=1)):
        super(SimpleRefineMaskHead, self).__init__()

        self.num_convs_instance = num_convs_instance
        self.conv_kernel_size_instance = conv_kernel_size_instance
        self.conv_in_channels_instance = conv_in_channels_instance
        self.conv_out_channels_instance = conv_out_channels_instance

        self.num_convs_semantic = num_convs_semantic
        self.conv_kernel_size_semantic = conv_kernel_size_semantic
        self.conv_in_channels_semantic = conv_in_channels_semantic
        self.conv_out_channels_semantic = conv_out_channels_semantic

        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        self.semantic_out_stride = semantic_out_stride
        self.stage_sup_size = stage_sup_size
        self.stage_num_classes = stage_num_classes
        self.pre_upsample_last_stage = pre_upsample_last_stage

        self._build_conv_layer('instance')
        self._build_conv_layer('semantic')
        self.loss_func = build_model(loss_cfg)

        assert len(self.stage_sup_size) > 1
        self.stages = nn.ModuleList()
        out_channel = conv_out_channels_instance
        stage_out_channels = [conv_out_channels_instance]
        for idx, out_size in enumerate(self.stage_sup_size[:-1]):
            in_channel = out_channel
            out_channel = in_channel // 2

            new_stage = SimpleSFMStage(
                semantic_in_channel=conv_out_channels_semantic,
                semantic_out_channel=in_channel,
                instance_in_channel=in_channel,
                instance_out_channel=out_channel,
                fusion_type=fusion_type,
                dilations=dilations,
                out_size=out_size,
                num_classes=self.stage_num_classes[idx],
                semantic_out_stride=semantic_out_stride,
                upsample_cfg=upsample_cfg)

            self.stages.append(new_stage)
            stage_out_channels.append(out_channel)

        self.stage_instance_logits = nn.ModuleList([
            nn.Conv2d(stage_out_channels[idx], num_classes, 1)
            for idx, num_classes in enumerate(self.stage_num_classes)])
        self.relu = nn.ReLU(inplace=True)

    def _build_conv_layer(self, name):
        out_channels = getattr(self, f'conv_out_channels_{name}')
        conv_kernel_size = getattr(self, f'conv_kernel_size_{name}')

        convs = []
        for i in range(getattr(self, f'num_convs_{name}')):
            in_channels = getattr(self, f'conv_in_channels_{name}') if i == 0 else out_channels
            conv = ConvModule(in_channels, out_channels, conv_kernel_size, dilation=1, padding=1)
            convs.append(conv)

        self.add_module(f'{name}_convs', nn.ModuleList(convs))

    def init_weights(self):
        for m in self.stage_instance_logits:
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(m.bias, 0)

    def forward(self, instance_feats, semantic_feat, rois, roi_labels):
        for conv in self.instance_convs:
            instance_feats = conv(instance_feats)

        for conv in self.semantic_convs:
            semantic_feat = conv(semantic_feat)

        stage_instance_preds = []
        for idx, stage in enumerate(self.stages):
            instance_logits = self.stage_instance_logits[idx](instance_feats)
            instance_logits = instance_logits[torch.arange(len(rois)), roi_labels][:, None]
            upsample_flag = self.pre_upsample_last_stage or idx < len(self.stages) - 1
            instance_feats = stage(instance_feats, instance_logits, semantic_feat, rois, upsample_flag)
            stage_instance_preds.append(instance_logits)

        # if use class-agnostic classifier for the last stage
        if self.stage_num_classes[-1] == 1:
            roi_labels = roi_labels.clamp(max=0)

        instance_preds = self.stage_instance_logits[-1](instance_feats)[torch.arange(len(rois)), roi_labels][:, None]
        if not self.pre_upsample_last_stage:
            instance_preds = F.interpolate(instance_preds, scale_factor=2, mode='bilinear', align_corners=True)
        stage_instance_preds.append(instance_preds)

        return stage_instance_preds

    def get_targets(self, pos_bboxes_list, pos_assigned_gt_inds_list, gt_masks_list):

        def _generate_instance_targets(pos_proposals, pos_assigned_gt_inds, gt_masks, mask_size=None):
            device = pos_proposals.device
            proposals_np = pos_proposals.cpu().numpy()
            maxh, maxw = gt_masks.height, gt_masks.width
            proposals_np[:, [0, 2]] = np.clip(proposals_np[:, [0, 2]], 0, maxw)
            proposals_np[:, [1, 3]] = np.clip(proposals_np[:, [1, 3]], 0, maxh)
            pos_assigned_gt_inds = pos_assigned_gt_inds.cpu().numpy()

            # crop and resize the instance mask
            resize_masks = gt_masks.crop_and_resize(
                proposals_np, _pair(mask_size), inds=pos_assigned_gt_inds, device=device,).to_ndarray()
            instance_targets = torch.from_numpy(resize_masks).float().to(device)  # Tensor(Bitmaps)

            return instance_targets

        stage_instance_targets_list = [[] for _ in range(len(self.stage_sup_size))]
        for pos_bboxes, pos_gt_ids, gt_masks in zip(pos_bboxes_list, pos_assigned_gt_inds_list, gt_masks_list):
            stage_instance_targets = [_generate_instance_targets(
                pos_bboxes, pos_gt_ids, gt_masks, mask_size=mask_size) for mask_size in self.stage_sup_size]
            for stage_idx in range(len(self.stage_sup_size)):
                stage_instance_targets_list[stage_idx].append(stage_instance_targets[stage_idx])
        stage_instance_targets = [torch.cat(targets) for targets in stage_instance_targets_list]

        return stage_instance_targets

    def loss(self, stage_instance_preds, stage_instance_targets):
        loss_instance = self.loss_func(stage_instance_preds, stage_instance_targets)
        return dict(loss_instance=loss_instance)

    def get_seg_masks(self, mask_pred, det_bboxes, det_labels, rcnn_test_cfg, ori_shape, scale_factor, rescale):
        mask_pred = mask_pred.sigmoid()
        device = mask_pred[0].device
        bboxes = det_bboxes[:, :4]
        labels = det_labels

        if rescale:
            img_h, img_w = ori_shape[:2]
        else:
            img_h = np.round(ori_shape[0] * scale_factor).astype(np.int32)
            img_w = np.round(ori_shape[1] * scale_factor).astype(np.int32)
            scale_factor = 1.0

        if not isinstance(scale_factor, (float, torch.Tensor)):
            scale_factor = bboxes.new_tensor(scale_factor)
        bboxes = bboxes / scale_factor

        N = len(mask_pred)
        # The actual implementation split the input into chunks,
        # and paste them chunk by chunk.
        if device.type == 'cpu':
            # CPU is most efficient when they are pasted one by one with
            # skip_empty=True, so that it performs minimal number of
            # operations.
            num_chunks = N
        else:
            # GPU benefits from parallelism for larger chunks,
            # but may have memory issue
            num_chunks = int(
                np.ceil(N * img_h * img_w * BYTES_PER_FLOAT / GPU_MEM_LIMIT))
            assert (num_chunks <= N), 'Default GPU_MEM_LIMIT is too small; try increasing it'
        chunks = torch.chunk(torch.arange(N, device=device), num_chunks)

        threshold = rcnn_test_cfg.mask_thr_binary
        im_mask = torch.zeros(
            N,
            img_h,
            img_w,
            device=device,
            dtype=torch.bool if threshold >= 0 else torch.uint8)

        if mask_pred.shape[1] > 1:
            mask_pred = mask_pred[range(N), labels][:, None]

        for inds in chunks:
            masks_chunk, spatial_inds = _do_paste_mask(
                mask_pred[inds],
                bboxes[inds],
                img_h,
                img_w,
                skip_empty=device.type == 'cpu')

            if threshold >= 0:
                masks_chunk = (masks_chunk >= threshold).to(dtype=torch.bool)
            else:
                # for visualization and debugging
                masks_chunk = (masks_chunk * 255).to(dtype=torch.uint8)

            im_mask[(inds, ) + spatial_inds] = masks_chunk

        im_segms = [im_mask[i].cpu().numpy() for i in range(N)]
        return im_segms
