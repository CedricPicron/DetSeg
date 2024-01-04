_base_ = '../../fqdet/base/v1.py'

model = dict(
    requires_masks=True,
    head_cfgs=[
        *_base_.model.head_cfgs,
        dict(
            type='ModRoIHead',
            cls_agn_masks=False,
            roi_ext_cfg=dict(
                type='MMDetRoIExtractor',
                in_key='feat_maps',
                boxes_key='roi_boxes',
                ids_key='roi_batch_ids',
                out_key='roi_feats',
                mmdet_roi_ext_cfg=dict(
                    type='mmdet.SingleRoIExtractor',
                    roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
                    out_channels=256,
                    featmap_strides=[4, 8, 16, 32],
                ),
            ),
            mask_logits_cfg=[
                dict(
                    type='QryMapFusion',
                    in_key='roi_feats',
                    qry_key='roi_qry_feats',
                    out_key='roi_feats',
                    pre_cat_cfg=[
                        dict(
                            type='nn.Linear',
                            in_features=256,
                            out_features=256,
                            bias=True,
                        ),
                        dict(
                            type='nn.ReLU',
                            inplace=True,
                        ),
                    ],
                    post_cat_cfg=[
                        dict(
                            type='nn.Conv2d',
                            in_channels=512,
                            out_channels=256,
                            kernel_size=1,
                            bias=True,
                        ),
                        dict(
                            type='nn.ReLU',
                            inplace=True,
                        ),
                        dict(
                            type='nn.Conv2d',
                            in_channels=256,
                            out_channels=256,
                            kernel_size=1,
                            bias=True,
                        ),
                    ],
                ),
                dict(
                    type='StorageApply',
                    in_key='roi_feats',
                    out_key='mask_logits',
                    module_cfg=[
                        dict(
                            type='mmdet.ConvModule',
                            num_layers=4,
                            in_channels=256,
                            out_channels=256,
                            kernel_size=3,
                            padding=1,
                        ),
                        dict(
                            type='nn.ConvTranspose2d',
                            in_channels=256,
                            out_channels=256,
                            kernel_size=2,
                            stride=2,
                        ),
                        dict(
                            type='nn.ReLU',
                            inplace=True,
                        ),
                        dict(
                            type='nn.Conv2d',
                            in_channels=256,
                            out_channels=80,
                            kernel_size=1,
                        ),
                    ],
                ),
                dict(
                    type='SelectClass',
                    in_key='mask_logits',
                    labels_key='roi_labels',
                    out_key='mask_logits',
                ),
                dict(
                    type='StorageCondition',
                    cond_key='is_training',
                    module_cfg=[
                        dict(
                            type='DenseRoIMaskTargets',
                            in_key='mask_logits',
                            boxes_key='roi_boxes',
                            tgt_ids_key='matched_tgt_ids',
                            out_key='mask_targets',
                        ),
                        dict(
                            type='StorageApply',
                            in_key='mask_logits',
                            out_key='mask_loss',
                            storage_kwargs={'mask_targets': 'label'},
                            filter_kwargs=[],
                            module_cfg=dict(
                                type='MaskLoss',
                                mask_loss_cfg=dict(
                                    type='mmdet.CrossEntropyLoss',
                                    use_sigmoid=True,
                                    reduction='sum',
                                    loss_weight=1.0,
                                ),
                            ),
                        ),
                        dict(
                            type='BinaryAccuracy',
                            pred_key='mask_logits',
                            tgt_key='mask_targets',
                            out_key='mask_acc',
                        ),
                        dict(
                            type='StorageTransfer',
                            in_keys=['mask_loss'],
                            dict_key='loss_dict',
                            transfer_mode='out',
                        ),
                        dict(
                            type='StorageTransfer',
                            in_keys=['mask_acc'],
                            dict_key='analysis_dict',
                            transfer_mode='out',
                        ),
                    ],
                ),
            ],
            roi_paster_cfg=dict(
                type='MMDetRoIPaster',
            ),
            dup_attrs=dict(
                type='nms',
                nms_candidates=1000,
                nms_thr=0.5,
            ),
            max_segs=100,
        ),
    ],
)
