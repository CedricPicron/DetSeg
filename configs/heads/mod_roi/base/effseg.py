_base_ = '../../fqdet/base/v1.py'

model = dict(
    requires_masks=True,
    head_cfgs=[
        *_base_.model.head_cfgs,
        dict(
            type='ModRoIHead',
            cls_agn_masks=True,
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
                map_ids_key='roi_map_ids',
            ),
            mask_logits_cfg=[
                dict(
                    type='PosMapFusion',
                    in_key='roi_feats',
                    out_key='roi_feats',
                    pos_cfg=dict(
                        type='SinePosEncoder2d',
                        feat_size=256,
                    ),
                ),
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
                    out_key='roi_feats',
                    module_cfg=dict(
                        type='mmdet.ConvModule',
                        num_layers=4,
                        in_channels=256,
                        out_channels=256,
                        kernel_size=3,
                        padding=1,
                    ),
                ),
                dict(
                    type='StorageApply',
                    in_key='roi_feats',
                    out_key='mask_logits',
                    module_cfg=[
                        dict(
                            type='nn.Conv2d',
                            in_channels=256,
                            out_channels=256,
                            kernel_size=1,
                        ),
                        dict(
                            type='nn.ReLU',
                            inplace=True,
                        ),
                        dict(
                            type='nn.Conv2d',
                            in_channels=256,
                            out_channels=1,
                            kernel_size=1,
                        ),
                    ],
                ),
                dict(
                    type='StorageApply',
                    in_key='roi_feats',
                    out_key='ref_logits',
                    module_cfg=[
                        dict(
                            type='nn.Conv2d',
                            in_channels=256,
                            out_channels=256,
                            kernel_size=1,
                        ),
                        dict(
                            type='nn.ReLU',
                            inplace=True,
                        ),
                        dict(
                            type='nn.Conv2d',
                            in_channels=256,
                            out_channels=1,
                            kernel_size=1,
                        ),
                    ],
                ),
                dict(
                    type='StorageCondition',
                    cond_key='is_training',
                    module_cfg=[
                        dict(
                            type='StorageTransfer',
                            in_keys=['masks'],
                            dict_key='tgt_dict',
                            out_keys=['tgt_map'],
                            transfer_mode='in',
                        ),
                        dict(
                            type='StorageApply',
                            in_key='tgt_map',
                            out_key='tgt_map',
                            module_cfg=[
                                dict(
                                    type='Float',
                                ),
                                dict(
                                    type='Unsqueeze',
                                    dim=1,
                                ),
                            ],
                        ),
                        dict(
                            type='MMDetRoIExtractor',
                            in_key='tgt_map',
                            boxes_key='roi_boxes',
                            ids_key='matched_tgt_ids',
                            out_key='roi_tgt_map',
                            mmdet_roi_ext_cfg=dict(
                                type='mmdet.SingleRoIExtractor',
                                roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
                                out_channels=1,
                                featmap_strides=[1],
                            ),
                        ),
                        dict(
                            type='BinaryTargets',
                            in_key='roi_tgt_map',
                            out_key='mask_targets',
                            threshold=0.5,
                        ),
                        dict(
                            type='AmbiguityTargets',
                            in_key='roi_tgt_map',
                            out_key='ref_targets',
                            low_bnd=0.0,
                            up_bnd=1.0,
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
                                    loss_weight=0.25,
                                ),
                            ),
                        ),
                        dict(
                            type='StorageApply',
                            in_key='ref_logits',
                            out_key='ref_loss',
                            storage_kwargs={'ref_targets': 'label'},
                            filter_kwargs=[],
                            module_cfg=dict(
                                type='MaskLoss',
                                mask_loss_cfg=dict(
                                    type='mmdet.CrossEntropyLoss',
                                    use_sigmoid=True,
                                    reduction='sum',
                                    loss_weight=0.25,
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
                            type='BinaryAccuracy',
                            pred_key='ref_logits',
                            tgt_key='ref_targets',
                            out_key='ref_acc',
                        ),
                        dict(
                            type='StorageTransfer',
                            in_keys=['mask_loss', 'ref_loss'],
                            out_keys=['mask_loss_0', 'ref_loss_0'],
                            dict_key='loss_dict',
                            transfer_mode='out',
                        ),
                        dict(
                            type='StorageTransfer',
                            in_keys=['mask_acc', 'ref_acc'],
                            out_keys=['mask_acc_0', 'ref_acc_0'],
                            dict_key='analysis_dict',
                            transfer_mode='out',
                        ),
                    ],
                ),
                *[dict(
                    type='StorageAdd',
                    module_key=f'fuse_td_{i}',
                    module_cfg=dict(
                        type='StorageApply',
                        in_key='act_feats',
                        out_key='fuse_feats',
                        module_cfg=[
                            dict(
                                type='nn.Linear',
                                in_features=2**(8-i),
                                out_features=2**(10-i),
                                bias=True,
                            ),
                            dict(
                                type='nn.ReLU',
                                inplace=True,
                            ),
                            dict(
                                type='View',
                                out_shape=(-1, 2**(8-i)),
                            ),
                            dict(
                                type='nn.Linear',
                                in_features=2**(8-i),
                                out_features=2**(8-i),
                                bias=True,
                            ),
                        ],
                    ),
                ) for i in range(3)],
                *[dict(
                    type='StorageAdd',
                    module_key=f'fuse_key_{i}',
                    module_cfg=dict(
                        type='StorageApply',
                        in_key='fuse_feats',
                        out_key='fuse_feats',
                        module_cfg=[
                            dict(
                                type='nn.Linear',
                                in_features=256 + 2**(8-i),
                                out_features=2**(8-i),
                                bias=True,
                            ),
                            dict(
                                type='nn.ReLU',
                                inplace=True,
                            ),
                            dict(
                                type='nn.Linear',
                                in_features=2**(8-i),
                                out_features=2**(8-i),
                                bias=True,
                            ),
                        ],
                    ),
                ) for i in range(3)],
                *[dict(
                    type='StorageAdd',
                    module_key=f'trans_{i}',
                    module_cfg=dict(
                        type='StorageApply',
                        in_key='trans_feats',
                        out_key='trans_feats',
                        module_cfg=[
                            dict(
                                type='nn.Linear',
                                in_features=2**(8-i),
                                out_features=2**(7-i),
                                bias=True,
                            ),
                            dict(
                                type='nn.ReLU',
                                inplace=True,
                            ),
                        ],
                    ),
                ) for i in range(3)],
                *[dict(
                    type='StorageAdd',
                    module_key=f'proc_{i}',
                    module_cfg=dict(
                        type='StorageApply',
                        in_key='act_feats',
                        out_key='act_feats',
                        storage_kwargs={
                            'pas_feats': 'aux_feats',
                            'sps_id_map': 'id_map',
                            'act_roi_ids': 'roi_ids',
                            'act_pos_ids': 'pos_ids',
                        },
                        module_cfg=[
                            dict(
                                type='nn.Linear',
                                in_features=2**(7-i),
                                out_features=2**(7-i),
                                bias=True,
                            ),
                            dict(
                                type='nn.ReLU',
                                inplace=True,
                            ),
                            dict(
                                type='ModuleSum',
                                sub_module_cfgs=[[
                                    dict(
                                        type='IdConv2d',
                                        in_channels=2**(7-i),
                                        out_channels=2**(7-i),
                                        kernel_size=3,
                                        dilation=dilation,
                                    ),
                                    dict(
                                        type='nn.ReLU',
                                        inplace=True,
                                    ),
                                ] for dilation in (1, 3, 5)],
                            ),
                            dict(
                                type='nn.Linear',
                                in_features=2**(7-i),
                                out_features=2**(7-i),
                                bias=True,
                            ),
                            dict(
                                type='nn.ReLU',
                                inplace=True,
                            ),
                            dict(
                                type='nn.Linear',
                                in_features=2**(7-i),
                                out_features=2**(7-i),
                                bias=True,
                            ),
                            dict(
                                type='nn.ReLU',
                                inplace=True,
                            ),
                        ],
                    ),
                ) for i in range(3)],
                *[dict(
                    type='StorageAdd',
                    module_key=f'mask_module_{i}',
                    module_cfg=dict(
                        type='StorageApply',
                        in_key='act_feats',
                        out_key='act_mask_logits',
                        module_cfg=[
                            dict(
                                type='nn.Linear',
                                in_features=2**(7-i),
                                out_features=2**(7-i),
                                bias=True,
                            ),
                            dict(
                                type='nn.ReLU',
                                inplace=True,
                            ),
                            dict(
                                type='nn.Linear',
                                in_features=2**(7-i),
                                out_features=1,
                                bias=True,
                            ),
                        ],
                    ),
                ) for i in range(3)],
                *[dict(
                    type='StorageAdd',
                    module_key=f'ref_module_{i}',
                    module_cfg=dict(
                        type='StorageApply',
                        in_key='act_feats',
                        out_key='ref_logits',
                        module_cfg=[
                            dict(
                                type='nn.Linear',
                                in_features=2**(7-i),
                                out_features=2**(7-i),
                                bias=True,
                            ),
                            dict(
                                type='nn.ReLU',
                                inplace=True,
                            ),
                            dict(
                                type='nn.Linear',
                                in_features=2**(7-i),
                                out_features=1,
                                bias=True,
                            ),
                            dict(
                                type='Squeeze',
                                dim=1,
                            ),
                        ],
                    ),
                ) for i in range(3)],
                dict(
                    type='MapToSps',
                    in_key='roi_feats',
                    out_act_key='act_feats',
                    out_pas_key='pas_feats',
                    out_id_key='sps_id_map',
                    out_grp_key='act_roi_ids',
                    out_pos_key='act_pos_ids',
                ),
                dict(
                    type='StorageApply',
                    in_key='roi_batch_ids',
                    out_key='act_batch_ids',
                    module_cfg=[
                        dict(
                            type='Unsqueeze',
                            dim=1,
                        ),
                        dict(
                            type='Expand',
                            size=[-1, 196],
                        ),
                        dict(
                            type='nn.Flatten',
                            start_dim=0,
                            end_dim=1,
                        ),
                    ],
                ),
                dict(
                    type='StorageApply',
                    in_key='roi_map_ids',
                    out_key='act_map_ids',
                    module_cfg=[
                        dict(
                            type='Unsqueeze',
                            dim=1,
                        ),
                        dict(
                            type='Expand',
                            size=[-1, 196],
                        ),
                        dict(
                            type='nn.Flatten',
                            start_dim=0,
                            end_dim=1,
                        ),
                    ],
                ),
                dict(
                    type='StorageApply',
                    in_key='ref_logits',
                    out_key='ref_logits',
                    module_cfg=[
                        dict(
                            type='Permute',
                            dims=[0, 2, 3, 1],
                        ),
                        dict(
                            type='nn.Flatten',
                            start_dim=0,
                            end_dim=3,
                        ),
                    ],
                ),
                dict(
                    type='StorageIterate',
                    num_iters=3,
                    iter_key='iter_id',
                    module_cfg=[
                        dict(
                            type='RefineBool',
                            in_key='act_feats',
                            out_key='ref_bool',
                            num_refines=10000,
                        ),
                        dict(
                            type='StorageCondition',
                            cond_key='ref_bool',
                            module_cfg=[
                                dict(
                                    type='Topk',
                                    in_key='ref_logits',
                                    out_ids_key='ref_ids',
                                    topk_kwargs=dict(
                                        k=10000,
                                        sorted=False,
                                    ),
                                ),
                                dict(
                                    type='IdsToMask',
                                    in_key='ref_ids',
                                    size_key='ref_logits',
                                    out_key='ref_mask',
                                ),
                                dict(
                                    type='SpsMask',
                                    in_act_key='act_feats',
                                    in_pas_key='pas_feats',
                                    in_id_key='sps_id_map',
                                    in_grp_key='act_roi_ids',
                                    in_pos_key='act_pos_ids',
                                    mask_key='ref_mask',
                                    out_act_key='act_feats',
                                    out_pas_key='pas_feats',
                                    out_id_key='sps_id_map',
                                    out_grp_key='act_roi_ids',
                                    out_pos_key='act_pos_ids',
                                ),
                                dict(
                                    type='GetItemStorage',
                                    in_key='act_batch_ids',
                                    index_key='ref_mask',
                                    out_key='act_batch_ids'
                                ),
                                dict(
                                    type='GetItemStorage',
                                    in_key='act_map_ids',
                                    index_key='ref_mask',
                                    out_key='act_map_ids'
                                ),
                            ],
                        ),
                        dict(
                            type='StorageGetApply',
                            module_key='fuse_td',
                            id_key='iter_id',
                        ),
                        dict(
                            type='SpsUpsample',
                            in_act_key='act_feats',
                            in_pas_key='pas_feats',
                            in_id_key='sps_id_map',
                            in_grp_key='act_roi_ids',
                            in_pos_key='act_pos_ids',
                            out_act_key='act_feats',
                            out_pas_key='pas_feats',
                            out_id_key='sps_id_map',
                            out_grp_key='act_roi_ids',
                            out_pos_key='act_pos_ids',
                        ),
                        dict(
                            type='Add',
                            in_keys=['act_feats', 'fuse_feats'],
                            out_key='act_feats',
                        ),
                        dict(
                            type='IdsToPts2d',
                            in_key='act_pos_ids',
                            size_key='sps_id_map',
                            out_key='act_pos_xy',
                        ),
                        dict(
                            type='StorageApply',
                            in_key='act_pos_xy',
                            out_key='act_pos_xy',
                            module_cfg=dict(
                                type='Unsqueeze',
                                dim=1,
                            ),
                        ),
                        dict(
                            type='GetItemStorage',
                            in_key='roi_boxes',
                            index_key='act_roi_ids',
                            out_key='act_boxes',
                        ),
                        dict(
                            type='BoxToImgPts',
                            in_key='act_pos_xy',
                            boxes_key='act_boxes',
                            out_key='act_pos_xy',
                        ),
                        dict(
                            type='StorageApply',
                            in_key='act_batch_ids',
                            out_key='act_batch_ids',
                            module_cfg=dict(
                                type='RepeatInterleave',
                                repeats=4,
                                dim=0,
                            ),
                        ),
                        dict(
                            type='StorageApply',
                            in_key='act_map_ids',
                            out_key='act_map_ids',
                            module_cfg=[
                                dict(
                                    type='AddValue',
                                    value=-1,
                                ),
                                dict(
                                    type='Clamp',
                                    min=0,
                                ),
                                dict(
                                    type='RepeatInterleave',
                                    repeats=4,
                                    dim=0,
                                ),
                            ]
                        ),
                        dict(
                            type='MapsSampler2d',
                            in_maps_key='feat_maps',
                            in_pts_key='act_pos_xy',
                            batch_ids_key='act_batch_ids',
                            map_ids_key='act_map_ids',
                            out_key='fuse_feats',
                            grid_sample_kwargs=dict(
                                mode='bilinear',
                                align_corners=False,
                            ),
                        ),
                        dict(
                            type='StorageApply',
                            in_key='fuse_feats',
                            out_key='fuse_feats',
                            module_cfg=dict(
                                type='Squeeze',
                                dim=1,
                            ),
                        ),
                        dict(
                            type='Cat',
                            in_keys=['act_feats', 'fuse_feats'],
                            out_key='fuse_feats',
                            dim=1,
                        ),
                        dict(
                            type='StorageGetApply',
                            module_key='fuse_key',
                            id_key='iter_id',
                        ),
                        dict(
                            type='Add',
                            in_keys=['act_feats', 'fuse_feats'],
                            out_key='act_feats',
                        ),
                        dict(
                            type='StorageCopy',
                            in_key='act_feats',
                            out_key='trans_feats',
                            copy_type='assign',
                        ),
                        dict(
                            type='StorageGetApply',
                            module_key='trans',
                            id_key='iter_id',
                        ),
                        dict(
                            type='StorageCopy',
                            in_key='trans_feats',
                            out_key='act_feats',
                            copy_type='assign',
                        ),
                        dict(
                            type='StorageCopy',
                            in_key='pas_feats',
                            out_key='trans_feats',
                            copy_type='assign',
                        ),
                        dict(
                            type='StorageGetApply',
                            module_key='trans',
                            id_key='iter_id',
                        ),
                        dict(
                            type='StorageCopy',
                            in_key='trans_feats',
                            out_key='pas_feats',
                            copy_type='assign',
                        ),
                        dict(
                            type='StorageGetApply',
                            module_key='proc',
                            id_key='iter_id',
                        ),
                        dict(
                            type='StorageGetApply',
                            module_key='mask_module',
                            id_key='iter_id',
                        ),
                        dict(
                            type='StorageGetApply',
                            module_key='ref_module',
                            id_key='iter_id',
                        ),
                        dict(
                            type='StorageCondition',
                            cond_key='is_inference',
                            module_cfg=[
                                dict(
                                    type='StorageApply',
                                    in_key='mask_logits',
                                    out_key='mask_logits',
                                    module_cfg=dict(
                                        type='Interpolate',
                                        scale_factor=2.0,
                                        mode='bilinear',
                                        align_corners=False,
                                    ),
                                ),
                                dict(
                                    type='GridInsert2d',
                                    in_key='mask_logits',
                                    grp_key='act_roi_ids',
                                    grid_key='act_pos_ids',
                                    feats_key='act_mask_logits',
                                ),
                            ],
                        ),
                    ],
                ),
            ],
            roi_paster_cfg=dict(
                type='MMDetRoIPaster',
                in_key='mask_scores',
                boxes_key='roi_boxes',
                out_key='mask_scores',
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
