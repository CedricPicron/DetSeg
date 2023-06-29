model = dict(
    type='BDE',
    name='bde',
    requires_masks=True,
    decoder_cfgs={
        '0_0': dict(
            type='QryInit',
            qry_init_cfg=dict(
                type='AnchorSelector',
                anchor_cfg=dict(
                    type='AnchorGenerator',
                    map_ids=(2, 3, 4, 5, 6, 7),
                    num_sizes=3,
                    scale_factor=4.0,
                    aspect_ratios=(0.5, 1.0, 2.0),
                ),
                pre_logits_cfg=[
                    dict(
                        type='ProjConv',
                        in_channels=256,
                        out_channels=256,
                        kernel_size=1,
                        norm='group',
                        skip=False,
                    ),
                    dict(
                        type='BottleneckConv',
                        num_layers=1,
                        in_channels=256,
                        bottle_channels=64,
                        out_channels=256,
                        kernel_size=3,
                        norm='group',
                        skip=True,
                    ),
                ],
                sel_attrs=dict(
                    mode='rel',
                    rel_thr=300,
                ),
                post_cfg=dict(
                    type='ModuleSelector',
                    module_cfg=dict(
                        type='OneStepMLP',
                        num_layers=1,
                        in_size=256,
                        out_size=256,
                        norm='layer',
                        act_fn='relu',
                        skip=False,
                    ),
                    num_modules=9,
                ),
                matcher_cfg=dict(
                    type='BoxMatcher',
                    qry_key='anchors',
                    share_qry_boxes=True,
                    match_tgt_labels=list(range(80)),
                    box_metric='iou',
                    sim_matcher_cfg=dict(
                        type='SimMatcher',
                        mode='static',
                        static_mode='rel',
                        rel_pos=5,
                        rel_neg=5,
                        get_top_qry_ids=True,
                        top_limit=15,
                    ),
                ),
                loss_cfg=dict(
                    type='SigmoidFocalLoss',
                    alpha=0.25,
                    gamma=2.0,
                    reduction='sum',
                    weight=1.0,
                ),
                init_prob=0.01,
            ),
            rename_dict={
                'sel_feats': 'qry_feats',
                'sel_boxes': 'prior_boxes',
            },
            qry_ids_key='thing_qry_ids',
        ),
        '0_1': dict(
            type='QryInit',
            qry_init_cfg=dict(
                type='EmbeddingSelector',
                num_embeds=53,
                embed_size=256,
            ),
            rename_dict={
                'embed_feats': 'qry_feats',
            },
            qry_ids_key='stuff_qry_ids',
        ),
        **{f'{i}_{i}': [
            dict(
                type='StorageMasking',
                with_in_tensor=True,
                mask_key='thing_qry_ids',
                mask_in_tensor=True,
                keys_to_mask=['batch_ids'],
                module_cfg=dict(
                    type='BoxCrossAttn',
                    attn_cfg=dict(
                        type='DeformableAttn',
                        in_size=256,
                        sample_size=256,
                        out_size=256,
                        norm='layer',
                        act_fn='',
                        skip=True,
                        version=1,
                        num_heads=8,
                        num_levels=6,
                        num_points=1,
                        val_size=256,
                    ),
                ),
            ),
            dict(
                type='SelfAttn1d',
                in_size=256,
                out_size=256,
                norm='layer',
                act_fn='',
                skip=True,
                num_heads=8,
            ),
            dict(
                type='TwoStepMLP',
                in_size=256,
                hidden_size=2048,
                out_size=256,
                norm1='layer',
                norm2='',
                act_fn1='',
                act_fn2='relu',
                skip=True,
            ),
        ] for i in range(1, 7)},
    },
    encoder_cfgs={},
    head_cfgs={
        '6_0': dict(
            type='BaseClsHead',
            logits_cfg=[
                dict(
                    type='nn.Linear',
                    in_features=256,
                    out_features=134,
                    bias=True,
                ),
            ],
            matcher_cfg=[
                dict(
                    type='TopMatcher',
                    ids_key='sel_top_ids',
                    qry_key='box_logits',
                    top_pos=15,
                    top_neg=15,
                    allow_multi_tgt=False,
                ),
                dict(
                    type='StorageCat',
                    keys_to_cat=[
                        'match_labels',
                        'matched_qry_ids',
                        'matched_tgt_ids',
                    ],
                    module_cfg=dict(
                        type='FixedClsMatcher',
                        qry_cls_labels=list(range(80, 133)),
                        qry_mask_key='stuff_qry_ids',
                    ),
                ),
            ],
            loss_cfg=dict(
                type='SigmoidFocalLoss',
                alpha=0.25,
                gamma=2.0,
                reduction='sum',
                weight=1.0,
            ),
        ),
        '6_1': dict(
            type='StorageMasking',
            with_in_tensor=False,
            mask_key='thing_qry_ids',
            keys_to_mask=[
                'qry_feats',
                'cls_logits',
            ],
            ids_mask_dicts=[
                dict(
                    ids_key='matched_qry_ids',
                    apply_keys=[
                        'matched_qry_ids',
                        'matched_tgt_ids',
                    ],
                ),
            ],
            module_cfg=dict(
                type='BaseBox2dHead',
                logits_cfg=[
                    dict(
                        type='nn.Linear',
                        in_features=256,
                        out_features=256,
                        bias=True,
                    ),
                    dict(
                        type='nn.LayerNorm',
                        normalized_shape=256,
                    ),
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
                    dict(
                        type='nn.Linear',
                        in_features=256,
                        out_features=4,
                        bias=True,
                    ),
                ],
                box_coder_cfg=dict(
                    type='RcnnBoxCoder',
                ),
                get_dets=True,
                dup_attrs=dict(
                    type='nms',
                    nms_candidates=1000,
                    nms_thr=0.65,
                ),
                max_dets=100,
                report_match_stats=True,
                matcher_cfg=None,
                loss_cfg=dict(
                    type='BoxLoss',
                    box_loss_type='mmdet_boxes',
                    box_loss_cfg=dict(
                        type='mmdet.EIoULoss',
                        reduction='sum',
                        loss_weight=1.85,
                    ),
                ),
            ),
        ),
        '6_2': dict(
            type='BaseSegHead',
            qry_dicts=[
                dict(
                    qry_mask_key='thing_qry_ids',
                    keys_to_qry_mask=[
                        'qry_feats',
                        'batch_ids',
                        'cls_logits',
                    ],
                    keys_to_sel=[
                        'seg_mask_logits',
                    ],
                    keys_to_mask=[
                        'qry_feats',
                        'batch_ids',
                        'pred_boxes',
                    ],
                    seg_mask_type='roi',
                    dup_type='box_nms',
                    dup_needs_masks=False,
                    nms_candidates=1000,
                    nms_thr=0.65,
                    name='thing',
                ),
                dict(
                    qry_mask_key='stuff_qry_ids',
                    keys_to_qry_mask=[
                        'qry_feats',
                        'batch_ids',
                        'cls_logits',
                    ],
                    keys_to_sel=[
                        'seg_mask_logits',
                    ],
                    keys_to_mask=[
                        'qry_feats',
                        'batch_ids',
                    ],
                    seg_mask_type='image',
                    name='stuff',
                ),
            ],
            qry_cfg=[
                dict(
                    type='nn.Linear',
                    in_features=256,
                    out_features=256,
                    bias=True,
                ),
                dict(
                    type='nn.LayerNorm',
                    normalized_shape=256,
                ),
                dict(
                    type='nn.Linear',
                    in_features=256,
                    out_features=256,
                    bias=True,
                ),
            ],
            key_cfg=dict(
                type='ApplyAll',
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
                        type='nn.Conv2d',
                        in_channels=256,
                        out_channels=256,
                        kernel_size=1,
                    ),
                ],
            ),
            roi_ext_cfg=dict(
                type='mmdet.SingleRoIExtractor',
                roi_layer=dict(type='RoIAlign', output_size=28, sampling_ratio=0),
                out_channels=256,
                featmap_strides=[4, 8, 16, 32],
            ),
            get_segs=True,
            seg_type='panoptic',
            max_segs=100,
            mask_thr=0.5,
            loss_sample_cfg=dict(
                type='PointRendSampling',
                num_points=12544,
                oversample_ratio=3.0,
                importance_ratio=0.75,
            ),
            loss_cfg=dict(
                type='MaskLoss',
                mask_loss_cfg=dict(
                    type='mmdet.CrossEntropyLoss',
                    use_sigmoid=True,
                    reduction='sum',
                    loss_weight=1.0,
                ),
            ),
        ),
    },
)
