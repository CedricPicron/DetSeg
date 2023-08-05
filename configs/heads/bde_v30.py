model = dict(
    type='BDE',
    name='bde',
    requires_masks=True,
    pos_cfg=dict(
        type='LearnedPos2d',
        pos_cfg=[
            dict(
                type='SinePosEncoder2d',
                feat_size=256,
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
        ],
    ),
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
        ),
        **{f'{i}_0': [
            dict(
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
        '7_0': dict(
            type='GetPosFromBoxes',
            boxes_key='pred_boxes',
            pos_module_key='pos_module',
            pos_feats_key='qry_pos_feats',
        ),
    },
    encoder_cfgs={
        '7_0': dict(
            type='GetPosFromMaps',
            pos_module_key='pos_module',
            pos_feats_key='key_pos_feats',
        ),
        '7_1': dict(
            type='Sparse3d',
            seq_feats_key='key_feats',
            act_map_ids=[1, 2, 3, 4, 5],
            pos_feats_key='key_pos_feats',
            get_pas_feats=False,
            get_id_maps=False,
            sparse_cfg=[
                dict(
                    type='CrossAttn1d',
                    in_size=256,
                    norm_cfg=dict(
                        type='nn.LayerNorm',
                        normalized_shape=256,
                    ),
                    qry_pos_key='act_pos_feats',
                    kv_feats_key='qry_feats',
                    key_pos_key='qry_pos_feats',
                    kv_size=256,
                    num_heads=8,
                    out_size=256,
                    skip=True,
                ),
            ],
        ),
        '8_0': dict(
            type='Sparse3d',
            seq_feats_key='key_feats',
            act_mask_key='seg_img_unc_mask',
            pos_feats_key='key_pos_feats',
            get_pas_feats=False,
            get_id_maps=False,
            sparse_cfg=[
                dict(
                    type='CrossAttn1d',
                    in_size=256,
                    norm_cfg=dict(
                        type='nn.LayerNorm',
                        normalized_shape=256,
                    ),
                    qry_pos_key='act_pos_feats',
                    kv_feats_key='qry_feats',
                    key_pos_key='qry_pos_feats',
                    kv_size=256,
                    num_heads=8,
                    out_size=256,
                    skip=True,
                ),
            ],
        ),
    },
    head_cfgs={
        '6_0': dict(
            type='BaseClsHead',
            logits_cfg=[
                dict(
                    type='nn.Linear',
                    in_features=256,
                    out_features=81,
                    bias=True,
                ),
            ],
            matcher_cfg=dict(
                type='TopMatcher',
                ids_key='sel_top_ids',
                qry_key='box_logits',
                top_pos=15,
                top_neg=15,
                allow_multi_tgt=False,
            ),
            soft_label_type='box_iou',
            loss_cfg=dict(
                type='mmdet.QualityFocalLoss',
                use_sigmoid=True,
                beta=2.0,
                reduction='sum',
                loss_weight=0.5,
                activated=False,
            ),
        ),
        '6_1': dict(
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
                    loss_weight=10.0,
                ),
            ),
            loss_reduction='tgt_sum',
        ),
        '7_0': dict(
            type='BaseSegHead',
            seg_qst_dicts=[
                dict(
                    name='mask',
                    loss_cfg=dict(
                        type='ModuleSum',
                        sub_module_cfgs=[
                            dict(
                                type='MaskLoss',
                                mask_loss_cfg=dict(
                                    type='mmdet.CrossEntropyLoss',
                                    use_sigmoid=True,
                                    loss_weight=25.0,
                                ),
                            ),
                            dict(
                                type='mmdet.DiceLoss',
                                use_sigmoid=True,
                                loss_weight=15.0,
                            ),
                        ],
                    ),
                    loss_reduction='tgt_sum',
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
            ],
            key_cfg=[
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
            key_map_ids=[1, 2, 3, 4, 5],
            get_unc_masks=True,
            unc_thr=100,
            get_segs=False,
        ),
        '8_0': dict(
            type='BaseSegHead',
            seg_qst_dicts=[
                dict(
                    name='mask',
                    loss_cfg=dict(
                        type='mmdet.CrossEntropyLoss',
                        use_sigmoid=True,
                        loss_weight=25.0,
                    ),
                    loss_reduction='tgt_sum',
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
            ],
            key_cfg=[
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
            update_mask_key='seg_qry_unc_mask',
            get_segs=True,
            seg_type='instance',
            dup_attrs=dict(
                type='box_nms',
                needs_masks=False,
                nms_candidates=1000,
                nms_thr=0.65,
            ),
            max_segs=100,
            mask_thr=0.5,
        ),
    },
)
