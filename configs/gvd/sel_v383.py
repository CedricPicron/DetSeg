model = dict(
    type='GVD',
    name='gvd',
    requires_masks=True,
    group_init_cfg=dict(
        mode='selected',
        sel_cfg=dict(
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
                rel_thr=600,
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
    ),
    dec_layer_cfg=[
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
    ],
    num_dec_layers=6,
    head_cfgs=[
        dict(
            type='BaseClsHead',
            apply_ids=[6],
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
        dict(
            type='BaseBox2dHead',
            apply_ids=[6],
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
        dict(
            type='RefineMaskRoIHead',
            mask_roi_extractor=dict(
                type='mmdet.SingleRoIExtractor',
                roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
                out_channels=256,
                featmap_strides=[4, 8, 16, 32],
            ),
            qry_cfg=[
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
            fuse_qry_cfg=[
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
            mask_head=dict(
                type='SimpleRefineMaskHead',
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
                stage_num_classes=[80, 80, 80, 1],
                stage_sup_size=[14, 28, 56, 112],
                pre_upsample_last_stage=False,
                upsample_cfg=dict(type='bilinear', scale_factor=2),
                loss_cfg=dict(
                    type='BARCrossEntropyLoss',
                    stage_instance_loss_weight=[0.5, 0.75, 0.75, 1.0],
                    boundary_width=2,
                    start_stage=1),
            ),
            dup_attrs=dict(
                type='nms',
                nms_candidates=1000,
                nms_thr=0.65,
            ),
            max_segs=100,
        ),
    ],
    head_apply_ids=[6],
)
