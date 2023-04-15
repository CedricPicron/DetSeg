model = dict(
    type='GVD',
    name='gvd',
    requires_masks=False,
    group_init_cfg=dict(
        mode='selected',
        sel_cfg=dict(
            type='AnchorSelector',
            anchor_cfg=dict(
                type='AnchorGenerator',
                map_ids=(3, 4, 5, 6, 7),
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
            box_encoder_cfg=[
                dict(
                    type='nn.Linear',
                    in_features=4,
                    out_features=256,
                    bias=True,
                ),
                dict(
                    type='OneStepMLP',
                    num_layers=1,
                    in_size=256,
                    out_size=256,
                    norm='layer',
                    act_fn='relu',
                    skip=True,
                ),
            ],
            matcher_cfg=dict(
                type='BoxMatcher',
                qry_key='anchors',
                tgt_key='boxes',
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
                num_levels=5,
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
            logits_cfg=[
                dict(
                    type='OneStepMLP',
                    in_size=256,
                    out_size=256,
                    norm='layer',
                    act_fn='relu',
                    skip=False,
                ),
                dict(
                    type='OneStepMLP',
                    num_layers=1,
                    in_size=256,
                    out_size=256,
                    norm='layer',
                    act_fn='relu',
                    skip=True,
                ),
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
                qry_key='cls_logits',
                top_pos=15,
                top_neg=15,
                allow_multi_tgt=False,
            ),
            loss_cfg=dict(
                type='SigmoidFocalLoss',
                alpha=0.25,
                gamma=2.0,
                reduction='sum',
                weight=1.0,
            ),
        ),
        dict(
            type='BaseBox2dHead',
            logits_cfg=[
                dict(
                    type='OneStepMLP',
                    in_size=256,
                    out_size=256,
                    norm='layer',
                    act_fn='relu',
                    skip=False,
                ),
                dict(
                    type='OneStepMLP',
                    num_layers=1,
                    in_size=256,
                    out_size=256,
                    norm='layer',
                    act_fn='relu',
                    skip=True,
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
                nms_thr=0.5,
            ),
            max_dets=100,
            matcher_cfg=None,
            loss_cfg=dict(
                type='SmoothL1Loss',
                beta=0.0,
                reduction='sum',
                weight=1.0,
            ),
        ),
    ],
    head_apply_ids=[6],
)