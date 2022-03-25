model = dict(
    type='GVD',
    requires_masks='False',
    group_init_cfg=dict(
        mode='learned',
        num_groups=300,
        feat_size=256,
    ),
    dec_layer_cfg=[
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
            hidden_size=1024,
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
            matcher_cfg=None,
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
            box_encoding='prior_boxes',
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
