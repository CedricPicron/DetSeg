model = dict(
    type='GCT',
    requires_masks=False,
    map_cfg=dict(
        type='ResNet',
        name='resnet50',
        out_ids=[4],
    ),
    graph_cfg=dict(
        type='Net',
        blocks_per_stage=[3],
        scale_factors=[4],
        scale_tags=['_features', '_shape', '_size'],
        scale_overwrites=[
            [0, 'in_proj_cfg', 'con_in_size', 1024],
            [0, 'in_proj_cfg', 'struc_in_size', 2],
            [0, 'block_cfg', 'edge_score_cfg', 3, 'out_features', 1],
            [0, 'block_cfg', 'edge_score_cfg', 3, 'init_cfg', 'val', 0.003],
        ],
        return_inter_stages=False,
        base_stage_cfg=dict(
            return_inter_blocks=False,
            in_proj_cfg=dict(
                type='GraphProjector',
                con_in_size=128,
                con_out_size=256,
                struc_in_size=32,
                struc_out_size=64,
            ),
            block_cfg=dict(
                type='GraphToGraph',
                left_zero_grad_thr=-0.1,
                right_zero_grad_thr=0.1,
                node_weight_iters=5,
                max_group_iters=100,
                con_temp=1.0,
                struc_temp=1.0,
                con_cross_cfg=[
                    dict(
                        type='GraphAttn',
                        in_size=256,
                        struc_size=64,
                        norm='layer',
                        act_fn='',
                        qk_size=64,
                        val_size=64,
                        num_heads=8,
                        skip=True,
                    ),
                ],
                edge_score_cfg=[
                    dict(
                        type='nn.Linear',
                        in_features=256,
                        out_features=64,
                        bias=True,
                    ),
                    dict(
                        type='nn.LayerNorm',
                        normalized_shape=64,
                    ),
                    dict(
                        type='NodeToEdge',
                        reduction='mul',
                        implementation='pytorch-custom',
                    ),
                    dict(
                        type='nn.Linear',
                        in_features=64,
                        out_features=1,
                        bias=True,
                        init_cfg=dict(type='Constant', layer='Linear', val=0.01, bias=-0.6),
                    ),
                ],
                con_self_cfg=[
                    dict(
                        type='TwoStepMLP',
                        in_size=256,
                        hidden_size=256,
                        norm1='layer',
                        norm2='',
                        act_fn1='',
                        act_fn2='relu',
                        skip=True,
                    )
                ],
                struc_self_cfg=[
                    dict(
                        type='TwoStepMLP',
                        in_size=64,
                        hidden_size=128,
                        norm1='layer',
                        norm2='',
                        act_fn1='',
                        act_fn2='relu',
                        skip=True,
                    )
                ],
            ),
        ),
    ),
    heads=dict(
        graph_box=dict(
            pred_cfg=[
                dict(
                    type='nn.Linear',
                    in_features=256,
                    out_features=64,
                    bias=True,
                ),
                dict(
                    type='OneStepMLP',
                    num_layers=2,
                    in_size=64,
                    norm='layer',
                    act_fn='relu',
                    skip=True,
                ),
                dict(
                    type='nn.Linear',
                    in_features=64,
                    out_features=4,
                    bias=True,
                ),
            ],
            loss_cfg=dict(
                type='SmoothL1Loss',
                reduction='sum',
                beta=0.0,
                weight=1.0,
            ),
        ),
    ),
)
