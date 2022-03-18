model = dict(
    type='GVD',
    requires_masks='False',
    group_init_cfg=dict(
        mode='learned',
        num_groups=300,
        feat_size=256,
    ),
)
