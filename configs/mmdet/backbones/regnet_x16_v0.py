backbone = dict(
        type='RegNet',
        arch='regnetx_1.6gf',
        out_indices=(1, 2, 3),
        out_sizes=(168, 408, 912),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='open-mmlab://regnetx_1.6gf'),
        )
