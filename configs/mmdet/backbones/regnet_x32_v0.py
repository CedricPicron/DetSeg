_base_ = 'regnet_x16_v0.py'

backbone = dict(
        arch='regnetx_3.2gf',
        out_sizes=(192, 432, 1008),
        init_cfg=dict(type='Pretrained', checkpoint='open-mmlab://regnetx_3.2gf'),
        )
