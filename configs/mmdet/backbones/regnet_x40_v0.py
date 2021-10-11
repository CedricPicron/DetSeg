_base_ = 'regnet_x16_v0.py'

backbone = dict(
        arch='regnetx_4.0gf',
        out_sizes=(240, 560, 1360),
        init_cfg=dict(type='Pretrained', checkpoint='open-mmlab://regnetx_4.0gf'),
        )
