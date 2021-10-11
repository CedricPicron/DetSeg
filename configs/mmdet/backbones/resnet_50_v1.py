_base_ = 'resnet_50_v0.py'

backbone = dict(
        norm_cfg=dict(type='BN', requires_grad=True),
        )
