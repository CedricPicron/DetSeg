_base_ = 'resnet_50_v0.py'

backbone = dict(
        norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
        init_cfg=dict(type='Pretrained', checkpoint='open-mmlab://detectron/resnet50_gn'),
        )
