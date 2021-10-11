_base_ = 'resnet_50_v2.py'

backbone = dict(
        conv_cfg=dict(type='ConvWS'),
        init_cfg=dict(type='Pretrained', checkpoint='open-mmlab://jhu/resnet50_gn_ws'),
        )
