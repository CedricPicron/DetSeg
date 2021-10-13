_base_ = 'resnet_50_v1.py'

backbone = dict(
        type='DetectoRS_ResNet',
        conv_cfg=dict(type='ConvAWS'),
        )
