_base_ = 'resnet_50_v5.py'

backbone = dict(
        sac=dict(type='SAC', use_deform=True),
        )
