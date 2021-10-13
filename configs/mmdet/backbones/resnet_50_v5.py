_base_ = 'resnet_50_v4.py'

backbone = dict(
        sac=dict(type='SAC', use_deform=False),
        stage_with_sac=(False, True, True, True),
        )
