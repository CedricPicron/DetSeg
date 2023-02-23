in_proj = dict(
    type='mmdet.ChannelMapper',
    in_channels=[512, 1024, 2048],
    kernel_size=1,
    out_channels=256,
    act_cfg=None,
    norm_cfg=dict(type='GN', num_groups=32),
    num_outs=5,
)
