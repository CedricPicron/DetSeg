out_proj = dict(
    type='mmdet.ChannelMapper',
    in_channels=[256, 256, 256, 256, 256],
    kernel_size=1,
    out_channels=256,
    act_cfg=None,
    norm_cfg=None,
    num_outs=5,
    out_ids=(3, 4, 5, 6, 7),
    out_sizes=(256, 256, 256, 256, 256),
)
