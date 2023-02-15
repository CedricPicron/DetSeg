proj = dict(
    type='ApplyOneToOne',
    sub_module_cfgs=[
        dict(
            type='nn.Conv2d',
            in_channels=256,
            out_channels=256,
            kernel_size=1,
        ) for _ in range(6)
    ],
    out_ids=(2, 3, 4, 5, 6, 7),
    out_sizes=(256, 256, 256, 256, 256, 256),
)
