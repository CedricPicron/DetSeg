core = dict(
    type='FPN',
    in_channels=[192, 384, 768, 1536],
    out_channels=256,
    num_outs=6,
    out_ids=(2, 3, 4, 5, 6, 7),
    out_sizes=(256, 256, 256, 256, 256, 256),
    add_extra_convs='on_lateral',
    relu_before_extra_convs=True,
)
