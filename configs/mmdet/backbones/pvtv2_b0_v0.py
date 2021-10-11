backbone = dict(
        type='PyramidVisionTransformerV2',
        embed_dims=32,
        num_layers=[2, 2, 2, 2],
        out_indices=[1, 2, 3],
        out_sizes=[64, 160, 256],
        init_cfg=dict(checkpoint='https://github.com/whai362/PVT/releases/download/v2/pvt_v2_b0.pth'),
        )
