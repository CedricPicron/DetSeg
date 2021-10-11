pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth'

backbone = dict(
        type='PyramidVisionTransformer',
        num_layers=[2, 2, 2, 2],
        out_indices=[1, 2, 3],
        out_sizes=[128, 320, 512],
        init_cfg=dict(checkpoint='https://github.com/whai362/PVT/releases/download/v2/pvt_tiny.pth'),
        )
