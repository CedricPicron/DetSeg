model = dict(
    type='DinoNeck',
    in_channels=[256, 512, 1024, 2048],
    in_strides=[4, 8, 16, 32],
    transformer_dropout=0.0,
    transformer_nheads=8,
    transformer_dim_feedforward=1024,
    transformer_enc_layers=6,
    conv_dim=256,
    mask_dim=256,
    transformer_in_ids=[1, 2, 3],
    common_stride=4,
    num_feature_levels=3,
    total_num_feature_levels=5,
    feature_order='high2low',
    out_ids=(2, 3, 4, 5, 6, 7),
    out_sizes=(256, 256, 256, 256, 256, 256),
    norm='GN',
)
