deform_encoder = dict(
    type='DeformEncoder',
    pos_embed_cfg=dict(
        type='mmdet.SinePositionalEncoding',
        num_feats=128,
        temperature=20,
        normalize=True,
        offset=0.0,
    ),
    with_lvl_embed=True,
    encoder_cfg=dict(
        type='',
    ),
)
