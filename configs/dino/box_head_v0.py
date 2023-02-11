model = dict(
    type='DinoHead',
    requires_masks=False,
    in_channels=256,
    num_classes=80,
    hidden_dim=256,
    num_queries=300,
    nheads=8,
    dim_feedforward=2048,
    dec_layers=9,
    mask_dim=256,
    enforce_input_project=False,
    two_stage=True,
    dn='standard',
    noise_scale=0.4,
    dn_num=100,
    initialize_box_type='no',
    initial_pred=True,
    learn_tgt=False,
    mask_classification=True,
    decoder_feat_ids=[0, 1, 2, 3],
    box_loss=True,
    semantic_ce_loss=False,
    deep_supervision=True,
    no_object_weight=0.1,
    class_weight=4.0,
    cost_class_weight=4.0,
    box_weight=5.0,
    cost_box_weight=5.0,
    giou_weight=2.0,
    cost_giou_weight=2.0,
    semantic_on=False,
    instance_on=True,
    instance_box_only=True,
    panoptic_on=False,
    test_topk_per_image=100,
)
