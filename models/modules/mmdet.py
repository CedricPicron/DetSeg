"""
Collection of additional MMDetection modules.
"""

from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence
from mmcv.runner.base_module import BaseModule
from mmdet.models.builder import DETECTORS
from mmdet.models.detectors import DeformableDETR
from mmdet.models.utils.builder import TRANSFORMER
from mmdet.models.utils.transformer import DeformableDetrTransformer, Transformer


@DETECTORS.register_module()
class DeformableDETRPlus(DeformableDETR):
    """
    Class implementing the DeformableDETRPlus module.
    """

    def forward_train(self, images, img_metas, *args, **kwargs):
        """
        Forward method of the DeformableDETRPlus module during training.

        Args:
            images (Images): Images structure containing the batched images.
            img_metas (List): List of size [num_images] containing additional image-specific information.
            args (Tuple): Tuple of additional arguments passed to underlying bounding box head.
            kwargs (Dict): Dictionary of additional keyword arguments passed to underlying bounding box head.

        Returns:
            loss_dict (Dict): Dictionary containing different loss terms.
        """

        # Add 'batch_input_shape' key to list of MMDetection image metas
        iW, iH = images.size(mode='with_padding')
        [img_meta.update({'batch_input_shape': (iH, iW)}) for img_meta in img_metas]

        # Get loss dictionary
        feat_maps = self.extract_feat(images)
        loss_dict = self.bbox_head.forward_train(feat_maps, img_metas, *args, **kwargs)

        return loss_dict


@TRANSFORMER.register_module()
class TransformerPlus(Transformer):
    """
    Class implementing the TransformerPlus module.

    Attributes:
        embed_dims (int): Integer containing the size of the transformer embeddings.
        encoder (nn.Module): Encoder module of the transformer.
        decoder (nn.Module): Decoder module of the transformer.
    """

    def __init__(self, encoder=None, decoder=None, init_cfg=None):
        """
        Initializes the TransformerPlus module.

        Args:
            encoder (Dict): Configuration dictionary specifying the transformer encoder (default=None).
            decoder (Dict): Configuration dictionary specifying the transformer decoder (default=None).
            init_cfg (Dict): Configuration dictionary controlling the base module initialization (default=None).
        """

        # Initialization of base module
        BaseModule.__init__(self, init_cfg=init_cfg)

        # Define Identity module used in case no encoder is specified
        class Identity(BaseModule):
            def __init__(self):
                super().__init__()

            def forward(self, query, **kwargs):
                return query

        # Build underlying encoder and decoder modules
        self.encoder = build_transformer_layer_sequence(encoder) if encoder is not None else Identity()
        self.decoder = build_transformer_layer_sequence(decoder)

        # Set attribute with size of transformer embeddings
        self.embed_dims = self.decoder.embed_dims


@TRANSFORMER.register_module()
class DeformableDetrTransformerPlus(DeformableDetrTransformer):
    """
    Class implementing the DeformableDetrTransformerPlus module.

    Attributes:
        num_feature_levels (int): Integer containing the number of expected input feature levels.
        as_two_stage (bool): Boolean indicating whether to generate queries from encoder output.
        two_stage_num_proposals (int): Integer containing the number of two-stage proposals.
    """

    def __init__(self, num_feature_levels=5, as_two_stage=False, two_stage_num_proposals=300, **kwargs):
        """
        Initializes the DeformableDetrTransformerPlus module.

        Args:
            num_feature_levels (int): Integer containing the number of expected input feature levels (default=5).
            as_two_stage (bool): Boolean indicating whether to generate queries from encoder output (default=False).
            two_stage_num_proposals (int): Integer containing the number of two-stage proposals (default=300).
        """

        # Initialization of underlying transformer
        TransformerPlus.__init__(self, **kwargs)

        # Set attributes
        self.num_feature_levels = num_feature_levels
        self.as_two_stage = as_two_stage
        self.two_stage_num_proposals = two_stage_num_proposals

        # Initialize the transformer layers
        self.init_layers()
        self.level_embeds.requires_grad_(False)
