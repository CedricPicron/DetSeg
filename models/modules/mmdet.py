"""
Collection of additional MMDetection modules.
"""

from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence
from mmcv.runner.base_module import BaseModule
from mmdet.models.utils.builder import TRANSFORMER
from mmdet.models.utils.transformer import DeformableDetrTransformer, Transformer


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

    def __init__(self, num_feature_levels=4, as_two_stage=False, two_stage_num_proposals=300, **kwargs):
        """
        Initializes the DeformableDetrTransformerPlus module.

        Args:
            num_feature_levels (int): Integer containing the number of expected input feature levels (default=4).
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
