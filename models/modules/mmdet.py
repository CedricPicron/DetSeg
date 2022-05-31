"""
Collection of modules based on existing MMDetection modules.
"""

from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence
from mmcv.runner import BaseModule
from mmdet.models.builder import DETECTORS, HEADS
from mmdet.models.detectors import DETR, SingleStageDetector
from mmdet.models.roi_heads.mask_heads import FCNMaskHead
from mmdet.models.utils.builder import TRANSFORMER
from mmdet.models.utils.transformer import DeformableDetrTransformer, Transformer

from models.build import MODELS


@DETECTORS.register_module()
@MODELS.register_module()
class DETRPlus(DETR):
    """
    Class implementing the DETRPlus module.
    """

    def __init__(self, *args, **kwargs):
        """
        Initializes the DETRPlus module.
        """

        # Initialization of SingleStageDetector module
        SingleStageDetector.__init__(self, *args, **kwargs)

    def forward_train(self, images, img_metas, *args, **kwargs):
        """
        Forward method of the DETRPlus module during training.

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

    def forward_test(self, images, img_metas, *args, **kwargs):
        """
        Forward method of the DETRPlus module during testing.

        Args:
            images (List): List [num_augs] with Images structures [num_images] containing the batched images.
            img_metas (List): List [num_augs] of lists [num_images] containing additional image-specific information.
            args (Tuple): Tuple of additional arguments passed to underlying test method.
            kwargs (Dict): Dictionary of additional keyword arguments passed to underlying test method.

        Returns:
            preds (List): List [num_images] of lists [num_classes] containing bounding box predictions.

        Raises:
            TypeError: Error when the 'images' or 'img_metas' input arguments are not lists.
            ValueError: Error when the size of the 'images' and 'img_metas' input arguments are different.
            ValueError: Error when batch size is not one and testing with with test-time augmentations.
            ValueError: Error when proposals are provided and testing with with test-time augmentations.
        """

        # Check inputs
        for var, name in [(images, 'images'), (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                error_msg = f"Input argument '{name}' must be a list, but got {type(var)}."
                raise TypeError(error_msg)

        if len(images) != len(img_metas):
            error_msg = f"The 'images' ({len(images)}) and 'img_metas' ({len(img_metas)}) inputs must have same size."
            raise ValueError(error_msg)

        # Add 'batch_input_shape' key to list of MMDetection image metas
        for images_i, img_metas_i in zip(images, img_metas):
            iW, iH = images_i.size(mode='with_padding')
            [img_meta.update({'batch_input_shape': (iH, iW)}) for img_meta in img_metas_i]

        # Get test method and prepare inputs
        if len(images) == 1:
            test_method = self.simple_test
            images = images[0]
            img_metas = img_metas[0]

            if 'proposals' in kwargs:
                kwargs['proposals'] = kwargs['proposals'][0]

        elif len(images[0]) != 1:
            error_msg = f"Batch size (got {len(images[0])}) must be one when testing with test-time augmentations."
            raise ValueError(error_msg)

        elif 'proposals' in kwargs:
            error_msg = "Proposals are not supported when testing with test-time augmentations."
            raise ValueError(error_msg)

        else:
            test_method = self.aug_test

        # Get bounding box predictions
        preds = test_method(images, img_metas, *args, **kwargs)

        return preds


@MODELS.register_module()
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


@MODELS.register_module()
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


@HEADS.register_module()
@MODELS.register_module()
class QueryFCNMaskHead(FCNMaskHead):
    """
    Class implementing the QueryFCNMaskHead module.

    The QueryFCNMaskHead module contains the same attributes as the FCNMaskHead module from MMDetection, except that
    the 'conv_logits' attribute was removed. These logits will instead be obtained using query-key dot products, where
    the queries are provided from an external source.
    """

    def __init__(self, **kwargs):
        """
        Initializes the QueryFCNMaskHead module.

        Args:
            kwargs (Dict): Keyword arguments passed to the parent __init__ method.
        """

        # Initialize module using parent __init__ method
        super().__init__(**kwargs)

        # Remove 'conv_logits' attribute
        del self.conv_logits

    def forward(self, qry_feats, key_feats):
        """
        Forward method of the QueryFCNMaskHead.

        Args:
            qry_feats (FloatTensor): Query features of shape [num_qrys, qry_feat_size].
            key_feats (FloatTensor): Map with key features of shape [num_qrys, key_feat_size, kH, kW].

        Returns:
            mask_logits (FloatTensor): Map with mask logits of shape [num_qrys, 1, mH, mW].
        """

        # Process key features
        for conv in self.convs:
            key_feats = conv(key_feats)

        if self.upsample is not None:
            key_feats = self.upsample(key_feats)

            if self.upsample_method == 'deconv':
                key_feats = self.relu(key_feats)

        # Get mask logits
        mask_logits = (qry_feats[:, :, None, None] * key_feats).sum(dim=1, keepdim=True)

        return mask_logits
