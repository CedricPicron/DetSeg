"""
Bidirectional Decoder-Encoder (BDE) head.
"""

from torch import nn

from models.build import build_model, MODELS


@MODELS.register_module()
class BDE(nn.Module):
    """
    Class implementing the Bidirectional Decoder-Encoder (BDE) head.

    Attributes:
        pos (nn.Module): Shared module computing position features.
        decoders (nn.ModuleDict): Dictionary with decoder modules for specified decoder-encoder-head iterations.
        encoders (nn.ModuleDict): Dictionary with encoder modules for specified decoder-encoder-head iterations.
        heads (nn.ModuleDict): Dictionary with head modules for specified decoder-encoder-head iterations.
        num_iters (int): Integer containing the number of decoder-encoder-head iterations.
    """

    def __init__(self, decoder_cfgs, encoder_cfgs, head_cfgs, metadata, pos_cfg=None):
        """
        Initializes the BDE head.

        Args:
            decoder_cfgs (Dict): Dictionary of configuration dictionaries specifying the decoder modules.
            encoder_cfgs (Dict): Dictionary of configuration dictionaries specifying the encoder modules.
            head_cfgs (Dict): Dictionary of configuration dictionaries specifying the head modules.
            metadata (detectron2.data.Metadata): Metadata instance containing additional dataset information.
            pos_cfg (Dict): Configuration dictionary specifying the shared position module (default=None).
        """

        # Initialization of default nn.Module
        super().__init__()

        # Build shared position module if needed
        self.pos = build_model(pos_cfg) if pos_cfg is not None else None

        # Build decoder, encoder and head modules
        self.decoders = nn.ModuleDict({key: build_model(cfg) for key, cfg in decoder_cfgs.items()})
        self.encoders = nn.ModuleDict({key: build_model(cfg) for key, cfg in encoder_cfgs.items()})
        self.heads = nn.ModuleDict({key: build_model(cfg, metadata=metadata) for key, cfg in head_cfgs.items()})

        # Get number of decoder-encoder-head iterations
        keys = list(decoder_cfgs.keys()) + list(encoder_cfgs.keys()) + list(head_cfgs.keys())
        self.num_iters = max(int(key.split('_')[0]) for key in keys) + 1

    def forward(self, feat_maps, images=None, tgt_dict=None, visualize=False, **kwargs):
        """
        Forward method of the BDE head.

        Args:
            feat_maps (List): List of size [num_maps] with feature maps of shape [batch_size, feat_size, fH, fW].
            images (Images): Images structure of size [batch_size] containing the batched images (default=None).
            tgt_dict (Dict): Target dictionary with ground-truth information used during trainval (default=None).
            visualize (bool): Boolean indicating whether to compute dictionary with visualizations (default=False).
            kwargs (Dict): Dictionary of additional keyword arguments passed to some underlying modules and methods.

        Returns:
            return_list (List): List of size [num_returns] possibly containing following items to return:
                - pred_dicts (List): list of size [num_pred_dicts] with prediction dictionaries (evaluation only);
                - loss_dict (Dict): dictionary with different weighted loss terms used during training (trainval only);
                - analysis_dict (Dict): dictionary with different analyses used for logging purposes only;
                - images_dict (Dict): dictionary with annotated images of predictions/targets (when visualize is True).
        """

        # Intialize storage, loss, analysis, prediction and images dictionaries
        storage_dict = {'feat_maps': feat_maps, 'images': images, 'pos_module': self.pos}
        loss_dict = {} if tgt_dict is not None else None
        analysis_dict = {}
        pred_dicts = [] if not self.training else None
        images_dict = {} if visualize else None

        # Add above dictionaries to kwargs
        kwargs.update({'storage_dict': storage_dict, 'tgt_dict': tgt_dict, 'loss_dict': loss_dict})
        kwargs.update({'analysis_dict': analysis_dict, 'pred_dicts': pred_dicts, 'images_dict': images_dict})

        # Perform decoder-encoder-head iterations
        for iter_id in range(self.num_iters):

            # Apply decoder modules if needed
            for key, decoder in self.decoders.items():
                apply_id = int(key.split('_')[0])

                if apply_id == iter_id:
                    qry_feats = storage_dict.get('qry_feats', None)
                    qry_feats = decoder(qry_feats, **kwargs)
                    storage_dict['qry_feats'] = qry_feats

            # Apply encoder modules if needed
            for key, encoder in self.encoders.items():
                apply_id = int(key.split('_')[0])

                if apply_id == iter_id:
                    feat_maps = storage_dict['feat_maps']
                    feat_maps = encoder(feat_maps, **kwargs)
                    storage_dict['feat_maps'] = feat_maps

            # Apply head modules in prediction mode if needed
            for key, head in self.heads.items():
                apply_id = int(key.split('_')[0])

                if apply_id == iter_id:
                    head(mode='pred', id=key, **kwargs)

            # Apply head modules in loss mode if needed
            if tgt_dict is not None:
                for key, head in self.heads.items():
                    apply_id = int(key.split('_')[0])

                    if apply_id == iter_id:
                        head(mode='loss', id=key, **kwargs)

        # Get list with items to return
        return_list = [analysis_dict]
        return_list.insert(0, loss_dict) if tgt_dict is not None else None
        return_list.insert(0, pred_dicts) if not self.training else None
        return_list.append(images_dict) if visualize else None

        return return_list
