"""
Map-Based Detector (MBD) head.
"""

from torch import nn


class MBD (nn.Module):
    """
    Class implementing the Map-Based Detector (MBD) head.

    Attributes:
        sbd (SBD): State-based detector (SBD) module computing the object features.
        train_sbd (bool): Boolean indicating whether underlying SBD module should be trained.

        metadata (detectron2.data.Metadata): Metadata instance containing additional dataset information.
    """

    def __init__(self, sbd_dict, metadata):
        """
        Initializes the MBD module.

        sbd_dict (Dict): State-based detector (SBD) dictionary containing following keys:
            - sbd (SBD): state-based detector (SBD) module computing the object features;
            - train_sbd (bool): boolean indicating whether underlying SBD module should be trained.

        metadata (detectron2.data.Metadata): Metadata instance containing additional dataset information.
        """

        # Initialization of default nn.Module
        super().__init__()

        # Set SBD attribute
        self.sbd = sbd_dict['sbd']
        self.train_sbd = sbd_dict['train_sbd']
        self.sbd.requires_grad_(self.train_sbd)

        # Set metadata attribute
        metadata.stuff_classes = metadata.thing_classes
        metadata.stuff_colors = metadata.thing_colors
        self.metadata = metadata

    def forward(self, feat_maps, tgt_dict=None, images=None, visualize=False, **kwargs):
        """
        Forward method of the MBD head.

        Args:
            feat_maps (List): List of size [num_maps] with feature maps of shape [batch_size, feat_size, fH, fW].

            tgt_dict (Dict): Optional target dictionary used during trainval containing at least following keys:
                - labels (LongTensor): tensor of shape [num_targets_total] containing the class indices;
                - boxes (Boxes): structure containing axis-aligned bounding boxes of size [num_targets_total];
                - sizes (LongTensor): tensor of shape [batch_size+1] with the cumulative target sizes of batch entries.

            images (Images): Images structure containing the batched images (default=None).
            visualize (bool): Boolean indicating whether to compute dictionary with visualizations (default=False).
            kwargs (Dict): Dictionary of keyword arguments not used by this module, but passed to some sub-modules.

        Returns:
            * If MBD module in training mode (i.e. during training):
                loss_dict (Dict): Dictionary of different weighted loss terms used during training.
                analysis_dict (Dict): Dictionary of different analyses used for logging purposes only.

            * If MBD module not in training mode and tgt_dict is not None (i.e. during validation):
                pred_dicts (List): List with SBD and MBD prediction dictionaries.
                loss_dict (Dict): Dictionary of different weighted loss terms used during training.
                analysis_dict (Dict): Dictionary of different analyses used for logging purposes only.

            * If MBD module not in training mode and tgt_dict is None (i.e. during testing):
                pred_dicts (List): List with SBD and MBD prediction dictionaries.
                analysis_dict (Dict): Dictionary of different analyses used for logging purposes only.

        Raises:
            NotImplementedError: Error when visualizations are requested.
        """

        # Check whether visualizations are requested
        if visualize:
            raise NotImplementedError

        # Get batch size
        batch_size = len(feat_maps[0])

        # Apply SBD module and extract output
        sbd_output = self.sbd(feat_maps, tgt_dict, images, stand_alone=False, visualize=visualize, **kwargs)

        if self.training:
            loss_dict, sbd_analysis_dict, pred_dict, obj_feats, sample_feats = sbd_output
        elif tgt_dict is not None:
            pred_dicts, loss_dict, sbd_analysis_dict, obj_feats, sample_feats = sbd_output
        else:
            pred_dicts, sbd_analysis_dict, obj_feats, sample_feats = sbd_output

        # Get desired prediction, loss and analysis dictionaries
        if not self.training:
            pred_dict = pred_dicts[-1]

        elif not self.train_sbd:
            loss_dict = {}

        analysis_dict = {f'sbd_{k}': v for k, v in sbd_analysis_dict.items() if 'dod_' not in k}
        analysis_dict.update({k: v for k, v in sbd_analysis_dict.items() if 'dod_' in k})

        # Get SBD boxes
        batch_ids = pred_dict['batch_ids']
        sbd_boxes = pred_dict['boxes']
        sbd_boxes = [sbd_boxes[batch_ids == i] for i in range(batch_size)]

        # Return desired dictionaries
        if self.training:
            return loss_dict, analysis_dict
        elif tgt_dict is not None:
            return pred_dicts, loss_dict, analysis_dict
        else:
            return pred_dicts, analysis_dict
