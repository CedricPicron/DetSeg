"""
Collection of RoI extractor modules.
"""

import torch
from torch import nn

from models.build import build_model, MODELS


@MODELS.register_module()
class MMDetRoIExtractor(nn.Module):
    """
    Class implementing the MMDetRoIExtractor module.

    Attributes:
        mmdet_roi_ext (nn.Module): Module containing the MMDetection RoI extractor.
        out_key (str): String with key to store output RoI features in storage dictionary.
    """

    def __init__(self, mmdet_roi_ext_cfg, out_key='roi_feats'):
        """
        Initializes the MMDetRoIExtractor module.

        Args:
            mmdet_roi_ext_cfg (Dict): Configuration dictionary specifying the underlying MMDetection RoI extractor.
            out_key (str): String with key to store output RoI features in storage dictionary (default='roi_feats').
        """

        # Initialization of default nn.Module
        super().__init__()

        # Build underlying MMDetection RoI extractor
        self.mmdet_roi_ext = build_model(mmdet_roi_ext_cfg)

        # Set additional attribute
        self.out_key = out_key

    def forward(self, storage_dict, **kwargs):
        """
        Forward method of the MMDetRoIExtractor module.

        Args:
            storage_dict (Dict): Storage dictionary containing at least following keys:
                - images (Images): Images structure containing the batched images of size [batch_size];
                - feat_maps (List): list of size [num_maps] with feature maps of shape [batch_size, feat_size, fH, fW];
                - roi_batch_ids (LongTensor): batch indices of RoIs of shape [num_rois];
                - roi_boxes (Boxes): 2D bounding boxes of RoIs of size [num_rois].

            kwargs (Dict): Dictionary of unused keyword arguments.

        Returns:
            storage_dict (Dict): Storage dictionary containing following additional key:
                - {self.out_key} (FloatTensor): Extracted RoI features of shape [num_rois, feat_size, rH, rW].
        """

        # Retrieve desired items from storage dictionary
        images = storage_dict['images']
        feat_maps = storage_dict['feat_maps']
        roi_batch_ids = storage_dict['roi_batch_ids']
        roi_boxes = storage_dict['roi_boxes'].clone()

        # Extract RoI features
        roi_boxes = roi_boxes.to_format('xyxy').to_img_scale(images).boxes.detach()
        roi_boxes = torch.cat([roi_batch_ids[:, None], roi_boxes], dim=1)

        roi_feat_maps = feat_maps[:self.mmdet_roi_ext.num_inputs]
        roi_feats = self.mmdet_roi_ext(roi_feat_maps, roi_boxes)

        # Store RoI features in storage dictionary
        storage_dict[self.out_key] = roi_feats

        return storage_dict
