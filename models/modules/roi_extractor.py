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
        in_key (str): String with key to retrieve input map or maps from storage dictionary.
        boxes_key (str): String with key to retrieve RoI boxes from storage dictionary.
        ids_key (str): String with key to retrieve RoI group indices from storage dictionary.
        out_key (str): String with key to store output RoI map in storage dictionary.
        mmdet_roi_ext (nn.Module): Module containing the MMDetection RoI extractor.
    """

    def __init__(self, in_key, boxes_key, ids_key, out_key, mmdet_roi_ext_cfg):
        """
        Initializes the MMDetRoIExtractor module.

        Args:
            in_key (str): String with key to retrieve input map or maps from storage dictionary.
            boxes_key (str): String with key to retrieve RoI boxes from storage dictionary.
            ids_key (str): String with key to retrieve RoI group indices from storage dictionary.
            out_key (str): String with key to store output RoI map in storage dictionary.
            mmdet_roi_ext_cfg (Dict): Configuration dictionary specifying the underlying MMDetection RoI extractor.
        """

        # Initialization of default nn.Module
        super().__init__()

        # Build underlying MMDetection RoI extractor
        self.mmdet_roi_ext = build_model(mmdet_roi_ext_cfg)

        # Set additional attributes
        self.in_key = in_key
        self.boxes_key = boxes_key
        self.ids_key = ids_key
        self.out_key = out_key

    def forward(self, storage_dict, **kwargs):
        """
        Forward method of the MMDetRoIExtractor module.

        Args:
            storage_dict (Dict): Storage dictionary containing at least following keys:
                - images (Images): Images structure containing the batched images of size [batch_size];
                - {self.in_key} (FloatTensor or List): map or list of maps of shape [num_groups, feat_size, fH, fW];
                - {self.boxes_key} (Boxes): 2D bounding boxes of RoIs of size [num_rois];
                - {self.ids_key} (LongTensor): group indices of RoIs of shape [num_rois].

            kwargs (Dict): Dictionary of unused keyword arguments.

        Returns:
            storage_dict (Dict): Storage dictionary containing following additional key:
                - {self.out_key} (FloatTensor): extracted RoI features of shape [num_rois, feat_size, rH, rW].
        """

        # Retrieve desired items from storage dictionary
        images = storage_dict['images']
        in_maps = storage_dict[self.in_key]
        roi_boxes = storage_dict[self.boxes_key].clone()
        roi_group_ids = storage_dict[self.ids_key]

        # Extract RoI features
        if torch.is_tensor(in_maps):
            in_maps = [in_maps]

        roi_boxes = roi_boxes.to_format('xyxy').to_img_scale(images).boxes.detach()
        roi_boxes = torch.cat([roi_group_ids[:, None], roi_boxes], dim=1)

        in_maps = in_maps[:self.mmdet_roi_ext.num_inputs]
        roi_feats = self.mmdet_roi_ext(in_maps, roi_boxes)

        # Store RoI features in storage dictionary
        storage_dict[self.out_key] = roi_feats

        return storage_dict
