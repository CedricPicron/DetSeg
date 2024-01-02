"""
Collection of mask-related modules.
"""

from mmdet.structures.mask import BitmapMasks
from mmdet.structures.mask.mask_target import mask_target_single
from mmengine.config import Config
import torch
from torch import nn

from models.build import MODELS


@MODELS.register_module()
class DenseRoIMaskTargets(nn.Module):
    """
    Class implementing the DenseRoIMaskTargets module.

    Attributes:
        in_key (str): String with key to retrieve mask predictions from storage dictionary.
        boxes_key (str): String with key to retrieve RoI boxes from storage dictionary.
        tgt_ids_key (str): String with key to retrieve target indices from storage dictionary.
        out_key (str): String with key to store mask targets in storage dictionary.
    """

    def __init__(self, in_key, boxes_key, tgt_ids_key, out_key):
        """
        Initializes the DenseRoIMaskTargets module.

        Args:
            in_key (str): String with key to retrieve mask predictions from storage dictionary (default='mask_logits').
            boxes_key (str): String with key to retrieve RoI boxes from storage dictionary (default='roi_boxes').
            tgt_ids_key (str): String with key to retrieve target indices (default='matched_tgt_ids').
            out_key (str): String with key to store mask targets in storage dictionary (default='mask_targets').
        """

        # Initialization of default nn.Module
        super().__init__()

        # Set additional attributes
        self.in_key = in_key
        self.boxes_key = boxes_key
        self.tgt_ids_key = tgt_ids_key
        self.out_key = out_key

    @torch.no_grad()
    def forward(self, storage_dict, tgt_dict, **kwargs):
        """
        Forward method of the DenseRoIMaskTargets module.

        Args:
            storage_dict (Dict): Storage dictionary containing at least following key:
                - images (Images): Images structure containing the batched images of size [batch_size];
                - {self.in_key} (FloatTensor): dense mask predictions of shape [num_rois, {1}, rH, rW];
                - {self.boxes_key} (Boxes): 2D bounding boxes of RoIs of size [num_rois];
                - {self.tgt_ids_key} (LongTensor): target indices corresponding to RoIs of shape [num_rois].

            tgt_dict (Dict): Target dictionary containing at least following key:
                - masks (BoolTensor): target segmentation masks of shape [num_targets, iH, iW].

            kwargs (Dict): Dictionary of unused keyword arguments.

        Returns:
            storage_dict (Dict): Storage dictionary containing following additional key:
                - {self.out_key} (FloatTensor): dense mask targets of shape [num_rois, rH, rW].
        """

        # Retrieve desired items from storage dictionary
        images = storage_dict['images']
        mask_preds = storage_dict[self.in_key]
        roi_boxes = storage_dict[self.boxes_key].clone()
        tgt_ids = storage_dict[self.tgt_ids_key]

        # Get mask targets
        tgt_masks = tgt_dict['masks'].cpu().numpy()
        tgt_masks = BitmapMasks(tgt_masks, *tgt_masks.shape[-2:])

        mask_size = tuple(mask_preds.size()[-2:])
        mask_tgt_cfg = Config({'mask_size': mask_size})

        roi_boxes = roi_boxes.to_format('xyxy').to_img_scale(images).boxes
        mask_targets = mask_target_single(roi_boxes, tgt_ids, tgt_masks, mask_tgt_cfg)

        # Store mask targets in storage dictionary
        storage_dict[self.out_key] = mask_targets

        return storage_dict
