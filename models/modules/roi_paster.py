"""
Collection of RoI paster modules.
"""

from mmdet.models.roi_heads.mask_heads.fcn_mask_head import _do_paste_mask
from torch import nn

from models.build import MODELS


@MODELS.register_module()
class MMDetRoIPaster(nn.Module):
    """
    Module implementing the MMDetRoIPaster module.

    Attributes:
        in_key (str): String with key to retrieve input map to paste from storage dictionary.
        out_key (str): String with key to store pasted output map in storage dictionary.
    """

    def __init__(self, in_key, boxes_key, out_key):
        """
        Initializes the MMDetRoIPaster module.

        Args:
            in_key (str): String with key to retrieve input map to paste from storage dictionary.
            boxes_key (str): String with key to retrieve RoI boxes from storage dictionary.
            out_key (str): String with key to store pasted output map in storage dictionary.
        """

        # Initialization of default nn.Module
        super().__init__()

        # Set additional attributes
        self.in_key = in_key
        self.boxes_key = boxes_key
        self.out_key = out_key

    def forward(self, storage_dict, **kwargs):
        """
        Forward method of the MMDetRoIPaster module.

        Args:
            storage_dict (Dict): Storage dictionary containing at least following keys:
                - images (Images): Images structure containing the batched images of size [batch_size];
                - {self.in_key} (FloatTensor): input map to be pasted of shape [num_rois, {1}, rH, rW];
                - {self.boxes_key} (Boxes): 2D bounding boxes of RoIs of size [num_rois].

            kwargs (Dict): Dictionary of unused keyword arguments.

        Returns:
            storage_dict (Dict): Storage dictionary containing following additional key:
                - {self.out_key} (FloatTensor): pasted output map of shape [num_rois, iH, iW].
        """

        # Retrieve desired items from storage dictionary
        images = storage_dict['images']
        in_map = storage_dict[self.in_key]
        roi_boxes = storage_dict[self.boxes_key].clone()

        # Get pasted output map
        if in_map.dim() == 3:
            in_map = in_map.unsqueeze(dim=1)

        iW, iH = images.size()
        roi_boxes = roi_boxes.to_format('xyxy').to_img_scale(images).boxes
        out_map = _do_paste_mask(in_map, roi_boxes, iH, iW, skip_empty=False)[0]

        # Store pasted output map in storage dictionary
        storage_dict[self.out_key] = out_map

        return storage_dict
