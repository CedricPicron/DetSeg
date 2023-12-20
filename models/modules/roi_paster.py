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
        in_key (str): String with key to retrieve input to paste from storage dictionary.
        out_key (str): String with key to store pasted output in storage dictionary.
    """

    def __init__(self, in_key='mask_scores', out_key='mask_scores'):
        """
        Initializes the MMDetRoIPaster module.

        Args:
            in_key (str): String with key to retrieve input to paste from storage dictionary (default='mask_scores').
            out_key (str): String with key to store pasted output in storage dictionary (default='mask_scores').
        """

        # Initialization of default nn.Module
        super().__init__()

        # Set additional attributes
        self.in_key = in_key
        self.out_key = out_key

    def forward(self, storage_dict, **kwargs):
        """
        Forward method of the MMDetRoIPaster module.

        Args:
            storage_dict (Dict): Storage dictionary containing at least following keys:
                - images (Images): Images structure containing the batched images of size [batch_size];
                - roi_boxes (Boxes): 2D bounding boxes of RoIs of size [num_rois];
                - {self.in_key} (FloatTensor): input tensor to be pasted of shape [num_rois, {1}, rH, rW].

            kwargs (Dict): Dictionary of unused keyword arguments.

        Returns:
            storage_dict (Dict): Storage dictionary containing following additional key:
                - {self.out_key} (FloatTensor): pasted output tensor of shape [num_rois, 1, iH, iW].
        """

        # Retrieve desired items from storage dictionary
        images = storage_dict['images']
        roi_boxes = storage_dict['roi_boxes'].clone()
        in_tensor = storage_dict[self.in_key]

        # Get pasted output tensor
        if in_tensor.dim() == 3:
            in_tensor = in_tensor.unsqueeze(dim=1)

        iW, iH = images.size()
        roi_boxes = roi_boxes.to_format('xyxy').to_img_scale(images).boxes
        out_tensor = _do_paste_mask(in_tensor, roi_boxes, iH, iW, skip_empty=False)[0]

        # Store pasted output tensor in storage dictionary
        storage_dict[self.out_key] = out_tensor

        return storage_dict
