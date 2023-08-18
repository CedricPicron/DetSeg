"""
Collection of utility functions related to masks.
"""

import numpy as np
from pycocotools import mask as coco_mask
import torchvision.transforms.functional as F
from torchvision.transforms.functional import InterpolationMode


def mask_inv_transform(in_masks, images, cum_masks_batch, interpolation=InterpolationMode.BILINEAR):
    """
    Transforms the given masks back to the original image space as encoded by the given Images structure.

    Args:
        in_masks (Tensor): Input masks of shape [num_masks, iH, iW].
        images (Images): Images structure [batch_size] containing batched images with their entire transform history.
        cum_masks_batch (LongTensor): Tensor with cumulative number of masks per batch entry of shape [batch_size+1].
        interpolation (InterpolationMode): Object with resize interpolation mode (default=InterpolationMode.BILINEAR).

    Returns:
        out_masks_list (List): List [batch_size] with output masks of shape [num_masks_i, oH, oW].

    Raises:
        ValueError: Error when one of the transforms has an unknown transform type.
    """

    # Resize input masks to image size after transforms if needed
    mH, mW = in_masks.size()[1:]
    iW, iH = images.size()

    if (mH != iH) or (mW != iW):
        in_masks = F.resize(in_masks, size=(iH, iW))

    # Get list with output masks
    out_masks_list = []

    for i, transforms_i in enumerate(images.transforms):
        i0 = cum_masks_batch[i].item()
        i1 = cum_masks_batch[i+1].item()
        masks_i = in_masks[i0:i1]

        for transform in reversed(transforms_i):
            transform_type = transform[0]

            if transform_type == 'crop':
                crop_deltas = transform[2]
                masks_i = F.pad(masks_i, crop_deltas)

            elif transform_type == 'hflip':
                masks_i = F.hflip(masks_i)

            elif transform_type == 'pad':
                (left, top, right, bottom) = transform[2]
                (width, height) = (right-left, bottom-top)
                masks_i = F.crop(masks_i, top, left, height, width)

            elif transform_type == 'resize':
                new_width, new_height = transform[2]

                if len(masks_i) > 0:
                    masks_i = F.resize(masks_i, (new_height, new_width), interpolation=interpolation)
                else:
                    masks_i = masks_i.new_zeros([0, new_height, new_width])

            else:
                error_msg = f"Unknown transform type (got '{transform_type}')."
                raise ValueError(error_msg)

        out_masks_list.append(masks_i)

    return out_masks_list


def mask_to_rle(masks):
    """
    Function encoding the dense input masks using run-length encoding (RLE).

    Args:
        masks (List): List of size [num_masks] containing the dense input masks of shape [height, width].

    Returns:
        rles (List): List of size [num_masks] containing the RLEs of the dense input masks.
    """

    # Get RLEs of dense input masks
    rles = [coco_mask.encode(np.array(mask[:, :, None].cpu(), order='F', dtype='uint8'))[0] for mask in masks]

    # Alter counts entry of RLEs
    for rle in rles:
        rle['counts'] = rle['counts'].decode('utf-8')

    return rles
