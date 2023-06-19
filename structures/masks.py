"""
Collection of utility functions related to masks.
"""

import numpy as np
import torch
from pycocotools import mask as coco_mask
import torchvision.transforms.functional as F


def mask_inv_transform(in_masks, images, batch_ids):
    """
    Transforms the given masks back to the original image space as encoded by the given Images structure.

    Args:
        in_masks (Tensor): Input masks (or mask scores) of shape [num_masks, in_height, in_width].
        images (Images): Images structure [batch_size] containing batched images with their entire transform history.
        batch_ids (LongTensor): Tensor containing the batch indices of each input mask of shape [num_masks].

    Returns:
        out_masks (Tensor): Output masks (or mask scores) of shape [num_masks, out_height, out_width].

    Raises:
        ValueError: Error when one of the transforms has an unknown transform type.
    """

    # Resize input masks to image size after transforms if needed
    num_masks, mH, mW = in_masks.size()
    iW, iH = images.size()

    if (mH != iH) or (mW != iW):
        in_masks = F.resize(in_masks, size=(iH, iW))

    # Get list with output masks
    mask_ids = torch.arange(num_masks, device=in_masks.device)
    masks_list = [None for _ in range(num_masks)]

    for i, transforms_i in enumerate(images.transforms):
        batch_mask = batch_ids == i
        mask_ids_i = mask_ids[batch_mask]
        masks_i = in_masks[batch_mask]

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
                    masks_i = F.resize(masks_i, (new_height, new_width))
                else:
                    masks_i = masks_i.new_zeros([0, new_height, new_width])

            else:
                error_msg = f"Unknown transform type (got '{transform_type}')."
                raise ValueError(error_msg)

        for j, mask_id in enumerate(mask_ids_i.tolist()):
            masks_list[mask_id] = masks_i[j]

    # Get output masks
    out_masks = torch.stack(masks_list, dim=0)

    return out_masks


def mask_to_rle(masks):
    """
    Function encoding the dense input masks using run-length encoding (RLE).

    Args:
        masks (BoolTensor): Dense input masks of shape [num_masks, height, width].

    Returns:
        rles (List): List of size [num_masks] containing the RLEs of the dense input masks.
    """

    # Get RLEs of dense input masks
    rles = [coco_mask.encode(np.array(mask[:, :, None].cpu(), order='F', dtype='uint8'))[0] for mask in masks]

    # Alter counts entry of RLEs
    for rle in rles:
        rle['counts'] = rle['counts'].decode('utf-8')

    return rles
