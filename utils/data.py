"""
Data loading and data structure utilities.
"""

import torch
import torch.nn.functional as F


def train_collate_fn(batch):
    """
    The collate function used during training data loading.

    Args:
        batch (List): List of size [batch_size] containing tuples of:
            - image (FloatTensor): tensor containing the image of shape [3, iH, iW].
            - tgt_dict (Dict): target dictionary containing following keys:
                - labels (IntTensor): tensor of shape [num_targets] containing the class indices;
                - boxes (FloatTensor): boxes of shape [num_targets, 4] in (center_x, center_y, width, height) format;
                - masks (ByteTensor, optional): segmentation masks of shape [num_targets, iH, iW].

    Returns:
        images (NestedTensor): NestedTensor consisting of:
            - images.tensor (FloatTensor): padded images of shape [batch_size, 3, max_iH, max_iW];
            - images.mask (BoolTensor): masks encoding inactive pixels of shape [batch_size, max_iH, max_iW].

        tgt_dict (Dict): New target dictionary with concatenated items across batch entries containing following keys:
            - labels (IntTensor): tensor of shape [num_targets_total] containing the class indices;
            - boxes (FloatTensor): boxes of shape [num_targets_total, 4] in (center_x, center_y, width, height) format;
            - sizes (IntTensor): tensor of shape [batch_size+1] with the cumulative target sizes of batch entries;
            - masks (ByteTensor, optional): padded segmentation masks of shape [num_targets_total, max_iH, max_iW].
    """

    # Get batch images and target dictionaries
    images, tgt_dicts = list(zip(*batch))

    # Compute nested tensor of batched images
    images = nested_tensor_from_image_list(images)

    # Concatenate target labels and target boxes across batch entries
    tgt_labels = torch.cat([tgt_dict['labels'] for tgt_dict in tgt_dicts], dim=0)
    tgt_boxes = torch.cat([tgt_dict['boxes'] for tgt_dict in tgt_dicts], dim=0)

    # Compute cumulative target sizes of batch entries
    tgt_sizes = [0] + [len(tgt_dict['labels']) for tgt_dict in tgt_dicts]
    tgt_sizes = torch.tensor(tgt_sizes, dtype=torch.int)
    tgt_sizes = torch.cumsum(tgt_sizes, dim=0)

    # Place target labels, boxes and sizes in new target dictionary
    tgt_dict = {'labels': tgt_labels, 'boxes': tgt_boxes, 'sizes': tgt_sizes}

    # Concatenate masks if provided and add to target dictionary
    if 'masks' in tgt_dicts[0]:
        max_iH, max_iW = images.tensor.shape[-2:]
        masks = [old_tgt_dict['masks'] for old_tgt_dict in tgt_dicts]
        padded_masks = [F.pad(mask, (0, max_iW-mask.shape[-1], 0, max_iH-mask.shape[-2])) for mask in masks]
        tgt_dict['masks'] = torch.cat(padded_masks, dim=0)

    return images, tgt_dict


def val_collate_fn(batch):
    """
    The collate function used during validation data loading.

    Args:
        batch (List): List of size [batch_size] containing tuples of:
            - image (FloatTensor): tensor containing the image of shape [3, iH, iW].
            - tgt_dict (Dict): target dictionary containing following keys:
                - labels (IntTensor): tensor of shape [num_targets] containing the class indices;
                - boxes (FloatTensor): boxes of shape [num_targets, 4] in (center_x, center_y, width, height) format;
                - masks (ByteTensor, optional): segmentation masks of shape [num_targets, iH, iW];
                - image_id (IntTensor): tensor of shape [1] containing the image id;
                - image_size (IntTensor): tensor of shape [2] containing the image size (before data augmentation).

    Returns:
        images (NestedTensor): NestedTensor consisting of:
            - images.tensor (FloatTensor): padded images of shape [batch_size, 3, max_iH, max_iW];
            - images.mask (BoolTensor): masks encoding inactive pixels of shape [batch_size, max_iH, max_iW].

         tgt_dict (Dict): New target dictionary with concatenated items across batch entries containing following keys:
            - labels (IntTensor): tensor of shape [num_targets_total] containing the class indices;
            - boxes (FloatTensor): boxes of shape [num_targets_total, 4] in (center_x, center_y, width, height) format;
            - sizes (IntTensor): tensor of shape [batch_size+1] with the cumulative target sizes of batch entries;
            - masks (ByteTensor, optional): padded segmentation masks of shape [num_targets_total, max_iH, max_iW].

        eval_dict (Dict): Dictionary containing following keys:
            - image_ids (IntTensor): tensor of shape [batch_size] containing the images ids;
            - image_sizes (IntTensor): tensor of shape [batch_size, 2] containing the image sizes.
    """

    # Get batch images and target dictionaries
    images, tgt_dicts = list(zip(*batch))

    # Compute nested tensor of batched images
    images = nested_tensor_from_image_list(images)

    # Concatenate target labels and target boxes across batch entries
    tgt_labels = torch.cat([tgt_dict['labels'] for tgt_dict in tgt_dicts], dim=0)
    tgt_boxes = torch.cat([tgt_dict['boxes'] for tgt_dict in tgt_dicts], dim=0)

    # Compute cumulative target sizes of batch entries
    tgt_sizes = [0] + [len(tgt_dict['labels']) for tgt_dict in tgt_dicts]
    tgt_sizes = torch.tensor(tgt_sizes, dtype=torch.int)
    tgt_sizes = torch.cumsum(tgt_sizes, dim=0)

    # Place target labels, boxes and sizes in new target dictionary
    tgt_dict = {'labels': tgt_labels, 'boxes': tgt_boxes, 'sizes': tgt_sizes}

    # Concatenate masks if provided and add to target dictionary
    if 'masks' in tgt_dicts[0]:
        max_iH, max_iW = images.tensor.shape[-2:]
        masks = [old_tgt_dict['masks'] for old_tgt_dict in tgt_dicts]
        padded_masks = [F.pad(mask, (0, max_iW-mask.shape[-1], 0, max_iH-mask.shape[-2])) for mask in masks]
        tgt_dict['masks'] = torch.cat(padded_masks, dim=0)

    # Place image ids and image sizes in evaluation dictionary
    image_ids = torch.cat([old_tgt_dict['image_id'] for old_tgt_dict in tgt_dicts], dim=0)
    image_sizes = torch.stack([old_tgt_dict['image_size'] for old_tgt_dict in tgt_dicts], dim=0)
    eval_dict = {'image_ids': image_ids, 'image_sizes': image_sizes}

    return images, tgt_dict, eval_dict


def nested_tensor_from_image_list(image_list):
    """
    Create nested tensor from list of image tensors.

    Args:
        image_list (List): List of size [batch_size] with image tensors of shape [3, iH, iW]. Each image tensor
                           is allowed to have a different image height (iH) and image width (iW).

    Returns:
        images (NestedTensor): NestedTensor consisting of:
            - images.tensor (FloatTensor): padded images of shape [batch_size, 3, max_iH, max_iW];
            - images.mask (BoolTensor): masks encoding inactive pixels of shape [batch_size, max_iH, max_iW].
    """

    # Compute different sizes
    batch_size = len(image_list)
    max_h, max_w = torch.tensor([list(img.shape[1:]) for img in image_list]).max(dim=0)[0].tolist()

    # Some renaming for code readability
    dtype = image_list[0].dtype
    device = image_list[0].device

    # Initialize the padded images and masks
    padded_images = torch.zeros(batch_size, 3, max_h, max_w, dtype=dtype, device=device)
    masks = torch.ones(batch_size, max_h, max_w, dtype=torch.bool, device=device)

    # Fill the padded images and masks
    for image, padded_image, mask in zip(image_list, padded_images, masks):
        h, w = image.shape[1:]
        padded_image[:, :h, :w].copy_(image)
        mask[:h, :w] = False

    # Create nested tensor from padded images and masks
    images = NestedTensor(padded_images, masks)

    return images


class NestedTensor(object):
    """
    Class implementing the NestedTensor data structure.

    Attributes:
        tensor (Tensor): A tensor with padded entries.
        mask (BoolTensor): A boolean mask encoding the padded entries of the tensor.
    """

    def __init__(self, tensor, mask):
        """
        Initializes the NestedTensor data structure.

        Args:
            tensor (Tensor): A tensor with padded entries.
            mask (BoolTensor): A boolean mask encoding the padded entries of the tensor.
        """

        self.tensor = tensor
        self.mask = mask

    def to(self, device):
        """
        Method to cast NestedTensor to specified device.

        Args:
            device (torch.device): The device to cast the NestedTensor to.

        Returns:
            cast_nested_tensor (NestedTensor): The new NestedTensor residing on the specified device.
        """

        cast_tensor = self.tensor.to(device)
        cast_mask = self.mask.to(device) if self.mask is not None else None
        cast_nested_tensor = NestedTensor(cast_tensor, cast_mask)

        return cast_nested_tensor

    def decompose(self):
        """
        Method decomposing the NestedTensor into its tensor and mask constituents.

        Returns:
            tensor (Tensor): The NestedTensor's tensor with padded entries.
            mask (BoolTensor): The NestedTensor's mask encoding the padded entries of the tensor.
        """

        return self.tensor, self.mask
