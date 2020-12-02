"""
Data loading and data structure utilities.
"""
import random

import torch
import torch.nn.functional as F
from torch.utils.data import Sampler


class SubsetSampler(Sampler):
    """
    Class implementing the SubsetSampler sampler.

    Attributes:
        dataset (torch.utils.data.Dataset): Dataset to sample from.
        subset_size (int): Positive integer containing the subset size.
        subset_offset (int): Positive integer containing the subset offset.
    """

    def __init__(self, dataset, subset_size, subset_offset=0, random_offset=False):
        """
        Initializes the SubsetSampler sampler.

        Args:
            dataset (torch.utils.data.Dataset): Dataset to sample from.
            subset_size (int): Positive integer specifying the desired subset size.
            subset_offset (int): Positive integer specifying the desired subset offset (default=0).
            random_offset (bool): Boolean indicating whether to generate a random subset offset (default=False).
        """

        self.dataset = dataset
        self.subset_size = min(subset_size, len(dataset))

        max_offset = len(dataset) - self.subset_size
        self.subset_offset = random.randint(0, max_offset) if random_offset else min(subset_offset, max_offset)

    def __iter__(self):
        """
        Implements the __iter__ method of the SubsetSampler sampler.

        Returns:
            An iterator containing the subset indices.
        """

        return iter(range(self.subset_offset, self.subset_offset+self.subset_size))

    def __len__(self):
        """
        Implements the __len__ method of the SubsetSampler sampler.

        Returns:
            An integer containing the subset size.
        """

        return self.subset_size


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
            - images.mask (BoolTensor): masks encoding padded pixels of shape [batch_size, max_iH, max_iW].

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
            - images.mask (BoolTensor): masks encoding padded pixels of shape [batch_size, max_iH, max_iW].

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
            - images.mask (BoolTensor): masks encoding padded pixels of shape [batch_size, max_iH, max_iW].
    """

    # Compute different sizes
    batch_size = len(image_list)
    max_iH, max_iW = torch.tensor([list(img.shape[1:]) for img in image_list]).max(dim=0)[0].tolist()

    # Some renaming for code readability
    dtype = image_list[0].dtype
    device = image_list[0].device

    # Initialize the padded images and masks
    padded_images = torch.zeros(batch_size, 3, max_iH, max_iW, dtype=dtype, device=device)
    masks = torch.ones(batch_size, max_iH, max_iW, dtype=torch.bool, device=device)

    # Fill the padded images and masks
    for image, padded_image, mask in zip(image_list, padded_images, masks):
        iH, iW = image.shape[1:]
        padded_image[:, :iH, :iW].copy_(image)
        mask[:iH, :iW] = False

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

    def decompose(self):
        """
        Method decomposing the NestedTensor into its tensor and mask constituents.

        Returns:
            tensor (Tensor): The NestedTensor's tensor with padded entries.
            mask (BoolTensor): The NestedTensor's mask encoding the padded entries of the tensor.
        """

        return self.tensor, self.mask

    def normalize(self, mean, std, inplace=False):
        """
        Method normalizing the NestedTensor's tensor channels at non-padded positions.

        We assume the NestedTensor tensor has following shape signature: [*, num_channels, iH, iW].

        Args:
            mean (FloatTensor): Tensor of shape [num_channels] containing the normalization means.
            std (FloatTensor): Tensor of shape [num_channels] containing the normalization standard deviations.
            inplace (bool): Boolean indicating whether to perform operation in place or not (default=False).

        Returns:
            norm_nested_tensor (NestedTensor): NestedTensor normalized at non-padded positions.
        """

        tensor, mask = self.decompose()
        tensor = tensor.clone() if not inplace else tensor

        tensor_view = tensor.movedim([-3, -2, -1], [-1, -3, -2])
        tensor_view[~mask, :] = tensor_view[~mask, :].sub_(mean).div_(std)
        norm_nested_tensor = NestedTensor(tensor, mask)

        return norm_nested_tensor

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
