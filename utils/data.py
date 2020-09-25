"""
Data loading utilities.
"""

import torch


def collate_fn(batch):
    """
    The collate function used during data loading.

    Args:
        batch (List): List of size [batch_size] containing tuples of:
            - image (FloatTensor): tensor containing the transformed image tensor of shape [3, H, W].
            - target (Dict): dictionary containing at least following two keys:
                - labels (IntTensor): tensor of shape [num_target_boxes] containing the class indices;
                - boxes (FloatTensor): tensor of shape [num_target_boxes, 4] containing the transformed target box
                                       coordinates in the (center_x, center_y, width, height) format.

    Returns:
        images (NestedTensor): NestedTensor consisting of:
            - images.tensor (FloatTensor): padded images of shape [batch_size, 3, H, W];
            - images.mask (BoolTensor): boolean masks encoding inactive pixels of shape [batch_size, H, W].

        tgt_dict (Dict): Dictionary containing following keys:
            - labels (IntTensor): tensor of shape [num_target_boxes_total] (with num_target_boxes_total the total
                                  number of objects across batch entries) containing the target class indices;
            - boxes (FloatTensor): tensor of shape [num_target_boxes_total, 4] with the target box coordinates;
            - sizes (IntTensor): tensor of shape [batch_size+1] containing the cumulative sizes of batch entries.
    """

    # Get batch images and target dictionaries
    images, targets = list(zip(*batch))

    # Compute images nested tensor
    images = nested_tensor_from_image_list(images)

    # Concatenating the target labels and boxes across batch entries
    tgt_labels = torch.cat([target['labels'] for target in targets], dim=0)
    tgt_boxes = torch.cat([target['boxes'] for target in targets], dim=0)

    # Computing the cumulative sizes of batch entries
    tgt_sizes = [0] + [len(target['labels']) for target in targets]
    tgt_sizes = torch.tensor(tgt_sizes, dtype=torch.int)
    tgt_sizes = torch.cumsum(tgt_sizes, dim=0)

    # Place labels, boxes and size in new target dictionary
    tgt_dict = {'labels': tgt_labels, 'boxes': tgt_boxes, 'sizes': tgt_sizes}

    return images, tgt_dict


def nested_tensor_from_image_list(image_list):
    """
    Create nested tensor from list of image tensors.

    Args:
        image_list (List): List of size [batch_size] with image tensors of shape [3, height, width]. Each image tensor
                           is allowed to have a different height and width.

    Returns:
        images (NestedTensor): NestedTensor consisting of:
            - images.tensor (FloatTensor): padded images of shape [batch_size, 3, max_height, max_width];
            - images.mask (BoolTensor): boolean masks of inactive pixels of shape [batch_size, max_height, max_width].
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
