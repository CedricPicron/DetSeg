"""
Images structure.
"""
from copy import deepcopy

import PIL
import torch
import torchvision.transforms.functional as F


class Images(object):
    """
    Class implementing the Images structure.

    Attributes:
        images: It is one of following two possibilities:
            1) images (PIL.Image.Image): Single image in PIL format;
            2) images (FloatTensor): Image tensor of shape [batch_size, num_channels, iH, iW].

        image_ids (List): List of size [num_images] containing the image id corresponding to each image.
        masks (BoolTensor): Tensor of shape [batch_size, iH, iW] indicating active pixels (as opposed to padding).
        transforms (List): List of size [num_images] with transforms corresponding to each image.
    """

    def __init__(self, images, image_ids=None, masks=None, transforms=None):
        """
        Initializes the Images structure.

        Args:
            images: We support following two possibilities:
                1) images (PIL.Image.Image): Single image in PIL format;
                2) images (FloatTensor): Image tensor of shape [batch_size, num_channels, iH, iW].

            image_ids: Optional input supporting following two possibilities:
                1) image_ids (int): Integer containing the dataset image id (works only with single image input);
                2) image_ids (List): List of size [batch_size] containing the image ids of each of the input images.

            masks (BoolTensor): Optional tensor of shape [batch_size, iH, iW] indicating active pixels.
            transforms (List): Optional list of size [num_images] with transforms corresponding to each image.
        """

        # Set images, image ids and transforms attributes
        self.images = images
        self.image_ids = image_ids if isinstance(image_ids, list) else [image_ids]
        self.transforms = transforms if transforms is not None else [[]]

        # Set masks attribute
        img_width, img_height = self.size()
        img_shape = (len(self), img_height, img_width)

        if masks is not None:
            check = img_shape == masks.shape
            assert check, f"The input images and masks have inconsistent shapes (got {img_shape} and {masks.shape})."
            self.masks = masks

        else:
            masks = torch.ones(img_shape, dtype=torch.bool)
            self.masks = masks.to(images.device) if isinstance(images, torch.Tensor) else masks

    def __getitem__(self, item):
        """
        Implements the __getitem__ method of the Images structure.

        Args:
            item: We support three possibilities:
                1) item (int): integer containing the index of the image to be returned;
                2) item (slice): one-dimensional slice slicing a subset of images;
                3) item (BoolTensor): tensor of shape [num_images] containing boolean values of images to be selected.

        Returns:
            selected_images (Images): New Images structure containing the selected images.
        """

        images_tensor = self.images[item].view(1, -1) if isinstance(item, int) else self.images[item]
        masks = self.masks[item].view(1, -1) if isinstance(item, int) else self.masks[item]

        image_ids = list(self.image_ids[item])
        transforms = [self.transforms[item]] if isinstance(item, int) else self.transforms[item]
        selected_images = Images(images_tensor, image_ids, masks, transforms)

        return selected_images

    def __len__(self):
        """
        Implements the __len__ method of the Images structure.

        It is measured as the number of images within the structure.

        Returns:
            num_images (int): Number of images within the structure.
        """

        num_images = len(self.image_ids)

        return num_images

    def __repr__(self):
        """
        Implements the __repr__ method of the Images structure.

        Returns:
            images_string (str): String containing information about the Images structure.
        """

        images_string = "Images structure:\n"
        images_string += f"   Size: {len(self)}\n"
        images_string += f"   Image ids: {self.image_ids}\n"
        images_string += f"   Transforms: {self.transforms}\n"

        images_string += f"   Images content: {self.images}\n"
        images_string += f"   Masks content: {self.masks}"

        return images_string

    @staticmethod
    def cat(images_list):
        """
        Concatenate list of Images structures into single Images structure.

        Args:
            image_list (List): List of size [num_structures] with Image structures to be concatenated.

        Returns:
            cat_images (Images): Images structure containing the concatenated input Images structures.
        """

        # Check whether all images have same number of channels
        channels_set = {s.images.shape[1] for s in images_list}
        assert len(channels_set) == 1, f"All images should have same number of channels (got {channels_set})."
        channels = channels_set[0]

        # Check whether all images reside on the same device
        device_set = {s.images.device for s in images_list}
        assert len(device_set) == 1, f"All images should reside on the same device (got {device_set})."
        device = device_set[0]

        # Initialize padded tensor of concatenated images and corresponding masks
        sizes = [structure.size() for structure in images_list]
        cum_sizes = torch.cumsum([0, *sizes], dim=0)

        batch_size = cum_sizes[-1]
        max_iH, max_iW = torch.tensor([list(s.images.shape[2:]) for s in images_list]).max(dim=0)[0].tolist()

        images_tensor = torch.zeros(batch_size, channels, max_iH, max_iW, dtype=torch.float, device=device)
        masks_tensor = torch.zeros(batch_size, max_iH, max_iW, dtype=torch.bool, device=device)

        # Fill the padded images and masks
        for i0, i1, structure in zip(cum_sizes[:-1], cum_sizes[1:], images_list):
            iH, iW = structure.images.shape[2:]
            images_tensor[i0:i1, :, :iH, :iW].copy_(structure.images)
            masks_tensor[i0:i1, :iH, :iW] = structure.masks

        # Get list of image ids and list of transforms
        image_ids = []
        transforms = []

        for structure in images_list:
            image_ids.extend(structure.image_ids)
            transforms.extend(structure.transforms)

        # Get concatenated Images structure
        cat_images = Images(images_tensor, image_ids, masks_tensor, transforms)

        return cat_images

    def clone(self):
        """
        Clones the Images structure into a new Images structure.

        Returns:
            cloned_images (Images): Cloned Images structure.
        """

        args = (self.images.clone(), deepcopy(self.image_ids), self.masks.clone(), deepcopy(self.transforms))
        cloned_images = Images(*args)

        return cloned_images

    def crop(self, crop_region):
        """
        Crops the images w.r.t. the given crop region.

        Args:
            crop_region (Tuple): Tuple delineating the cropped region in (left, top, width, height) format.
        """

        # Crop the images w.r.t. the given crop region
        left, top, width, height = crop_region
        self.images = F.crop(self.images, top, left, height, width)
        self.masks = F.crop(self.masks, top, left, height, width)

        # Add crop operation to list of transforms
        for img_transforms in self.transforms:
            img_transforms.append(tuple('crop', crop_region))

    def hflip(self):
        """
        Flips the images horizontally.
        """

        # Flip the images horizontally
        self.images = F.hflip(self.images)
        self.masks = F.hflip(self.masks)

        # Add horizontal flip operation to list of transforms
        for img_transforms in self.transforms:
            img_transforms.append(tuple('hflip'))

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

    def pad(self, padding):
        """
        Pads the images according to the given padding vector.

        Args:
            padding (Tuple): Padding vector of size [4] with padding values in (left, top, right, bottom) format.
        """

        # Pad the images according to the given padding vector
        self.images = F.pad(padding)
        self.masks = F.pad(padding)

        # Add padding operation to list of transforms
        for img_transforms in self.transforms:
            img_transforms.append(tuple('pad', padding))

    def resize(self, size, max_size=None):
        """
        Resizes the images according to the given resize specifications.

        Args:
            size: It can be one of following two possibilities:
                1) size (int): integer containing the minimum size (width or height) to resize to;
                2) size (Tuple): tuple of size [2] containing respectively the width and height to resize to.

            max_size (int): Overrules above size whenever this value is exceded in width or height (defaults to None).

        Returns:
            resize_ratio (Tuple): Tuple of size [2] containing the resize ratio as (width_ratio, height_ratio).
        """

        # Get size in (width, height) format
        if isinstance(size, int):
            image_size = self.images.size()
            size = tuple(size*x/min(image_size) for x in image_size)  

        # Overrule with max size if given and necessary
        if max_size is not None:
            if max_size < max(size):
                size = tuple(max_size*x/max(size) for x in size)

        # Resize image according to size tuple and get resize ratio
        self.images = F.resize(self.images, (size[1], size[0]))
        resize_ratio = tuple(x/y for x, y in zip(size, image_size))

        # Add resize operation to list of transforms
        for img_transforms in self.transforms:
            img_transforms.append(tuple('resize', resize_ratio))

        return resize_ratio

    def size(self):
        """
        Gets the image size (i.e. width and height) of images within the Images structure.

        Returns:
            img_size (Tuple): Tuple of size [2] containing the image size in (width, height) format.
        """

        if isinstance(self.images, PIL.Image.Image):
            img_size = self.images.size

        elif isinstance(self.images, torch.Tensor):
            iH, iW = self.images.shape[2:]
            img_size = (iW, iH)

        return img_size

    def to(self, *args, **kwargs):
        """
        Performs type and/or device conversion for the tensors within the Images structure.

        Returns:
            Updated Images structure with converted tensors.
        """

        self.images.to(*args, **kwargs)
        self.masks.to(*args, **kwargs)

        return self

    def to_tensor(self):
        """
        Converts PIL Image to tensor with batch size 1.
        """

        if isinstance(self.images, PIL.Image.Image):
            self.images = F.to_tensor(self.image).unsqueeze(0)
