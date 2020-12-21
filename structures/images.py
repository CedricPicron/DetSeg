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
        self.image_ids = image_ids if isinstance(image_ids, list) else [image_ids] * len(self)
        self.transforms = transforms if transforms is not None else [[]]

        # Set masks attribute
        img_width, img_height = self.size()
        img_shape = (len(self), img_height, img_width)

        if masks is not None:
            check = img_shape == masks.shape
            assert check, f"The input images and masks have inconsistent shapes (got {img_shape} and {masks.shape})."
            self.masks = masks.to(images.device)

        else:
            masks = torch.ones(img_shape, dtype=torch.bool)
            self.masks = masks.to(images.device) if isinstance(images, torch.Tensor) else masks

    def __getitem__(self, key):
        """
        Implements the __getitem__ method of the Images structure.

        Args:
            key: We support three possibilities:
                1) key (int): integer containing the index of the image to be returned;
                2) key (slice): one-dimensional slice slicing a subset of images to be returned;
                3) key (BoolTensor): tensor of shape [num_images] containing boolean values of images to be returned.

        Returns:
            item (Images): New Images structure containing the selected images.
        """

        images_tensor = self.images[key].view(1, -1) if isinstance(key, int) else self.images[key]
        masks = self.masks[key].view(1, -1) if isinstance(key, int) else self.masks[key]

        image_ids = list(self.image_ids[key])
        transforms = [self.transforms[key]] if isinstance(key, int) else self.transforms[key]
        item = Images(images_tensor, image_ids, masks, transforms)

        return item

    def __len__(self):
        """
        Implements the __len__ method of the Images structure.

        It is measured as the number of images within the structure.

        Returns:
            num_images (int): Number of images within the structure.
        """

        if isinstance(self.images, PIL.Image.Image):
            num_images = 1

        elif isinstance(self.images, torch.Tensor):
            num_images = len(self.images)

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

        # Check whether all Images structures have same number of channels
        channels_set = {s.images.shape[1] for s in images_list}
        assert len(channels_set) == 1, f"All images should have same number of channels (got {channels_set})."
        channels = channels_set.pop()

        # Check whether all Images structures reside on the same device
        device_set = {s.images.device for s in images_list}
        assert len(device_set) == 1, f"All images should reside on the same device (got {device_set})."
        device = device_set.pop()

        # Get list of image ids and list of transforms
        image_ids = []
        transforms = []

        for structure in images_list:
            image_ids.extend(structure.image_ids)
            transforms.extend(structure.transforms)

        # Initialize padded tensor of concatenated images and corresponding masks
        sizes = [0] + [len(structure.images) for structure in images_list]
        cum_sizes = torch.cumsum(torch.tensor(sizes), dim=0)

        batch_size = cum_sizes[-1]
        max_iH, max_iW = torch.tensor([list(s.images.shape[2:]) for s in images_list]).max(dim=0)[0].tolist()

        images_tensor = torch.zeros(batch_size, channels, max_iH, max_iW, dtype=torch.float, device=device)
        masks_tensor = torch.zeros(batch_size, max_iH, max_iW, dtype=torch.bool, device=device)

        # Fill the padded images and masks, and update list of transforms
        for i0, i1, structure in zip(cum_sizes[:-1], cum_sizes[1:], images_list):
            iH, iW = structure.images.shape[2:]
            images_tensor[i0:i1, :, :iH, :iW].copy_(structure.images)
            masks_tensor[i0:i1, :iH, :iW] = structure.masks

            padding_transform = ('pad', (0, 0, max_iW-iW, max_iH-iH))
            [transforms[i].append(padding_transform) for i in range(i0, i1)]

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

        Returns:
            self (Images): Updated Images structure with cropped images.
        """

        # Crop the images masks w.r.t. the given crop region
        left, top, width, height = crop_region
        self.images = F.crop(self.images, top, left, height, width)
        self.masks = F.crop(self.masks, top, left, height, width)

        # Add crop operation to list of transforms
        for img_transforms in self.transforms:
            img_transforms.append(('crop', crop_region))

        return self

    def hflip(self):
        """
        Flips the images horizontally.

        Returns:
            self (Images): Updated Images structure with horizontally flipped images.
        """

        # Horizontally flip the images and masks
        self.images = F.hflip(self.images)
        self.masks = F.hflip(self.masks)

        # Add horizontal flip operation to list of transforms
        img_width, _ = self.size()
        for img_transforms in self.transforms:
            img_transforms.append(('hflip', img_width))

        return self

    def normalize(self, mean, std, inplace=False):
        """
        Method normalizing the channels of the images tensor at non-padded positions.

        Args:
            mean (FloatTensor): Tensor of shape [num_channels] containing the normalization means.
            std (FloatTensor): Tensor of shape [num_channels] containing the normalization standard deviations.
            inplace (bool): Boolean indicating whether to perform operation in place or not (default=False).

        Returns:
            images (FloatTensor): Images tensor normalized at non-padded positions.
        """

        images = self.images.clone() if not inplace else self.images
        images_view = images.movedim([-3, -2, -1], [-1, -3, -2])
        images_view[self.masks, :] = images_view[self.masks, :].sub_(mean).div_(std)

        return images

    def pad(self, padding):
        """
        Pads the images according to the given padding vector.

        Args:
            padding (Tuple): Padding vector of size [4] with padding values in (left, top, right, bottom) format.7

        Returns:
            self (Images): Updated Images structure with padded images.
        """

        # Pad the images and masks according to the given padding vector
        self.images = F.pad(padding)
        self.masks = F.pad(padding)

        # Add padding operation to list of transforms
        for img_transforms in self.transforms:
            img_transforms.append(('pad', padding))

        return self

    def resize(self, size, max_size=None):
        """
        Resizes the images according to the given resize specifications.

        Args:
            size: It can be one of following two possibilities:
                1) size (int): integer containing the minimum size (width or height) to resize to;
                2) size (Tuple): tuple of size [2] containing respectively the width and height to resize to.

            max_size (int): Overrules above size whenever this value is exceded in width or height (defaults to None).

        Returns:
            self (Images): Updated Images structure with resized images.
            resize_ratio (Tuple): Tuple of size [2] containing the resize ratio as (width_ratio, height_ratio).
        """

        # Get size in (width, height) format
        if isinstance(size, int):
            image_size = self.size()
            size = tuple(size*x/min(image_size) for x in image_size)

        # Overrule with max size if given and necessary
        if max_size is not None:
            if max_size < max(size):
                size = tuple(max_size*x/max(size) for x in size)

        # Resize the images and masks
        size = tuple(int(x) for x in size)
        self.images = F.resize(self.images, (size[1], size[0]))
        self.masks = F.resize(self.masks, (size[1], size[0]))

        # Get the resize ratio
        resize_ratio = tuple(x/y for x, y in zip(size, image_size))

        # Add resize operation to list of transforms
        for img_transforms in self.transforms:
            img_transforms.append(('resize', resize_ratio))

        return self, resize_ratio

    def size(self, with_padding=True):
        """
        Gets the image size (with or without padding) of images within the Images structure.

        Args:
            with_padding (bool): Boolean indicating whether padding should be counted towards image size or not.

        Returns:
            If 'with_padding' is True:
                img_size (Tuple): Tuple of size [2] containing the image size in (width, height) format.

            If 'with_padding' is False:
                img_sizes (List): List of size [num_images] containing the (width, height) image sizes without padding.
        """

        # Get global image size with padding
        if isinstance(self.images, PIL.Image.Image):
            img_size = self.images.size

        elif isinstance(self.images, torch.Tensor):
            iH, iW = self.images.shape[2:]
            img_size = (iW, iH)

        # Return image size with padding if requested
        if with_padding:
            return img_size

        # Get image sizes without padding if requested
        img_sizes = [img_size for i in range(len(self))]

        for i, transforms in enumerate(self.transforms):
            for transform in transforms.reverse():
                if transform[0] != 'pad':
                    break

                padding = transform[1]
                img_sizes[i][0] = img_sizes[i][0] - padding[0] - padding[2]
                img_sizes[i][1] = img_sizes[i][1] - padding[1] - padding[3]

        return img_sizes

    def to(self, *args, **kwargs):
        """
        Performs type and/or device conversion for the tensors within the Images structure.

        Returns:
            self (Images): Updated Images structure with converted tensors.
        """

        self.images = self.images.to(*args, **kwargs)
        self.masks = self.masks.to(*args, **kwargs)

        return self

    def to_tensor(self):
        """
        Converts PIL Image to tensor with batch size 1.

        Returns:
            self (Images): Updated Images structure with PIL Image converted to tensor.
        """

        if isinstance(self.images, PIL.Image.Image):
            self.images = F.to_tensor(self.images).unsqueeze(0)

        return self
