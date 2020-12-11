"""
Transforms on both image and corresponding target dictionaries.
"""
import random

import torch
from torchvision.ops.misc import interpolate
import torchvision.transforms as T
import torchvision.transforms.functional as F

from utils.box_ops import box_xyxy_to_cxcywh


# 1. Transforms based on cropping
def crop(image, tgt_dict, crop_region):
    """
    Function cropping the input image w.r.t. the given crop region. The input target dictionary is updated accordingly.

    Args:
        image (PIL.Image.Image): Image in PIL format to be cropped.
        tgt_dict (Dict): Target dictionary corresponding to the image, with following required and optional keys:
            - labels (LongTensor, required): tensor of shape [num_targets] containing the class indices;
            - boxes (FloatTensor, optional): boxes of shape [num_targets, 4] in (left, top, right, bottom) format;
            - masks (ByteTensor, optional): segmentation masks of shape [num_targets, height, width].
        crop_region (Tuple): Tuple delineating the cropped region in (top, left, height, width) format.

    Returns:
        cropped_image (PIL.Image.Image): Image cropped by the given crop region.
        tgt_dict (Dict): Updated target dictionary corresponding to the new cropped image.
    """

    # Crop the image w.r.t. the given crop region
    top, left, height, width = crop_region
    cropped_image = F.crop(image, top, left, height, width)

    # Update bounding boxes and identify targets no longer in the image
    if 'boxes' in tgt_dict:
        cropped_boxes = tgt_dict['boxes'] - torch.as_tensor([left, top, left, top])
        cropped_boxes = cropped_boxes.clamp(min=0)
        max_size = torch.as_tensor([width, height], dtype=torch.float32)
        cropped_boxes = torch.min(cropped_boxes.reshape(-1, 2, 2), max_size)
        keep = torch.all(cropped_boxes[:, 1, :] > cropped_boxes[:, 0, :], dim=1)
        tgt_dict['boxes'] = cropped_boxes.reshape(-1, 4)

    # Update segmentation masks and identify targets no longer in the image (if not already done)
    if 'masks' in tgt_dict:
        tgt_dict['masks'] = tgt_dict['masks'][:, top:top+height, left:left+width]
        keep = tgt_dict['masks'].flatten(1).any(1) if 'boxes' not in tgt_dict else keep

    # Remove targets that no longer appear in the cropped image
    if 'boxes' in tgt_dict or 'masks' in tgt_dict:
        for key in tgt_dict.keys():
            if key in ['labels', 'boxes', 'masks']:
                tgt_dict[key] = tgt_dict[key][keep]

    return cropped_image, tgt_dict


class CenterCrop(object):
    """
    Class implementing the CenterCrop transform.

    Attributes:
        crop_size (Tuple): Tuple of size [2] containing the crop height and crop width respectively.
    """

    def __init__(self, crop_size):
        """
        Initializes the CenterCrop transform.

        Args:
            crop_size (Tuple): Tuple of size [2] containing the crop height and crop width respectively.
        """

        self.crop_size = crop_size

    def __call__(self, image, tgt_dict):
        """
        The __call__ method corresponding to the CenterCrop transform.

        Args:
            image (PIL.Image.Image): Image in PIL format to be cropped.
            tgt_dict (Dict): Target dictionary corresponding to the input image.

        Returns:
            cropped_image (PIL.Image.Image): Image cropped in the center and of size 'crop_size'.
            tgt_dict (Dict): Updated target dictionary corresponding to the new cropped image.
        """

        crop_height, crop_width = self.size
        crop_top = int(round((image.height - crop_height) / 2.))
        crop_left = int(round((image.width - crop_width) / 2.))

        crop_region = (crop_top, crop_left, crop_height, crop_width)
        cropped_image, tgt_dict = crop(image, tgt_dict, crop_region)

        return cropped_image, tgt_dict


class RandomCrop(object):
    """
    Class implementing the RandomCrop transform.

    Attributes:
        crop_size (Tuple): Tuple of size [2] containing the crop height and crop width respectively.
    """

    def __init__(self, crop_size):
        """
        Initializes the RandomCrop transform.

        Args:
            crop_size (Tuple): Tuple of size [2] containing the crop height and crop width respectively.
        """

        self.crop_size = crop_size

    def __call__(self, image, tgt_dict):
        """
        The __call__ method corresponding to the RandomCrop transform.

        Args:
            image (PIL.Image.Image): Image in PIL format to be cropped.
            tgt_dict (Dict): Target dictionary corresponding to the input image.

        Returns:
            cropped_image (PIL.Image.Image): Image cropped at a random position and of size 'crop_size'.
            tgt_dict (Dict): Updated target dictionary corresponding to the new cropped image.
        """

        crop_region = T.RandomCrop.get_params(image, self.crop_size)
        cropped_image, tgt_dict = crop(image, tgt_dict, crop_region)

        return cropped_image, tgt_dict


class RandomSizeCrop(object):
    """
    Class implementing the RandomSizeCrop transform.

    Attributes:
        min_crop_size (int): Integer containing the minimum crop height and crop width.
        max_crop_size (int): Integer containing the maximum crop height and crop width.
    """

    def __init__(self, min_crop_size, max_crop_size):
        """
        Initializes the RandomSizeCrop transform.

        Args:
            min_crop_size (int): Integer containing the minimum crop height and crop width.
            max_crop_size (int): Integer containing the maximum crop height and crop width.
        """

        self.min_crop_size = min_crop_size
        self.max_crop_size = max_crop_size

    def __call__(self, image, tgt_dict):
        """
        The __call__ method corresponding to the RandomSizeCrop transform.

        Args:
            image (PIL.Image.Image): Image in PIL format to be cropped.
            tgt_dict (Dict): Target dictionary corresponding to the input image.

        Returns:
            cropped_image (PIL.Image.Image): Image cropped at a random position and of allowed random size.
            tgt_dict (Dict): Updated target dictionary corresponding to the new cropped image.
        """

        crop_width = random.randint(self.min_crop_size, min(image.width, self.max_crop_size))
        crop_height = random.randint(self.min_crop_size, min(image.height, self.max_crop_size))

        crop_region = T.RandomCrop.get_params(image, [crop_height, crop_width])
        cropped_image, tgt_dict = crop(image, tgt_dict, crop_region)

        return cropped_image, tgt_dict


# 2. Transforms based on horizontal flipping
def hflip(image, tgt_dict):
    """
    Function flipping the input image horizontally and updating the target dictionary accordingly.

    Args:
        image (PIL.Image.Image): Image in PIL format to be flipped horizontally.
        tgt_dict (Dict): Target dictionary corresponding to the image, potentially containing following keys:
            - boxes (FloatTensor, optional): boxes of shape [num_targets, 4] in (left, top, right, bottom) format;
            - masks (ByteTensor, optional): segmentation masks of shape [num_targets, height, width].

    Returns:
        flipped_image (PIL.Image.Image): The horizontally flipped input image.
        tgt_dict (Dict): Updated target dictionary corresponding to the new flipped image.
    """

    # Flip the image horizontally
    flipped_image = F.hflip(image)

    # Update the bounding boxes
    if 'boxes' in tgt_dict:
        boxes = tgt_dict['boxes']
        mirrored_boxes = boxes[:, [2, 1, 0, 3]] * torch.as_tensor([-1, 1, -1, 1])
        tgt_dict['boxes'] = mirrored_boxes + torch.as_tensor([image.width, 0, image.width, 0])

    # Update the segmentation masks
    if 'masks' in tgt_dict:
        tgt_dict['masks'] = tgt_dict['masks'].flip([-1])

    return flipped_image, tgt_dict


class RandomHorizontalFlip(object):
    """
    Class implementing the RandomHorizontalFlip transform.

    Attributes:
        flip_prob (float): Value between 0 and 1 determining the flip probability.
    """

    def __init__(self, flip_prob=0.5):
        """
        Initializes the RandomHorizontalFlip transform.

        Args:
            flip_prob (float): Value between 0 and 1 determining the flip probability.
        """

        self.flip_prob = flip_prob

    def __call__(self, image, tgt_dict):
        """
        The __call__ method of the RandomHorizontalFlip transform.

        Args:
            image (PIL.Image.Image): Image in PIL format, potentially to be flipped horizontally.
            tgt_dict (Dict): Target dictionary corresponding to the input image.

        Returns:
            image (PIL.Image.Image): The input image, horizontally flipped or not.
            tgt_dict (Dict): The (potentially updated) target dictionary corresponding to the returned image.
        """

        if random.random() < self.flip_prob:
            image, tgt_dict = hflip(image, tgt_dict)

        return image, tgt_dict


def resize(image, target, size, max_size=None):
    # size can be min_size (scalar) or (w, h) tuple

    def get_size_with_aspect_ratio(image_size, size, max_size=None):
        w, h = image_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def get_size(image_size, size, max_size=None):
        if isinstance(size, (list, tuple)):
            return size[::-1]
        else:
            return get_size_with_aspect_ratio(image_size, size, max_size)

    size = get_size(image.size, size, max_size)
    rescaled_image = F.resize(image, size)

    if target is None:
        return rescaled_image, None

    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(rescaled_image.size, image.size))
    ratio_width, ratio_height = ratios

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        scaled_boxes = boxes * torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height])
        target["boxes"] = scaled_boxes

    h, w = size
    target["size"] = torch.tensor([h, w])

    if "masks" in target:
        target['masks'] = interpolate(
            target['masks'][:, None].float(), size, mode="nearest")[:, 0] > 0.5

    return rescaled_image, target


def pad(image, target, padding):
    # assumes that we only pad on the bottom right corners
    padded_image = F.pad(image, (0, 0, padding[0], padding[1]))
    if target is None:
        return padded_image, None
    target = target.copy()
    # should we do something wrt the original size?
    target["size"] = torch.tensor(padded_image.size[::-1])
    if "masks" in target:
        target['masks'] = torch.nn.functional.pad(target['masks'], (0, padding[0], 0, padding[1]))
    return padded_image, target


class RandomResize(object):
    def __init__(self, sizes, max_size=None):
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes
        self.max_size = max_size

    def __call__(self, img, target):
        size = random.choice(self.sizes)

        return resize(img, target, size, self.max_size)


class RandomPad(object):
    def __init__(self, max_pad):
        self.max_pad = max_pad

    def __call__(self, img, target):
        pad_x = random.randint(0, self.max_pad)
        pad_y = random.randint(0, self.max_pad)

        return pad(img, target, (pad_x, pad_y))


class RandomSelect(object):
    """
    Randomly selects between transforms1 and transforms2,
    with probability p for transforms1 and (1 - p) for transforms2
    """
    def __init__(self, transforms1, transforms2, p=0.5):
        self.transforms1 = transforms1
        self.transforms2 = transforms2
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return self.transforms1(img, target)

        return self.transforms2(img, target)


class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        target = target.copy()
        h, w = image.shape[-2:]

        if 'boxes' in target:
            boxes = target['boxes']
            boxes = box_xyxy_to_cxcywh(boxes)
            boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
            target['boxes'] = boxes

        return image, target


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for transform in self.transforms:
            image, target = transform(image, target)

        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("

        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)

        format_string += "\n)"

        return format_string
