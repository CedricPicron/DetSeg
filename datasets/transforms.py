"""
Transforms for both image and bounding box.
"""
import random

import PIL
import torch
from torchvision.ops.misc import interpolate
import torchvision.transforms as T
import torchvision.transforms.functional as F

from utils.box_ops import box_xyxy_to_cxcywh


def crop(image, tgt_dict, region):
    cropped_image = F.crop(image, *region)

    tgt_dict = tgt_dict.copy()
    i, j, h, w = region

    # should we do something wrt the original size?
    tgt_dict["size"] = torch.tensor([h, w])

    fields = ["labels", "area", "iscrowd"]

    if "boxes" in tgt_dict:
        boxes = tgt_dict["boxes"]
        max_size = torch.as_tensor([w, h], dtype=torch.float32)
        cropped_boxes = boxes - torch.as_tensor([j, i, j, i])
        cropped_boxes = torch.min(cropped_boxes.reshape(-1, 2, 2), max_size)
        cropped_boxes = cropped_boxes.clamp(min=0)
        area = (cropped_boxes[:, 1, :] - cropped_boxes[:, 0, :]).prod(dim=1)
        tgt_dict["boxes"] = cropped_boxes.reshape(-1, 4)
        tgt_dict["area"] = area
        fields.append("boxes")

    if "masks" in tgt_dict:
        # FIXME should we update the area here if there are no boxes?
        tgt_dict['masks'] = tgt_dict['masks'][:, i:i + h, j:j + w]
        fields.append("masks")

    # remove elements for which the boxes or masks that have zero area
    if "boxes" in tgt_dict or "masks" in tgt_dict:
        # favor boxes selection when defining which elements to keep
        # this is compatible with previous implementation
        if "boxes" in tgt_dict:
            cropped_boxes = tgt_dict['boxes'].reshape(-1, 2, 2)
            keep = torch.all(cropped_boxes[:, 1, :] > cropped_boxes[:, 0, :], dim=1)
        else:
            keep = tgt_dict['masks'].flatten(1).any(1)

        for field in fields:
            tgt_dict[field] = tgt_dict[field][keep]

    return cropped_image, tgt_dict


def hflip(image, tgt_dict):
    flipped_image = F.hflip(image)

    w, h = image.size

    tgt_dict = tgt_dict.copy()
    if "boxes" in tgt_dict:
        boxes = tgt_dict["boxes"]
        boxes = boxes[:, [2, 1, 0, 3]] * torch.as_tensor([-1, 1, -1, 1]) + torch.as_tensor([w, 0, w, 0])
        tgt_dict["boxes"] = boxes

    if "masks" in tgt_dict:
        tgt_dict['masks'] = tgt_dict['masks'].flip(-1)

    return flipped_image, tgt_dict


def resize(image, tgt_dict, size, max_size=None):
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

    if tgt_dict is None:
        return rescaled_image, None

    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(rescaled_image.size, image.size))
    ratio_width, ratio_height = ratios

    tgt_dict = tgt_dict.copy()
    if "boxes" in tgt_dict:
        boxes = tgt_dict["boxes"]
        scaled_boxes = boxes * torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height])
        tgt_dict["boxes"] = scaled_boxes

    if "area" in tgt_dict:
        area = tgt_dict["area"]
        scaled_area = area * (ratio_width * ratio_height)
        tgt_dict["area"] = scaled_area

    h, w = size
    tgt_dict["size"] = torch.tensor([h, w])

    if "masks" in tgt_dict:
        tgt_dict['masks'] = interpolate(
            tgt_dict['masks'][:, None].float(), size, mode="nearest")[:, 0] > 0.5

    return rescaled_image, tgt_dict


def pad(image, tgt_dict, padding):
    # assumes that we only pad on the bottom right corners
    padded_image = F.pad(image, (0, 0, padding[0], padding[1]))
    if tgt_dict is None:
        return padded_image, None
    tgt_dict = tgt_dict.copy()
    # should we do something wrt the original size?
    tgt_dict["size"] = torch.tensor(padded_image.size[::-1])
    if "masks" in tgt_dict:
        tgt_dict['masks'] = torch.nn.functional.pad(tgt_dict['masks'], (0, padding[0], 0, padding[1]))
    return padded_image, tgt_dict


class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, tgt_dict):
        region = T.RandomCrop.get_params(img, self.size)

        return crop(img, tgt_dict, region)


class RandomSizeCrop(object):
    def __init__(self, min_size: int, max_size: int):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, img: PIL.Image.Image, tgt_dict: dict):
        w = random.randint(self.min_size, min(img.width, self.max_size))
        h = random.randint(self.min_size, min(img.height, self.max_size))
        region = T.RandomCrop.get_params(img, [h, w])

        return crop(img, tgt_dict, region)


class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, tgt_dict):
        image_width, image_height = img.size
        crop_height, crop_width = self.size
        crop_top = int(round((image_height - crop_height) / 2.))
        crop_left = int(round((image_width - crop_width) / 2.))

        return crop(img, tgt_dict, (crop_top, crop_left, crop_height, crop_width))


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, tgt_dict):
        if random.random() < self.p:
            return hflip(img, tgt_dict)

        return img, tgt_dict


class RandomResize(object):
    def __init__(self, sizes, max_size=None):
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes
        self.max_size = max_size

    def __call__(self, img, tgt_dict):
        size = random.choice(self.sizes)

        return resize(img, tgt_dict, size, self.max_size)


class RandomPad(object):
    def __init__(self, max_pad):
        self.max_pad = max_pad

    def __call__(self, img, tgt_dict):
        pad_x = random.randint(0, self.max_pad)
        pad_y = random.randint(0, self.max_pad)

        return pad(img, tgt_dict, (pad_x, pad_y))


class RandomSelect(object):
    """
    Randomly selects between transforms1 and transforms2,
    with probability p for transforms1 and (1 - p) for transforms2
    """
    def __init__(self, transforms1, transforms2, p=0.5):
        self.transforms1 = transforms1
        self.transforms2 = transforms2
        self.p = p

    def __call__(self, img, tgt_dict):
        if random.random() < self.p:
            return self.transforms1(img, tgt_dict)

        return self.transforms2(img, tgt_dict)


class ToTensor(object):
    def __call__(self, img, tgt_dict):
        return F.to_tensor(img), tgt_dict


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, tgt_dict):
        image = F.normalize(image, mean=self.mean, std=self.std)
        tgt_dict = tgt_dict.copy()
        h, w = image.shape[-2:]

        if "boxes" in tgt_dict:
            boxes = tgt_dict["boxes"]
            boxes = box_xyxy_to_cxcywh(boxes)
            boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
            tgt_dict["boxes"] = boxes

        return image, tgt_dict


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, tgt_dict):
        for transform in self.transforms:
            image, tgt_dict = transform(image, tgt_dict)

        return image, tgt_dict

    def __repr__(self):
        format_string = self.__class__.__name__ + "("

        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)

        format_string += "\n)"

        return format_string
