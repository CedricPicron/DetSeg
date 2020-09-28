"""
COCO dataset and build function.
"""
from pathlib import Path

from PIL import Image
from pycocotools.coco import COCO
import torch
from torchvision.datasets.vision import VisionDataset

import datasets.transforms as T


class CocoDataset(VisionDataset):
    """
    Class implementing the COCO dataset.

    Attributes:
        coco (COCO): Object containing the COCO dataset annotations.
        image_ids (List): List of image indices, sorted in ascending order.
    """

    def __init__(self, image_folder, annotation_file, transforms):
        """
        Initializes the CocoDataset dataset.

        Args:
            image_folder (Path): Path to image folder containing COCO images.
            annotation_file (Path): Path to annotation file with COCO annotations.
            transforms (object): The transforms to be applied on both image and its bounding boxes.
        """

        super().__init__(image_folder, transforms=transforms)
        self.coco = COCO(annotation_file)
        self.image_ids = list(sorted(self.coco.imgs.keys()))

    def __getitem__(self, index):
        """
        Implements the __getitem__ method of the CocoDataset dataset.

        Args:
            index (int): Index selecting one of the dataset images.

        Returns:
            image (FloatTensor): Tensor containing the transformed image tensor of shape [3, H, W].
            target (Dict): Dictionary containing following keys:
                - labels (IntTensor): tensor of shape [num_target_boxes] containing the class indices;
                - boxes (FloatTensor): tensor of shape [num_target_boxes, 4] containing the transformed target box
                                       coordinates in the (center_x, center_y, width, height) format.
        """

        # Load image
        image_id = self.image_ids[index]
        image_path = self.root / self.coco.loadImgs(image_id)[0]['file_name']
        image = Image.open(image_path).convert('RGB')

        # Load annotations
        annotation_ids = self.coco.getAnnIds(imgIds=image_id)
        annotations = self.coco.loadAnns(annotation_ids)
        annotations = [obj for obj in annotations if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        # Get object class labels
        labels = [obj["category_id"] for obj in annotations]
        labels = torch.tensor(labels, dtype=torch.int64)

        # Get object boxes in (left, top, width, height) format
        boxes = [obj["bbox"] for obj in annotations]
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)

        # Transform boxes to (left, top, right, bottom) format
        boxes[:, 2:] += boxes[:, :2]

        # Crop boxes such that they fit within the image
        w, h = image.size
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        # Only keep objects with well-defined boxes
        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        labels = labels[keep]
        boxes = boxes[keep]

        # Some additional properties useful for conversion to coco api
        area = torch.tensor([obj["area"] for obj in annotations])[keep]
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in annotations])[keep]

        # Place target properties into target dictionary
        target = {'labels': labels, 'boxes': boxes, 'area': area, 'iscrowd': iscrowd}

        # Perform image and bounding box transformations
        image, target = self.transforms(image, target)

        return image, target

    def __len__(self):
        """
        Implements the __len__ method of the CocoDataset dataset.

        Returns:
            The dataset length measured as the number of images in the dataset.
        """

        return len(self.image_ids)


def get_coco_transforms():
    """
    Function returning the COCO training and validation transforms.

    Returns:
        train_transforms (object): The COCO training transforms.
        val_transforms (object): The COCO validation transforms.
    """

    hflip = T.RandomHorizontalFlip()
    crop = T.Compose([T.RandomResize([400, 500, 600]), T.RandomSizeCrop(384, 600)])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
    default_resize = T.RandomResize(scales, max_size=1333)
    cropped_resize = T.Compose([crop, default_resize])

    train_resize = T.RandomSelect(default_resize, cropped_resize)
    val_resize = T.RandomResize([800], max_size=1333)

    to_tensor = T.ToTensor()
    normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    train_transforms = T.Compose([hflip, train_resize, to_tensor, normalize])
    val_transforms = T.Compose([val_resize, to_tensor, normalize])

    return train_transforms, val_transforms


def build_coco(args):
    """
    Build training and validation COCO dataset from command-line arguments.

    Args:
        args (argparse.Namespace): Command-line arguments.

    Returns:
        train_dataset (torch.utils.data.Dataset): The specified COCO training dataset.
        val_dataset (torch.utils.data.Dataset): The specified COCO validation dataset.
    """

    coco_root = Path() / 'datasets' / 'coco'
    train_image_folder = coco_root / 'images' / 'train2017'
    val_image_folder = coco_root / 'images' / 'val2017'
    train_annotation_file = coco_root / 'annotations' / 'instances_train2017.json'
    val_annotation_file = coco_root / 'annotations' / 'instances_val2017.json'

    train_transforms, val_transforms = get_coco_transforms()
    train_dataset = CocoDataset(train_image_folder, train_annotation_file, train_transforms)
    val_dataset = CocoDataset(val_image_folder, val_annotation_file, val_transforms)

    return train_dataset, val_dataset
