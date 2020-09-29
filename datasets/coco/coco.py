"""
COCO dataset and build function.
"""
from pathlib import Path

from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import torch
import torch.nn.functional as F
from torchvision.datasets.vision import VisionDataset

import datasets.transforms as T
from utils.box_ops import box_cxcywh_to_xywh


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
                                       coordinates in the (center_x, center_y, width, height) format;
                - image_id (IntTensor): tensor of shape [1] containing the image id;
                - image_size (IntTensor): tensor of shape [2] containing the image size (before data augmentation);
                - area (FloatTensor): tesnor of shape [num_target_boxes] containing the area of each target box;
                - iscrowd (IntTensor): tensor of shape [num_target_boxes] containing the iscrowd annotations.
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

        # Place target properties into target dictionary
        target = {'labels': labels, 'boxes': boxes}

        # Add some additional properties useful during evaluation
        target['image_id'] = torch.tensor([image_id])
        target['image_size'] = torch.tensor([int(h), int(w)])

        target['area'] = torch.tensor([obj['area'] for obj in annotations])[keep]
        target['iscrowd'] = torch.tensor([obj['iscrowd'] if 'iscrowd' in obj else 0 for obj in annotations])[keep]

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


class CocoEvaluator(object):
    """
    Evaluator object capable of evaluating predictions and storing them.

    Attributes:
        coco (COCO): Object containing the COCO dataset annotations.
        eval_types (List): List of strings containing the evaluation metrics to be used.
        sub_evaluators (Dict): Dictionary of sub-evaluators, each of them evaluating one metric from eval_types.
        image_ids (List): List of evaluated image ids.
        image_evals (Dict): Dictionary of lists (in same order as image_ids) of image evaluations for each metric.
    """

    def __init__(self, coco, eval_types=['bbox']):
        """
        Initializes the CocoEvaluator evaluator.

        Args:
            coco (COCO): Object containing the COCO dataset annotations.
            eval_types (List): List of strings containing the evaluation metrics to be used.
        """

        self.coco = coco
        self.eval_types = eval_types
        self.sub_evaluators = {eval_type: COCOeval(coco, iouType=eval_type) for eval_type in eval_types}

        self.image_ids = []
        self.image_evals = {eval_type: [] for eval_type in eval_types}

    def update(self, pred_dict, eval_dict):
        """
        Updates the evaluator object with the given predictions.

        Args:
            pred_dict (Dict): Dictionary containing at least following keys:
                - logits (FloatTensor): classification logits of shape [num_slots_total, num_classes];
                - boxes (FloatTensor): normalized box coordinates of shape [num_slots_total, 4];
                - batch_idx (IntTensor): batch indices of slots (in ascending order) of shape [num_slots_total];

            eval_dict (Dict): Dictionary containing following keys:
                - image_ids (IntTensor): tensor of shape [batch_size] containing the images ids;
                - image_sizes (IntTensor): tensor of shape [batch_size, 2] containing the image sizes.
        """

        # Compute max scores and resulting labels among object classes
        probs = F.softmax(pred_dict['logits'], dim=-1)
        scores, labels = probs[:, :-1].max(dim=-1)

        # Convert boxes to (left, top, right, bottom) format
        boxes = box_cxcywh_to_xywh(pred_dict['boxes'])

        # Convert boxes from relative to absolute coordinates
        batch_idx = pred_dict['batch_idx']
        image_sizes = eval_dict['image_sizes']

        h, w = image_sizes[batch_idx, :].unbind(1)
        scale = torch.stack([w, h, w, h], dim=1)
        boxes = scale*boxes

        # Get image ids and update image_ids attribute
        image_ids = eval_dict['image_ids']
        self.image_ids.extend(image_ids)

        # Go from tensors to lists
        image_ids = image_ids[batch_idx].tolist()
        labels = labels.tolist()
        boxes = boxes.tolist()
        scores = scores.tolist()

        for eval_type in self.eval_types:
            if eval_type == 'bbox':
                continue


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
        evaluator (object): The COCO evaluator capable of evaluating predictions and storing them.
    """

    coco_root = Path() / 'datasets' / 'coco'
    train_image_folder = coco_root / 'images' / 'train2017'
    val_image_folder = coco_root / 'images' / 'val2017'
    train_annotation_file = coco_root / 'annotations' / 'instances_train2017.json'
    val_annotation_file = coco_root / 'annotations' / 'instances_val2017.json'

    train_transforms, val_transforms = get_coco_transforms()
    train_dataset = CocoDataset(train_image_folder, train_annotation_file, train_transforms)
    val_dataset = CocoDataset(val_image_folder, val_annotation_file, val_transforms)
    evaluator = CocoEvaluator(val_dataset.coco)

    return train_dataset, val_dataset, evaluator
