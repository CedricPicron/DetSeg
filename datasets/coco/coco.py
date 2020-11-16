"""
COCO dataset/evaluator and build function.
"""
import copy
import contextlib
import os
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as coco_mask
import torch
import torch.nn.functional as F
from torchvision.datasets.vision import VisionDataset

import datasets.transforms as T
from utils.box_ops import box_cxcywh_to_xywh
import utils.distributed as distributed


class CocoDataset(VisionDataset):
    """
    Class implementing the COCO dataset.

    Attributes:
        coco (COCO): Object containing the COCO dataset annotations.
        image_ids (List): List of image indices, sorted in ascending order.
        requires_masks (bool): Bool indicating whether target dictionaries require segmentation masks.
    """

    def __init__(self, image_folder, annotation_file, transforms, requires_masks):
        """
        Initializes the CocoDataset dataset.

        Args:
            image_folder (Path): Path to image folder containing COCO images.
            annotation_file (Path): Path to annotation file with COCO annotations.
            transforms (object): The transforms to be applied on both image and its bounding boxes.
            requires_masks (bool): Bool indicating whether target dictionaries require segmentation masks.
        """

        with open(os.devnull, 'w') as devnull:
            with contextlib.redirect_stdout(devnull):
                super().__init__(image_folder, transforms=transforms)
                self.coco = COCO(annotation_file)
                self.image_ids = list(sorted(self.coco.imgs.keys()))
                self.requires_masks = requires_masks

    @staticmethod
    def get_masks(annotations, height, width):
        """
        Get segmentation masks from COCO annotations.

        Args:
            annotations (List): List of size [num_targets] with COCO annotation dictionaries with key:
                - segmentation (List): list of polygons delineating the segmentation mask related to the annotation.
            height (int): Height of image corresponding to the input annotations.
            width (int): Width of image corresponding to the input annotations.

        Returns:
            masks (ByteTensor): Tensor of shape [num_targets, height, width] containing the segmentation masks.
        """

        # Get segmentations, with each segmentation represented as a list of polygons
        segmentations = [annotation['segmentation'] for annotation in annotations]

        # Get segmentation masks corresponding to each segmentation
        masks = torch.zeros(len(segmentations), height, width, dtype=torch.uint8)
        for i, polygons in enumerate(segmentations):
            rle_objs = coco_mask.frPyObjects(polygons, height, width)
            mask = coco_mask.decode(rle_objs)
            mask = mask[..., None] if len(mask.shape) < 3 else mask
            masks[i] = torch.as_tensor(mask, dtype=torch.uint8).any(dim=2)

        return masks

    def __getitem__(self, index):
        """
        Implements the __getitem__ method of the CocoDataset dataset.

        Args:
            index (int): Index selecting one of the dataset images.

        Returns:
            image (FloatTensor): Tensor containing the image of shape [3, height, width].
            tgt_dict (Dict): Target dictionary containing following keys:
                - labels (IntTensor): tensor of shape [num_targets] containing the class indices;
                - boxes (FloatTensor): boxes of shape [num_targets, 4] in (center_x, center_y, width, height) format;
                - masks (ByteTensor, optional): segmentation masks of shape [num_targets, height, width];
                - image_id (IntTensor): tensor of shape [1] containing the image id;
                - image_size (IntTensor): tensor of shape [2] containing the image size (before data augmentation).
        """

        # Load image
        image_id = self.image_ids[index]
        image_path = self.root / self.coco.loadImgs(image_id)[0]['file_name']
        image = Image.open(image_path).convert('RGB')

        # Load annotations
        annotation_ids = self.coco.getAnnIds(imgIds=image_id)
        annotations = self.coco.loadAnns(annotation_ids)

        # Remove crowd annotations
        annotations = [anno for anno in annotations if 'iscrowd' not in anno or anno['iscrowd'] == 0]

        # Get object class labels
        labels = [annotation["category_id"] for annotation in annotations]
        labels = torch.tensor(labels, dtype=torch.int64)

        # Get object boxes in (left, top, width, height) format
        boxes = [annotation["bbox"] for annotation in annotations]
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)

        # Transform boxes to (left, top, right, bottom) format
        boxes[:, 2:] += boxes[:, :2]

        # Crop boxes such that they fit within the image
        width, height = image.size
        boxes[:, 0::2].clamp_(min=0, max=width)
        boxes[:, 1::2].clamp_(min=0, max=height)

        # Only keep objects with well-defined boxes
        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        labels = labels[keep]
        boxes = boxes[keep]

        # Place target properties into target dictionary
        tgt_dict = {'labels': labels, 'boxes': boxes}

        # Get segmentation masks if required and add to target dictionary
        if self.requires_masks:
            masks = self.get_masks(annotations, height, width)
            tgt_dict['masks'] = masks[keep]

        # Perform image and bounding box transformations
        image, tgt_dict = self.transforms(image, tgt_dict)

        # Add additional properties to target dictionary, useful during evaluation
        tgt_dict['image_id'] = torch.tensor([image_id])
        tgt_dict['image_size'] = torch.tensor([int(height), int(width)])

        return image, tgt_dict

    def __len__(self):
        """
        Implements the __len__ method of the CocoDataset dataset.

        Returns:
            The dataset length measured as the number of images in the dataset.
        """

        return len(self.image_ids)


class CocoEvaluator(object):
    """
    Evaluator object capable of computing evaluations from predictions on COCO data, and storing them.

    Attributes:
        coco (COCO): Object containing the COCO dataset annotations.
        metrics (List): List of strings containing the evaluation metrics to be used.
        sub_evaluators (Dict): Dictionary of sub-evaluators, each of them evaluating one metric.
        image_ids (List): List of evaluated image ids.
        image_evals (Dict): Dictionary of lists containing image evaluations for each metric.
    """

    def __init__(self, coco, metrics=['bbox']):
        """
        Initializes the CocoEvaluator evaluator.

        Args:
            coco (COCO): Object containing the COCO dataset annotations.
            metrics (List): List of strings containing the evaluation metrics to be used.
        """

        self.coco = coco
        self.metrics = metrics
        self.sub_evaluators = {metric: COCOeval(coco, iouType=metric) for metric in self.metrics}

        self.image_ids = []
        self.image_evals = {metric: [] for metric in self.metrics}

    def reset(self):
        """
        Resets the CocoEvaluator evaluator by reinitializing its image_ids and image_evals attributes.
        """

        self.image_ids = []
        self.image_evals = {metric: [] for metric in self.metrics}

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
        image_ids = eval_dict['image_ids'].tolist()
        self.image_ids.extend(image_ids)

        # Get image id for every prediction
        pred_image_ids = eval_dict['image_ids'][batch_idx]

        # Go from tensors to lists
        pred_image_ids = pred_image_ids.tolist()
        labels = labels.tolist()
        boxes = boxes.tolist()
        scores = scores.tolist()

        # Perform evaluation for every evaluation type
        for metric in self.metrics:
            result_dicts = []

            if metric == 'bbox':
                for i, image_id in enumerate(pred_image_ids):
                    result_dict = {}
                    result_dict['image_id'] = image_id
                    result_dict['category_id'] = labels[i]
                    result_dict['bbox'] = boxes[i]
                    result_dict['score'] = scores[i]
                    result_dicts.append(result_dict)

            with open(os.devnull, 'w') as devnull:
                with contextlib.redirect_stdout(devnull):
                    coco_api_predictions = COCO.loadRes(self.coco, result_dicts)

            sub_evaluator = self.sub_evaluators[metric]
            sub_evaluator.cocoDt = coco_api_predictions
            sub_evaluator.params.imgIds = image_ids

            image_evals = CocoEvaluator.evaluate(sub_evaluator)
            self.image_evals[metric].append(image_evals)

    @staticmethod
    def evaluate(sub_evaluator):
        """
        Evaluates a metric from given sub_evaluator.

        Copied from pycocotools, but without print statements and with additional post-processing.

        Args:
            sub_evaluator (COCOeval): Object used for evaluating a specific metric on COCO.
        """

        # Copied from pycocotools, but without print statements
        p = sub_evaluator.params
        p.imgIds = list(np.unique(p.imgIds))

        if p.useCats:
            p.catIds = list(np.unique(p.catIds))

        p.maxDets = sorted(p.maxDets)
        sub_evaluator.params = p

        sub_evaluator._prepare()
        catIds = p.catIds if p.useCats else [-1]

        if p.iouType == 'segm' or p.iouType == 'bbox':
            computeIoU = sub_evaluator.computeIoU
        elif p.iouType == 'keypoints':
            computeIoU = sub_evaluator.computeOks

        sub_evaluator.ious = {(imgId, catId): computeIoU(imgId, catId)
                              for imgId in p.imgIds
                              for catId in catIds}

        evaluateImg = sub_evaluator.evaluateImg
        maxDet = p.maxDets[-1]
        evalImgs = [evaluateImg(imgId, catId, areaRng, maxDet)
                    for catId in catIds
                    for areaRng in p.areaRng
                    for imgId in p.imgIds]

        # Some post-processing (not in pycocotools)
        evalImgs = np.asarray(evalImgs).reshape(len(catIds), len(p.areaRng), len(p.imgIds))
        sub_evaluator._paramsEval = copy.deepcopy(sub_evaluator.params)

        return evalImgs

    def synchronize_between_processes(self):
        """
        Synchronization of CocoEvaluator objects across different processes.
        """

        for metric in self.metrics:
            self.image_evals[metric] = np.concatenate(self.image_evals[metric], axis=2)
            CocoEvaluator.sync_evaluator(self.sub_evaluators[metric], self.image_ids, self.image_evals[metric])

    @staticmethod
    def sync_evaluator(sub_evaluator, image_ids, image_evals):
        """
        Synchronization of sub-evaluators across different processes.

        Args:
            sub_evaluator (COCOeval): Object used for evaluating a specific metric on COCO.
            image_ids (List): List of image indices processed by this CocoEvaluator.
            image_evals (List): List of image evaluations processed by this CocoEvaluator.
        """

        # Merge image indices and image evaluations across processes
        merged_image_ids, merged_image_evals = CocoEvaluator.merge(image_ids, image_evals)

        # Update sub-evaluator with merged image indices and evaluations
        sub_evaluator.evalImgs = merged_image_evals
        sub_evaluator.params.imgIds = merged_image_ids
        sub_evaluator._paramsEval = copy.deepcopy(sub_evaluator.params)

    @staticmethod
    def merge(image_ids, image_evals):
        """
        Merges image indices and image evaluations across different processes.

        Args:
            image_ids (List): List of image indices processed by this CocoEvaluator.
            image_evals (List): List of image evaluations processed by this CocoEvaluator.

        Returns:
            merged_image_ids (List): List of merged image indices from all processes.
            merged_image_evals (List): List of merged image evaluations from all processes.
        """

        # Gather image indices and image evaluations across processes
        gathered_image_ids = distributed.all_gather(image_ids)
        gathered_image_evals = distributed.all_gather(image_evals)

        # Create merged lists
        merged_image_ids = []
        for image_ids in gathered_image_ids:
            merged_image_ids.extend(image_ids)

        merged_image_evals = []
        for image_evals in gathered_image_evals:
            merged_image_evals.append(image_evals)

        # Keep only unique (and in sorted order) images
        merged_image_ids = np.array(merged_image_ids)
        merged_image_evals = np.concatenate(merged_image_evals, axis=2)

        merged_image_ids, idx = np.unique(merged_image_ids, return_index=True)
        merged_image_evals = merged_image_evals[..., idx]

        merged_image_ids = list(merged_image_ids)
        merged_image_evals = list(merged_image_evals.flatten())

        return merged_image_ids, merged_image_evals

    def accumulate(self):
        """
        Accumulates evaluations for each metric and stores them in their corresponding sub-evaluator.
        """

        with open(os.devnull, 'w') as devnull:
            with contextlib.redirect_stdout(devnull):
                for sub_evaluator in self.sub_evaluators.values():
                    sub_evaluator.accumulate()

    def summarize(self):
        """
        Summarizes evaluations for each metric and prints them.
        """

        for metric, sub_evaluator in self.sub_evaluators.items():
            print(f"Evaluation metric: {metric}")
            sub_evaluator.summarize()


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
        train_dataset (CocoDataset): The specified COCO training dataset.
        val_dataset (CocoDataset): The specified COCO validation dataset.
        evaluator (CocoEvaluator): COCO evaluator capable of computing evaluations from predictions and storing them.
    """

    coco_root = Path() / 'datasets' / 'coco'
    train_image_folder = coco_root / 'images' / 'train2017'
    val_image_folder = coco_root / 'images' / 'val2017'
    train_annotation_file = coco_root / 'annotations' / 'instances_train2017.json'
    val_annotation_file = coco_root / 'annotations' / 'instances_val2017.json'

    train_transforms, val_transforms = get_coco_transforms()
    requires_masks = True if args.meta_arch in ['BiViNet'] else False

    train_dataset = CocoDataset(train_image_folder, train_annotation_file, train_transforms, requires_masks)
    val_dataset = CocoDataset(val_image_folder, val_annotation_file, val_transforms, requires_masks)
    evaluator = CocoEvaluator(val_dataset.coco)

    return train_dataset, val_dataset, evaluator
