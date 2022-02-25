"""
COCO dataset/evaluator and build function.
"""
import contextlib
import json
import os
from pathlib import Path

from detectron2.data import MetadataCatalog
from detectron2.data.datasets.builtin import _PREDEFINED_SPLITS_COCO as COCO_SPLITS
from detectron2.layers import batched_nms
import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as coco_mask
import torch
from torch.utils.data import Dataset

from datasets.transforms import get_train_transforms, get_eval_transforms
from structures.boxes import Boxes
from structures.images import Images
import utils.distributed as distributed


class CocoDataset(Dataset):
    """
    Class implementing the CocoDataset dataset.

    Attributes:
        root (Path): Path to directory with COCO images.
        transforms (List): List [num_transforms] of transforms applied to image (and targets if available).
        metadata (detectron2.data.Metadata): Metadata instance containing additional dataset information.

        coco (COCO): Optional object containing COCO annotations.
        image_ids (List): List [num_images] of image indices, sorted in ascending order.
        file_names (List): Optional list [num_images] of image file names aligned with image_ids list.

        requires_masks (bool): Boolean indicating whether target dictionaries require masks.
    """

    def __init__(self, image_dir, transforms, metadata, annotation_file=None, info_file=None, requires_masks=False):
        """
        Initializes the CocoDataset dataset.

        Args:
            image_dir (Path): Path to directory with COCO images.
            transforms (List): List [num_transforms] of transforms applied to image (and targets if available).
            metadata (detectron2.data.Metadata): Metadata instance containing additional dataset information.
            annotation_file (Path): Path to annotation file with COCO annotations (default=None).
            info_file (Path): Path to file with additional image information, but no annotations (default=None).
            requires_masks (bool): Boolean indicating whether target dictionaries require masks (default=False).

        Raises:
            ValueError: Error when no annotation or info file is provided.
        """

        # Set base attributes
        self.root = image_dir
        self.transforms = transforms
        self.metadata = metadata

        # Process annotation or info file
        if annotation_file is not None:
            with open(os.devnull, 'w') as devnull:
                with contextlib.redirect_stdout(devnull):
                    self.coco = COCO(annotation_file)
                    self.image_ids = list(sorted(self.coco.imgs.keys()))

        elif info_file is not None:
            with open(info_file) as json_file:
                data = json.load(json_file)
                filenames = {img['id']: img['file_name'] for img in data['images']}
                filenames = dict(sorted(filenames.items()))

                self.image_ids = list(filenames.keys())
                self.filenames = list(filenames.values())

        else:
            error_msg = "No annotation or info file was provided during CocoDataset initialization."
            raise ValueError(error_msg)

        # Set additional attributes
        self.requires_masks = requires_masks

    @staticmethod
    def get_masks(annotations, iH, iW):
        """
        Get segmentation masks from COCO annotations.

        Args:
            annotations (List): List of size [num_targets] with COCO annotation dictionaries with key:
                - segmentation (List): list of polygons delineating the segmentation mask related to the annotation.

            iH (int): Height of image corresponding to the input annotations.
            iW (int): Width of image corresponding to the input annotations.

        Returns:
            masks (ByteTensor): Tensor of shape [num_targets, iH, iW] containing the segmentation masks.
        """

        # Get segmentations, with each segmentation represented as a list of polygons
        segmentations = [annotation['segmentation'] for annotation in annotations]

        # Get segmentation masks corresponding to each segmentation
        masks = torch.zeros(len(segmentations), iH, iW, dtype=torch.uint8)
        for i, polygons in enumerate(segmentations):
            rle_objs = coco_mask.frPyObjects(polygons, iH, iW)
            mask = coco_mask.decode(rle_objs)
            mask = mask[..., None] if len(mask.shape) < 3 else mask
            masks[i] = torch.as_tensor(mask, dtype=torch.uint8).any(dim=2)

        return masks

    def __getitem__(self, index):
        """
        Implements the __getitem__ method of the CocoDataset dataset.

        Args:
            index (int): Index selecting one of the dataset (image, transform) combinations.

        Returns:
            image (Images): Structure containing the image tensor after data augmentation.

            tgt_dict (Dict): Target dictionary potentially containing following keys (empty when no annotations):
                - labels (LongTensor): tensor of shape [num_targets] containing the class indices;
                - boxes (Boxes): structure containing axis-aligned bounding boxes of size [num_targets];
                - masks (ByteTensor, optional): segmentation masks of shape [num_targets, iH, iW].

        Raises:
            ValueError: Error when neither the 'coco' attribute nor the 'filenames' attribute is set.
        """

        # Get image index and transform
        image_id = self.image_ids[index // len(self.transforms)]
        transform = self.transforms[index % len(self.transforms)]

        # Load image and place it into Images structure
        if hasattr(self, 'coco'):
            image_path = self.root / self.coco.loadImgs(image_id)[0]['file_name']
        elif hasattr(self, 'filenames'):
            image_path = self.root / self.filenames[index // len(self.transforms)]
        else:
            error_msg = "Neither the 'coco' attribute nor the 'filenames' attribute is set."
            raise ValueError(error_msg)

        image = Image.open(image_path).convert('RGB')
        image = Images(image, image_id)

        # Initialize empty target dictionary
        tgt_dict = {}

        # Add targets to target dictionary if annotations are provided
        if hasattr(self, 'coco'):

            # Load annotations and remove crowd annotations
            annotation_ids = self.coco.getAnnIds(imgIds=image_id)
            annotations = self.coco.loadAnns(annotation_ids)
            annotations = [anno for anno in annotations if 'iscrowd' not in anno or anno['iscrowd'] == 0]

            # Get object class labels (in contiguous id space)
            id_dict = self.metadata.thing_dataset_id_to_contiguous_id
            labels = [id_dict[annotation['category_id']] for annotation in annotations]
            labels = torch.tensor(labels, dtype=torch.int64)

            # Get object boxes in (left, top, right, bottom) format
            boxes = [annotation['bbox'] for annotation in annotations]
            boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
            boxes = Boxes(boxes, format='xywh').to_format('xyxy')

            # Clip boxes and only keep targets with well-defined boxes
            boxes, well_defined = boxes.clip(image.size())
            tgt_dict['labels'] = labels[well_defined]
            tgt_dict['boxes'] = boxes[well_defined]

            # Get segmentation masks if required and add to target dictionary
            if self.requires_masks:
                iW, iH = image.size()
                masks = self.get_masks(annotations, iH, iW)
                tgt_dict['masks'] = masks[well_defined]

        # Perform image and target dictionary transformations
        image, tgt_dict = transform(image, tgt_dict)

        # Only keep targets with well-defined boxes
        if 'boxes' in tgt_dict:
            well_defined = tgt_dict['boxes'].well_defined()

            for key in tgt_dict.keys():
                if key in ['labels', 'boxes', 'masks']:
                    tgt_dict[key] = tgt_dict[key][well_defined]

        return image, tgt_dict

    def __len__(self):
        """
        Implements the __len__ method of the CocoDataset dataset.

        Returns:
            dataset_length (int): Dataset length measured as the number of images times the number of transforms.
        """

        # Get dataset length
        dataset_length = len(self.image_ids) * len(self.transforms)

        return dataset_length


class CocoEvaluator(object):
    """
    Evaluator object capable of computing evaluations from predictions on COCO data and storing them.

    Attributes:
        image_ids (List): List of evaluated image ids.
        metadata (detectron2.data.Metadata): Metadata instance containing additional dataset information.
        metrics (List): List of strings containing the evaluation metrics to be used.
        result_dicts (Dict): Dictionary of lists containing prediction results in COCO results format for each metric.

        eval_nms (bool): Boolean indicating whether to perform NMS during evaluation.
        nms_thr (float): IoU threshold used during evaluation NMS to remove duplicate detections.

        Additional attributes when the evaluation dataset contains annotations:
            coco (COCO): Object containing the COCO dataset annotations.
            sub_evaluators (Dict): Dictionary of sub-evaluators, each of them evaluating one metric.
    """

    def __init__(self, eval_dataset, metrics=['bbox'], nms_thr=0.5):
        """
        Initializes the CocoEvaluator evaluator.

        Args:
            eval_dataset (CocoDataset): The evaluation dataset.
            metrics (List): List of strings containing the evaluation metrics to be used (default=['bbox']).
            nms_thr (float): IoU threshold used during evaluation NMS to remove duplicate detections (default=0.5).
        """

        # Set base attributes
        self.image_ids = []
        self.metadata = eval_dataset.metadata
        self.metrics = metrics
        self.result_dicts = {metric: [] for metric in self.metrics}

        # Set NMS attributes
        self.eval_nms = len(eval_dataset.transforms) > 1
        self.nms_thr = nms_thr

        # Set additional attributes when evaluation dataset has annotations
        if hasattr(eval_dataset, 'coco'):
            self.coco = eval_dataset.coco
            self.sub_evaluators = {metric: COCOeval(self.coco, iouType=metric) for metric in self.metrics}

    def reset(self):
        """
        Resets the image_ids and result_dicts attributes of the CocoEvaluator evaluator.
        """

        # Reset image_ids and result_dicts attributes
        self.image_ids = []
        self.result_dicts = {metric: [] for metric in self.metrics}

    def update(self, images, pred_dict):
        """
        Updates result dictionaries of the evaluator object based on the given images and corresponding predictions.

        Args:
            images (Images): Images structure containing the batched images.

            pred_dict (Dict): Prediction dictionary containing following keys:
                - labels (LongTensor): predicted class indices of shape [num_preds_total];
                - boxes (Boxes): structure containing axis-aligned bounding boxes of size [num_preds_total];
                - scores (FloatTensor): normalized prediction scores of shape [num_preds_total];
                - batch_ids (LongTensor): batch indices of predictions of shape [num_preds_total].

        Raises:
            ValueError: Error when evaluator contains an unknown evaluation metric.
        """

        # Update the image_ids attribute
        self.image_ids.extend(images.image_ids)

        # Transform boxes to original image space and convert them to (left, top, width, height) format
        boxes, well_defined = pred_dict['boxes'].transform(images, inverse=True)
        boxes = boxes.to_format('xywh')

        # Get well-defined predictions and convert them to lists
        labels = pred_dict['labels'][well_defined].tolist()
        boxes = boxes.boxes[well_defined].tolist()
        scores = pred_dict['scores'][well_defined].tolist()
        batch_ids = pred_dict['batch_ids'][well_defined].tolist()

        # Get image id for every prediction
        image_ids = torch.as_tensor(images.image_ids)
        image_ids = image_ids[batch_ids].tolist()

        # Get labels in original non-contiguous id space
        inv_id_dict = {v: k for k, v in self.metadata.thing_dataset_id_to_contiguous_id.items()}
        labels = [inv_id_dict[label] for label in labels]

        # Perform evaluation for every evaluation metric
        for metric in self.metrics:

            # Get result dictionaries
            result_dicts = []

            if metric == 'bbox':
                for i, image_id in enumerate(image_ids):
                    result_dict = {}
                    result_dict['image_id'] = image_id
                    result_dict['category_id'] = labels[i]
                    result_dict['bbox'] = boxes[i]
                    result_dict['score'] = scores[i]
                    result_dicts.append(result_dict)

                for image_id in images.image_ids:
                    if image_id not in image_ids:
                        result_dict = {}
                        result_dict['image_id'] = image_id
                        result_dict['category_id'] = 0
                        result_dict['bbox'] = [0.0, 0.0, 0.0, 0.0]
                        result_dict['score'] = 0.0
                        result_dicts.append(result_dict)

            else:
                error_msg = f"CocoEvaluator object contains an unknown evaluation metric (got '{metric}')."
                raise ValueError(error_msg)

            # Update result_dicts attribute
            self.result_dicts[metric].extend(result_dicts)

    def evaluate(self, device='cpu'):
        """
        Perform evaluation by finalizing the result dictionaries and by comparing with ground-truth (if available).

        Args:
            device (str): String containing the type of device used during NMS (default='cpu').

        Raises:
            ValueError: Error when none of the allowed NMS metrics match one of the evaluation metrics.
        """

        # Synchronize image indices and make them unique
        gathered_image_ids = distributed.all_gather(self.image_ids)
        self.image_ids = [image_id for list in gathered_image_ids for image_id in list]
        self.image_ids = list(np.unique(self.image_ids))

        # Synchronize result dictionaries for each evaluation metric
        for metric in self.metrics:
            gathered_result_dicts = distributed.all_gather(self.result_dicts[metric])
            self.result_dicts[metric] = [result_dict for list in gathered_result_dicts for result_dict in list]

        # Peform NMS if requested
        if self.eval_nms:
            boxes = {image_id: [] for image_id in self.image_ids}
            scores = {image_id: [] for image_id in self.image_ids}
            labels = {image_id: [] for image_id in self.image_ids}
            result_ids = {image_id: [] for image_id in self.image_ids}
            keep_result_ids = []

            allowed_metrics = ['bbox', 'segm', 'keypoints']
            nms_metrics = [metric for metric in self.metrics if metric in allowed_metrics]

            if len(nms_metrics) > 0:
                nms_metric = nms_metrics[0]
            else:
                error_msg = f"NMS requires at least one metric from {allowed_metrics}, but got {self.metrics}."
                raise ValueError(error_msg)

            for result_id, result_dict in enumerate(self.result_dicts[nms_metric]):
                image_id = result_dict['image_id']
                label = result_dict['category_id']
                box = result_dict['bbox'].copy()
                score = result_dict['score']

                box[2] = box[0] + box[2]
                box[3] = box[1] + box[3]

                boxes[image_id].append(box)
                scores[image_id].append(score)
                labels[image_id].append(label)
                result_ids[image_id].append(result_id)

            for image_id in self.image_ids:
                boxes_i = torch.tensor(boxes[image_id], device=device)
                scores_i = torch.tensor(scores[image_id], device=device)
                labels_i = torch.tensor(labels[image_id], device=device)
                result_ids_i = torch.tensor(result_ids[image_id], device=device)

                keep_ids = batched_nms(boxes_i, scores_i, labels_i, iou_threshold=self.nms_thr)[:100]
                keep_result_ids_i = result_ids_i[keep_ids].tolist()
                keep_result_ids.extend(keep_result_ids_i)

            for metric in self.metrics:
                result_dicts = self.result_dicts[metric]
                result_dicts = [result_dict for i, result_dict in enumerate(result_dicts) if i in keep_result_ids]
                self.result_dicts[metric] = result_dicts

        # Compare with ground-truth annotations if available
        if hasattr(self, 'coco'):
            with open(os.devnull, 'w') as devnull:
                with contextlib.redirect_stdout(devnull):
                    for metric in self.metrics:
                        sub_evaluator = self.sub_evaluators[metric]
                        sub_evaluator.cocoDt = self.coco.loadRes(self.result_dicts[metric])
                        sub_evaluator.params.imgIds = self.image_ids
                        sub_evaluator.evaluate()
                        sub_evaluator.accumulate()

        # Print evaluation summary for each evaluation metric
        for metric in self.metrics:
            print(f"Evaluation metric: {metric}")
            self.sub_evaluators[metric].summarize()


def build_coco(args):
    """
    Build COCO datasets and evaluator from command-line arguments.

    Args:
        args (argparse.Namespace): Command-line arguments.

    Returns:
        datasets (Dict): Dictionary of datasets potentially containing following keys:
            - train (CocoDataset): the training dataset (only present during training);
            - eval (CocoDataset): the evaluation dataset (always present).

        evaluator (object): Object capable of computing evaluations from predictions and storing them.

     Raises:
        ValueError: Raised when unknown evaluator type is provided in args.evaluator.
    """

    # Get root directory containing datasets
    root = Path() / 'datasets'

    # Initialize empty datasets dictionary
    datasets = {}

    # Get training dataset if needed
    if not args.eval:
        image_dir = root / COCO_SPLITS['coco'][f'coco_{args.train_split}'][0]
        train_transforms = get_train_transforms(args.train_transforms_type)
        metadata = MetadataCatalog.get(f'coco_{args.train_split}')
        annotation_file = root / COCO_SPLITS['coco'][f'coco_{args.train_split}'][1]
        datasets['train'] = CocoDataset(image_dir, train_transforms, metadata, annotation_file=annotation_file)

    # Get evaluation dataset
    image_dir = root / COCO_SPLITS['coco'][f'coco_{args.eval_split}'][0]
    eval_transforms = get_eval_transforms(args.eval_transforms_type)
    metadata = MetadataCatalog.get(f'coco_{args.eval_split}')
    annotation_file = root / COCO_SPLITS['coco'][f'coco_{args.eval_split}'][1] if 'val' in args.eval_split else None
    info_file = root / COCO_SPLITS['coco'][f'coco_{args.eval_split}'][1] if 'test' in args.eval_split else None

    file_kwargs = {'annotation_file': annotation_file, 'info_file': info_file}
    datasets['eval'] = CocoDataset(image_dir, eval_transforms, metadata, **file_kwargs)

    # Get evaluator
    if args.evaluator == 'detection':
        evaluator = CocoEvaluator(datasets['eval'], metrics=['bbox'], nms_thr=args.eval_nms_thr)
    elif args.evaluator == 'none':
        evaluator = None
    else:
        raise ValueError(f"Unknown evaluator type '{args.evaluator}' was provided.")

    return datasets, evaluator
