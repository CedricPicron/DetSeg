"""
COCO dataset/evaluator and build function.
"""
import contextlib
import json
import logging
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

from boundary_iou.coco_instance_api.coco import COCO
from boundary_iou.coco_instance_api.cocoeval import COCOeval
from boundary_iou.coco_panoptic_api.evaluation import pq_compute
from boundary_iou.lvis_instance_api.eval import LVISEval
from boundary_iou.lvis_instance_api.lvis import LVIS
from boundary_iou.lvis_instance_api.results import LVISResults
from detectron2.data import MetadataCatalog
from detectron2.data.datasets.builtin import _PREDEFINED_SPLITS_COCO as SPLITS
from detectron2.layers import batched_nms
import numpy as np
import pandas as pd
from panopticapi.converters.detection2panoptic_coco_format import convert_detection_to_panoptic_coco_format
from PIL import Image
from pycocotools import mask as coco_mask
import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import InterpolationMode

from datasets.transforms import get_train_transforms, get_eval_transforms
from structures.boxes import Boxes
from structures.images import Images
from structures.masks import mask_inv_transform, mask_to_rle
import utils.distributed as distributed


# Add splits with LVIS annotations
SPLITS['coco']['coco_lvis_v0.5_train'] = ('coco/train2017', 'coco/annotations/instances_train_lvis_v0.5.json')
SPLITS['coco']['coco_lvis_v0.5_val'] = ('coco/val2017', 'coco/annotations/instances_val_lvis_v0.5.json')
SPLITS['coco']['coco_lvis_v1_train'] = ('coco/train2017', 'coco/annotations/instances_train_lvis_v1.json')
SPLITS['coco']['coco_lvis_v1_val'] = ('coco/val2017', 'coco/annotations/instances_val_lvis_v1.json')

SPLITS['lvis'] = {}
SPLITS['lvis']['lvis_v0.5_val_cocofied'] = ('coco/val2017', 'coco/annotations/lvis_v0.5_val_cocofied.json')

names = ('coco_lvis_v0.5_train', 'coco_lvis_v0.5_val', 'coco_lvis_v1_train', 'coco_lvis_v1_val')
thing_dataset_id_to_contiguous_id = MetadataCatalog.get('coco_2017_train').thing_dataset_id_to_contiguous_id
thing_classes = MetadataCatalog.get('coco_2017_train').thing_classes
thing_colors = MetadataCatalog.get('coco_2017_train').thing_colors

for name in names:
    MetadataCatalog.get(name).json_file = f"datasets/{SPLITS['coco'][name][1]}"
    MetadataCatalog.get(name).image_root = f"dataset/{SPLITS['coco'][name][0]}"
    MetadataCatalog.get(name).evaluator_type = 'coco'
    MetadataCatalog.get(name).thing_dataset_id_to_contiguous_id = thing_dataset_id_to_contiguous_id
    MetadataCatalog.get(name).thing_classes = thing_classes
    MetadataCatalog.get(name).thing_colors = thing_colors

# Add splits with panoptic annotations
PANOPTIC_SPLITS = {}
PANOPTIC_SPLITS['coco_2017_train_panoptic'] = ('coco/train2017', 'coco/annotations/panoptic_train2017_converted.json')
PANOPTIC_SPLITS['coco_2017_val_panoptic'] = ('coco/val2017', 'coco/annotations/panoptic_val2017_converted.json')
SPLITS['coco'].update(PANOPTIC_SPLITS)

# Merge thing and stuff items
for split in SPLITS['coco']:
    split_metadata = MetadataCatalog.get(split)

    split_metadata.classes = split_metadata.thing_classes
    split_metadata.colors = split_metadata.thing_colors

    thing_val = getattr(split_metadata, 'thing_dataset_id_to_contiguous_id', {})
    stuff_val = getattr(split_metadata, 'stuff_dataset_id_to_contiguous_id', {})
    merge_val = {**thing_val, **stuff_val}

    split_metadata._RENAMED.pop('dataset_id_to_contiguous_id', None)
    setattr(split_metadata, 'dataset_id_to_contiguous_id', merge_val)


class CocoDataset(Dataset):
    """
    Class implementing the CocoDataset dataset.

    Attributes:
        transforms (List): List [num_transforms] of transforms applied to image (and targets if available).
        metadata (detectron2.data.Metadata): Metadata instance containing additional dataset information.
        requires_masks (bool): Boolean indicating whether target dictionaries require masks.
        coco (COCO): Object containing COCO annotations in COCO format (only when annotation file is provided).
        image_ids (List): List [num_images] with image ids.
        image_paths (List): List [num_images] with image paths.
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
        self.transforms = transforms
        self.metadata = metadata
        self.requires_masks = requires_masks

        # Process annotation or info file
        if annotation_file is not None:
            with open(os.devnull, 'w') as devnull:
                with contextlib.redirect_stdout(devnull):
                    self.coco = COCO(annotation_file)
                    self.coco.img_ids = self.image_ids = list(sorted(self.coco.imgs.keys()))
                    self.image_paths = [f'{image_dir}/{img_id:012}.jpg' for img_id in self.image_ids]

        elif info_file is not None:
            with open(info_file) as json_file:
                info_dict = json.load(json_file)
                self.image_ids = [img['id'] for img in info_dict['images']]
                self.image_paths = [f"{image_dir}/{img['file_name']}" for img in info_dict['images']]

        else:
            error_msg = "No annotation or info file was provided during CocoDataset initialization."
            raise ValueError(error_msg)

    @staticmethod
    def get_masks(annotations, iH, iW):
        """
        Get segmentation masks from polygon segmentation annotations.

        Args:
            annotations (List): List [num_targets] with annotation dictionaries at least containing the key:
                - segmentation (List): list of polygons or RLEs delineating the segmentation mask of the annotation.

            iH (int): Height of image corresponding to the input annotations.
            iW (int): Width of image corresponding to the input annotations.

        Returns:
            masks (BoolTensor): Tensor containing the segmentation masks of shape [num_targets, iH, iW].
        """

        # Get segmentations with each segmentation represented as a list of polygons
        segmentations = [annotation['segmentation'] for annotation in annotations]

        # Get segmentation masks corresponding to each segmentation
        masks = torch.empty(len(segmentations), iH, iW, dtype=torch.bool)

        for i, mask_i in enumerate(segmentations):

            if isinstance(mask_i, list):
                mask_i = coco_mask.frPyObjects(mask_i, iH, iW)
            elif isinstance(mask_i['counts'], list):
                mask_i = coco_mask.frPyObjects(mask_i, iH, iW)

            mask_i = coco_mask.decode(mask_i)
            mask_i = mask_i[:, :, None] if len(mask_i.shape) == 2 else mask_i
            masks[i] = torch.as_tensor(mask_i, dtype=torch.bool).any(dim=2)

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
                - masks (BoolTensor): segmentation masks of shape [num_targets, iH, iW].
        """

        # Load image and place it into Images structure
        contiguous_img_id = index // len(self.transforms)
        image_path = self.image_paths[contiguous_img_id]
        image = Image.open(image_path).convert('RGB')

        img_id = self.image_ids[contiguous_img_id]
        image = Images(image, img_id)

        # Initialize empty target dictionary
        tgt_dict = {}

        # Add targets to target dictionary if annotations are provided
        if hasattr(self, 'coco'):

            # Load annotations and remove crowd annotations
            annotation_ids = self.coco.getAnnIds(imgIds=img_id)
            annotations = self.coco.loadAnns(annotation_ids)
            annotations = [anno for anno in annotations if 'iscrowd' not in anno or anno['iscrowd'] == 0]

            # Get object class labels (in contiguous label id space)
            id_dict = self.metadata.dataset_id_to_contiguous_id
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
        transform = self.transforms[index % len(self.transforms)]
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
        dataset_length = len(self.image_paths) * len(self.transforms)

        return dataset_length


class CocoEvaluator(object):
    """
    Class implementing the CocoEvaluator evaluator.

    Attributes:
        image_ids (List): List of evaluated image ids.
        metadata (detectron2.data.Metadata): Metadata instance containing additional dataset information.
        metrics (List): List with strings containing the evaluation metrics to be used.
        result_dicts (Dict): Dictionary of lists containing prediction results in COCO results format for each metric.
        result_format (str): String containing the type of result format.

        eval_nms (bool): Boolean indicating whether to perform NMS during evaluation.
        nms_thr (float): IoU threshold used during evaluation NMS to remove duplicate detections.

        If result_format is 'panoptic':
            image_dicts (List): List of dictionaries containing information about each dataset image.

        has_gt_anns (bool): Boolean indicating whether evaluation dataset has ground-truth annotations.

        If ground-annotations are available:
            ann_format (str): String containing the annotation format.

            If annotation format is 'coco':
                coco (COCO): Object containing the COCO annotations.

            If annotation format is 'lvis':
                lvis (LVIS): Object containing the LVIS annotations.
    """

    def __init__(self, eval_dataset, ann_format=None, annotation_file=None, info_file=None, metrics=None, nms_thr=0.5,
                 result_format='default'):
        """
        Initializes the CocoEvaluator evaluator.

        Args:
            eval_dataset (CocoDataset): The evaluation dataset.
            ann_format (str): String containing the annotation format (default=None).
            annotation_file (Path): Path to annotation file with COCO annotations (default=None).
            info_file (Path): Path to file with additional image information, but no annotations (default=None).
            metrics (List): List with strings containing the evaluation metrics to be used (default=None).
            nms_thr (float): IoU threshold used during evaluation NMS to remove duplicate detections (default=0.5).
            result_format (str): String containing the type of result format (default='default').

        Raises:
            ValueError: Error when neither annotation nor info file is provided when using 'panoptic' result format.
        """

        # Set base attributes
        self.image_ids = []
        self.metadata = eval_dataset.metadata
        self.metrics = metrics if metrics is not None else []
        self.result_dicts = []
        self.result_format = result_format

        # Set NMS attributes
        self.eval_nms = len(eval_dataset.transforms) > 1
        self.nms_thr = nms_thr

        # Get list of image dictionaries if needed
        if self.result_format == 'panoptic':
            if annotation_file is not None:
                json_file_name = annotation_file
            elif info_file is not None:
                json_file_name = info_file
            else:
                error_msg = "Neither annotation nor info file provided when using 'panoptic' result format."
                raise ValueError(error_msg)

            with open(json_file_name) as json_file:
                self.image_dicts = json.load(json_file)['images']

        # Set attributes related to ground-truth annotations
        self.has_gt_anns = hasattr(eval_dataset, 'coco')

        if self.has_gt_anns:
            self.ann_format = ann_format if ann_format is not None else 'coco'

            if self.ann_format == 'coco':
                self.coco = eval_dataset.coco

            elif self.ann_format == 'lvis':
                self.lvis = LVIS(annotation_file)

    def add_metrics(self, metrics):
        """
        Adds given evaluation metrics to the CocoEvaluator evaluator.

        Args:
            metrics (List): List with strings containing the evaluation metrics to be added.
        """

        # Add evalutation metrics
        self.metrics.extend(metrics)

    def reset(self):
        """
        Resets the image_ids and result_dicts attributes of the CocoEvaluator evaluator.
        """

        # Reset image_ids and result_dicts attributes
        self.image_ids = []
        self.result_dicts = []

    def update(self, images, pred_dict):
        """
        Updates result dictionaries of the evaluator object based on the given images and corresponding predictions.

        Args:
            images (Images): Images structure containing the batched images.

            pred_dict (Dict): Prediction dictionary potentially containing following keys:
                - labels (LongTensor): predicted class indices of shape [num_preds];
                - boxes (Boxes): structure containing axis-aligned bounding boxes of size [num_preds];
                - masks (FloatTensor): predicted segmentation masks of shape [num_preds, iH, iW];
                - scores (FloatTensor): normalized prediction scores of shape [num_preds];
                - batch_ids (LongTensor): batch indices of predictions of shape [num_preds].

        Raises:
            ValueError: Error when CocoEvaluator evaluator has unknown result format.
            ValueError: Error when no box predictions are provided when using evaluation NMS.
        """

        # Update the image_ids attribute
        self.image_ids.extend(images.image_ids)

        # Extract predictions from prediction dictionary
        labels = pred_dict['labels']
        boxes = pred_dict.get('boxes', None)
        masks = pred_dict.get('masks', None)
        scores = pred_dict['scores']
        batch_ids = pred_dict['batch_ids']

        # Transform boxes to original image space and convert them to desired format
        if boxes is not None:
            boxes, well_defined = boxes.transform(images, inverse=True)
            boxes = boxes[well_defined].to_format('xywh')

            labels = labels[well_defined]
            masks = masks[well_defined] if masks is not None else None
            scores = scores[well_defined]
            batch_ids = batch_ids[well_defined]

        # Get segmentation masks in original image space
        if masks is not None:
            batch_size = len(images)
            device = batch_ids.device

            pred_sizes = [0] + [(batch_ids == i).sum() for i in range(batch_size)]
            pred_sizes = torch.tensor(pred_sizes, device=device).cumsum(dim=0)

            if self.result_format == 'default':
                interpolation = InterpolationMode.BILINEAR

            elif self.result_format == 'panoptic':
                interpolation = InterpolationMode.NEAREST

            else:
                error_msg = f"Unknown result format for CocoEvaluator evaluator (got '{self.result_format}')."
                raise ValueError(error_msg)

            masks = mask_inv_transform(masks, images, pred_sizes, interpolation=interpolation)
            masks = [mask_ij for masks_i in masks for mask_ij in masks_i]

            if len(masks) > 0:
                masks = torch.stack(masks, dim=0)
                non_empty = masks.flatten(1).sum(dim=1) > 0

                labels = labels[non_empty]
                boxes = boxes[non_empty] if boxes is not None else None
                masks = masks[non_empty]
                scores = scores[non_empty]
                batch_ids = batch_ids[non_empty]

            masks = [mask for mask in masks]
            masks = mask_to_rle(masks)

        # Get image indices corresponding to predictions
        image_ids = torch.as_tensor(images.image_ids)
        image_ids = image_ids[batch_ids.cpu()]

        # Convert labels to original non-contiguous id space
        orig_ids = list(self.metadata.dataset_id_to_contiguous_id.keys())
        orig_ids = labels.new_tensor(orig_ids)
        labels = orig_ids[labels]

        # Convert desired tensors to lists
        image_ids = image_ids.tolist()
        labels = labels.tolist()
        boxes = boxes.boxes.tolist() if boxes is not None else None
        scores = scores.tolist()

        # Get result dictionary
        result_dict = {}
        result_dict['image_id'] = image_ids
        result_dict['category_id'] = labels
        result_dict['score'] = scores

        if boxes is not None:
            result_dict['bbox'] = boxes

        elif self.eval_nms:
            error_msg = "Box predictions must be provided when using evaluation NMS."
            raise ValueError(error_msg)

        if masks is not None:
            result_dict['segmentation'] = masks

        # Get list of result dictionaries
        result_dicts = pd.DataFrame(result_dict).to_dict(orient='records')

        # Make sure there is at least one result dictionary per image
        if self.result_format == 'instance':
            for image_id in images.image_ids:
                if image_id not in image_ids:
                    result_dict = {}
                    result_dict['image_id'] = image_id
                    result_dict['category_id'] = 1
                    result_dict['score'] = 0.0

                    if boxes is not None:
                        result_dict['bbox'] = [0.0, 0.0, 0.0, 0.0]

                    if masks is not None:
                        result_dict['segmentation'] = {'size': [0, 0], 'counts': ''}

                    result_dicts.append(result_dict)

        # Update result_dicts attribute
        self.result_dicts.extend(result_dicts)

    def evaluate(self, device='cpu', output_dir=None, save_results=False, save_name='results'):
        """
        Perform evaluation by finalizing the result dictionaries and by comparing with ground-truth (if available).

        Args:
            device (str): String containing the type of device used during NMS (default='cpu').
            output_dir (Path): Path to output directory to save result dictionaries (default=None).
            save_results (bool): Boolean indicating whether to save the results (default=False).
            save_name (str) String containing the name of the saved result file (default='results').

        Returns:
            eval_dict (Dict): Dictionary with evaluation results for each metric (if annotations are available).

        Raises:
            ValueError: Error when CocoEvaluator evaluator has unknown result format.
            ValueError: Error when CocoEvaluator evaluator has unknown annotation format.
        """

        # Synchronize image indices and make them unique
        gathered_image_ids = distributed.all_gather(self.image_ids)
        self.image_ids = [image_id for list in gathered_image_ids for image_id in list]
        self.image_ids = list(np.unique(self.image_ids))

        # Synchronize result dictionaries
        gathered_result_dicts = distributed.all_gather(self.result_dicts)
        self.result_dicts = [result_dict for list in gathered_result_dicts for result_dict in list]

        # Return if not main process
        if not distributed.is_main_process():
            return

        # Peform NMS if requested
        if self.eval_nms:
            boxes = {image_id: [] for image_id in self.image_ids}
            scores = {image_id: [] for image_id in self.image_ids}
            labels = {image_id: [] for image_id in self.image_ids}
            result_ids = {image_id: [] for image_id in self.image_ids}
            keep_result_ids = []

            for result_id, result_dict in enumerate(self.result_dicts):
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

            result_ids = set(range(len(self.result_dicts)))
            keep_result_ids = set(keep_result_ids)
            drop_result_ids = result_ids - keep_result_ids

            data = pd.DataFrame(self.result_dicts)
            data.drop(index=drop_result_ids, inplace=True)
            self.result_dicts = data.to_dict('records')

        # Get results in panoptic format if needed
        if self.result_format == 'panoptic':
            for i, result_dict in enumerate(self.result_dicts):
                result_dict['id'] = i

            ann_dict = {}
            ann_dict['images'] = [img_dict for img_dict in self.image_dicts if img_dict['id'] in self.image_ids]
            ann_dict['annotations'] = self.result_dicts

            tmp_dir = TemporaryDirectory()
            tmp_dir_path = Path(tmp_dir.name)
            in_json_file = tmp_dir_path / 'in.json'

            with open(in_json_file, 'w') as json_file:
                json.dump(ann_dict, json_file)

            in_json_file = str(in_json_file)
            seg_folder = str(tmp_dir_path / f'{save_name}')
            out_json_file = str(tmp_dir_path / f'{save_name}.json')
            cat_json = 'datasets/coco/annotations/panoptic_coco_categories.json'

            with open(os.devnull, 'w') as devnull:
                with contextlib.redirect_stdout(devnull):
                    args = (in_json_file, seg_folder, out_json_file, cat_json)
                    convert_detection_to_panoptic_coco_format(*args)

        # Save result dictionaries if needed
        if output_dir is not None:
            if save_results or not self.has_gt_anns:
                zip_file_name = output_dir / f'{save_name}.zip'

                if self.result_format == 'default':
                    json_file_name = output_dir / f'{save_name}.json'

                    with open(json_file_name, 'w') as json_file:
                        json.dump(self.result_dicts, json_file)

                    with ZipFile(zip_file_name, 'w') as zip_file:
                        zip_file.write(json_file_name, arcname=f'{save_name}.json')
                        os.remove(json_file_name)

                elif self.result_format == 'panoptic':
                    with ZipFile(zip_file_name, 'w') as zip_file:
                        zip_file.write(out_json_file, arcname=f'{save_name}.json')
                        zip_file.write(seg_folder, arcname=f'{save_name}')

                else:
                    error_msg = f"Unknown result format '{self.result_format}' for CocoEvaluator evaluator."
                    raise ValueError(error_msg)

        # Return if no ground-truth annotations are available
        if not self.has_gt_anns:
            if self.result_format == 'panoptic':
                tmp_dir.cleanup()
            return

        # Compare predictions with ground-truth annotations
        eval_dict = {}

        if self.result_format == 'default':
            for metric_id, metric in enumerate(self.metrics):

                if self.ann_format == 'coco':
                    with open(os.devnull, 'w') as devnull:
                        with contextlib.redirect_stdout(devnull):

                            if metric_id == 0:
                                coco_res = self.coco.loadRes(self.result_dicts)

                            sub_evaluator = COCOeval(self.coco, coco_res, iouType=metric)
                            sub_evaluator.params.imgIds = self.image_ids
                            sub_evaluator.evaluate()
                            sub_evaluator.accumulate()

                    print()
                    print(f"Evaluation metric: {metric}")
                    sub_evaluator.summarize()
                    eval_dict[metric] = sub_evaluator.stats.tolist()

                elif self.ann_format == 'lvis':
                    prev_log_lvl = logging.root.manager.disable
                    logging.disable()

                    with open(os.devnull, 'w') as devnull:
                        with contextlib.redirect_stdout(devnull):

                            if metric_id == 0:
                                lvis_res = LVISResults(self.lvis, self.result_dicts)

                            sub_evaluator = LVISEval(self.lvis, lvis_res, iou_type=metric)
                            sub_evaluator.params.img_ids = self.image_ids

                    sub_evaluator.evaluate()
                    sub_evaluator.accumulate()
                    sub_evaluator.summarize(show_freq_groups=False)
                    logging.disable(prev_log_lvl)

                    print()
                    print(f"Evaluation metric: {metric}")
                    sub_evaluator.print_results()
                    eval_dict[metric] = list(sub_evaluator.get_results().values())

                else:
                    error_msg = f"Unknown annotation format '{self.evaluator_type}' for CocoEvaluator evaluator."
                    raise ValueError(error_msg)

        elif self.result_format == 'panoptic':
            gt_json_file = 'datasets/coco/annotations/panoptic_val2017.json'
            gt_folder = 'datasets/coco/annotations/panoptic_val2017'
            args = (gt_json_file, out_json_file, gt_folder, seg_folder)

            for metric in self.metrics:
                print()
                print(f"Evaluation metric: {metric}")
                eval_dict[metric] = pq_compute(*args, gt_img_ids=self.image_ids, iou_type=metric, verbose=False)
                eval_dict[metric].pop('per_class')

            tmp_dir.cleanup()

        else:
            error_msg = f"Unknown result format '{self.result_format}' for CocoEvaluator evaluator."
            raise ValueError(error_msg)

        return eval_dict


def build_coco(args):
    """
    Build COCO datasets and evaluator from command-line arguments.

    Args:
        args (argparse.Namespace): Command-line arguments.

    Returns:
        datasets (Dict): Dictionary of datasets potentially containing following keys:
            - train (CocoDataset): the training dataset (only present during training);
            - eval (CocoDataset): the evaluation dataset (always present).

        evaluator (CocoEvaluator): Object capable of computing evaluations from predictions and storing them.
    """

    # Get root directory containing datasets
    root = Path() / 'datasets'

    # Initialize empty datasets dictionary
    datasets = {}

    # Get training dataset if needed
    if not args.eval:
        image_dir = root / SPLITS['coco'][f'coco_{args.train_split}'][0]
        train_transforms = get_train_transforms(f'coco_{args.train_transforms_type}')
        metadata = MetadataCatalog.get(f'coco_{args.train_split}')
        annotation_file = root / SPLITS['coco'][f'coco_{args.train_split}'][1]
        datasets['train'] = CocoDataset(image_dir, train_transforms, metadata, annotation_file=annotation_file)

    # Get evaluation dataset
    image_dir = root / SPLITS['coco'][f'coco_{args.eval_split}'][0]
    eval_transforms = get_eval_transforms(f'coco_{args.eval_transforms_type}')
    metadata = MetadataCatalog.get(f'coco_{args.eval_split}')
    annotation_file = root / SPLITS['coco'][f'coco_{args.eval_split}'][1] if 'val' in args.eval_split else None
    info_file = root / SPLITS['coco'][f'coco_{args.eval_split}'][1] if 'test' in args.eval_split else None

    file_kwargs = {'annotation_file': annotation_file, 'info_file': info_file}
    datasets['eval'] = CocoDataset(image_dir, eval_transforms, metadata, **file_kwargs)

    # Get evaluator
    nms_thr = args.eval_nms_thr

    if 'lvis' in args.eval_split and 'val' in args.eval_split:
        file_kwargs['annotation_file'] = root / SPLITS['lvis'][f'{args.eval_split}_cocofied'][1]
        evaluator = CocoEvaluator(datasets['eval'], ann_format='lvis', nms_thr=nms_thr, **file_kwargs)

    elif 'panoptic' in args.eval_split:
        evaluator = CocoEvaluator(datasets['eval'], nms_thr=nms_thr, result_format='panoptic', **file_kwargs)

    else:
        evaluator = CocoEvaluator(datasets['eval'], nms_thr=nms_thr)

    return datasets, evaluator
