"""
Cityscapes dataset/evaluator and build function.
"""
from pathlib import Path

from detectron2.data import MetadataCatalog
from detectron2.data.datasets.cityscapes import load_cityscapes_instances
from PIL import Image
import torch
from torch.utils.data import Dataset

from datasets.coco.coco import CocoDataset, CocoEvaluator
from datasets.transforms import get_train_transforms, get_eval_transforms
from structures.boxes import Boxes
from structures.images import Images


class CityscapesDataset(Dataset):
    """
    Class implementing the CityscapesDataset dataset.

    Attributes:
        root (Path): Path to main directory from which datasets directory can be accessed.
        transforms (List): List [num_transforms] of transforms applied to image (and targets if available).
        metadata (detectron2.data.Metadata): Metadata instance containing additional dataset information.
        requires_masks (bool): Boolean indicating whether target dictionaries require masks.
        image_dicts (List): List [num_images] of dictionaries containing image-specific information.
        has_annotations (bool): Boolean indicating whether annotations are available.
    """

    def __init__(self, image_dir, info_dir, transforms, metadata, requires_masks=False):
        """
        Initializes the Cityscapes dataset.

        Args:
            image_dir (Path): Path to directory with Cityscapes images.
            info_dir (Path): Path to directory with information about Cityscapes images.
            transforms (List): List [num_transforms] of transforms applied to image (and targets if available).
            metadata (detectron2.data.Metadata): Metadata instance containing additional dataset information.
            requires_masks (bool): Boolean indicating whether target dictionaries require masks (default=False).
        """

        # Set base attributes
        self.root = Path()
        self.transforms = transforms
        self.metadata = metadata
        self.requires_masks = requires_masks

        # Get image dictionaries containing information about each image
        self.image_dicts = load_cityscapes_instances(image_dir, info_dir)

        # Set attribute indicating whether annotations are available
        self.has_annotations = False

        for image_dict in self.image_dicts:
            if len(image_dict['annotations']) > 0:
                self.has_annotations = True
                break

        # Filter image dictionaries if needed
        if self.has_annotations:
            image_dicts = []

            for image_dict in self.image_dicts:
                image_dict['annotations'] = [ann for ann in image_dict['annotations'] if not ann['iscrowd']]
                image_dicts.append(image_dict) if len(image_dict['annotations']) > 0 else None

            self.image_dicts = image_dicts

    def __getitem__(self, index):
        """
        Implements the __getitem__ method of the CityscapesDataset dataset.

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
        image_dict = self.image_dicts[contiguous_img_id]
        image_path = self.root / image_dict['file_name']

        image = Image.open(image_path).convert('RGB')
        image = Images(image, contiguous_img_id)

        # Initialize empty target dictionary
        tgt_dict = {}

        # Add targets to target dictionary if annotations are provided
        if self.has_annotations:

            # Get annotations
            annotations = image_dict['annotations']

            # Get object class labels
            labels = [annotation['category_id'] for annotation in annotations]
            labels = torch.tensor(labels, dtype=torch.int64)

            # Get object boxes in (left, top, right, bottom) format
            boxes = [annotation['bbox'] for annotation in annotations]
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            boxes = Boxes(boxes, format='xyxy')

            # Clip boxes and only keep targets with well-defined boxes
            boxes, well_defined = boxes.clip(image.size())
            tgt_dict['labels'] = labels[well_defined]
            tgt_dict['boxes'] = boxes[well_defined]

            # Get segmentation masks if required and add to target dictionary
            if self.requires_masks:
                iW, iH = image.size()
                masks = CocoDataset.get_masks(annotations, iH, iW)
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
        Implements the __len__ method of the CityscapesDataset dataset.

        Returns:
            dataset_length (int): Dataset length measured as the number of images times the number of transforms.
        """

        # Get dataset length
        dataset_length = len(self.image_dicts) * len(self.transforms)

        return dataset_length


class CityscapesEvaluator(CocoEvaluator):
    """
    Class implementing the CityscapesEvaluator evaluator.
    """


def build_cityscapes(args):
    """
    Build Cityscapes datasets and evaluator from command-line arguments.

    Args:
        args (argparse.Namespace): Command-line arguments.

    Returns:
        datasets (Dict): Dictionary of datasets potentially containing following keys:
            - train (CityscapesDataset): the training dataset (only present during training);
            - eval (CityscapesDataset): the evaluation dataset (always present).

        evaluator (CityscapesEvaluator): Object capable of computing evaluations from predictions and storing them.
    """

    # Get root directory containing datasets
    root = Path() / 'datasets'

    # Initialize empty datasets dictionary
    datasets = {}

    # Get training dataset if needed
    if not args.eval:
        image_dir = root / 'cityscapes/leftImg8bit/train'
        info_dir = root / 'cityscapes/gtFine/train'
        train_transforms = get_train_transforms(f'cityscapes_{args.train_transforms_type}')
        metadata = MetadataCatalog.get('cityscapes_fine_instance_seg_train')
        datasets['train'] = CityscapesDataset(image_dir, info_dir, train_transforms, metadata)

    # Get evaluation dataset
    image_dir = root / f'cityscapes/leftImg8bit/{args.eval_split}'
    info_dir = root / f'cityscapes/gtFine/{args.eval_split}'
    eval_transforms = get_eval_transforms(f'cityscapes_{args.eval_transforms_type}')
    metadata = MetadataCatalog.get(f'cityscapes_fine_instance_seg_{args.eval_split}')
    datasets['eval'] = CityscapesDataset(image_dir, info_dir, eval_transforms, metadata)

    # Get evaluator
    evaluator = CityscapesEvaluator(datasets['eval'], nms_thr=args.eval_nms_thr)

    return datasets, evaluator
