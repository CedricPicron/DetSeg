"""
Cityscapes dataset/evaluator and build function.
"""
from pathlib import Path

from boundary_iou.coco_instance_api.coco import COCO
from detectron2.data import MetadataCatalog
from detectron2.data.datasets.cityscapes import load_cityscapes_instances

from datasets.coco.coco import CocoDataset, CocoEvaluator
from datasets.transforms import get_train_transforms, get_eval_transforms


class CityscapesDataset(CocoDataset):
    """
    Class implementing the CityscapesDataset dataset.

    Attributes:
        root (Path): Path to main directory from which datasets directory can be accessed.
        transforms (List): List [num_transforms] of transforms applied to image (and targets if available).
        metadata (detectron2.data.Metadata): Metadata instance containing additional dataset information.
        requires_masks (bool): Boolean indicating whether target dictionaries require masks.
        coco (COCO): Optional object containing Cityscapes annotations in COCO format.
        image_paths (List): List [num_images] with image paths relative to the root path.
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
        img_dicts = load_cityscapes_instances(image_dir, info_dir)

        # Check whether annotations are available
        has_annotations = 'test' not in str(info_dir)

        # Get COCO object storing Cityscapes annotations
        if has_annotations:
            img_dicts = [img_dict for img_dict in img_dicts if len(img_dict['annotations']) > 0]
            annotation_file = None
            self.coco = COCO(annotation_file)

        # Get list with image paths
        self.image_paths = [img_dict['file_name'] for img_dict in img_dicts]


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
