"""
Dataset/evaluator build function.
"""

from detectron2.data import MetadataCatalog

from .coco.coco import build_coco


def build_dataset(args):
    """
    Build training and validation dataset from command-line arguments.

    It also adds the 'args.num_classes', 'args.train_metadata' and 'args.val_metadata' to the 'args' namespace.

    Args:
        args (argparse.Namespace): Command-line arguments.

    Returns:
        train_dataset (torch.utils.data.Dataset): The specified training dataset.
        val_dataset (torch.utils.data.Dataset): The specified validation dataset.
        evaluator (object): Object capable of computing evaluations from predictions and storing them.

    Raises:
        ValueError: Raised when unknown dataset name is provided in args.dataset.
    """

    if args.dataset == 'coco':
        args.num_classes = 91
        args.train_metadata = MetadataCatalog.get('coco_2017_train')
        args.val_metadata = MetadataCatalog.get('coco_2017_val')
        train_dataset, val_dataset, evaluator = build_coco(args)
    else:
        raise ValueError(f"Unknown dataset name '{args.dataset}' was provided.")

    return train_dataset, val_dataset, evaluator
