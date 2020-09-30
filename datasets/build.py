"""
Dataset/evaluator build function.
"""

from .coco.coco import build_coco


def build_dataset(args):
    """
    Build training and validation dataset from command-line arguments.

    Note that is also adds the number of classes (args.num_classes) to the command-line arguments.

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
        train_dataset, val_dataset, evaluator = build_coco(args)
    else:
        raise ValueError(f"Unknown dataset name {args.dataset} was provided.")

    return train_dataset, val_dataset, evaluator
