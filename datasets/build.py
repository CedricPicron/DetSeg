"""
Dataset/evaluator build function.
"""

from datasets.coco.coco import build_coco


def build_dataset(args):
    """
    Build training and validation dataset from command-line arguments.

    It also adds the 'num_classes' attribute to the 'args' namespace.

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
        args.num_classes = 80
        train_dataset, val_dataset, evaluator = build_coco(args)

    else:
        raise ValueError(f"Unknown dataset name '{args.dataset}' was provided.")

    return train_dataset, val_dataset, evaluator
