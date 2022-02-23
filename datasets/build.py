"""
Dataset/evaluator build function.
"""

from datasets.coco.coco import build_coco


def build_dataset(args):
    """
    Build datasets and evaluator from command-line arguments.

    It also adds args.metadata and args.num_classes to the args namespace.

    Args:
        args (argparse.Namespace): Command-line arguments.

    Returns:
        datasets (Dict): Dictionary of datasets potentially containing following keys:
            - train (torch.utils.data.Dataset): the training dataset (only present during training);
            - eval (torch.utils.data.Dataset): the evaluation dataset (always present).

        evaluator (object): Object capable of computing evaluations from predictions and storing them.

    Raises:
        ValueError: Raised when unknown dataset name is provided in args.dataset.
    """

    # Build datasets and evaluator
    if args.dataset == 'coco':
        datasets, evaluator = build_coco(args)
    else:
        raise ValueError(f"Unknown dataset name '{args.dataset}' was provided.")

    # Add args.metadata and args.num_classes to the args namespace
    args.metadata = datasets['eval'].metadata

    if hasattr(args.metadata, 'thing_dataset_id_to_contiguous_id'):
        args.num_classes = len(args.metadata.thing_dataset_id_to_contiguous_id)
    else:
        args.num_classes = len(args.metadata.thing_classes)

    return datasets, evaluator
