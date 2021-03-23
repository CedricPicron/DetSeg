"""
Collection of training, evaluation and saving functions.
"""
from copy import deepcopy
import json
import math
from pathlib import Path
import sys

from PIL import Image
import torch

from utils.logging import MetricLogger
import utils.distributed as distributed


def train(model, dataloader, optimizer, max_grad_norm, epoch, print_freq=10):
    """
    Trains model for one epoch.

    Args:
        model (nn.Module): Module computing predictions from images.
        dataloader (torch.utils.data.Dataloader): Training dataloader.
        optimizer (torch.optim.Optimizer): Optimizer updating the model parameters during training.
        max_grad_norm (float): Maximum norm of optimizer update (clipped if larger).
        epoch (int): Current training epoch.
        print_freq (int): Logger print frequency (default=10).

    Returns:
        train_stats (Dict): Dictionary containing the epoch training statistics.
    """

    # Get device and set model in training mode
    device = next(model.parameters()).device
    model.train()

    # Initialize metric logger
    metric_logger = MetricLogger(delimiter="  ")
    header = f"Train epoch {epoch}:"

    # Iterate over training images
    for images, tgt_dict in metric_logger.log_every(dataloader, print_freq, header):

        # Place images and target dictionary on correct device
        images = images.to(device)
        tgt_dict = {k: v.to(device) for k, v in tgt_dict.items()}

        # Train model with optimizer and report loss and analysis dictionaries
        loss_dict, analysis_dict = model(images, tgt_dict, optimizer, max_grad_norm=max_grad_norm)

        # Average analysis and loss dictionaries over all GPUs for logging purposes
        analysis_dict = distributed.reduce_dict(analysis_dict)
        loss_dict = distributed.reduce_dict(loss_dict)
        loss = sum(loss_dict.values()).item()

        # Check whether loss is finite
        if not math.isfinite(loss):
            print(f"Loss dictionary: {loss_dict}")
            print(f"Loss is {loss}, stopping training.")
            sys.exit(1)

        # Update logger
        metric_logger.update(**analysis_dict, **loss_dict, loss=loss)

    # Get epoch training statistics
    metric_logger.synchronize_between_processes()
    train_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    # Add epoch learning rates of different parameter families to training statistics
    param_families = model.module.get_param_families() if hasattr(model, 'module') else model.get_param_families()
    for i, param_family in enumerate(param_families):
        train_stats[f'lr_{param_family}'] = optimizer.param_groups[i]['lr']

    return train_stats


@torch.no_grad()
def evaluate(model, dataloader, evaluator=None, epoch=None, print_freq=10):
    """
    Evaluates model.

    Args:
        model (nn.Module): Module to be evaluated computing predictions from images.
        dataloader (torch.utils.data.Dataloader): Validation dataloader.
        evaluator (object): Object computing evaluations from predictions and storing them (default=None).
        epoch (int): Training epochs completed (default=None).
        print_freq (int): Logger print frequency (default=10).

    Returns:
        val_stats (Dict): Dictionary containing the validation statistics.
        evaluators (List): List of updated evaluator objects containing the evaluations (or None).
    """

    # Get device and set model in evaluation mode
    device = next(model.parameters()).device
    model.eval()

    # Initialize evaluators
    if evaluator is not None:
        evaluator.reset()

        sample_images, _ = next(iter(dataloader))
        num_pred_dicts = len(model(sample_images.to(device)))
        evaluators = [deepcopy(evaluator) for _ in range(num_pred_dicts)]

    else:
        evaluators = None

    # Initialize metric logger
    metric_logger = MetricLogger(delimiter="  ")
    header = "Validation:" if epoch is None else f"Val epoch {epoch}:"

    # Iterate over validation images
    for images, tgt_dict in metric_logger.log_every(dataloader, print_freq, header):

        # Place images and target dictionary on correct device
        images = images.to(device)
        tgt_dict = {k: v.to(device) for k, v in tgt_dict.items()}

        # Get prediction, loss and analysis dictionaries
        val_kwargs = {'extended_analysis': True}
        pred_dicts, loss_dict, analysis_dict = model(images, tgt_dict, **val_kwargs)

        # Average analysis and loss dictionaries over all GPUs for logging purposes
        analysis_dict = distributed.reduce_dict(analysis_dict)
        loss_dict = distributed.reduce_dict(loss_dict)
        loss = sum(loss_dict.values()).item()

        # Check whether loss is finite
        if not math.isfinite(loss):
            print(f"Loss dictionary: {loss_dict}")
            print(f"Loss is {loss}, stopping training.")
            sys.exit(1)

        # Update logger
        metric_logger.update(**analysis_dict, **loss_dict, loss=loss)

        # Update evaluators
        if evaluators is not None:
            for evaluator, pred_dict in zip(evaluators, pred_dicts):
                evaluator.update(images, pred_dict)

    # Accumulate predictions from all images and summarize
    if evaluators is not None:
        for evaluator in evaluators:
            evaluator.synchronize_between_processes()
            evaluator.accumulate()
            evaluator.summarize()

    # Get epoch validation statistics
    metric_logger.synchronize_between_processes()
    val_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    if evaluators is not None:
        for i, evaluator in enumerate(evaluators, 1):
            for metric in evaluator.metrics:
                val_stats[f'eval_{i}_{metric}'] = evaluator.sub_evaluators[metric].stats.tolist()

    return val_stats, evaluators


@torch.no_grad()
def visualize(model, dataloader, output_dir):
    """
    Visualizes model predictions and corresponding ground-truth.

    Args:
        model (nn.Module): Module to be visualized computing predictions from images.
        dataloader (torch.utils.data.Dataloader): Visualization dataloader.
        output_dir (Path): Path object containing the path to the output directory used for saving.
    """

    # Get device and set model in evaluation mode
    device = next(model.parameters()).device
    model.eval()

    # Initialize metric logger
    metric_logger = MetricLogger(delimiter="  ", window_size=1)
    logger_log_every_kwargs = {'print_freq': 1, 'header': 'Visualization:'}

    # Iterate over images to be visualized
    for images, tgt_dict in metric_logger.log_every(dataloader, **logger_log_every_kwargs):

        # Place images and target dictionary on correct device
        images = images.to(device)
        tgt_dict = {k: v.to(device) for k, v in tgt_dict.items()}

        # Get images, loss and analysis dictionary
        images_dict, loss_dict, analysis_dict = model(images, tgt_dict, visualize=True)

        # Save images
        for key, image in images_dict.items():
            key_parts = key.split('_')
            image_id = images.image_ids[int(key_parts[-1])]

            filename = ('_').join([str(image_id), *key_parts[:-1]])
            Image.fromarray(image).save(f'{output_dir / filename}.png')

        # Update logger
        metric_logger.update(**analysis_dict, **loss_dict, loss=sum(loss_dict.values()).item())


def save_checkpoint(args, epoch, model, optimizer, scheduler):
    """
    Function used for checkpoint saving.

    No checkpoints are saved when args.output_dir is an empty string.

    Args:
        args (argparse.Namespace): Command-line arguments.
        epoch (int): Training epochs completed.
        model (nn.Module): Model module to be saved.
        optimizer (torch.optim.Optimizer): Optimizer to be saved.
        scheduler (torch.optim.lr_scheduler): Scheduler to be saved.
    """

    if args.output_dir and distributed.is_main_process():
        output_dir = Path(args.output_dir)
        checkpoint_paths = [output_dir / 'checkpoint.pth']

        # Extra checkpoint before LR drop
        if epoch in args.lr_drops:
            checkpoint_paths.append(output_dir / f'checkpoint{epoch}.pth')

        # Checkpoint saving
        for checkpoint_path in checkpoint_paths:
            checkpoint = {'args': args, 'epoch': epoch}
            checkpoint['model'] = model.state_dict()
            checkpoint['optimizer'] = optimizer.state_dict()
            checkpoint['scheduler'] = scheduler.state_dict()
            torch.save(checkpoint, checkpoint_path)


def save_log(output_dir, epoch, train_stats, val_stats):
    """
    Function used for log saving.

    No logs are saved when output_dir is and empty string.

    Args:
        output_dir (str): String containing the path to the output directory used for saving.
        epoch (int): Training epochs completed.
        train_stats (Dict): Dictionary containing the training statistics.
        val_stats (Dict): Dictionary containing the val statistics.
    """

    if output_dir and distributed.is_main_process():
        output_dir = Path(output_dir)

        log_dict = {'epoch': epoch}
        log_dict.update({f'train_{k}': v for k, v in train_stats.items()})
        log_dict.update({f'val_{k}': v for k, v in val_stats.items()})

        with (output_dir / 'log.txt').open('a') as log_file:
            log_file.write(json.dumps(log_dict) + "\n")
