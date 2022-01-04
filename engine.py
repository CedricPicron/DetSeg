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


def train(model, dataloader, optimizer, epoch, max_grad_norm=-1, print_freq=10):
    """
    Trains model for one epoch on data from the given dataloader.

    Args:
        model (nn.Module): Module computing predictions from images.
        dataloader (torch.utils.data.Dataloader): Training dataloader.
        optimizer (torch.optim.Optimizer): Optimizer updating the model parameters during training.
        epoch (int): Integer containing the current training epoch.
        max_grad_norm (float): Maximum gradient norm of parameters throughout model (default=-1).
        print_freq (int): Integer containing the logger print frequency (default=10).

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

    # Add epoch learning rates of different parameter groups to training statistics
    for i, param_group_name in enumerate(optimizer.param_group_names):
        train_stats[f'lr_{param_group_name}'] = optimizer.param_groups[i]['lr']

    return train_stats


@torch.no_grad()
def evaluate(model, dataloader, evaluator=None, epoch=None, output_dir=None, print_freq=10, save_stats=True,
             visualize=False):
    """
    Evaluates model on data from given dataloader. It additionally computes visualizations if 'visualize' is set.

    Args:
        model (nn.Module): Module to be evaluated computing predictions from images.
        dataloader (torch.utils.data.Dataloader): Validation dataloader.
        evaluator (object): Object computing evaluations from predictions and storing them (default=None).
        epoch (int): Integer containing the number of training epochs completed (default=None).
        output_dir (Path): Path to directory to save evaluations and potentially visualizations (default=None).
        print_freq (int): Integer containing the logger print frequency (default=10).
        save_stats (bool): Boolean indicating whether to save validation statistics (default=True).
        visualize (bool): Boolean indicating whether visualizations should be computed (default=False).

    Returns:
        val_stats (Dict): Dictionary containing the validation statistics.
    """

    # Get device, set model in evaluation mode and initialize evaluators
    device = next(model.parameters()).device
    model.eval()
    evaluators = None

    # Get one evaluator per prediction dictionary
    if evaluator is not None:
        evaluator.reset()
        sample_images, _ = next(iter(dataloader))
        num_pred_dicts = len(model(sample_images.to(device))[0])
        evaluators = [deepcopy(evaluator) for _ in range(num_pred_dicts)]

    # Make visualization directory within output directory if needed
    if visualize and output_dir is not None:
        vis_dir = output_dir / 'visualization'
        vis_dir.mkdir(exist_ok=True)

    # Initialize metric logger
    metric_logger = MetricLogger(delimiter="  ")
    header = "Validation:" if epoch is None else f"Val epoch {epoch}:"

    # Iterate over validation images
    for images, tgt_dict in metric_logger.log_every(dataloader, print_freq, header):

        # Place images and target dictionary on correct device
        images = images.to(device)
        tgt_dict = {k: v.to(device) for k, v in tgt_dict.items()}

        # Get prediction, loss and analysis dictionaries
        output_dicts = model(images, tgt_dict, extended_analysis=True, visualize=visualize)
        pred_dicts, loss_dict, analysis_dict = output_dicts[:3]

        # Average analysis and loss dictionaries over all GPUs for logging purposes
        analysis_dict = distributed.reduce_dict(analysis_dict)
        loss_dict = distributed.reduce_dict(loss_dict)
        loss = sum(loss_dict.values()).item()

        # Update logger
        metric_logger.update(**analysis_dict, **loss_dict, loss=loss)

        # Update evaluators
        if evaluators is not None:
            for evaluator, pred_dict in zip(evaluators, pred_dicts):
                evaluator.update(images, pred_dict)

        # Save visualizations to visualization directory
        if visualize and output_dir is not None:
            images_dict = output_dicts[3]

            for key, image in images_dict.items():
                key_parts = key.split('_')
                image_id = images.image_ids[int(key_parts[-1])]

                filename = ('_').join([str(image_id), *key_parts[:-1]])
                Image.fromarray(image).save(f'{vis_dir / filename}.png')

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

    # Save evaluations to output directory
    if distributed.is_main_process() and output_dir is not None:
        if save_stats:
            with (output_dir / 'eval.txt').open('w') as eval_file:
                eval_file.write(json.dumps(val_stats) + "\n")

        if evaluators is not None:
            for i, evaluator in enumerate(evaluators, 1):
                for metric in evaluator.metrics:
                    evaluations = evaluator.sub_evaluators[metric].eval
                    torch.save(evaluations, output_dir / f'eval_{i}_{metric}.pth')

    return val_stats


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
