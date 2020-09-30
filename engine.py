"""
Collection of training, evaluation and saving functions.
"""
import json
import math
from pathlib import Path
import sys

import torch
from torch.nn.utils import clip_grad_norm_

from utils.logging import MetricLogger
import utils.distributed as distributed


def train(model, criterion, dataloader, optimizer, max_grad_norm, epoch, print_freq=10):
    """
    Train model for one epoch.

    Args:
        model (nn.Module): Module computing predictions from images.
        criterion (nn.Module): Module comparing predictions with targets.
        dataloader (torch.utils.data.Dataloader): Training dataloader.
        optimizer (torch.optim.Optimizer): Optimizer used for optimizing the model from gradients.
        max_grad_norm (float): Maximum gradient norm (clipped if larger).
        epoch (int): Current training epoch.
        print_freq (int): Logger print frequency (defaults to 10).

    Returns:
        train_stats (Dict): Dictionary containing the epoch training statistics.
    """

    device = next(model.parameters()).device
    model.train()
    criterion.train()

    metric_logger = MetricLogger(delimiter="  ")
    header = f"Train epoch {epoch}:"

    for images, tgt_dict in metric_logger.log_every(dataloader, print_freq, header):
        images = images.to(device)
        tgt_dict = {k: v.to(device) for k, v in tgt_dict.items()}

        # Get loss and analysis dictionaries
        pred_list = model(images)
        loss_dict, analysis_dict = criterion(pred_list, tgt_dict)
        loss = sum(loss_dict.values())

        # Update model parameters
        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), max_grad_norm) if max_grad_norm > 0 else None
        optimizer.step()

        # Average analysis and loss dictionaries over all GPUs for logging purposes
        analysis_dict = distributed.reduce_dict(analysis_dict)
        loss_dict = distributed.reduce_dict(loss_dict)
        loss = sum(loss_dict.values()).item()

        # Check whether loss if finite
        if not math.isfinite(loss):
            print(f"Loss dictionary: {loss_dict}")
            print(f"Loss is {loss}, stopping training.")
            sys.exit(1)

        # Update logger
        metric_logger.update(**analysis_dict, **loss_dict, loss=loss)

    # Get epoch training statistics
    metric_logger.synchronize_between_processes()
    train_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    train_stats['lr'] = optimizer.param_groups[0]['lr']

    return train_stats


@torch.no_grad()
def evaluate(model, criterion, dataloader, evaluator, epoch=None, print_freq=10):
    """
    Evaluate model.

    Args:
        model (nn.Module): Module to be evaluated computing predictions from images.
        criterion (nn.Module): Module comparing predictions with targets.
        dataloader (torch.utils.data.Dataloader): Validation dataloader.
        evaluator (object): Object capable of computing evaluations from predictions and storing them.
        epoch (int): Training epochs completed (defaults to None).
        print_freq (int): Logger print frequency (defaults to 10).

    Returns:
        val_stats (Dict): Dictionary containing the validation statistics.
        evaluator (object): Updated evaluator object containing the evaluations.
    """

    device = next(model.parameters()).device
    model.eval()
    criterion.eval()
    evaluator.reset()

    metric_logger = MetricLogger(delimiter="  ")
    header = "Validation:" if epoch is None else f"Val epoch {epoch}:"

    for images, tgt_dict, eval_dict in metric_logger.log_every(dataloader, print_freq, header):
        images = images.to(device)
        tgt_dict = {k: v.to(device) for k, v in tgt_dict.items()}
        eval_dict = {k: v.to(device) for k, v in eval_dict.items()}

        # Get loss and analysis dictionaries
        pred_list = model(images)
        loss_dict, analysis_dict = criterion(pred_list, tgt_dict)
        loss = sum(loss_dict.values())

        # Average analysis and loss dictionaries over all GPUs for logging purposes
        analysis_dict = distributed.reduce_dict(analysis_dict)
        loss_dict = distributed.reduce_dict(loss_dict)
        loss = sum(loss_dict.values()).item()

        # Check whether loss if finite
        if not math.isfinite(loss):
            print(f"Loss dictionary: {loss_dict}")
            print(f"Loss is {loss}, stopping training.")
            sys.exit(1)

        # Update logger
        metric_logger.update(**analysis_dict, **loss_dict, loss=loss)

        # Update evaluator
        pred_dict = pred_list[0]
        evaluator.update(pred_dict, eval_dict)

    # Accumulate predictions from all images and summarize
    evaluator.synchronize_between_processes()
    evaluator.accumulate()
    evaluator.summarize()

    # Get epoch validation statistics
    metric_logger.synchronize_between_processes()
    val_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    for metric in evaluator.metrics:
        val_stats[f'eval_{metric}'] = evaluator.sub_evaluators[metric].stats.tolist()

    return val_stats, evaluator


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

        # Extra checkpoint before LR drop and every 100 epochs
        if epoch % args.lr_drop == 0 or epoch % 100 == 0:
            checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')

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
        output_dir (str): String containg the path to the output directory used for saving.
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
