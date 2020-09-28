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


def train(model, criterion, dataloader, optimizer, epoch, max_grad_norm, print_freq=10):
    """
    Train model for one epoch.

    Args:
        model (nn.Module): Module computing predictions from images.
        criterion (nn.Module): Module comparing predictions with targets.
        dataloader (torch.utils.data.Dataloader): Training dataloader.
        optimizer (torch.optim.Optimizer): Optimizer used for optimizing the model from gradients.
        epoch (int): Current training epoch.
        max_grad_norm (float): Maximum gradient norm (clipped if larger).
        print_freq (int): Logger print frequency.
    """

    device = next(model.parameters()).device
    model.train()
    criterion.train()

    metric_logger = MetricLogger(delimiter="  ")
    header = f"Epoch {epoch}:"

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
def evaluate(model, criterion, postprocessors, dataloader, base_ds, device, output_dir, print_freq=10):
    device = next(model.parameters()).device
    model.eval()
    criterion.eval()

    metric_logger = MetricLogger(delimiter="  ")
    header = "Test:"

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    for images, tgt_dict in metric_logger.log_every(dataloader, print_freq, header):
        images = images.to(device)
        tgt_dict = {k: v.to(device) for k, v in tgt_dict.items()}

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

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    return stats, coco_evaluator


def save_checkpoint(args, epoch, model, optimizer, scheduler):
    """
    Function used for checkpoint saving.

    No checkpoints are saved when args.output_dir is an empty string.

    Args:
        args (argparse.Namespace): Command-line arguments.
        epoch (int): Number of epochs trained.
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


def save_log(output_dir, epoch, train_stats, test_stats):
    """
    Function used for log saving.

    No logs are saved when output_dir is and empty string.

    Args:
        output_dir (str): String containg the path to the output directory used for saving.
        epoch (int): Number of epochs trained.
        train_stats (Dict): Dictionary containing the training statistics.
        test_stats (Dict): Dictionary containing the test statistics.
    """

    if output_dir and distributed.is_main_process():
        output_dir = Path(output_dir)

        log_dict = {'epoch', epoch}
        log_dict.update({f'train_{k}': v for k, v in train_stats.items()})
        log_dict.update({f'test_{k}': v for k, v in test_stats.items()})

        with (output_dir / 'log.txt').open('a') as log_file:
            log_file.write(json.dumps(log_dict) + "\n")
