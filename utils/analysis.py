"""
Collection of analysis utilities.
"""
import json
import logging
import warnings

from detectron2.utils.analysis import flop_count_operators, _IGNORED_OPS
import torch
from torch.utils.benchmark import Timer

add_ignore_ops = (
    'aten::clone',
    'aten::cumsum',
    'aten::diff',
    'aten::movedim',
    'aten::prod',
    'aten::repeat_interleave',
    'aten::rsqrt',
    'aten::sub_',
    'aten::sum',
    'aten::topk',
)

_IGNORED_OPS.update(add_ignore_ops)


def analyze_model(model, dataloader, optimizer, max_grad_norm=-1, num_samples=100, output_dir=None):
    """
    Function analyzing the given model by computing various model properties during both training and inference.

    Args:
        model (nn.Module): Module containing the model to be analyzed.
        dataloader (torch.utils.data.Dataloader): Dataloader used to obtain the FLOPS and FPS metrics.
        optimizer (torch.optim.Optimizer): Optimizer used during training to update the model parameters.
        max_grad_norm (float): Maximum gradient norm of parameters throughout model (default=-1).
        num_samples (int): Integer containing the nubmer of batches sampled from both dataloaders (default=100).
        output_dir (Path): Path to directory to save the analysis results (default=None).
    """

    # Print string indicating start of model analysis in training mode
    print("\n===============================")
    print("|  Model analysis (training)  |")
    print("===============================\n")

    # Get number of digits of num_samples
    num_digits = len(str(num_samples))

    # Get model device and set model in training mode
    device = next(model.parameters()).device
    model.train()

    # Set optimizer learning rates equal to zero
    for param_group in optimizer.param_groups:
        param_group['lr'] = 0.0

    # Get number of model parameters that are trained
    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_parameters = num_parameters / 10**6

    # Initialize average training FPS variable
    avg_train_fps = 0.0

    # Iterate over dataloader
    for i, (images, tgt_dict) in enumerate(dataloader, 1):

        # Place images and target dictionary on correct device
        images = images.to(device)
        tgt_dict = {k: v.to(device) for k, v in tgt_dict.items()}

        # Get tuple of model inputs
        inputs = (images, tgt_dict, optimizer, max_grad_norm)

        # Get training FPS
        globals_dict = {'model': model, 'inputs': inputs}
        timer = Timer(stmt="model(*inputs)", globals=globals_dict)

        train_fps = 1 / timer.timeit(number=1).median
        avg_train_fps += train_fps / num_samples

        # Print training FPS
        print_str = f"Analysis training [{i:{num_digits}d}/{num_samples}]:  "
        print_str += f"train_fps: {train_fps:.2f} FPS"
        print(print_str)

        # Break when 'num_samples' batches are processed
        if i == num_samples:
            break

    # Get maximum memory utilization during training
    max_train_mem = torch.cuda.max_memory_allocated(device)
    max_train_mem = max_train_mem / 1024**3

    # Print string indicating start of model analysis in inference mode
    print("\n================================")
    print("|  Model analysis (inference)  |")
    print("================================\n")

    # Set model in evaluation mode and reset maximum memory statistic
    model.eval()
    torch.cuda.reset_peak_memory_stats(device)

    # Initialize average inference FLOPS and FPS variables
    avg_inf_flops = 0.0
    avg_inf_fps = 0.0

    # Iterate over dataloader
    for i, (images, _) in enumerate(dataloader, 1):

        # Disable some logging and ignore warnings from second iteration onwards
        if i == 2:
            jit_analysis_logger = logging.getLogger('fvcore.nn.jit_analysis')
            jit_analysis_logger.disabled = True
            warnings.filterwarnings('ignore')

        # Place images on correct device
        images = images.to(device)

        # Get tuple of model inputs
        inputs = (images,)

        # Get inference FLOPS
        inf_flops = sum(flop_count_operators(model, inputs).values())
        avg_inf_flops += inf_flops / num_samples

        # Get inference FPS
        globals_dict = {'model': model, 'inputs': inputs}
        timer = Timer(stmt="model(*inputs)", globals=globals_dict)

        inf_fps = 1 / timer.timeit(number=1).median
        avg_inf_fps += inf_fps / num_samples

        # Print batch inference FLOPS and FPS
        print_str = f"Analysis inference [{i:{num_digits}d}/{num_samples}]:  "
        print_str += f"inf_flops: {inf_flops:.1f} GFLOPS  "
        print_str += f"inf_fps: {inf_fps:.2f} FPS"
        print(print_str)

        # Break when 'num_samples' batches are processed
        if i == num_samples:

            if i >= 2:
                jit_analysis_logger.disabled = False
                warnings.filterwarnings('default')

            break

    # Get maximum memory utilization during inference
    max_inf_mem = torch.cuda.max_memory_allocated(device)
    max_inf_mem = max_inf_mem / 1024**3

    # Print results of model analysis
    print("\n==============================")
    print("|  Model analysis (results)  |")
    print("==============================\n")

    print(f"Number of parameters: {num_parameters:.1f} M")
    print(f"Average training FPS: {avg_train_fps:.2f} FPS")
    print(f"Maximum training GPU memory: {max_train_mem:.2f} GB\n")

    print(f"Average inference FLOPS: {avg_inf_flops:.1f} GFLOPS")
    print(f"Average inference FPS: {avg_inf_fps:.2f} FPS")
    print(f"Maximum inference GPU memory: {max_inf_mem:.2f} GB\n")

    # Save results if output directory is provided
    if output_dir is not None:

        # Get dictionary with model analysis results
        result_dict = {'num_params': num_parameters, 'train_fps': avg_train_fps, 'train_mem': max_train_mem}
        result_dict = {**result_dict, 'inf_flops': avg_inf_flops, 'inf_fps': avg_inf_fps, 'inf_mem': max_inf_mem}

        # Save result dictionary as json
        with (output_dir / 'model_analysis.json').open('w') as result_file:
            json.dump(result_dict, result_file)
