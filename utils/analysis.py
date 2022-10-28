"""
Collection of analysis utilities.
"""
import json

import torch
from torch.utils.benchmark import Timer

from utils.flops import FlopCountAnalysis, adj_conv2d_flop_jit, msda_flop_jit, roi_align_mmcv_flop_jit

EXTRA_OPS = {
    'aten::abs': None,
    'aten::affine_grid_generator': None,
    'aten::argmin': None,
    'aten::avg_pool2d': None,
    'aten::clone': None,
    'aten::cos': None,
    'aten::cumsum': None,
    'aten::diff': None,
    'aten::expand_as': None,
    'aten::flip': None,
    'aten::le': None,
    'aten::linspace': None,
    'aten::lt': None,
    'aten::movedim': None,
    'aten::ne': None,
    'aten::prod': None,
    'aten::pow': None,
    'aten::repeat_interleave': None,
    'aten::rsqrt': None,
    'aten::scatter_': None,
    'aten::sin': None,
    'aten::sub_': None,
    'aten::sum': None,
    'aten::topk': None,
    'aten::_unique2': None,
    'aten::where': None,
    'prim::PythonOp.AdjConv2d': adj_conv2d_flop_jit,
    'prim::PythonOp.MSDeformAttnFunction': msda_flop_jit,
    'prim::PythonOp.RoIAlignFunction': roi_align_mmcv_flop_jit,
}


def analyze_model(model, dataloader, optimizer, max_grad_norm=-1, num_samples=100, output_dir=None):
    """
    Function analyzing the given model by computing various model properties during both training and inference.

    Args:
        model (nn.Module): Module containing the model to be analyzed.
        dataloader (torch.utils.data.Dataloader): Dataloader used to obtain the FLOPs and FPS metrics.
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

    # Initialize average inference FLOPs and FPS variables
    avg_inf_flops = 0.0
    avg_inf_fps = 0.0

    # Iterate over dataloader
    for i, (images, _) in enumerate(dataloader, 1):

        # Place images on correct device
        images = images.to(device)

        # Get tuple of model inputs
        inputs = (images,)

        # Get inference FLOPs
        inf_flops = FlopCountAnalysis(model, inputs)
        inf_flops.set_op_handle(**EXTRA_OPS)

        inf_flops.unsupported_ops_warnings(i == 1)
        inf_flops.uncalled_modules_warnings(i == 1)
        inf_flops.tracer_warnings('no_tracer_warning') if i == 1 else inf_flops.tracer_warnings('none')

        inf_flops = inf_flops.total() / 10**9
        avg_inf_flops += inf_flops / num_samples

        # Get inference FPS
        globals_dict = {'model': model, 'inputs': inputs}
        timer = Timer(stmt="model(*inputs)", globals=globals_dict)

        inf_fps = 1 / timer.timeit(number=1).median
        avg_inf_fps += inf_fps / num_samples

        # Print batch inference FLOPs and FPS
        print_str = f"Analysis inference [{i:{num_digits}d}/{num_samples}]:  "
        print_str += f"inf_flops: {inf_flops:.1f} GFLOPs  "
        print_str += f"inf_fps: {inf_fps:.2f} FPS"
        print(print_str)

        # Break when 'num_samples' batches are processed
        if i == num_samples:
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

    print(f"Average inference FLOPs: {avg_inf_flops:.1f} GFLOPs")
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
