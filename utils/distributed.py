"""
Distributed training utilities.
"""
import os

import torch


def init_distributed_mode(args):
    if 'RANK' not in os.environ or 'WORLD_SIZE' not in os.environ:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True
    args.rank = int(os.environ["RANK"])
    args.world_size = int(os.environ['WORLD_SIZE'])

    args.gpu = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(args.gpu)

    print('| distributed init (rank {}): {}'.format(args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend='nccl', init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)

    set_master_only_printing(args.rank == 0)


def set_master_only_printing(is_master):
    """
    Disables printing when not in master process.
    """
    import builtins
    builtin_print = builtins.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    builtins.print = print


def is_dist_avail_and_initialized():
    if not torch.distributed.is_available():
        return False
    if not torch.distributed.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return torch.distributed.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return torch.distributed.get_rank()


def is_main_process():
    return get_rank() == 0


@torch.no_grad()
def reduce_dict(input_dict, average=True):
    """
    Reduce dictionary values across all processes.

    Args:
        input_dict (Dict): Input dictionary for which values will be reduced.
        average (bool): If true, average values, else simply take the sum.

    Returns:
        reduced_dict (Dict): Dictionary with reduced values for all input keys.
    """

    if not is_dist_avail_and_initialized():
        return input_dict

    keys = sorted(input_dict.keys())
    values = torch.stack([input_dict[key] for key in keys], dim=0)
    torch.distributed.all_reduce(values)

    values = values/get_world_size() if average else values
    reduced_dict = {k: v for k, v in zip(keys, values)}

    return reduced_dict
