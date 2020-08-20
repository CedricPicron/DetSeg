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
    This function disables printing when not in master process
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


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)
