""" Distributed training/validation utils

Hacked together by / Copyright 2020 Ross Wightman
"""
import logging
import os
from typing import Optional

import torch
from torch import distributed as dist

from .model import unwrap_model

_logger = logging.getLogger(__name__)


def reduce_tensor(tensor, n):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= n
    return rt


def distribute_bn(model, world_size, reduce=False):
    # ensure every node has the same running bn stats
    for bn_name, bn_buf in unwrap_model(model).named_buffers(recurse=True):
        if ('running_mean' in bn_name) or ('running_var' in bn_name):
            if reduce:
                # average bn stats across whole group
                torch.distributed.all_reduce(bn_buf, op=dist.ReduceOp.SUM)
                bn_buf /= float(world_size)
            else:
                # broadcast bn stats from rank 0 to whole group
                torch.distributed.broadcast(bn_buf, 0)


def is_global_primary(args):
    return args.rank == 0


def is_local_primary(args):
    return args.local_rank == 0


def is_primary(args, local=False):
    return is_local_primary(args) if local else is_global_primary(args)


def is_distributed_env():
    if 'WORLD_SIZE' in os.environ:
        return int(os.environ['WORLD_SIZE']) > 1
    if 'SLURM_NTASKS' in os.environ:
        return int(os.environ['SLURM_NTASKS']) > 1
    return False


def world_info_from_env():
    local_rank = 0
    for v in ('LOCAL_RANK', 'MPI_LOCALRANKID', 'SLURM_LOCALID', 'OMPI_COMM_WORLD_LOCAL_RANK'):
        if v in os.environ:
            local_rank = int(os.environ[v])
            break

    global_rank = 0
    for v in ('RANK', 'PMI_RANK', 'SLURM_PROCID', 'OMPI_COMM_WORLD_RANK'):
        if v in os.environ:
            global_rank = int(os.environ[v])
            break

    world_size = 1
    for v in ('WORLD_SIZE', 'PMI_SIZE', 'SLURM_NTASKS', 'OMPI_COMM_WORLD_SIZE'):
        if v in os.environ:
            world_size = int(os.environ[v])
            break

    return local_rank, global_rank, world_size


def init_distributed_device(args):
    # Distributed training = training on more than one GPU.
    # Works in both single and multi-node scenarios.
    args.distributed = False
    args.world_size = 1
    args.rank = 0  # global rank
    args.local_rank = 0
    result = init_distributed_device_so(
        device=getattr(args, 'device', 'cuda'),
        dist_backend=getattr(args, 'dist_backend', None),
        dist_url=getattr(args, 'dist_url', None),
    )
    args.device = result['device']
    args.world_size = result['world_size']
    args.rank = result['global_rank']
    args.local_rank = result['local_rank']
    args.distributed = result['distributed']
    device = torch.device(args.device)
    return device


def init_distributed_device_so(
        device: str = 'cuda',
        dist_backend: Optional[str] = None,
        dist_url: Optional[str] = None,
):
    # Distributed training = training on more than one GPU.
    # Works in both single and multi-node scenarios.
    distributed = False
    world_size = 1
    global_rank = 0
    local_rank = 0
    device_type, *device_idx = device.split(':', maxsplit=1)

    if dist_backend is None:
        # FIXME: verify that ROCm transform nccl to rccl
        dist_backends = {
            "xpu": "ccl",
            "hpu": "hccl",
            "cuda": "nccl",
            "npu": "hccl",
        }
        dist_backend = dist_backends.get(device_type, 'gloo')
    dist_url = dist_url or 'env://'

    # TBD, support horovod?
    # if args.horovod:
    #     import horovod.torch as hvd
    #     assert hvd is not None, "Horovod is not installed"
    #     hvd.init()
    #     args.local_rank = int(hvd.local_rank())
    #     args.rank = hvd.rank()
    #     args.world_size = hvd.size()
    #     args.distributed = True
    #     os.environ['LOCAL_RANK'] = str(args.local_rank)
    #     os.environ['RANK'] = str(args.rank)
    #     os.environ['WORLD_SIZE'] = str(args.world_size)
    if is_distributed_env():
        if 'SLURM_PROCID' in os.environ:
            # DDP via SLURM
            local_rank, global_rank, world_size = world_info_from_env()
            # SLURM var -> torch.distributed vars in case needed
            os.environ['LOCAL_RANK'] = str(local_rank)
            os.environ['RANK'] = str(global_rank)
            os.environ['WORLD_SIZE'] = str(world_size)
            torch.distributed.init_process_group(
                backend=dist_backend,
                init_method=dist_url,
                world_size=world_size,
                rank=global_rank,
            )
        else:
            # DDP via torchrun, torch.distributed.launch
            local_rank, _, _ = world_info_from_env()
            torch.distributed.init_process_group(
                backend=dist_backend,
                init_method=dist_url,
            )
            world_size = torch.distributed.get_world_size()
            global_rank = torch.distributed.get_rank()
        distributed = True

    if device_type == 'cuda':
        assert torch.cuda.is_available(), f'CUDA is not available but {device} was specified.'
    if device_type == 'npu':
        assert torch.npu.is_available(), f'Ascend NPU is not available but {device} was specified.'

    if distributed and device != 'cpu':
        # Ignore manually specified device index in distributed mode and
        # override with resolved local rank, fewer headaches in most setups.
        if device_idx:
            _logger.warning(f'device index {device_idx[0]} removed from specified ({device}).')
        device = f'{device_type}:{local_rank}'

    if device.startswith('cuda:'):
        torch.cuda.set_device(device)

    return dict(
        device=device,
        global_rank=global_rank,
        local_rank=local_rank,
        world_size=world_size,
        distributed=distributed,
    )
