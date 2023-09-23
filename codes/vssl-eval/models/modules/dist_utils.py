"""
copied from https://github.com/facebookresearch/SlowFast/blob/5b5d9ecb15a54188943af0cbf5f7c420d8409018/slowfast/utils/distributed.py
"""


import torch
import torch.distributed as dist


def all_gather(tensors):
    """
    All gathers the provided tensors from all processes across machines.
    Args:
        tensors (list): tensors to perform all gather across all processes in
        all machines.
    """

    gather_list = []
    output_tensor = []
    world_size = dist.get_world_size()
    for tensor in tensors:
        tensor_placeholder = [
            torch.ones_like(tensor) for _ in range(world_size)
        ]
        dist.all_gather(tensor_placeholder, tensor, async_op=False)
        gather_list.append(tensor_placeholder)
    for gathered_tensor in gather_list:
        output_tensor.append(torch.cat(gathered_tensor, dim=0))
    return output_tensor


def all_reduce(tensors, average=True):
    """
    All reduce the provided tensors from all processes across machines.
    Args:
        tensors (list): tensors to perform all reduce across all processes in
        all machines.
        average (bool): scales the reduced tensor by the number of overall
        processes across all machines.
    """

    for tensor in tensors:
        dist.all_reduce(tensor, async_op=False)
    if average:
        world_size = dist.get_world_size()
        for tensor in tensors:
            tensor.mul_(1.0 / world_size)
    return tensors


def is_master_proc(num_gpus=8):
    """
    Determines if the current process is the master process.
    """
    if torch.distributed.is_initialized():
        return dist.get_rank() % num_gpus == 0
    else:
        return True


def is_root_proc():
    """
    Determines if the current process is the root process.
    """
    if torch.distributed.is_initialized():
        return dist.get_rank() == 0
    else:
        return True


def get_rank():
    """
    Get the rank of the current process.
    """
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()
    
class AllGatherWithGradient(torch.autograd.Function):
    """AllGatherWithGradient"""

    @staticmethod
    def forward(ctx, input):
        world_size = dist.get_world_size()
        x_gather = [torch.ones_like(input) for _ in range(world_size)]
        torch.distributed.all_gather(x_gather, input, async_op=False)
        x_gather = torch.cat(x_gather, dim=0)
        return x_gather

    @staticmethod
    def backward(ctx, grad_output):

        reduction = torch.distributed.all_reduce(grad_output, async_op=True)
        reduction.wait()

        world_size = dist.get_world_size()
        N = grad_output.size(0)
        mini_batchsize = N // world_size
        cur_gpu = torch.distributed.get_rank()
        grad_output = grad_output[
            cur_gpu * mini_batchsize : (cur_gpu + 1) * mini_batchsize
        ]
        return grad_output