"""Lightweight DDP helpers shared across training scripts.

Usage in train script:
    from ddp_utils import init_ddp, cleanup_ddp, is_main, gather_for_contrastive

    rank, world_size, local_rank = init_ddp()  # 0,1,0 in single-GPU mode
    device = torch.device(f"cuda:{local_rank}")
    if world_size > 1:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[local_rank])
    if is_main():
        ... # only rank 0 writes log, saves ckpt
    cleanup_ddp()

Launched via:
    torchrun --nproc_per_node=N src/script.py [args]

Falls back to single-GPU when LOCAL_RANK is not set in env (no DDP overhead).
"""
from __future__ import annotations

import os
import torch
import torch.distributed as dist


def init_ddp() -> tuple[int, int, int]:
    """Initialize torch.distributed if LOCAL_RANK env is set; otherwise no-op.

    Returns (rank, world_size, local_rank). In single-GPU mode returns (0, 1, 0).
    """
    if "LOCAL_RANK" not in os.environ:
        return 0, 1, 0
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return dist.get_rank(), dist.get_world_size(), local_rank


def cleanup_ddp() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main() -> bool:
    """True on rank 0 (or single-GPU mode)."""
    return not dist.is_initialized() or dist.get_rank() == 0


def world_size() -> int:
    return dist.get_world_size() if dist.is_initialized() else 1


def gather_for_contrastive(t: torch.Tensor) -> torch.Tensor:
    """All-gather tensor across ranks for contrastive loss.

    Each rank computes its own forward and produces a [bs_local, D] tensor.
    For InfoNCE we want negatives from all ranks → concat into [bs_local * world, D].
    Backwards flows only into the rank's own slice (other ranks' tensors are detached).
    """
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return t
    world = dist.get_world_size()
    rank = dist.get_rank()
    gathered = [torch.zeros_like(t) for _ in range(world)]
    dist.all_gather(gathered, t)
    # Replace our own slice with the autograd-connected tensor.
    gathered[rank] = t
    return torch.cat(gathered, dim=0)


def reduce_mean(t: torch.Tensor) -> torch.Tensor:
    """All-reduce a tensor by mean across ranks (for logging)."""
    if not dist.is_initialized():
        return t
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t / dist.get_world_size()
