"""MoCo v2 (He et al. 2020) — fourth-paradigm contrastive SSL baseline.

Sits between SimCLR (in-batch negatives, batch=128 → 127 negs) and BYOL/SimSiam
(zero negatives) on the negatives-count spectrum:

    SimCLR (127 neg/anchor) >> MoCo v2 (K=8192 queue neg) >> BYOL/SimSiam (0 neg)

Architecture: query encoder + projection head + EMA-updated key encoder + key projection.
Queue stores past projected keys; InfoNCE uses queue + current key positives.

Tests the "negatives anchor EML" mechanism: if EML head wins when there ARE negatives
(SimCLR) and crashes when there aren't (SimSiam/BYOL), what happens at queue-size 8192?

Usage:
    python src/moco_cifar.py --head eml --aug eml-aug --dataset imagenet100 \
        --epochs 100 --batch-size 128 --lr 5e-2 --momentum-tau 0.999 \
        --moco-k 8192 --moco-t 0.2 --seed 0 --out-dir logs/in100_moco_D_eml_emlaug_s0
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets

from simclr_cifar import (
    TwoViewTransform, simclr_augmentation,
    make_cifar_resnet18, make_imagenet_resnet18,
    make_cifar_eml_resnet18, make_imagenet_eml_resnet18,
)
from eml import build_projection_head


class MoCoModel(nn.Module):
    """MoCo v2: query encoder + projection + EMA key encoder + projection + queue.

    Forward returns InfoNCE loss between query projections and key projections
    against a K-sized queue of past keys.
    """

    def __init__(self, head_kind: str, dataset: str,
                 hidden_dim: int = 512, proj_dim: int = 128,
                 moco_k: int = 8192, moco_t: float = 0.2) -> None:
        super().__init__()
        self.K = moco_k
        self.T = moco_t
        use_eml_encoder = head_kind.startswith("p2")
        proj_kind = head_kind.split("_", 1)[1] if head_kind.startswith("p2_") else head_kind
        if dataset in ("imagenet100", "imagenet1k"):
            self.q_encoder = make_imagenet_eml_resnet18() if use_eml_encoder else make_imagenet_resnet18()
        else:
            self.q_encoder = make_cifar_eml_resnet18() if use_eml_encoder else make_cifar_resnet18()
        self.q_projection = build_projection_head(proj_kind, in_dim=512,
                                                  hidden_dim=hidden_dim, out_dim=proj_dim)
        # Key encoder = deep copy of query; no grad, updated by EMA.
        self.k_encoder = copy.deepcopy(self.q_encoder)
        self.k_projection = copy.deepcopy(self.q_projection)
        for p in (*self.k_encoder.parameters(), *self.k_projection.parameters()):
            p.requires_grad = False
        # Queue: K projected keys (proj_dim). Normalized.
        self.register_buffer("queue", F.normalize(torch.randn(proj_dim, moco_k), dim=0))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def momentum_update(self, tau: float) -> None:
        for kp, qp in zip(self.k_encoder.parameters(), self.q_encoder.parameters()):
            kp.data.mul_(tau).add_((1.0 - tau) * qp.data)
        for kp, qp in zip(self.k_projection.parameters(), self.q_projection.parameters()):
            kp.data.mul_(tau).add_((1.0 - tau) * qp.data)

    @torch.no_grad()
    def dequeue_and_enqueue(self, keys: torch.Tensor) -> None:
        bs = keys.size(0)
        ptr = int(self.queue_ptr)
        # Wrap if past end.
        if ptr + bs <= self.K:
            self.queue[:, ptr:ptr + bs] = keys.T
        else:
            tail = self.K - ptr
            self.queue[:, ptr:] = keys[:tail].T
            self.queue[:, :bs - tail] = keys[tail:].T
        self.queue_ptr[0] = (ptr + bs) % self.K

    def forward(self, x_q, x_k):
        """Compute InfoNCE between q (query) and k (key) with queue negatives."""
        # Query path
        q = F.normalize(self.q_projection(self.q_encoder(x_q)), dim=1)
        # Key path (no grad)
        with torch.no_grad():
            k = F.normalize(self.k_projection(self.k_encoder(x_k)), dim=1)
        # Positives: q · k (per-sample dot)
        l_pos = torch.einsum("nc,nc->n", q, k).unsqueeze(-1)  # [N, 1]
        # Negatives: q · queue (queue is [C, K])
        l_neg = torch.einsum("nc,ck->nk", q, self.queue.clone().detach())  # [N, K]
        # Concat → [N, 1+K]
        logits = torch.cat([l_pos, l_neg], dim=1) / self.T
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=q.device)
        loss = F.cross_entropy(logits, labels)
        return loss, k.detach()


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[setup] device={device} head={args.head} aug={args.aug} "
          f"dataset={args.dataset} epochs={args.epochs} method=moco K={args.moco_k} T={args.moco_t} tau={args.momentum_tau}")

    image_size = 224 if args.dataset in ("imagenet100", "imagenet1k") else 32
    transform = TwoViewTransform(simclr_augmentation(
        image_size=image_size, dataset=args.dataset, aug=args.aug,
        eml_aug_strength=args.eml_aug_strength, eml_aug_p=args.eml_aug_p,
        jitter_strength=args.jitter_strength, jitter_p=args.jitter_p,
        gray_p=args.gray_p,
    ))
    if args.dataset == "cifar10":
        ds = datasets.CIFAR10(args.data_root, train=True, download=True, transform=transform)
    elif args.dataset == "cifar100":
        ds = datasets.CIFAR100(args.data_root, train=True, download=True, transform=transform)
    elif args.dataset in ("imagenet100", "imagenet1k"):
        train_dir = Path(args.data_root) / "train"
        if not train_dir.exists():
            train_dir = Path(args.data_root) / "val"
        ds = datasets.ImageFolder(str(train_dir), transform=transform)
    else:
        raise ValueError(args.dataset)

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, pin_memory=True,
                        drop_last=True, persistent_workers=args.num_workers > 0)

    model = MoCoModel(head_kind=args.head, dataset=args.dataset,
                      hidden_dim=args.hidden_dim, proj_dim=args.proj_dim,
                      moco_k=args.moco_k, moco_t=args.moco_t).to(device)
    # Only optimize query net (key receives EMA, no grad).
    opt_params = list(model.q_encoder.parameters()) + list(model.q_projection.parameters())
    opt = torch.optim.SGD(opt_params, lr=args.lr, momentum=0.9, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    log_f = (out / "train.jsonl").open("a")
    nan_steps = 0
    step = 0
    t_start = time.time()
    steps_per_epoch = len(ds) // args.batch_size
    log_interval = max(1, steps_per_epoch // 10)

    for epoch in range(args.epochs):
        model.train()
        ep_loss = 0.0; ep_n = 0; ep_start = time.time()
        for (x1, x2), _ in loader:
            x1 = x1.to(device, non_blocking=True)
            x2 = x2.to(device, non_blocking=True)
            # Symmetric MoCo: average two directions
            loss_a, k_a = model(x1, x2)
            loss_b, k_b = model(x2, x1)
            loss = 0.5 * (loss_a + loss_b)
            if not torch.isfinite(loss):
                nan_steps += 1
                opt.zero_grad(set_to_none=True); continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            model.momentum_update(args.momentum_tau)
            # Enqueue both directions of keys
            model.dequeue_and_enqueue(k_a)
            model.dequeue_and_enqueue(k_b)
            ep_loss += loss.item() * x1.size(0); ep_n += x1.size(0); step += 1
            if step % log_interval == 0:
                print(f"  [ep {epoch:03d} step {step}/{steps_per_epoch}] "
                      f"loss={loss.item():.4f} {time.time()-ep_start:.0f}s", flush=True)
        sched.step()
        avg = ep_loss / max(ep_n, 1)
        record = {"epoch": epoch, "loss": avg, "lr": opt.param_groups[0]["lr"],
                  "nan_steps_total": nan_steps, "elapsed_s": time.time() - t_start,
                  "K": args.moco_k, "T": args.moco_t, "tau": args.momentum_tau}
        log_f.write(json.dumps(record) + "\n"); log_f.flush()
        print(f"[ep {epoch:03d}] loss={avg:.4f} lr={record['lr']:.2e} "
              f"nan={nan_steps} ({time.time()-ep_start:.0f}s)", flush=True)

    log_f.close()
    ckpt = out / "encoder_final.pt"
    torch.save({"encoder": model.q_encoder.state_dict(), "head": args.head,
                "method": "moco", "momentum_tau": args.momentum_tau,
                "moco_k": args.moco_k, "moco_t": args.moco_t,
                "args": vars(args)}, ckpt)
    print(f"[done] saved {ckpt} (nan_steps={nan_steps})")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--head", choices=["mlp", "eml", "p2_mlp", "signed_power", "tanh", "beml"], default="mlp")
    p.add_argument("--aug", choices=["standard", "eml-aug", "strong-jitter"], default="standard")
    p.add_argument("--eml-aug-strength", type=float, default=0.35)
    p.add_argument("--eml-aug-p", type=float, default=0.8)
    p.add_argument("--jitter-strength", type=float, default=0.4)
    p.add_argument("--jitter-p", type=float, default=0.8)
    p.add_argument("--gray-p", type=float, default=0.2)
    p.add_argument("--dataset", choices=["cifar10", "cifar100", "imagenet100", "imagenet1k"], default="imagenet100")
    p.add_argument("--data-root", default="data")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=5e-2)
    p.add_argument("--momentum-tau", type=float, default=0.999,
                   help="MoCo key encoder EMA momentum (default 0.999)")
    p.add_argument("--moco-k", type=int, default=8192,
                   help="MoCo queue size (default 8192 for IN100)")
    p.add_argument("--moco-t", type=float, default=0.2,
                   help="MoCo InfoNCE temperature (default 0.2)")
    p.add_argument("--hidden-dim", type=int, default=512)
    p.add_argument("--proj-dim", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out-dir", required=True)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
