"""DINO (Caron et al. 2021 ICCV) — self-distillation SSL.

Two networks: student (online) + teacher (EMA of student, no grad).
Student predicts teacher's softmax output via cross-entropy on K-dim probabilities.
Teacher is centered + sharpened to avoid collapse.
No contrastive negatives, no stop-grad alone (relies on momentum + sharpen-centering).

This is the most cited modern (post-2020) ResNet-compatible SSL paper.
Useful test of EML head — DINO uses linear projector to K-dim distribution.
If EML head + DINO objective fail: aligns with "EML needs InfoNCE-style softmax" thesis.

Usage:
    python src/dino_cifar.py --head eml --aug eml-aug --dataset imagenet1k \\
        --epochs 100 --batch-size 256 --lr 5e-4 --seed 0 \\
        --out-dir logs/in1k_dino_D_eml_emlaug_s0
"""
from __future__ import annotations

import argparse
import copy
import json
import math
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


class DINOHead(nn.Module):
    """3-layer MLP projector + L2-normalize + last linear with weight_norm (no bias).
    Produces K-dim logits (typically K=65536 in paper; we use K=4096 for IN1k bs=256).
    """
    def __init__(self, in_dim: int = 512, hidden_dim: int = 512, bottleneck_dim: int = 256,
                 out_dim: int = 4096, proj_kind: str = "mlp"):
        super().__init__()
        # MLP projector (can be swapped with EML head)
        self.projection = build_projection_head(proj_kind, in_dim=in_dim,
                                                 hidden_dim=hidden_dim, out_dim=bottleneck_dim)
        # Final linear with weight normalization, no bias
        self.last_layer = nn.utils.parametrizations.weight_norm(
            nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.parametrizations.weight.original0.data.fill_(1.0)
        self.last_layer.parametrizations.weight.original0.requires_grad = False

    def forward(self, x):
        h = self.projection(x)
        h = F.normalize(h, dim=-1)
        return self.last_layer(h)


class DINOModel(nn.Module):
    def __init__(self, head_kind: str, dataset: str,
                 out_dim: int = 4096, bottleneck_dim: int = 256):
        super().__init__()
        use_eml_encoder = head_kind.startswith("p2")
        proj_kind = head_kind.split("_", 1)[1] if head_kind.startswith("p2_") else head_kind
        if dataset in ("imagenet100", "imagenet1k"):
            self.s_encoder = make_imagenet_eml_resnet18() if use_eml_encoder else make_imagenet_resnet18()
        else:
            self.s_encoder = make_cifar_eml_resnet18() if use_eml_encoder else make_cifar_resnet18()
        self.s_head = DINOHead(in_dim=512, bottleneck_dim=bottleneck_dim,
                               out_dim=out_dim, proj_kind=proj_kind)
        # Teacher: deep copy, no grad
        self.t_encoder = copy.deepcopy(self.s_encoder)
        self.t_head = copy.deepcopy(self.s_head)
        for p in (*self.t_encoder.parameters(), *self.t_head.parameters()):
            p.requires_grad = False
        # Center buffer for teacher (running mean of teacher's outputs)
        self.register_buffer("center", torch.zeros(1, out_dim))

    @torch.no_grad()
    def momentum_update(self, tau: float):
        for tp, sp in zip(self.t_encoder.parameters(), self.s_encoder.parameters()):
            tp.data.mul_(tau).add_((1 - tau) * sp.data)
        for tp, sp in zip(self.t_head.parameters(), self.s_head.parameters()):
            tp.data.mul_(tau).add_((1 - tau) * sp.data)

    @torch.no_grad()
    def update_center(self, teacher_out: torch.Tensor, m: float = 0.9):
        batch_center = teacher_out.mean(dim=0, keepdim=True)
        self.center.mul_(m).add_((1 - m) * batch_center)

    def forward_student(self, x):
        return self.s_head(self.s_encoder(x))

    @torch.no_grad()
    def forward_teacher(self, x):
        return self.t_head(self.t_encoder(x))


def dino_loss(s_out: torch.Tensor, t_out: torch.Tensor, center: torch.Tensor,
              s_temp: float = 0.1, t_temp: float = 0.04) -> torch.Tensor:
    """Cross-entropy between student softmax and teacher softmax (centered+sharpened)."""
    s_log_softmax = F.log_softmax(s_out / s_temp, dim=-1)
    t_softmax = F.softmax((t_out - center) / t_temp, dim=-1)
    return -(t_softmax * s_log_softmax).sum(dim=-1).mean()


def train(args):
    torch.manual_seed(args.seed)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[setup] device={device} head={args.head} aug={args.aug} "
          f"dataset={args.dataset} epochs={args.epochs} method=dino "
          f"K={args.out_dim} t_temp={args.t_temp} s_temp={args.s_temp}")

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

    model = DINOModel(head_kind=args.head, dataset=args.dataset,
                      out_dim=args.out_dim, bottleneck_dim=args.bottleneck_dim).to(device)
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    log_f = (out / "train.jsonl").open("a")
    nan_steps = 0; step = 0
    t_start = time.time()
    steps_per_epoch = len(ds) // args.batch_size
    log_interval = max(1, steps_per_epoch // 10)

    for epoch in range(args.epochs):
        model.train()
        ep_loss = 0.0; ep_n = 0; ep_start = time.time()
        # Cosine warmup of momentum tau from 0.996 → 1.0
        tau = args.momentum_tau + (1.0 - args.momentum_tau) * (0.5 - 0.5 * math.cos(math.pi * epoch / args.epochs))
        for (x1, x2), _ in loader:
            x1 = x1.to(device, non_blocking=True)
            x2 = x2.to(device, non_blocking=True)
            # Student gets both views
            s1 = model.forward_student(x1)
            s2 = model.forward_student(x2)
            # Teacher gets both views (no grad)
            t1 = model.forward_teacher(x1)
            t2 = model.forward_teacher(x2)
            # Cross loss: student view1 vs teacher view2, and vice versa
            loss = 0.5 * (dino_loss(s1, t2, model.center, args.s_temp, args.t_temp) +
                         dino_loss(s2, t1, model.center, args.s_temp, args.t_temp))
            if not torch.isfinite(loss):
                nan_steps += 1
                opt.zero_grad(set_to_none=True); continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            model.momentum_update(tau)
            # Update center with the averaged teacher output
            model.update_center(torch.cat([t1, t2], dim=0), m=args.center_m)
            ep_loss += loss.item() * x1.size(0); ep_n += x1.size(0); step += 1
            if step % log_interval == 0:
                print(f"  [ep {epoch:03d} step {step}/{steps_per_epoch}] "
                      f"loss={loss.item():.3f} tau={tau:.4f} {time.time()-ep_start:.0f}s", flush=True)
        sched.step()
        avg = ep_loss / max(ep_n, 1)
        record = {"epoch": epoch, "loss": avg, "lr": opt.param_groups[0]["lr"],
                  "tau": tau, "nan_steps_total": nan_steps,
                  "elapsed_s": time.time() - t_start}
        log_f.write(json.dumps(record) + "\n"); log_f.flush()
        print(f"[ep {epoch:03d}] loss={avg:.3f} lr={record['lr']:.2e} tau={tau:.4f} "
              f"nan={nan_steps} ({time.time()-ep_start:.0f}s)", flush=True)

    log_f.close()
    ckpt = out / "encoder_final.pt"
    # Save student encoder (matches downstream probe interface)
    torch.save({"encoder": model.s_encoder.state_dict(), "head": args.head,
                "method": "dino", "args": vars(args)}, ckpt)
    print(f"[done] saved {ckpt} (nan_steps={nan_steps})")


def parse_args():
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
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=0.04)
    p.add_argument("--momentum-tau", type=float, default=0.996)
    p.add_argument("--out-dim", type=int, default=4096)
    p.add_argument("--bottleneck-dim", type=int, default=256)
    p.add_argument("--s-temp", type=float, default=0.1)
    p.add_argument("--t-temp", type=float, default=0.04)
    p.add_argument("--center-m", type=float, default=0.9)
    p.add_argument("--num-workers", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out-dir", required=True)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
