"""SimSiam baseline with togglable projection head (mlp | eml) and augmentation.

Pairs with simclr_cifar.py — same encoder + augmentation pipeline + 4-condition
(A/B/C/D) scheme, but replaces the SimCLR InfoNCE loss with SimSiam's
asymmetric stop-grad + predictor scheme (Chen & He, 2021).

Method:
    x  ─┬─► aug(x) ─► encoder ─► projection ─z1 ─► predictor ─p1
        │
        └─► aug(x) ─► encoder ─► projection ─z2 ─► predictor ─p2
    loss = (1 − cos(p1, sg(z2)) + 1 − cos(p2, sg(z1))) / 2

Reuses simclr_cifar.py for ResNet-18 builders and the augmentation pipeline so
that --aug standard|eml-aug|strong-jitter behaves identically across methods.

Usage:
    python src/simsiam_cifar.py --head eml --aug eml-aug --dataset imagenet100 \\
        --epochs 200 --batch-size 256 --lr 5e-2 --seed 0 \\
        --out-dir logs/in100_simsiam_D_eml_emlaug_s0
"""
from __future__ import annotations

import argparse
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


class Predictor(nn.Module):
    """2-layer MLP predictor with BN+ReLU between, no activation on output."""

    def __init__(self, dim: int, hidden_dim: int | None = None) -> None:
        super().__init__()
        hidden_dim = hidden_dim or dim
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)


class SimSiamModel(nn.Module):
    """SimSiam wrapper. Inserts BN after the projection head — without it the
    representation collapses within ~5 epochs (loss saturates at −1)."""

    def __init__(self, head_kind: str, dataset: str,
                 hidden_dim: int = 512, proj_dim: int = 128,
                 pred_hidden: int | None = None,
                 latent_predict: bool = False, power: float = 3.0) -> None:
        super().__init__()
        self.latent_predict = latent_predict
        use_eml_encoder = head_kind.startswith("p2")
        proj_kind = head_kind.split("_", 1)[1] if head_kind.startswith("p2_") else head_kind
        if dataset in ("imagenet100", "imagenet1k"):
            self.encoder = make_imagenet_eml_resnet18() if use_eml_encoder else make_imagenet_resnet18()
        else:
            self.encoder = make_cifar_eml_resnet18() if use_eml_encoder else make_cifar_resnet18()
        if latent_predict:
            # I-JEPA 式潜接口(无 EMA 版,仅 stop-grad):去 projection-MLP/proj_bn,512-d 表征空间预测;
            # 与 BYOL latent-predict 的唯一差别 = 无 EMA → 干净检验"EMA 是否潜接口防坍缩关键"。
            if head_kind == "headless":
                self.projection = nn.Identity()
            else:
                self.projection = build_projection_head(proj_kind, in_dim=512,
                                                        hidden_dim=hidden_dim, out_dim=512, power=power)
            self.proj_bn = nn.Identity()
            self.predictor = Predictor(dim=512, hidden_dim=pred_hidden)
        else:
            self.projection = build_projection_head(proj_kind, in_dim=512,
                                                    hidden_dim=hidden_dim, out_dim=proj_dim, power=power)
            self.proj_bn = nn.BatchNorm1d(proj_dim, affine=False)  # SimSiam anti-collapse
            self.predictor = Predictor(dim=proj_dim, hidden_dim=pred_hidden)

    def forward(self, x):
        h = self.encoder(x)
        z = self.proj_bn(self.projection(h))
        p = self.predictor(z)
        return z, p


def negative_cosine(p: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """1 − cos(p, sg(z)). Stop-grad applied on z."""
    z = z.detach()
    p = F.normalize(p, dim=1)
    z = F.normalize(z, dim=1)
    return -(p * z).sum(dim=1).mean()


def simsiam_loss(p1, z1, p2, z2) -> torch.Tensor:
    """Symmetric SimSiam loss."""
    return 0.5 * (negative_cosine(p1, z2) + negative_cosine(p2, z1))


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[setup] device={device} head={args.head} aug={args.aug} "
          f"dataset={args.dataset} epochs={args.epochs} method=simsiam")

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

    model = SimSiamModel(head_kind=args.head, dataset=args.dataset,
                         hidden_dim=args.hidden_dim, proj_dim=args.proj_dim,
                         latent_predict=args.latent_predict, power=args.power).to(device)
    # SimSiam paper recipe: SGD + momentum 0.9 + weight_decay 1e-4 + cosine lr.
    # Empirically AdamW + this lr collapses; SGD is much more stable here.
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
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
            z1, p1 = model(x1)
            z2, p2 = model(x2)
            loss = simsiam_loss(p1, z1, p2, z2)
            if not torch.isfinite(loss):
                nan_steps += 1
                opt.zero_grad(set_to_none=True); continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            ep_loss += loss.item() * x1.size(0); ep_n += x1.size(0); step += 1
            if step % log_interval == 0:
                print(f"  [ep {epoch:03d} step {step}/{steps_per_epoch}] "
                      f"loss={loss.item():.4f} {time.time()-ep_start:.0f}s", flush=True)
        sched.step()
        avg = ep_loss / max(ep_n, 1)
        record = {"epoch": epoch, "loss": avg, "lr": opt.param_groups[0]["lr"],
                  "nan_steps_total": nan_steps, "elapsed_s": time.time() - t_start}
        log_f.write(json.dumps(record) + "\n"); log_f.flush()
        print(f"[ep {epoch:03d}] loss={avg:.4f} lr={record['lr']:.2e} "
              f"nan={nan_steps} ({time.time()-ep_start:.0f}s)", flush=True)

    log_f.close()
    ckpt = out / "encoder_final.pt"
    torch.save({"encoder": model.encoder.state_dict(), "head": args.head,
                "method": "simsiam", "args": vars(args)}, ckpt)
    print(f"[done] saved {ckpt} (nan_steps={nan_steps})")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--head", choices=["mlp", "eml", "p2_mlp", "signed_power", "tanh", "beml", "headless"], default="mlp")
    p.add_argument("--latent-predict", action="store_true",
                   help="I-JEPA 式潜接口(无 EMA,仅 stop-grad):去 projection/proj_bn,512-d 表征空间预测")
    p.add_argument("--power", type=float, default=3.0, help="signed_power head 幂次")
    p.add_argument("--aug", choices=["standard", "eml-aug", "strong-jitter"], default="standard")
    p.add_argument("--eml-aug-strength", type=float, default=0.35)
    p.add_argument("--eml-aug-p", type=float, default=0.8)
    p.add_argument("--jitter-strength", type=float, default=0.4)
    p.add_argument("--jitter-p", type=float, default=0.8)
    p.add_argument("--gray-p", type=float, default=0.2)
    p.add_argument("--dataset", choices=["cifar10", "cifar100", "imagenet100", "imagenet1k"], default="imagenet100")
    p.add_argument("--data-root", default="data")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=5e-2)
    p.add_argument("--hidden-dim", type=int, default=512)
    p.add_argument("--proj-dim", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out-dir", required=True)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
