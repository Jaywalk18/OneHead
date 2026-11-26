"""SimCLR with togglable projection head (mlp | eml) on CIFAR-10/100/ImageNet-100.

Single-file trainer with a torchvision loader,
no Lightning/Hydra.

Usage:
    # CIFAR-10
    python src/simclr_cifar.py --head eml --dataset cifar10 --epochs 200 \\
        --batch-size 256 --lr 6e-4 --out-dir logs/simclr_eml_cifar10

    # ImageNet-100 (224x224, standard ResNet-18)
    python src/simclr_cifar.py --head eml --dataset imagenet100 --epochs 200 \\
        --batch-size 128 --lr 3e-4 --out-dir logs/simclr_eml_in100
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import datasets, transforms
from torchvision.models import resnet18

from eml import build_projection_head, EMLActivation, EMLPhotometricAugment
from ddp_utils import init_ddp, cleanup_ddp, is_main, gather_for_contrastive


# ---------- ResNet-18 variants -----------------------------------------------

def _replace_relu_with_eml(module: nn.Module) -> None:
    """Recursively replace all nn.ReLU with EMLActivation."""
    for name, child in module.named_children():
        if isinstance(child, nn.ReLU):
            setattr(module, name, EMLActivation())
        else:
            _replace_relu_with_eml(child)


def make_cifar_resnet18(feature_dim: int = 512) -> nn.Module:
    """Standard SimCLR-on-CIFAR tweak: 3x3 first conv, no maxpool, no fc."""
    m = resnet18(weights=None)
    m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    m.fc = nn.Identity()  # output feature_dim = 512
    assert feature_dim == 512, "resnet18 fixed feature_dim=512"
    return m


def make_imagenet_resnet18(feature_dim: int = 512) -> nn.Module:
    """Standard ResNet-18 for 224x224 ImageNet (7x7 conv1, maxpool, no fc)."""
    m = resnet18(weights=None)
    m.fc = nn.Identity()  # output feature_dim = 512
    assert feature_dim == 512, "resnet18 fixed feature_dim=512"
    return m


def make_cifar_eml_resnet18(feature_dim: int = 512) -> nn.Module:
    """CIFAR ResNet-18 with EML replacing all ReLU activations (P2)."""
    m = make_cifar_resnet18(feature_dim)
    _replace_relu_with_eml(m)
    return m


def make_imagenet_eml_resnet18(feature_dim: int = 512) -> nn.Module:
    """ImageNet ResNet-18 with EML replacing all ReLU activations (P2)."""
    m = make_imagenet_resnet18(feature_dim)
    _replace_relu_with_eml(m)
    return m


def make_vit_s(feature_dim: int = 384) -> nn.Module:
    """ViT-Small/16 via timm (num_classes=0 → pooled [B,384] feature vector). 224px inputs only.
    与 backbone 无关的机制(L2-norm / VICReg 方差项 / Barlow 的 BN(affine=False))全在 projector/loss 里,
    换 ViT 只替换这个 encoder —— 用于验证崩溃/救回不是 ResNet 特有(跨 BN→LN、跨残差结构)。"""
    import timm
    m = timm.create_model("vit_small_patch16_224", pretrained=False, num_classes=0)
    assert feature_dim == 384, "vit_small_patch16_224 fixed embed_dim=384"
    return m


def build_backbone(backbone: str, dataset: str, use_eml_encoder: bool):
    """Returns (encoder_module, feature_dim)。
    backbone='resnet18'(默认,与既有行为逐字节一致)或 'vit_s'(ViT-S/16,仅 224px;不支持 P2/EML-in-encoder)。"""
    if backbone == "vit_s":
        if use_eml_encoder:
            raise ValueError("ViT backbone 不支持 P2(EML-in-encoder);ViT 只跑 P3 projector heads")
        if dataset not in ("imagenet100", "imagenet1k"):
            raise ValueError("vit_s 需要 224px 输入(imagenet100/imagenet1k)")
        return make_vit_s(), 384
    if dataset in ("imagenet100", "imagenet1k"):
        enc = make_imagenet_eml_resnet18() if use_eml_encoder else make_imagenet_resnet18()
    else:
        enc = make_cifar_eml_resnet18() if use_eml_encoder else make_cifar_resnet18()
    return enc, 512


# ---------- SimCLR augmentation pipeline ------------------------------------

class TwoViewTransform:
    def __init__(self, base_transform: transforms.Compose) -> None:
        self.base = base_transform

    def __call__(self, x):
        return self.base(x), self.base(x)


def simclr_augmentation(
    image_size: int = 32,
    dataset: str = "cifar10",
    aug: str = "standard",
    eml_aug_strength: float = 0.35,
    eml_aug_p: float = 0.8,
    jitter_strength: float = 0.4,
    jitter_p: float = 0.8,
    gray_p: float = 0.2,
) -> transforms.Compose:
    """Configurable SimCLR view generator.

    aug:
      - "standard"     : ColorJitter(0.4, p=0.8) + Grayscale(p=0.2)  (default baseline)
      - "strong-jitter": ColorJitter(0.8, p=1.0) + Grayscale(p=0.2)  (codex strong-aug control)
      - "eml-aug"      : standard pipeline + guarded EML photometric op after ToTensor
                         (codex EML-Aug main line; defaults strength=0.35 p=0.8)
    Explicit --jitter-strength / --jitter-p / --gray-p flags override the preset above.
    """
    if aug == "strong-jitter":
        jitter_strength = jitter_strength if jitter_strength != 0.4 else 0.8
        jitter_p = jitter_p if jitter_p != 0.8 else 1.0
    color_jitter = transforms.ColorJitter(
        jitter_strength, jitter_strength, jitter_strength, 0.25 * jitter_strength
    )
    if dataset in ("imagenet100", "imagenet1k"):
        norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    else:
        norm = transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616))
    ops: list = [
        transforms.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([color_jitter], p=jitter_p),
        transforms.RandomGrayscale(p=gray_p),
        transforms.ToTensor(),
    ]
    if aug == "eml-aug":
        ops.append(EMLPhotometricAugment(strength=eml_aug_strength, p=eml_aug_p))
    elif aug not in ("standard", "strong-jitter"):
        raise ValueError(f"unknown --aug: {aug!r}")
    ops.append(norm)
    return transforms.Compose(ops)


# ---------- NT-Xent / InfoNCE loss ------------------------------------------

def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.5,
                 l2norm: bool = True) -> torch.Tensor:
    """Standard SimCLR loss on a batch of (z1, z2) pairs.

    l2norm: 若 False，跳过对 z 的 L2 归一化（理论预测 3 检验：抽掉 contrastive 的
        径向不变性后，预测无界放大器 head(eml/signed_power) 从全健康翻转为崩溃）。
    """
    n = z1.size(0)
    z = torch.cat([z1, z2], dim=0)  # (2n, d)
    if l2norm:
        z = F.normalize(z, dim=1)
    sim = z @ z.t() / temperature  # (2n, 2n)
    mask = torch.eye(2 * n, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(mask, float("-inf"))
    targets = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(z.device)
    return F.cross_entropy(sim, targets)


# ---------- Model wrapper ---------------------------------------------------

class SimCLRModel(nn.Module):
    def __init__(self, head_kind: str, dataset: str = "cifar10",
                 hidden_dim: int = 512, proj_dim: int = 128, backbone: str = "resnet18",
                 power: float = 3.0) -> None:
        super().__init__()
        use_eml_encoder = head_kind.startswith("p2")
        proj_kind = head_kind.split("_", 1)[1] if head_kind.startswith("p2_") else head_kind
        self.encoder, feat_dim = build_backbone(backbone, dataset, use_eml_encoder)
        self.projection = build_projection_head(proj_kind, in_dim=feat_dim, hidden_dim=hidden_dim,
                                                out_dim=proj_dim, power=power)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        return self.projection(h)


# ---------- Train loop ------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    rank, world_size, local_rank = init_ddp()
    ddp = world_size > 1

    out = Path(args.out_dir)
    if is_main():
        out.mkdir(parents=True, exist_ok=True)

    if ddp:
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    if is_main():
        print(f"[setup] device={device} head={args.head} dataset={args.dataset} "
              f"epochs={args.epochs} world={world_size} per_gpu_bs={args.batch_size}")

    if args.dataset in ("imagenet100", "imagenet1k"):
        image_size = 224
    else:
        image_size = 32
    transform = TwoViewTransform(simclr_augmentation(
        image_size=image_size,
        dataset=args.dataset,
        aug=args.aug,
        eml_aug_strength=args.eml_aug_strength,
        eml_aug_p=args.eml_aug_p,
        jitter_strength=args.jitter_strength,
        jitter_p=args.jitter_p,
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

    sampler = DistributedSampler(ds, shuffle=True, drop_last=True) if ddp else None
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )

    model = SimCLRModel(head_kind=args.head, dataset=args.dataset,
                        hidden_dim=args.hidden_dim, proj_dim=args.proj_dim,
                        backbone=args.backbone, power=args.power).to(device)
    if ddp:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    log_f = (out / "train.jsonl").open("a") if is_main() else None
    nan_steps = 0
    step = 0
    t_start = time.time()
    steps_per_epoch = len(loader)
    log_interval = max(1, steps_per_epoch // 10)

    for epoch in range(args.epochs):
        if ddp:
            sampler.set_epoch(epoch)
        model.train()
        epoch_loss = 0.0
        epoch_n = 0
        ep_start = time.time()
        for (x1, x2), _ in loader:
            x1 = x1.to(device, non_blocking=True)
            x2 = x2.to(device, non_blocking=True)
            z1 = model(x1)
            z2 = model(x2)
            z1_g = gather_for_contrastive(z1)
            z2_g = gather_for_contrastive(z2)
            loss = nt_xent_loss(z1_g, z2_g, temperature=args.temperature, l2norm=not args.no_l2norm)
            if not torch.isfinite(loss):
                nan_steps += 1
                opt.zero_grad(set_to_none=True)
                continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * x1.size(0)
            epoch_n += x1.size(0)
            step += 1
            if step % log_interval == 0 and is_main():
                elapsed = time.time() - ep_start
                print(f"  [ep {epoch:03d} step {step}/{steps_per_epoch}] "
                      f"loss={loss.item():.4f} {elapsed:.0f}s", flush=True)
        sched.step()
        avg = epoch_loss / max(epoch_n, 1)
        if is_main():
            record = {"epoch": epoch, "loss": avg, "lr": opt.param_groups[0]["lr"],
                      "nan_steps_total": nan_steps, "elapsed_s": time.time() - t_start,
                      "world_size": world_size}
            log_f.write(json.dumps(record) + "\n"); log_f.flush()
            print(f"[ep {epoch:03d}] loss={avg:.4f} lr={record['lr']:.2e} "
                  f"nan={nan_steps} ({time.time()-ep_start:.0f}s)", flush=True)

    if is_main():
        log_f.close()
        encoder_state = (model.module.encoder if ddp else model.encoder).state_dict()
        ckpt = out / "encoder_final.pt"
        torch.save({"encoder": encoder_state, "head": args.head, "args": vars(args)}, ckpt)
        print(f"[done] saved {ckpt} (nan_steps={nan_steps})")
    cleanup_ddp()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--head", choices=["mlp", "eml", "p2_mlp", "signed_power", "tanh", "beml"], default="eml")
    p.add_argument("--backbone", choices=["resnet18", "vit_s"], default="resnet18",
                   help="encoder backbone;vit_s=ViT-Small/16(timm,仅224px,不支持P2)")
    p.add_argument("--no-l2norm", action="store_true",
                   help="跳过 contrastive 损失里的 L2 归一化(理论预测3检验:抽掉径向不变性,预测放大器翻转为崩)")
    p.add_argument("--power", type=float, default=3.0,
                   help="signed_power head 的幂次(剂量-响应:对比式预测全 power 免疫,因 L2-norm 除掉尺度)")
    p.add_argument("--dataset", choices=["cifar10", "cifar100", "imagenet100", "imagenet1k"], default="cifar10")
    p.add_argument("--aug", choices=["standard", "eml-aug", "strong-jitter"], default="standard",
                   help="view-generation pipeline; eml-aug ports codex EML photometric op")
    p.add_argument("--eml-aug-strength", type=float, default=0.35)
    p.add_argument("--eml-aug-p", type=float, default=0.8)
    p.add_argument("--jitter-strength", type=float, default=0.4)
    p.add_argument("--jitter-p", type=float, default=0.8)
    p.add_argument("--gray-p", type=float, default=0.2)
    p.add_argument("--data-root", default="data")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=6e-4)
    p.add_argument("--temperature", type=float, default=0.5)
    p.add_argument("--hidden-dim", type=int, default=512)
    p.add_argument("--proj-dim", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out-dir", required=True)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
