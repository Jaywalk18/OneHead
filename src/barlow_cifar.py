"""Barlow Twins (Zbontar et al. 2021 ICML) — non-contrastive SSL via redundancy reduction.

Tests "EML is InfoNCE-specific" hypothesis: Barlow Twins uses cross-correlation
matrix loss, NO contrastive negatives, NO stop-grad, NO momentum encoder.

Loss: empirical cross-correlation matrix C between two view embeddings (after BN),
    L = sum_i (1 - C_ii)^2 + lambda * sum_i!=j C_ij^2
where C_ij = sum_b (z1_bi * z2_bj) / (N * std(z1_i) * std(z2_j))

Default: lambda=5e-3 (paper IN1k value), proj_dim=8192 (paper) — we scale down
to 2048 for ResNet-18 IN100 setup parity with VICReg.

Usage:
    python src/barlow_cifar.py --head eml --aug eml-aug --dataset imagenet100 \\
        --epochs 100 --batch-size 128 --lr 5e-2 --seed 0 \\
        --out-dir logs/in100_barlow_D_eml_emlaug_s0
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
    build_backbone,
)
from eml import build_projection_head


def off_diagonal(M):
    """Return off-diagonal elements of a square matrix flattened."""
    n, m = M.shape
    assert n == m
    return M.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def barlow_loss(z1, z2, bn, lambd=5e-3, var_w=0.0, use_bn=True):
    """Barlow Twins loss.
    z1, z2: [N, D] embeddings
    bn: BatchNorm1d(D, affine=False) — normalize before correlation
    lambd: off-diagonal redundancy reduction weight
    var_w: 若 > 0，叠加 VICReg 式逐维方差 hinge（作用在 BN 前的 raw z 上）。
        这是 unified-theory 预测 1 的检验项：给 Barlow 装一个"活的（梯度非零）尺度恢复项"，
        预测能把有界 head(tanh/beml) 从退化救回。默认 0 = 原始 Barlow。
    """
    N = z1.size(0)
    if use_bn:
        z1n = bn(z1)
        z2n = bn(z2)
    else:
        # 不做 BN:用 raw projector 输出算 cross-correlation。理论检验——BN(affine=False) 在 forward 里
        # 把 std 抹平到 1(梯度死),正是 Barlow 救不回放大器的疑似病根;去掉 BN 后,对角项 (C_ii-1)²
        # 变成作用在 raw 二阶矩上的【梯度活】约束(等价于 VICReg 的活方差项),预测能救回有界/放大算子。
        z1n = z1
        z2n = z2
    C = (z1n.T @ z2n) / N  # [D, D] cross-correlation
    on_diag = (torch.diagonal(C) - 1.0).pow(2).sum()
    off_diag = off_diagonal(C).pow(2).sum()
    loss = on_diag + lambd * off_diag
    if var_w > 0.0:
        std1 = torch.sqrt(z1.var(dim=0) + 1e-4)
        std2 = torch.sqrt(z2.var(dim=0) + 1e-4)
        var_loss = torch.relu(1.0 - std1).mean() + torch.relu(1.0 - std2).mean()
        loss = loss + var_w * var_loss
    return loss, on_diag.item(), off_diag.item()


class BarlowModel(nn.Module):
    def __init__(self, head_kind: str, dataset: str,
                 hidden_dim: int = 2048, proj_dim: int = 2048, backbone: str = "resnet18",
                 power: float = 3.0) -> None:
        super().__init__()
        use_eml_encoder = head_kind.startswith("p2")
        proj_kind = head_kind.split("_", 1)[1] if head_kind.startswith("p2_") else head_kind
        self.encoder, feat_dim = build_backbone(backbone, dataset, use_eml_encoder)
        self.projection = build_projection_head(proj_kind, in_dim=feat_dim,
                                                hidden_dim=hidden_dim, out_dim=proj_dim, power=power)
        # BatchNorm without affine — for normalizing before cross-correlation
        self.bn = nn.BatchNorm1d(proj_dim, affine=False)

    def forward(self, x):
        h = self.encoder(x)
        z = self.projection(h)
        return z


def train(args):
    torch.manual_seed(args.seed)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[setup] device={device} head={args.head} aug={args.aug} "
          f"dataset={args.dataset} epochs={args.epochs} method=barlow "
          f"lambda={args.lambd} proj_dim={args.proj_dim}")

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

    model = BarlowModel(head_kind=args.head, dataset=args.dataset,
                        hidden_dim=args.hidden_dim, proj_dim=args.proj_dim,
                        backbone=args.backbone, power=args.power).to(device)
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
        ep_on = ep_off = 0.0
        for (x1, x2), _ in loader:
            x1 = x1.to(device, non_blocking=True)
            x2 = x2.to(device, non_blocking=True)
            z1 = model(x1)
            z2 = model(x2)
            loss, l_on, l_off = barlow_loss(z1, z2, model.bn, lambd=args.lambd, var_w=args.var_w,
                                            use_bn=not args.no_bn)
            if not torch.isfinite(loss):
                nan_steps += 1
                opt.zero_grad(set_to_none=True); continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            ep_loss += loss.item() * x1.size(0)
            ep_on += l_on; ep_off += l_off
            ep_n += x1.size(0); step += 1
            if step % log_interval == 0:
                print(f"  [ep {epoch:03d} step {step}/{steps_per_epoch}] "
                      f"loss={loss.item():.2f} on_diag={l_on:.2f} off_diag={l_off:.0f} "
                      f"{time.time()-ep_start:.0f}s", flush=True)
        sched.step()
        n_batches = max(steps_per_epoch, 1)
        avg = ep_loss / max(ep_n, 1)
        record = {"epoch": epoch, "loss": avg,
                  "on_diag": ep_on / n_batches, "off_diag": ep_off / n_batches,
                  "lr": opt.param_groups[0]["lr"],
                  "nan_steps_total": nan_steps, "elapsed_s": time.time() - t_start}
        log_f.write(json.dumps(record) + "\n"); log_f.flush()
        print(f"[ep {epoch:03d}] loss={avg:.2f} on={record['on_diag']:.2f} "
              f"off={record['off_diag']:.0f} lr={record['lr']:.2e} "
              f"nan={nan_steps} ({time.time()-ep_start:.0f}s)", flush=True)

    log_f.close()
    ckpt = out / "encoder_final.pt"
    torch.save({"encoder": model.encoder.state_dict(), "head": args.head,
                "method": "barlow", "args": vars(args)}, ckpt)
    print(f"[done] saved {ckpt} (nan_steps={nan_steps})")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--head", choices=["mlp", "eml", "p2_mlp", "signed_power", "tanh", "beml"], default="mlp")
    p.add_argument("--power", type=float, default=3.0,
                   help="signed_power head 的幂次(剂量-响应:Barlow BN 梯度死无救援,预测各 power 都崩)")
    p.add_argument("--no-bn", action="store_true",
                   help="去掉 Barlow 的 BN(affine=False),用 raw z 算 cross-correlation。理论检验:"
                        "对角项变 raw 二阶矩的梯度活约束,预测能救回(定位 pred1b 失败真因=那颗梯度死的 BN)")
    p.add_argument("--backbone", choices=["resnet18", "vit_s"], default="resnet18",
                   help="encoder backbone;vit_s=ViT-Small/16(timm,仅224px,不支持P2)")
    p.add_argument("--var-w", type=float, default=0.0,
                   help="VICReg式逐维方差hinge权重(作用在BN前raw z);>0给Barlow装活尺度项(理论预测1检验)")
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
    p.add_argument("--lambd", type=float, default=5e-3)
    p.add_argument("--hidden-dim", type=int, default=2048)
    p.add_argument("--proj-dim", type=int, default=2048)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out-dir", required=True)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
