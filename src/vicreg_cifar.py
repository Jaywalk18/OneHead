"""VICReg (Bardes et al. 2022 ICLR) — non-contrastive SSL via variance-invariance-covariance regularization.

Tests "EML is InfoNCE-specific" hypothesis: VICReg uses NO contrastive negatives,
NO stop-grad, NO momentum encoder — just regularization terms on the embeddings.

If EML head fails on VICReg (cond D < cond A), it confirms EML's benefit is tied
to the InfoNCE objective specifically, not to "non-contrastive SSL" in general.

Loss (per view pair):
    sim_loss = MSE(z1, z2)
    std_loss = mean(max(0, gamma - sqrt(Var(z) + eps)))  for each view
    cov_loss = sum_i!=j (cov(z)_ij^2 / d)  for each view
    total = lambda_sim * sim + mu * std + nu * cov

Default hparams from paper: lambda=25, mu=25, nu=1, gamma=1, eps=1e-4.

Usage:
    python src/vicreg_cifar.py --head eml --aug eml-aug --dataset imagenet100 \\
        --epochs 100 --batch-size 128 --lr 5e-2 --seed 0 \\
        --out-dir logs/in100_vicreg_D_eml_emlaug_s0
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


def vicreg_loss(z1, z2, sim_w=25.0, std_w=25.0, cov_w=1.0, gamma=1.0, eps=1e-4):
    """VICReg loss on a batch of paired embeddings (z1, z2) ∈ R^{N×D}."""
    N, D = z1.shape
    # Invariance (similarity): MSE
    sim_loss = F.mse_loss(z1, z2)
    # Variance: each dim should have std >= gamma
    z1c = z1 - z1.mean(dim=0, keepdim=True)
    z2c = z2 - z2.mean(dim=0, keepdim=True)
    std1 = torch.sqrt(z1c.var(dim=0) + eps)
    std2 = torch.sqrt(z2c.var(dim=0) + eps)
    std_loss = 0.5 * (F.relu(gamma - std1).mean() + F.relu(gamma - std2).mean())
    # Covariance: off-diagonal of cov matrix should be 0
    cov1 = (z1c.T @ z1c) / (N - 1)
    cov2 = (z2c.T @ z2c) / (N - 1)
    off_diag = lambda M: M.flatten()[:-1].view(M.size(0) - 1, M.size(0) + 1)[:, 1:].flatten()
    cov_loss = 0.5 * (off_diag(cov1).pow(2).sum() / D + off_diag(cov2).pow(2).sum() / D)
    return sim_w * sim_loss + std_w * std_loss + cov_w * cov_loss, sim_loss.item(), std_loss.item(), cov_loss.item()


@torch.no_grad()
def effective_rank(Z):
    """eff_rank = exp(Shannon entropy of normalized covariance eigenvalues).
    坍缩=1.0(rank-1),健康=数十。Z: N×D。对非有限值稳健(返回 1.0)。"""
    Z = Z.detach().float()
    if not torch.isfinite(Z).all():
        return 1.0
    Zc = Z - Z.mean(dim=0, keepdim=True)
    C = (Zc.T @ Zc) / max(1, Z.size(0) - 1)
    w = torch.linalg.eigvalsh(C).clamp_min(0.0)
    s = w.sum()
    if s <= 0:
        return 1.0
    p = (w / s)
    p = p[p > 1e-12]
    return float(torch.exp(-(p * torch.log(p)).sum()).item())


class VICRegModel(nn.Module):
    def __init__(self, head_kind: str, dataset: str,
                 hidden_dim: int = 2048, proj_dim: int = 2048, backbone: str = "resnet18",
                 power: float = 3.0) -> None:
        super().__init__()
        # VICReg paper uses 3-layer expander 8192-8192-8192. We use a smaller variant
        # to match parameter parity with SimCLR's 512→128 projector + EML head.
        # hidden=2048, proj=2048 is a reasonable scaled-down choice for ResNet-18 IN100.
        use_eml_encoder = head_kind.startswith("p2")
        proj_kind = head_kind.split("_", 1)[1] if head_kind.startswith("p2_") else head_kind
        self.encoder, feat_dim = build_backbone(backbone, dataset, use_eml_encoder)
        # NB: build_projection_head with proj_kind='mlp' or 'eml' from src/eml.py
        self.projection = build_projection_head(proj_kind, in_dim=feat_dim,
                                                hidden_dim=hidden_dim, out_dim=proj_dim, power=power)

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
          f"dataset={args.dataset} epochs={args.epochs} method=vicreg "
          f"sim_w={args.sim_w} std_w={args.std_w} cov_w={args.cov_w}")

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

    model = VICRegModel(head_kind=args.head, dataset=args.dataset,
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
        ep_sim = ep_std = ep_cov = 0.0
        ep_m2 = 0.0; ep_m2_n = 0; last_z = None   # 2阶矩(mean z^2)+ 末批 z(算 eff_rank)
        for (x1, x2), _ in loader:
            x1 = x1.to(device, non_blocking=True)
            x2 = x2.to(device, non_blocking=True)
            z1 = model(x1)
            z2 = model(x2)
            loss, l_sim, l_std, l_cov = vicreg_loss(
                z1, z2, sim_w=args.sim_w, std_w=args.std_w, cov_w=args.cov_w,
                gamma=args.gamma, eps=args.eps)
            # 监控:embedding 二阶矩(放大器会让它无界增长)
            with torch.no_grad():
                m2 = z1.detach().pow(2).mean()
                if torch.isfinite(m2):
                    ep_m2 += m2.item(); ep_m2_n += 1; last_z = z1.detach()
            if not torch.isfinite(loss):
                nan_steps += 1
                opt.zero_grad(set_to_none=True); continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if args.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            opt.step()
            ep_loss += loss.item() * x1.size(0)
            ep_sim += l_sim; ep_std += l_std; ep_cov += l_cov
            ep_n += x1.size(0); step += 1
            if step % log_interval == 0:
                print(f"  [ep {epoch:03d} step {step}/{steps_per_epoch}] "
                      f"loss={loss.item():.3f} sim={l_sim:.3f} std={l_std:.3f} cov={l_cov:.3f} "
                      f"{time.time()-ep_start:.0f}s", flush=True)
        sched.step()
        n_batches = max(steps_per_epoch, 1)
        avg = ep_loss / max(ep_n, 1)
        m2_mean = ep_m2 / max(ep_m2_n, 1)
        erank = effective_rank(last_z) if last_z is not None else 1.0
        record = {"epoch": epoch, "loss": avg,
                  "sim_loss": ep_sim / n_batches, "std_loss": ep_std / n_batches,
                  "cov_loss": ep_cov / n_batches,
                  "z_second_moment": m2_mean, "eff_rank": erank,
                  "lr": opt.param_groups[0]["lr"],
                  "nan_steps_total": nan_steps, "elapsed_s": time.time() - t_start}
        log_f.write(json.dumps(record) + "\n"); log_f.flush()
        print(f"[ep {epoch:03d}] loss={avg:.3f} sim={record['sim_loss']:.3f} "
              f"std={record['std_loss']:.3f} cov={record['cov_loss']:.3f} "
              f"m2={m2_mean:.3g} erank={erank:.1f} "
              f"lr={record['lr']:.2e} nan={nan_steps} ({time.time()-ep_start:.0f}s)", flush=True)

    log_f.close()
    ckpt = out / "encoder_final.pt"
    torch.save({"encoder": model.encoder.state_dict(), "head": args.head, "head_state": model.projection.state_dict(),
                "method": "vicreg", "args": vars(args)}, ckpt)
    print(f"[done] saved {ckpt} (nan_steps={nan_steps})")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--head", choices=["mlp", "eml", "p2_mlp", "signed_power", "tanh", "beml", "scalenorm", "fixedprofile"], default="mlp")
    p.add_argument("--backbone", choices=["resnet18", "vit_s"], default="resnet18",
                   help="encoder backbone;vit_s=ViT-Small/16(timm,仅224px,不支持P2)")
    p.add_argument("--power", type=float, default=3.0,
                   help="signed_power head 的幂次(剂量-响应:power=1 恒等无放大,↑则放大越强越易崩)")
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
    p.add_argument("--sim-w", type=float, default=25.0)
    p.add_argument("--std-w", type=float, default=25.0)
    p.add_argument("--cov-w", type=float, default=1.0)
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--eps", type=float, default=1e-4)
    p.add_argument("--clip-grad", type=float, default=0.0,
                   help="梯度裁剪范数(0=关);>0 防放大器 NaN 发散,让坍缩优雅可测(剂量-响应用)")
    p.add_argument("--hidden-dim", type=int, default=2048)
    p.add_argument("--proj-dim", type=int, default=2048)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out-dir", required=True)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
