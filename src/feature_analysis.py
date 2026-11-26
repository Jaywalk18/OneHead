"""Feature-space analysis for frozen SSL encoders.

Computes interpretable metrics on the 512-d ResNet-18 representations to
look beyond linear-probe top-1. The goal is to surface "what EML does to
the feature space" — alignment / uniformity / effective rank / class
scatter / augmentation invariance — so the paper can build mechanism
claims, not just accuracy claims.

Metrics:
  - alignment      : E[||f(x_a) - f(x_b)||^2]  over paired augmented views (smaller = better)
  - uniformity     : log E[exp(-2||f(x) - f(y)||^2)]  over random pairs (smaller = more uniform on sphere)
  - eff_rank       : exp(Shannon entropy of normalized singular values) of feature matrix
  - class_scatter  : tr(S_b) / tr(S_w)  Fisher-style class separability
  - aug_invariance : variance of feature across multiple photometric augs of same image

Writes feature_metrics.json next to the ckpt.

Usage:
    python src/feature_analysis.py --ckpt logs/in100_*/encoder_final.pt \\
        --dataset imagenet100 --data-root /path/to/in100 \\
        --num-pairs 5000 --num-samples 10000
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

# Limit CPU thread usage to avoid load-avg blow-up when running multiple instances.
# Each feature_analysis pass does big matmul / SVD on CPU; default OpenMP would
# grab all cores. Setting BEFORE torch import so it propagates to BLAS / OMP.
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import torch
torch.set_num_threads(4)
torch.set_num_interop_threads(2)
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from simclr_cifar import (
    make_cifar_resnet18, make_imagenet_resnet18,
    make_cifar_eml_resnet18, make_imagenet_eml_resnet18,
)


def eval_transform(dataset: str):
    if dataset in ("imagenet100", "imagenet1k"):
        return transforms.Compose([
            transforms.Resize(256), transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616)),
    ])


def photo_aug_transform(dataset: str):
    """SimCLR-style photometric aug for augmentation-invariance probing."""
    if dataset in ("imagenet100", "imagenet1k"):
        norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        size = 224
    else:
        norm = transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616))
        size = 32
    return transforms.Compose([
        transforms.Resize(256) if size == 224 else transforms.Lambda(lambda x: x),
        transforms.CenterCrop(size),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        norm,
    ])


def load_encoder(ckpt_path: str, dataset: str, device) -> nn.Module:
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    head_kind = state.get("head", "mlp")
    use_eml_encoder = head_kind.startswith("p2")
    if dataset in ("imagenet100", "imagenet1k"):
        enc = make_imagenet_eml_resnet18() if use_eml_encoder else make_imagenet_resnet18()
    else:
        enc = make_cifar_eml_resnet18() if use_eml_encoder else make_cifar_resnet18()
    enc.load_state_dict(state["encoder"])
    enc.eval()
    for p in enc.parameters(): p.requires_grad_(False)
    return enc.to(device)


@torch.no_grad()
def cache_features(encoder, dataset, device, batch_size=128, num_workers=4):
    """Run encoder over dataset once. Returns (features [N, D], labels [N])."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
    feats, labels = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        feats.append(encoder(x).cpu())
        labels.append(y)
    return torch.cat(feats), torch.cat(labels)


def metric_uniformity(feats: torch.Tensor, t: float = 2.0, max_pairs: int = 1000000) -> float:
    """log E[exp(-t * ||f(x) - f(y)||^2)] over random pairs. Smaller = more uniform."""
    f = F.normalize(feats, dim=1)
    N = f.size(0)
    # Use random pairs if N^2 is too large
    if N * N > max_pairs:
        n = int(math.sqrt(max_pairs))
        idx = torch.randperm(N)[:n]
        f = f[idx]
        N = n
    # pairwise sq dist via 2 - 2*cos
    sim = f @ f.T
    dist_sq = 2.0 - 2.0 * sim
    # Exclude self-pairs
    mask = ~torch.eye(N, dtype=torch.bool, device=f.device)
    return float(torch.log(torch.exp(-t * dist_sq[mask]).mean() + 1e-12))


def metric_alignment(encoder, dataset, device, n_pairs: int = 1000,
                     base_dataset_for_imgs=None, aug=None) -> float:
    """E[||f(x_a) - f(x_b)||^2] where x_a, x_b are two augmentations of same image.

    Uses base_dataset_for_imgs (no transform) to get raw PIL images, applies aug 2x.
    """
    f = aug
    rng = random.Random(0)
    idxs = rng.sample(range(len(base_dataset_for_imgs)), min(n_pairs, len(base_dataset_for_imgs)))
    feats_a, feats_b = [], []
    with torch.no_grad():
        for batch_start in range(0, len(idxs), 64):
            batch_idxs = idxs[batch_start:batch_start + 64]
            imgs_a = torch.stack([f(base_dataset_for_imgs[i][0]) for i in batch_idxs]).to(device)
            imgs_b = torch.stack([f(base_dataset_for_imgs[i][0]) for i in batch_idxs]).to(device)
            za = F.normalize(encoder(imgs_a), dim=1)
            zb = F.normalize(encoder(imgs_b), dim=1)
            feats_a.append(za.cpu()); feats_b.append(zb.cpu())
    fa = torch.cat(feats_a); fb = torch.cat(feats_b)
    return float(((fa - fb) ** 2).sum(dim=1).mean())


def metric_eff_rank(feats: torch.Tensor) -> float:
    """exp(H(sigma_normalized)) — effective rank a la Roy & Vetterli 2007.
    Higher = more spread across dimensions.
    """
    f = feats - feats.mean(dim=0, keepdim=True)
    # SVD on potentially large matrix → use torch.linalg.svdvals
    sv = torch.linalg.svdvals(f)
    p = sv / (sv.sum() + 1e-12)
    H = -(p * torch.log(p + 1e-12)).sum()
    return float(torch.exp(H))


def metric_class_scatter(feats: torch.Tensor, labels: torch.Tensor) -> dict:
    """Fisher discriminant ratio components:
    S_w = sum_c sum_{x in c} (x - mu_c)(x - mu_c)^T
    S_b = sum_c n_c (mu_c - mu)(mu_c - mu)^T
    Returns tr(S_b), tr(S_w), ratio.
    """
    classes = labels.unique()
    n_total = labels.numel()
    mu_global = feats.mean(dim=0)
    tr_Sw = 0.0; tr_Sb = 0.0
    for c in classes:
        mask = labels == c
        if mask.sum() < 2: continue
        feats_c = feats[mask]
        mu_c = feats_c.mean(dim=0)
        diffs_w = feats_c - mu_c
        tr_Sw += float((diffs_w ** 2).sum())
        diff_b = mu_c - mu_global
        tr_Sb += float(mask.sum()) * float((diff_b ** 2).sum())
    return {"tr_Sb": tr_Sb, "tr_Sw": tr_Sw, "fisher_ratio": tr_Sb / (tr_Sw + 1e-12)}


def metric_aug_invariance(encoder, base_dataset, device, n_imgs: int = 500, n_augs: int = 5,
                          aug=None) -> float:
    """Variance of feature representation across multiple photometric augs of same image.
    Lower = more invariant.
    """
    rng = random.Random(1)
    idxs = rng.sample(range(len(base_dataset)), min(n_imgs, len(base_dataset)))
    per_img_var = []
    with torch.no_grad():
        for i in idxs:
            img_raw = base_dataset[i][0]
            views = torch.stack([aug(img_raw) for _ in range(n_augs)]).to(device)
            feats = F.normalize(encoder(views), dim=1)
            # per-dim variance, then sum (= total variance)
            v = feats.var(dim=0).sum()
            per_img_var.append(float(v))
    return float(sum(per_img_var) / len(per_img_var))


def analyze(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[analyze] {args.ckpt} device={device}")
    encoder = load_encoder(args.ckpt, args.dataset, device)

    tfm = eval_transform(args.dataset)
    aug = photo_aug_transform(args.dataset)
    if args.dataset == "cifar10":
        val_set = datasets.CIFAR10(args.data_root, train=False, transform=tfm, download=True)
        train_set = datasets.CIFAR10(args.data_root, train=True, transform=tfm, download=True)
        raw_set = datasets.CIFAR10(args.data_root, train=True, transform=None, download=True)
    elif args.dataset == "cifar100":
        val_set = datasets.CIFAR100(args.data_root, train=False, transform=tfm, download=True)
        train_set = datasets.CIFAR100(args.data_root, train=True, transform=tfm, download=True)
        raw_set = datasets.CIFAR100(args.data_root, train=True, transform=None, download=True)
    elif args.dataset in ("imagenet100", "imagenet1k"):
        val_dir = Path(args.data_root) / "val"
        train_dir = Path(args.data_root) / "train"
        val_set = datasets.ImageFolder(str(val_dir), transform=tfm)
        train_set = datasets.ImageFolder(str(train_dir), transform=tfm)
        raw_set = datasets.ImageFolder(str(train_dir), transform=None)
    else:
        raise ValueError(args.dataset)

    # Subset train for speed
    if args.num_samples and args.num_samples < len(train_set):
        rng = random.Random(42)
        idx = rng.sample(range(len(train_set)), args.num_samples)
        train_subset = Subset(train_set, idx)
    else:
        train_subset = train_set

    t0 = time.time()
    print(f"[cache] val: {len(val_set)}")
    val_feats, val_labels = cache_features(encoder, val_set, device, args.batch_size, args.num_workers)
    print(f"[cache] train: {len(train_subset)}")
    train_feats, train_labels = cache_features(encoder, train_subset, device, args.batch_size, args.num_workers)
    print(f"[cache] done in {time.time() - t0:.1f}s")

    metrics = {"ckpt": str(args.ckpt)}
    metrics["uniformity_val"] = metric_uniformity(val_feats)
    metrics["uniformity_train"] = metric_uniformity(train_feats)
    metrics["eff_rank_val"] = metric_eff_rank(val_feats)
    metrics["eff_rank_train"] = metric_eff_rank(train_feats)
    metrics["class_scatter_val"] = metric_class_scatter(val_feats, val_labels)
    metrics["class_scatter_train"] = metric_class_scatter(train_feats, train_labels)
    print(f"[metrics so far] uniform_val={metrics['uniformity_val']:.4f} "
          f"eff_rank_val={metrics['eff_rank_val']:.2f} "
          f"fisher_val={metrics['class_scatter_val']['fisher_ratio']:.3f}")

    print(f"[alignment] sampling {args.num_pairs} image pairs ...")
    metrics["alignment"] = metric_alignment(
        encoder, train_set, device, n_pairs=args.num_pairs,
        base_dataset_for_imgs=raw_set, aug=aug,
    )
    print(f"[aug_inv] sampling {args.num_inv_imgs} imgs × {args.num_augs} augs ...")
    metrics["aug_invariance"] = metric_aug_invariance(
        encoder, raw_set, device,
        n_imgs=args.num_inv_imgs, n_augs=args.num_augs, aug=aug,
    )
    print(f"[done] alignment={metrics['alignment']:.4f} aug_inv={metrics['aug_invariance']:.4f}")

    out = Path(args.ckpt).parent / "feature_metrics.json"
    out.write_text(json.dumps(metrics, indent=2))
    print(f"[saved] {out}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--dataset", choices=["cifar10", "cifar100", "imagenet100", "imagenet1k"], default="imagenet100")
    p.add_argument("--data-root", default="data")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--num-samples", type=int, default=10000, help="train subset for class scatter")
    p.add_argument("--num-pairs", type=int, default=1000, help="alignment image pairs")
    p.add_argument("--num-inv-imgs", type=int, default=500, help="aug-invariance image count")
    p.add_argument("--num-augs", type=int, default=5, help="augs per image for invariance")
    return p.parse_args()


if __name__ == "__main__":
    analyze(parse_args())
