"""Semi-supervised linear probe on a fraction of labels.

Same as linear_probe.py but trains on a per-class label subset (1% / 10% by
default) of the train set. The frozen encoder is unchanged; we only restrict
the supervised signal at probe time. Standard SSL eval metric (SimCLR Table 3
style, simplified to linear-only for speed).

Outputs `probe_subset_fXXXX.json` alongside the encoder ckpt, where XXXX is
the fraction × 1000 (e.g., f0010 = 1%, f0100 = 10%).
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from simclr_cifar import (
    make_cifar_resnet18, make_imagenet_resnet18,
    make_cifar_eml_resnet18, make_imagenet_eml_resnet18,
)


def eval_transform(dataset: str) -> transforms.Compose:
    if dataset in ("imagenet100", "imagenet1k"):
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616)),
    ])


def load_encoder(ckpt_path: str, dataset: str, device: torch.device) -> nn.Module:
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    head_kind = state.get("head", "mlp")
    use_eml_encoder = head_kind.startswith("p2")
    if dataset in ("imagenet100", "imagenet1k"):
        enc = make_imagenet_eml_resnet18() if use_eml_encoder else make_imagenet_resnet18()
    else:
        enc = make_cifar_eml_resnet18() if use_eml_encoder else make_cifar_resnet18()
    enc.load_state_dict(state["encoder"])
    enc.eval()
    for p in enc.parameters():
        p.requires_grad_(False)
    return enc.to(device)


def subsample_per_class(train_set, fraction: float, seed: int) -> Subset:
    """Deterministic per-class subsample. Picks ceil(fraction * count) per label."""
    rng = random.Random(seed)
    by_label: dict[int, list[int]] = {}
    targets = getattr(train_set, "targets", None)
    if targets is None:
        targets = [train_set[i][1] for i in range(len(train_set))]
    for idx, lbl in enumerate(targets):
        by_label.setdefault(int(lbl), []).append(idx)
    kept: list[int] = []
    for lbl, idxs in by_label.items():
        idxs = list(idxs)
        rng.shuffle(idxs)
        n_keep = max(1, int(round(len(idxs) * fraction)))
        kept.extend(idxs[:n_keep])
    kept.sort()
    return Subset(train_set, kept)


@torch.no_grad()
def cache_features(encoder: nn.Module, dataset, device: torch.device,
                   batch_size: int, num_workers: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Run encoder once over the dataset, return (features [N, D], labels [N]) on CPU."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
    feats, labels = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        feats.append(encoder(x).detach().cpu())
        labels.append(y)
    return torch.cat(feats), torch.cat(labels)


def probe(args: argparse.Namespace) -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    encoder = load_encoder(args.ckpt, args.dataset, device)
    tfm = eval_transform(args.dataset)

    if args.dataset == "cifar10":
        n_cls = 10
        full_train = datasets.CIFAR10(args.data_root, train=True, download=True, transform=tfm)
        test_set = datasets.CIFAR10(args.data_root, train=False, download=True, transform=tfm)
    elif args.dataset == "cifar100":
        n_cls = 100
        full_train = datasets.CIFAR100(args.data_root, train=True, download=True, transform=tfm)
        test_set = datasets.CIFAR100(args.data_root, train=False, download=True, transform=tfm)
    elif args.dataset in ("imagenet100", "imagenet1k"):
        n_cls = 1000 if args.dataset == "imagenet1k" else 100
        train_dir = Path(args.data_root) / "train"
        val_dir = Path(args.data_root) / "val"
        full_train = datasets.ImageFolder(str(train_dir), transform=tfm)
        test_set = datasets.ImageFolder(
            str(val_dir if val_dir.exists() else Path(args.data_root) / "test"),
            transform=tfm,
        )
    else:
        raise ValueError(args.dataset)

    train_subset = subsample_per_class(full_train, args.label_fraction, seed=args.subset_seed)
    print(f"[subset] kept {len(train_subset)} / {len(full_train)} "
          f"({100.0 * len(train_subset) / len(full_train):.2f}%) "
          f"@ fraction={args.label_fraction} seed={args.subset_seed}")

    # ----- 1) Cache encoder features once (frozen encoder, no train-time aug). -----
    t_cache = time.time()
    tr_feats, tr_labels = cache_features(encoder, train_subset, device,
                                         args.batch_size, args.num_workers)
    val_feats, val_labels = cache_features(encoder, test_set, device,
                                           args.batch_size, args.num_workers)
    tr_feats, tr_labels = tr_feats.to(device), tr_labels.to(device)
    val_feats, val_labels = val_feats.to(device), val_labels.to(device)
    print(f"[cache] train={tuple(tr_feats.shape)} val={tuple(val_feats.shape)} "
          f"in {time.time() - t_cache:.1f}s")

    # ----- 2) Train linear head on cached features (pure matmul, very fast). -----
    head = nn.Linear(tr_feats.shape[1], n_cls).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    crit = nn.CrossEntropyLoss()

    best_acc = 0.0
    t0 = time.time()
    n_train = tr_feats.shape[0]
    bs = args.batch_size
    tag = f"f{int(round(args.label_fraction * 1000)):04d}"
    curve_path = Path(args.ckpt).parent / f"probe_subset_{tag}_curve.jsonl"
    curve_f = curve_path.open("w")
    for ep in range(args.epochs):
        head.train()
        perm = torch.randperm(n_train, device=device)
        running_loss = 0.0
        n_batches = 0
        for i in range(0, n_train, bs):
            idx = perm[i:i + bs]
            logits = head(tr_feats[idx])
            loss = crit(logits, tr_labels[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            running_loss += loss.item()
            n_batches += 1
        sched.step()

        head.eval()
        with torch.no_grad():
            preds = head(val_feats).argmax(1)
            acc = (preds == val_labels).float().mean().item()
        best_acc = max(best_acc, acc)
        curve_f.write(json.dumps({
            "epoch": ep, "test_acc": round(acc, 6),
            "train_loss": round(running_loss / max(n_batches, 1), 6),
            "lr": opt.param_groups[0]["lr"],
            "elapsed_s": round(time.time() - t0, 2),
        }) + "\n")
        curve_f.flush()
        if ep % 10 == 0 or ep == args.epochs - 1:
            print(f"[semisup ep {ep:03d}] test_acc={acc:.4f} best={best_acc:.4f} "
                  f"elapsed={time.time() - t0:.1f}s")
    curve_f.close()

    out = Path(args.ckpt).parent / f"probe_subset_{tag}.json"
    payload = {
        "test_acc_final": round(acc, 6),
        "test_acc_best": round(best_acc, 6),
        "epochs": args.epochs,
        "label_fraction": args.label_fraction,
        "subset_seed": args.subset_seed,
        "n_train_kept": len(train_subset),
        "n_train_total": len(full_train),
        "ckpt": str(args.ckpt),
        "curve_path": str(curve_path),
    }
    out.write_text(json.dumps(payload, indent=2))
    print(f"[done] {out}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--dataset", choices=["cifar10", "cifar100", "imagenet100", "imagenet1k"], default="imagenet100")
    p.add_argument("--data-root", default="data")
    p.add_argument("--label-fraction", type=float, required=True,
                   help="Fraction of train labels to keep per class (e.g., 0.01, 0.1)")
    p.add_argument("--subset-seed", type=int, default=0,
                   help="Deterministic seed for subset sampling")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num-workers", type=int, default=4)
    return p.parse_args()


if __name__ == "__main__":
    probe(parse_args())
