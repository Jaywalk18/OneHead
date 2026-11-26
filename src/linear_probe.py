"""Linear-probe evaluation for a frozen SSL-pretrained encoder.

Loads the encoder from a SimCLR checkpoint, freezes it, trains a linear head
on top with supervised labels, reports test accuracy.

Supports CIFAR-10, CIFAR-100, and ImageNet-100.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from simclr_cifar import (
    make_cifar_resnet18, make_imagenet_resnet18,
    make_cifar_eml_resnet18, make_imagenet_eml_resnet18,
    build_backbone,
)


def eval_transform(dataset: str = "cifar10") -> transforms.Compose:
    if dataset in ("imagenet100", "imagenet1k"):
        return transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616)),
        ]
    )


def load_encoder(ckpt_path: str, dataset: str, device: torch.device):
    """Returns (encoder, feat_dim). backbone 从 checkpoint 的 args 里读(默认 resnet18,向后兼容旧 ckpt)。"""
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    head_kind = state.get("head", "mlp")
    use_eml_encoder = head_kind.startswith("p2")
    backbone = state.get("args", {}).get("backbone", "resnet18")
    enc, feat_dim = build_backbone(backbone, dataset, use_eml_encoder)
    enc.load_state_dict(state["encoder"])
    enc.eval()
    for p in enc.parameters():
        p.requires_grad_(False)
    return enc.to(device), feat_dim


def probe(args: argparse.Namespace) -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    encoder, feat_dim = load_encoder(args.ckpt, args.dataset, device)
    tfm = eval_transform(args.dataset)

    if args.dataset == "cifar10":
        n_cls = 10
        train_set = datasets.CIFAR10(args.data_root, train=True, download=True, transform=tfm)
        test_set = datasets.CIFAR10(args.data_root, train=False, download=True, transform=tfm)
    elif args.dataset == "cifar100":
        n_cls = 100
        train_set = datasets.CIFAR100(args.data_root, train=True, download=True, transform=tfm)
        test_set = datasets.CIFAR100(args.data_root, train=False, download=True, transform=tfm)
    elif args.dataset in ("imagenet100", "imagenet1k"):
        n_cls = 1000 if args.dataset == "imagenet1k" else 100
        train_dir = Path(args.data_root) / "train"
        val_dir = Path(args.data_root) / "val"
        train_set = datasets.ImageFolder(str(train_dir), transform=tfm)
        test_set = datasets.ImageFolder(str(val_dir if val_dir.exists() else Path(args.data_root) / "test"), transform=tfm)
    else:
        raise ValueError(args.dataset)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    head = nn.Linear(feat_dim, n_cls).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    crit = nn.CrossEntropyLoss()

    t0 = time.time()
    curve_path = Path(args.ckpt).parent / "probe_curve.jsonl"
    curve_f = curve_path.open("w")
    best_acc = 0.0
    train_loss_ep = 0.0
    for ep in range(args.epochs):
        head.train()
        running_loss = 0.0
        n_batches = 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.no_grad():
                feat = encoder(x)
            logits = head(feat)
            loss = crit(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            running_loss += loss.item()
            n_batches += 1
        sched.step()
        train_loss_ep = running_loss / max(n_batches, 1)

        head.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                logits = head(encoder(x))
                correct += (logits.argmax(1) == y).sum().item()
                total += y.size(0)
        acc = correct / total
        best_acc = max(best_acc, acc)
        curve_f.write(json.dumps({
            "epoch": ep, "test_acc": round(acc, 6),
            "train_loss": round(train_loss_ep, 6),
            "lr": opt.param_groups[0]["lr"],
            "elapsed_s": round(time.time() - t0, 1),
        }) + "\n")
        curve_f.flush()
        print(f"[probe ep {ep:03d}] test_acc={acc:.4f} best={best_acc:.4f} elapsed={time.time() - t0:.1f}s")

    curve_f.close()
    out = Path(args.ckpt).parent / "probe_result.json"
    out.write_text(json.dumps({
        "test_acc": round(acc, 6),
        "test_acc_best": round(best_acc, 6),
        "epochs": args.epochs,
        "ckpt": str(args.ckpt),
        "curve_path": str(curve_path),
    }, indent=2))
    print(f"[done] {out}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--dataset", choices=["cifar10", "cifar100", "imagenet100", "imagenet1k"], default="cifar10")
    p.add_argument("--data-root", default="data")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num-workers", type=int, default=4)
    return p.parse_args()


if __name__ == "__main__":
    probe(parse_args())
