"""BYOL baseline (SimSiam + momentum encoder).

Adds an EMA-updated target network to SimSiam. The hypothesis being tested:
- SimSiam (no negatives, no momentum) → D fails (loss-aligned EML can't anchor)
- BYOL (no negatives, but momentum target stabilizes) → does EML still fail?
- If BYOL also fails → confirms "contrastive negatives" are the key, not "momentum"
- If BYOL works → momentum encoder rescues EML, points to stability rather than contrast as the issue

Uses simsiam_cifar's projection + predictor architecture; adds a target encoder/projector
updated by EMA. Loss = symmetric negative cosine between online predictor and target projection.

Usage:
    python src/byol_cifar.py --head eml --aug eml-aug --dataset imagenet100 \\
        --epochs 200 --batch-size 256 --lr 5e-2 --momentum-tau 0.996 \\
        --seed 0 --out-dir logs/in100_byol_D_eml_emlaug_s0
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
from simsiam_cifar import Predictor, negative_cosine


class BYOLModel(nn.Module):
    """BYOL: online (encoder + projection + predictor) + target (encoder + projection, EMA).

    Target net is updated by exponential moving average of online net every step:
        target_w ← τ · target_w + (1 − τ) · online_w
    """

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
            # I-JEPA 式潜接口:无 projection-MLP、无 proj_bn,在 512-d 表征空间预测 EMA 目标;
            # 算子 g 插在表征→predictor 接口(headless=identity,即纯 I-JEPA)。
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
            self.proj_bn = nn.BatchNorm1d(proj_dim, affine=False)
            self.predictor = Predictor(dim=proj_dim, hidden_dim=pred_hidden)
        # Target net: deep copy of (encoder, projection, proj_bn); no predictor.
        self.target_encoder = copy.deepcopy(self.encoder)
        self.target_projection = copy.deepcopy(self.projection)
        self.target_proj_bn = copy.deepcopy(self.proj_bn)
        for p in (*self.target_encoder.parameters(),
                  *self.target_projection.parameters(),
                  *self.target_proj_bn.parameters()):
            p.requires_grad = False

    @torch.no_grad()
    def momentum_update(self, tau: float) -> None:
        for tw, ow in zip(self.target_encoder.parameters(), self.encoder.parameters()):
            tw.data.mul_(tau).add_((1.0 - tau) * ow.data)
        for tw, ow in zip(self.target_projection.parameters(), self.projection.parameters()):
            tw.data.mul_(tau).add_((1.0 - tau) * ow.data)
        for tw, ow in zip(self.target_proj_bn.parameters(), self.proj_bn.parameters()):
            tw.data.mul_(tau).add_((1.0 - tau) * ow.data)

    def forward(self, x):
        """Online: features + projection + bn + predictor. Returns p (predictor output)."""
        h = self.encoder(x)
        z = self.proj_bn(self.projection(h))
        p = self.predictor(z)
        return p

    @torch.no_grad()
    def target_forward(self, x):
        h = self.target_encoder(x)
        z = self.target_proj_bn(self.target_projection(h))
        return z


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[setup] device={device} head={args.head} aug={args.aug} "
          f"dataset={args.dataset} epochs={args.epochs} method=byol tau={args.momentum_tau}")

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

    model = BYOLModel(head_kind=args.head, dataset=args.dataset,
                      hidden_dim=args.hidden_dim, proj_dim=args.proj_dim,
                      latent_predict=args.latent_predict, power=args.power).to(device)
    # Only optimize online net (target receives EMA updates, no grad).
    opt_params = list(model.encoder.parameters()) + list(model.projection.parameters()) \
                 + list(model.predictor.parameters())
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
            # Online forward
            p1 = model(x1)
            p2 = model(x2)
            # Target forward (no grad, no momentum update yet)
            with torch.no_grad():
                z1_t = model.target_forward(x1)
                z2_t = model.target_forward(x2)
            # Symmetric loss: online predictor matches target projection of OTHER view
            loss = 0.5 * (negative_cosine(p1, z2_t) + negative_cosine(p2, z1_t))
            if not torch.isfinite(loss):
                nan_steps += 1
                opt.zero_grad(set_to_none=True); continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            model.momentum_update(args.momentum_tau)
            ep_loss += loss.item() * x1.size(0); ep_n += x1.size(0); step += 1
            if step % log_interval == 0:
                print(f"  [ep {epoch:03d} step {step}/{steps_per_epoch}] "
                      f"loss={loss.item():.4f} {time.time()-ep_start:.0f}s", flush=True)
        sched.step()
        avg = ep_loss / max(ep_n, 1)
        record = {"epoch": epoch, "loss": avg, "lr": opt.param_groups[0]["lr"],
                  "nan_steps_total": nan_steps, "elapsed_s": time.time() - t_start,
                  "tau": args.momentum_tau}
        log_f.write(json.dumps(record) + "\n"); log_f.flush()
        print(f"[ep {epoch:03d}] loss={avg:.4f} lr={record['lr']:.2e} "
              f"nan={nan_steps} ({time.time()-ep_start:.0f}s)", flush=True)

    log_f.close()
    ckpt = out / "encoder_final.pt"
    torch.save({"encoder": model.encoder.state_dict(), "head": args.head,
                "method": "byol", "momentum_tau": args.momentum_tau,
                "args": vars(args)}, ckpt)
    print(f"[done] saved {ckpt} (nan_steps={nan_steps})")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--head", choices=["mlp", "eml", "p2_mlp", "signed_power", "tanh", "beml", "headless"], default="mlp")
    p.add_argument("--latent-predict", action="store_true",
                   help="I-JEPA 式潜接口:去 projection/proj_bn,在 512-d 表征空间预测 EMA 目标;算子 g 插表征接口")
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
    p.add_argument("--momentum-tau", type=float, default=0.996,
                   help="BYOL EMA momentum for target network update")
    p.add_argument("--hidden-dim", type=int, default=512)
    p.add_argument("--proj-dim", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out-dir", required=True)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
