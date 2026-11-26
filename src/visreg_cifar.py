"""VISReg-style (Wu, Balestriero, Levine 2026, arXiv:2606.02572) — our reimplementation.

VICReg 的 covariance 项 → Sliced-Wasserstein "sketching"(强制每个随机 1D 投影匹配标准高斯,
即匹配完整分布形状而非仅二阶矩);**保留 variance 项做 scale 控制**。
我们只取其损失作为判据里的一个【新防坍缩机制类】,不复现其 IN-22K SOTA。

判据预测:VISReg 保留了梯度活的 variance 项 → operator-compatibility 应与 VICReg 同款
(有界算子安全、放大器过阈值坍缩),不应像对比式那样免疫、也不像 Barlow 那样全崩。

用法同 vicreg_cifar.py,另有 --sketch-w / --n-proj。
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets

from simclr_cifar import TwoViewTransform, simclr_augmentation, build_backbone
from eml import build_projection_head
from vicreg_cifar import effective_rank   # 复用 eff_rank


def sliced_wasserstein_gauss(z, n_proj=64):
    """每个 z 的随机 1D 投影 vs 标准高斯的 sliced-Wasserstein-2(排序后均方差)。
    强制 z 的分布形状逼近 N(0,1) — VISReg 的 sketching 项(替代 covariance)。"""
    N, D = z.shape
    proj = torch.randn(D, n_proj, device=z.device, dtype=z.dtype)
    proj = proj / (proj.norm(dim=0, keepdim=True) + 1e-8)
    zp, _ = torch.sort(z @ proj, dim=0)                       # N×n_proj 排序
    tgt, _ = torch.sort(torch.randn(N, n_proj, device=z.device, dtype=z.dtype), dim=0)
    return ((zp - tgt) ** 2).mean()


def visreg_loss(z1, z2, sim_w=25.0, std_w=25.0, sketch_w=1.0, n_proj=64, gamma=1.0, eps=1e-4):
    sim_loss = F.mse_loss(z1, z2)
    z1c = z1 - z1.mean(0, keepdim=True); z2c = z2 - z2.mean(0, keepdim=True)
    std1 = torch.sqrt(z1c.var(0) + eps); std2 = torch.sqrt(z2c.var(0) + eps)
    std_loss = 0.5 * (F.relu(gamma - std1).mean() + F.relu(gamma - std2).mean())
    sk = 0.5 * (sliced_wasserstein_gauss(z1, n_proj) + sliced_wasserstein_gauss(z2, n_proj))
    total = sim_w * sim_loss + std_w * std_loss + sketch_w * sk
    return total, sim_loss.item(), std_loss.item(), sk.item()


class VISRegModel(nn.Module):
    def __init__(self, head_kind, dataset, hidden_dim=2048, proj_dim=2048,
                 backbone="resnet18", power=3.0):
        super().__init__()
        use_eml_encoder = head_kind.startswith("p2")
        proj_kind = head_kind.split("_", 1)[1] if head_kind.startswith("p2_") else head_kind
        self.encoder, feat_dim = build_backbone(backbone, dataset, use_eml_encoder)
        self.projection = build_projection_head(proj_kind, in_dim=feat_dim,
                                                hidden_dim=hidden_dim, out_dim=proj_dim, power=power)

    def forward(self, x):
        return self.projection(self.encoder(x))


def train(args):
    torch.manual_seed(args.seed)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[setup] device={device} head={args.head} aug={args.aug} dataset={args.dataset} "
          f"epochs={args.epochs} method=visreg sim_w={args.sim_w} std_w={args.std_w} "
          f"sketch_w={args.sketch_w} n_proj={args.n_proj} power={args.power}")

    image_size = 224 if args.dataset in ("imagenet100", "imagenet1k") else 32
    transform = TwoViewTransform(simclr_augmentation(
        image_size=image_size, dataset=args.dataset, aug=args.aug,
        eml_aug_strength=args.eml_aug_strength, eml_aug_p=args.eml_aug_p,
        jitter_strength=args.jitter_strength, jitter_p=args.jitter_p, gray_p=args.gray_p))
    if args.dataset == "cifar10":
        ds = datasets.CIFAR10(args.data_root, train=True, download=True, transform=transform)
    elif args.dataset == "cifar100":
        ds = datasets.CIFAR100(args.data_root, train=True, download=True, transform=transform)
    else:
        train_dir = Path(args.data_root) / "train"
        if not train_dir.exists(): train_dir = Path(args.data_root) / "val"
        ds = datasets.ImageFolder(str(train_dir), transform=transform)

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                        pin_memory=True, drop_last=True, persistent_workers=args.num_workers > 0)
    model = VISRegModel(args.head, args.dataset, args.hidden_dim, args.proj_dim,
                        args.backbone, args.power).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    log_f = (out / "train.jsonl").open("a")
    nan_steps = 0; step = 0; t0 = time.time()
    spe = len(ds) // args.batch_size; li = max(1, spe // 10)
    for epoch in range(args.epochs):
        model.train()
        el = en = 0.0; es = est = esk = 0.0; em2 = 0.0; em2n = 0; lastz = None; t_ep = time.time()
        for (x1, x2), _ in loader:
            x1 = x1.to(device, non_blocking=True); x2 = x2.to(device, non_blocking=True)
            z1 = model(x1); z2 = model(x2)
            loss, l_s, l_v, l_k = visreg_loss(z1, z2, args.sim_w, args.std_w, args.sketch_w,
                                              args.n_proj, args.gamma, args.eps)
            with torch.no_grad():
                m2 = z1.detach().pow(2).mean()
                if torch.isfinite(m2): em2 += m2.item(); em2n += 1; lastz = z1.detach()
            if not torch.isfinite(loss):
                nan_steps += 1; opt.zero_grad(set_to_none=True); continue
            opt.zero_grad(set_to_none=True); loss.backward()
            if args.clip_grad > 0: torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            opt.step()
            el += loss.item() * x1.size(0); es += l_s; est += l_v; esk += l_k
            en += x1.size(0); step += 1
            if step % li == 0:
                print(f"  [ep {epoch:03d} step {step}/{spe}] loss={loss.item():.3f} "
                      f"sim={l_s:.3f} std={l_v:.3f} sketch={l_k:.3f} {time.time()-t_ep:.0f}s", flush=True)
        sched.step()
        nb = max(spe, 1); avg = el / max(en, 1)
        m2m = em2 / max(em2n, 1); er = effective_rank(lastz) if lastz is not None else 1.0
        rec = {"epoch": epoch, "loss": avg, "sim_loss": es/nb, "std_loss": est/nb,
               "sketch_loss": esk/nb, "z_second_moment": m2m, "eff_rank": er,
               "lr": opt.param_groups[0]["lr"], "nan_steps_total": nan_steps,
               "elapsed_s": time.time()-t0}
        log_f.write(json.dumps(rec) + "\n"); log_f.flush()
        print(f"[ep {epoch:03d}] loss={avg:.3f} sim={rec['sim_loss']:.3f} std={rec['std_loss']:.3f} "
              f"sketch={rec['sketch_loss']:.3f} m2={m2m:.3g} erank={er:.1f} "
              f"lr={rec['lr']:.2e} nan={nan_steps} ({time.time()-t_ep:.0f}s)", flush=True)
    log_f.close()
    ckpt = out / "encoder_final.pt"
    torch.save({"encoder": model.encoder.state_dict(), "head": args.head,
                "method": "visreg", "args": vars(args)}, ckpt)
    print(f"[done] saved {ckpt} (nan_steps={nan_steps})")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--head", choices=["mlp", "eml", "p2_mlp", "signed_power", "tanh", "beml"], default="mlp")
    p.add_argument("--backbone", choices=["resnet18", "vit_s"], default="resnet18")
    p.add_argument("--power", type=float, default=3.0)
    p.add_argument("--aug", choices=["standard", "eml-aug", "strong-jitter"], default="eml-aug")
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
    p.add_argument("--sketch-w", type=float, default=1.0)
    p.add_argument("--n-proj", type=int, default=64)
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--eps", type=float, default=1e-4)
    p.add_argument("--clip-grad", type=float, default=10.0)
    p.add_argument("--hidden-dim", type=int, default=2048)
    p.add_argument("--proj-dim", type=int, default=2048)
    p.add_argument("--num-workers", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out-dir", required=True)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
