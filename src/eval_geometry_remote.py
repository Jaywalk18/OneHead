# -*- coding: utf-8 -*-
"""远程 inference-only:对 VISReg 剂量 encoder 算 alignment/uniformity 金标准(Wang-Isola 2020)
+ RankMe(Garrido 2023,label-free 表征秩)+ 真 kNN-acc(IN-100 val)。
acc 之外的几何剂量-响应:随 power↑,uniformity 退化 / alignment 漂移 / RankMe 塌 → geometry-space 复现坍缩。
用法(Zhou): CUDA_VISIBLE_DEVICES=4 CUDA_DEVICE_ORDER=PCI_BUS_ID python src/eval_geometry_remote.py \
    --enc-dir /path/to/encoders --data-root /path/to/imagenet100 \
    --n-img 2000 --out logs/geometry_eval.json"""
import os, sys, json, glob, argparse
import numpy as np
import torch, torch.nn.functional as F
from torchvision import datasets, transforms
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from simclr_cifar import build_backbone, simclr_augmentation, TwoViewTransform

def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--enc-dir", required=True); p.add_argument("--data-root", required=True)
    p.add_argument("--n-img", type=int, default=2000); p.add_argument("--out", required=True)
    p.add_argument("--knn-k", type=int, default=20)
    return p.parse_args()

@torch.no_grad()
def extract(enc, loader, dev, two=False):
    z1s, z2s, ys = [], [], []
    for batch in loader:
        if two:
            (a, b), y = batch
            z1s.append(F.normalize(enc(a.to(dev)), dim=1).cpu())
            z2s.append(F.normalize(enc(b.to(dev)), dim=1).cpu())
        else:
            x, y = batch
            z1s.append(F.normalize(enc(x.to(dev)), dim=1).cpu())
        ys.append(y)
    Z1 = torch.cat(z1s); Y = torch.cat(ys)
    return (Z1, torch.cat(z2s) if two else None, Y)

def uniformity(z, t=2.0):
    d = torch.pdist(z, p=2).pow(2)
    return float(torch.log(torch.exp(-t*d).mean()))

def alignment(z1, z2):
    return float((z1-z2).norm(dim=1).pow(2).mean())

def rankme(z, eps=1e-7):
    s = torch.linalg.svdvals(z.float())
    p = s/(s.sum()+eps) + eps
    return float(torch.exp(-(p*torch.log(p)).sum()))

def knn_acc(z, y, k=20):
    sim = z @ z.t(); sim.fill_diagonal_(-2.0)
    idx = sim.topk(k, dim=1).indices
    pred = []
    for i in range(z.size(0)):
        v, c = torch.unique(y[idx[i]], return_counts=True); pred.append(int(v[c.argmax()]))
    return float((torch.tensor(pred) == y).float().mean())

def main():
    a = parse(); dev = "cuda" if torch.cuda.is_available() else "cpu"
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    eval_tf = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),
                                  transforms.ToTensor(), norm])
    valdir = os.path.join(a.data_root, "val")
    base = datasets.ImageFolder(valdir)
    g = torch.Generator().manual_seed(0)
    idx = torch.randperm(len(base), generator=g)[:a.n_img].tolist()
    encs = sorted(glob.glob(os.path.join(a.enc_dir, "**", "*.pt"), recursive=True))
    print(f"{len(encs)} encoders, {len(idx)} imgs, dev={dev}", flush=True)
    res = {}
    for ep in encs:
        ck = torch.load(ep, map_location="cpu"); args = ck["args"]
        ad = args if isinstance(args, dict) else vars(args)
        enc, _ = build_backbone(ad["backbone"], ad["dataset"], False)
        enc.load_state_dict(ck["encoder"]); enc = enc.to(dev).eval()
        # eval-view metrics
        ds_e = datasets.ImageFolder(valdir, transform=eval_tf); ds_e.samples = [base.samples[i] for i in idx]
        le = torch.utils.data.DataLoader(ds_e, batch_size=128, num_workers=8)
        Z, _, Y = extract(enc, le, dev)
        # alignment: two SSL views
        ssl = TwoViewTransform(simclr_augmentation(224, ad["dataset"], aug=ad.get("aug", "standard"),
              eml_aug_strength=ad.get("eml_aug_strength", .35), eml_aug_p=ad.get("eml_aug_p", .8),
              jitter_strength=ad.get("jitter_strength", .4), jitter_p=ad.get("jitter_p", .8), gray_p=ad.get("gray_p", .2)))
        ds_a = datasets.ImageFolder(valdir, transform=ssl); ds_a.samples = [base.samples[i] for i in idx[:1000]]
        la = torch.utils.data.DataLoader(ds_a, batch_size=128, num_workers=8)
        Z1, Z2, _ = extract(enc, la, dev, two=True)
        m = dict(power=ad.get("power"), seed=ad.get("seed"),
                 alignment=round(alignment(Z1, Z2), 4), uniformity=round(uniformity(Z), 4),
                 rankme=round(rankme(Z), 3), knn=round(knn_acc(Z, Y, a.knn_k), 4))
        tag = os.path.splitext(os.path.basename(ep))[0]
        res[tag] = m; print(tag, m, flush=True)
    json.dump(res, open(a.out, "w"), indent=2)
    print("DONE ->", a.out, flush=True)

if __name__ == "__main__":
    main()
