"""Real ImageNet-C corruption k-NN eval for IN100-trained SimCLR encoders.

Wraps the codex T3 evaluator's `extract_features` + `load_encoder` primitives,
but solves the wnid-vs-English-name mismatch that left
`_filter_imagenet_c_to_in100` returning 0 classes:

  * Gallery: the standard IN100 train ImageFolder (English-named subdirs).
    `torchvision.datasets.ImageFolder` indexes classes 0..99 by sorted English
    names; that is the canonical label space for this evaluator's outputs.
  * Test (per corruption x severity): walk
    `<imagenet_c_root>/<corruption>/<severity>/<wnid>/...`, restrict to the 100
    wnids that map to IN100 English classes (via in100_classes.json), and
    label each image with its English class's index.

Output JSON schema (one file per checkpoint):
    {
      "checkpoint": "logs/in100_A_mlp_std_s0_zhou/encoder_final.pt",
      "tag": "A_mlp_std_s0",
      "imagenet_c_real": {
        "brightness": {"1": top1, "3": top1, "5": top1},
        "contrast":   {...},
        ...
      },
      "wnids_used": 100,
      "k": 200,
      "knn_chunk": 256,
      "seconds": <wall>,
      "device": "cuda:6"
    }

Usage
-----
    python scripts_t3/run_imagenet_c_eval.py \
        --checkpoint logs/in100_A_mlp_std_s0_zhou/encoder_final.pt \
        --tag A_mlp_std_s0 \
        --in100-root /path/to/imagenet100 \
        --imagenet-c-root /path/to/imagenet-c \
        --types brightness contrast gaussian_noise defocus_blur jpeg_compression \
        --severities 1 3 5 \
        --device cuda:6 \
        --out logs/imagenet_c_A_mlp_std_s0_zhou.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from PIL import Image

# Reuse codex T3 primitives.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from image_linear_probe_in100 import (  # type: ignore
    IMAGENET_MEAN,
    IMAGENET_STD,
    IMG_SIZE,
    extract_features,
    load_encoder,
)


def eval_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class WnidImageFolder(Dataset):
    """ImageNet-C subset where each wnid dir maps back to an IN100 English label idx.

    `class_dirs`: list of (wnid_path, label_idx) tuples already filtered to
    intersect with the IN100 wnid set.
    """

    def __init__(self, class_dirs: list[tuple[Path, int]], transform):
        self.transform = transform
        self.samples: list[tuple[str, int]] = []
        for cls_dir, idx in class_dirs:
            for img in sorted(cls_dir.iterdir()):
                if img.suffix.lower() in (".jpeg", ".jpg", ".png", ".bmp"):
                    self.samples.append((str(img), idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        path, label = self.samples[i]
        with Image.open(path) as img:
            img = img.convert("RGB")
            return self.transform(img), label


def knn_top1(test_feat: torch.Tensor, test_labels: torch.Tensor,
             gallery_feat: torch.Tensor, gallery_labels: torch.Tensor,
             k: int, chunk: int) -> float:
    correct = 0
    total = 0
    for start in range(0, test_feat.shape[0], chunk):
        feat = test_feat[start:start + chunk]
        sim = feat @ gallery_feat.T
        idx = sim.topk(k=min(k, gallery_feat.shape[0]), dim=1).indices
        votes = gallery_labels[idx]
        pred = torch.mode(votes, dim=1).values
        target = test_labels[start:start + chunk]
        correct += (pred == target).sum().item()
        total += target.numel()
    return correct / max(1, total)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--tag", type=str, required=True,
                   help="short label embedded in JSON, e.g. A_mlp_std_s0")
    p.add_argument("--in100-root", type=Path,
                   required=True)
    p.add_argument("--imagenet-c-root", type=Path,
                   required=True)
    p.add_argument("--class-map", type=Path,
                   default=HERE / "in100_classes.json")
    p.add_argument("--types", nargs="+",
                   default=["brightness", "contrast", "gaussian_noise",
                            "defocus_blur", "jpeg_compression"])
    p.add_argument("--severities", nargs="+", type=int, default=[1, 3, 5])
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--knn-k", type=int, default=200)
    p.add_argument("--knn-chunk", type=int, default=256)
    p.add_argument("--device", type=str, default="cuda:6")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--force-p2", action="store_true",
                   help="Force EML-in-encoder regardless of ckpt 'head' field (for cond D)")
    p.add_argument("--skip-existing", action="store_true",
                   help="If --out already exists with all requested types, skip eval.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.skip_existing and args.out.is_file():
        try:
            existing = json.loads(args.out.read_text(encoding="utf-8"))
            have = set(existing.get("imagenet_c_real", {}).keys())
            need = set(args.types)
            if need.issubset(have):
                print(f"[ic-eval] {args.out} already covers requested types {sorted(need)}; skipping.")
                return
        except Exception:
            pass

    device = torch.device(args.device)
    print(f"[ic-eval] device={device}")

    # 1. Load class mapping.
    if not args.class_map.is_file():
        raise SystemExit(f"[ic-eval] missing class map {args.class_map}; "
                         f"run scripts_t3/imagenet100_class_map.py first.")
    cmap = json.loads(args.class_map.read_text(encoding="utf-8"))
    by_english: dict[str, str] = cmap["by_english"]
    by_wnid: dict[str, str] = cmap["by_wnid"]

    # 2. Build gallery (IN100 train, English-named) and capture canonical label idx mapping.
    train_dir = args.in100_root / "train"
    if not train_dir.is_dir():
        raise SystemExit(f"[ic-eval] missing IN100 train dir {train_dir}")
    tf = eval_transform()
    gallery_ds = datasets.ImageFolder(str(train_dir), transform=tf)
    english_to_idx: dict[str, int] = gallery_ds.class_to_idx  # sorted-English -> 0..99
    print(f"[ic-eval] gallery classes: {len(english_to_idx)}")

    # Build the wnid -> English-class-idx lookup the test loader will use.
    wnid_to_idx: dict[str, int] = {}
    for english, wnid in by_english.items():
        if english in english_to_idx:
            wnid_to_idx[wnid] = english_to_idx[english]
    print(f"[ic-eval] wnid -> idx coverage: {len(wnid_to_idx)} / {len(by_english)}")

    # 3. Load encoder + extract gallery features.
    t0 = time.perf_counter()
    encoder = load_encoder(args.checkpoint, device,
                           force_p2=True if args.force_p2 else None)

    gallery_loader = DataLoader(
        gallery_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,
    )
    print(f"[ic-eval] extracting gallery features ({len(gallery_ds)} imgs)...")
    g_feat, g_labels = extract_features(encoder, gallery_loader, device)
    g_feat = F.normalize(g_feat, dim=1).to(device)
    g_labels = g_labels.to(device)
    print(f"[ic-eval] gallery features ready: shape={tuple(g_feat.shape)}")

    # 4. Loop over corruptions x severities.
    if args.out.is_file():
        try:
            results_root = json.loads(args.out.read_text(encoding="utf-8"))
        except Exception:
            results_root = {}
    else:
        results_root = {}
    results: dict = results_root.get("imagenet_c_real", {})

    for ctype in args.types:
        ctype_dir = args.imagenet_c_root / ctype
        if not ctype_dir.is_dir():
            print(f"[ic-eval] missing corruption dir {ctype_dir}; skipping")
            continue
        results.setdefault(ctype, {})
        for sev in args.severities:
            sev_dir = ctype_dir / str(sev)
            if not sev_dir.is_dir():
                print(f"[ic-eval] missing {sev_dir}; skipping")
                results[ctype][str(sev)] = None
                continue
            class_dirs: list[tuple[Path, int]] = []
            for wnid, idx in wnid_to_idx.items():
                cand = sev_dir / wnid
                if cand.is_dir():
                    class_dirs.append((cand, idx))
            if not class_dirs:
                print(f"[ic-eval] no IN100 wnids under {sev_dir}; skipping")
                results[ctype][str(sev)] = None
                continue
            ds = WnidImageFolder(class_dirs, tf)
            if len(ds) == 0:
                print(f"[ic-eval] {ctype} sev={sev}: 0 images")
                results[ctype][str(sev)] = None
                continue
            loader = DataLoader(
                ds, batch_size=args.batch_size, shuffle=False,
                num_workers=args.workers, pin_memory=True,
            )
            t_feat, t_labels = extract_features(encoder, loader, device)
            t_feat = F.normalize(t_feat, dim=1).to(device)
            t_labels = t_labels.to(device)
            top1 = knn_top1(t_feat, t_labels, g_feat, g_labels,
                            k=args.knn_k, chunk=args.knn_chunk)
            results[ctype][str(sev)] = round(top1, 6)
            print(f"[ic-eval] {ctype} sev={sev} top1={top1:.4f} (n={len(ds)}, "
                  f"wnids={len(class_dirs)})")

    out_json = {
        "checkpoint": str(args.checkpoint),
        "tag": args.tag,
        "imagenet_c_real": results,
        "wnids_used": len(wnid_to_idx),
        "k": args.knn_k,
        "knn_chunk": args.knn_chunk,
        "seconds": round(time.perf_counter() - t0, 2),
        "device": str(device),
        "in100_root": str(args.in100_root),
        "imagenet_c_root": str(args.imagenet_c_root),
        "types": args.types,
        "severities": args.severities,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out_json, indent=2), encoding="utf-8")
    print(f"\n[ic-eval] wrote {args.out}  ({out_json['seconds']:.1f}s)")


if __name__ == "__main__":
    main()
