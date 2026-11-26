"""Downstream transfer-learning probe for IN100-trained SimCLR encoders.

Evaluates a frozen encoder on a downstream classification dataset by running
both a linear probe (Adam, ~40 epochs on frozen features) and a cosine k-NN
probe (k=200) and writing a single JSON summary file.

Supported downstream datasets:

    food101  -> ~/datasets/food-101/{meta,images}/
    pets     -> ~/datasets/OpenDataLab___Oxford-IIIT_Pets/raw/{images,annotations}/
    flowers  -> ~/datasets/OpenDataLab___Oxford_102_Flower/raw/{jpg,imagelabels.mat,setid.mat}
    dtd      -> ~/datasets/OpenDataLab___DTD/raw/dtd/{images,labels}/   (partition 1)

NOT YET SUPPORTED (v1):
    cars     -> missing cars_test_annos_withlabels.mat in the OpenDataLab tarball
                (only cars_train_annos.mat is available; without test labels we
                cannot reproduce the standard 196-class test top-1).

Output JSON schema (one file per (checkpoint, dataset) pair):
    {
      "checkpoint": <path>, "tag": <run_id>,
      "dataset": "food101"|...,
      "n_train": int, "n_test": int, "n_classes": int,
      "linear_top1": float, "knn_top1": float,
      "epochs": int, "batch_size": int, "lr": float, "knn_k": int,
      "seconds": float, "device": str
    }

Usage example
-------------

    ~/.conda/envs/solo/bin/python scripts_t3/run_transfer_probe.py \
        --checkpoint logs/in100_A_mlp_std_s0/encoder_final.pt \
        --tag A_mlp_std_s0 \
        --dataset food101 \
        --data-root ~/datasets/food-101 \
        --out logs/transfer_food101_A_mlp_std_s0.json \
        --epochs 40 --batch-size 256 --lr 1e-3 --num-workers 8 \
        --device cuda:1 --knn-k 200 --skip-existing
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import resnet18


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
FEATURE_DIM = 512
IMG_SIZE = 224


# ---------- encoder loading (replicates codex T3 load_encoder logic) --------

def _make_in100_encoder(use_eml: bool) -> nn.Module:
    """Standard 224x224 ResNet-18, fc=Identity. Mirrors
    `make_imagenet_resnet18` in `src/simclr_cifar.py`.
    `use_eml=True` is the P2 (EML-in-encoder) variant."""
    m = resnet18(weights=None)
    m.fc = nn.Identity()
    if use_eml:
        try:
            from eml import EMLActivation  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "Checkpoint advertises P2 (EML-in-encoder) but `eml.EMLActivation` "
                "is not importable. Add the repository's src directory to "
                "PYTHONPATH or pass --no-p2."
            ) from exc

        def _replace(module: nn.Module) -> None:
            for name, child in module.named_children():
                if isinstance(child, nn.ReLU):
                    setattr(module, name, EMLActivation())
                else:
                    _replace(child)
        _replace(m)
    return m


def load_encoder(path: Path, device: torch.device,
                 force_p2: Optional[bool] = None) -> nn.Module:
    """Load an IN100 SimCLR encoder. Tolerant to several payload shapes:
      1. dict with key "encoder" + "head" (`src/simclr_cifar.py` format)
      2. dict with key "encoder_state_dict" (codex CIFAR trainer)
      3. bare state_dict
    """
    payload = torch.load(path, map_location=device)
    use_eml = False
    state: dict
    if isinstance(payload, dict):
        if "encoder" in payload and isinstance(payload["encoder"], dict):
            state = payload["encoder"]
            head_kind = str(payload.get("head", ""))
            use_eml = head_kind.startswith("p2")
        elif "encoder_state_dict" in payload:
            state = payload["encoder_state_dict"]
        else:
            state = payload
    else:
        raise RuntimeError(f"unrecognized checkpoint payload type: {type(payload)!r}")

    if force_p2 is not None:
        use_eml = force_p2

    encoder = _make_in100_encoder(use_eml=use_eml).to(device)
    missing, unexpected = encoder.load_state_dict(state, strict=False)
    leftover_missing = [k for k in missing if not k.startswith("fc.")]
    leftover_unexpected = [k for k in unexpected if not k.startswith("fc.")]
    if leftover_missing or leftover_unexpected:
        print(f"[load_encoder] non-fc key mismatch missing={leftover_missing[:3]} "
              f"unexpected={leftover_unexpected[:3]}")
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    return encoder


# ---------- transforms -------------------------------------------------------

def train_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def eval_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


# ---------- generic file-list dataset ---------------------------------------

class FileListDataset(Dataset):
    """Lightweight (path, label) list dataset. Avoids ImageFolder's class
    enumeration constraints — we already know the (path, int_label) mapping
    from each downstream dataset's split files."""

    def __init__(self, samples: list[tuple[str, int]], transform: Callable):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        path, label = self.samples[i]
        with Image.open(path) as img:
            img = img.convert("RGB")
            return self.transform(img), label


# ---------- dataset adapters ------------------------------------------------

def build_food101(data_root: Path) -> tuple[list, list, int]:
    """food-101 root layout (DLU): <root>/images/<class>/<id>.jpg and
    <root>/meta/{train,test,classes}.txt."""
    root = Path(data_root).expanduser()
    meta = root / "meta"
    images = root / "images"
    if not meta.is_dir() or not images.is_dir():
        raise FileNotFoundError(f"food-101 expects {root}/{{meta,images}}; got {root}")
    classes = [c.strip() for c in (meta / "classes.txt").read_text(
        encoding="utf-8").splitlines() if c.strip()]
    cls_to_idx = {c: i for i, c in enumerate(classes)}

    def _load(split: str) -> list[tuple[str, int]]:
        out = []
        for line in (meta / f"{split}.txt").read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            cls, stem = line.split("/")
            out.append((str(images / cls / f"{stem}.jpg"), cls_to_idx[cls]))
        return out

    return _load("train"), _load("test"), len(classes)


def build_pets(data_root: Path) -> tuple[list, list, int]:
    """Oxford-IIIT Pets (DLU): <root>/{images,annotations}/.
    annotations/trainval.txt + annotations/test.txt give `<stem> <cid_1based>
    <species> <breed_id>` per line. 37 classes."""
    root = Path(data_root).expanduser()
    images = root / "images"
    annos = root / "annotations"
    if not images.is_dir() or not annos.is_dir():
        raise FileNotFoundError(f"pets expects {root}/{{images,annotations}}; got {root}")

    def _load(split_file: str) -> list[tuple[str, int]]:
        out = []
        for line in (annos / split_file).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            stem, cid = parts[0], int(parts[1]) - 1  # cid is 1-indexed
            out.append((str(images / f"{stem}.jpg"), cid))
        return out

    return _load("trainval.txt"), _load("test.txt"), 37


def build_flowers(data_root: Path) -> tuple[list, list, int]:
    """Oxford-102 Flower (DLU): <root>/jpg/image_NNNNN.jpg, plus imagelabels.mat
    and setid.mat. Standard split: trnid + valid -> train, tstid -> test.
    102 classes."""
    from scipy.io import loadmat  # noqa: WPS433 — local import keeps top-level fast
    root = Path(data_root).expanduser()
    jpg = root / "jpg"
    labels_mat = loadmat(str(root / "imagelabels.mat"))
    setid_mat = loadmat(str(root / "setid.mat"))
    if not jpg.is_dir():
        raise FileNotFoundError(f"flowers expects {root}/jpg/; got {root}")

    labels = labels_mat["labels"].squeeze().astype(int)  # 1..102, 1-indexed
    trnid = setid_mat["trnid"].squeeze().astype(int)
    valid = setid_mat["valid"].squeeze().astype(int)
    tstid = setid_mat["tstid"].squeeze().astype(int)

    def _to_samples(ids) -> list[tuple[str, int]]:
        out = []
        for i in ids:
            path = jpg / f"image_{int(i):05d}.jpg"
            out.append((str(path), int(labels[int(i) - 1]) - 1))
        return out

    # Standard transfer convention: trnid + valid -> train, tstid -> test
    train = _to_samples(list(trnid)) + _to_samples(list(valid))
    test = _to_samples(list(tstid))
    return train, test, 102


def build_dtd(data_root: Path, partition: int = 1) -> tuple[list, list, int]:
    """DTD (DLU): <root>/dtd/{images,labels}/. Partition files
    train<p>.txt and test<p>.txt list relative paths `<class>/<img>.jpg`.
    47 classes. Default partition = 1."""
    root = Path(data_root).expanduser()
    base = root / "dtd"
    if not base.is_dir():
        # Tolerate caller passing the inner directory directly.
        base = root
    images = base / "images"
    labels = base / "labels"
    if not images.is_dir() or not labels.is_dir():
        raise FileNotFoundError(f"dtd expects {base}/{{images,labels}}; got {base}")

    classes = sorted(p.name for p in images.iterdir() if p.is_dir())
    cls_to_idx = {c: i for i, c in enumerate(classes)}

    def _load(name: str) -> list[tuple[str, int]]:
        out = []
        for line in (labels / name).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            cls = line.split("/")[0]
            out.append((str(images / line), cls_to_idx[cls]))
        return out

    # Use partition 1 (most common in literature).
    return _load(f"train{partition}.txt"), _load(f"test{partition}.txt"), len(classes)


def build_sun397(data_root: Path) -> tuple[list, list, int]:
    """SUN397 (Zhou OpenDataLab): <root>/SUN397/{a..z}/<class>/<img>.jpg, 397 classes.

    No standard partition files in OpenDataLab tarball. Uses deterministic per-class
    50/50 train/test split (sorted by filename then seeded shuffle, seed=42).
    """
    import random as _random
    root = Path(data_root).expanduser()
    base = root / "SUN397"
    cn_path = base / "ClassName.txt"
    if not cn_path.is_file():
        raise FileNotFoundError(f"sun397 expects {base}/ClassName.txt; got {root}")
    class_paths = [ln.strip().lstrip("/") for ln in cn_path.read_text().splitlines() if ln.strip()]
    assert len(class_paths) == 397, f"expected 397 classes, got {len(class_paths)}"
    train, test = [], []
    rng = _random.Random(42)
    for ci, cp in enumerate(class_paths):
        cls_dir = base / cp
        if not cls_dir.is_dir():
            print(f"[sun397 warn] missing dir {cls_dir}")
            continue
        imgs = sorted(p.name for p in cls_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
        rng.shuffle(imgs)
        half = len(imgs) // 2
        for j, name in enumerate(imgs):
            entry = (str(cls_dir / name), ci)
            (train if j < half else test).append(entry)
    return train, test, 397


DATASET_BUILDERS: dict[str, Callable[[Path], tuple[list, list, int]]] = {
    "sun397": build_sun397,
    "food101": build_food101,
    "pets": build_pets,
    "flowers": build_flowers,
    "dtd": build_dtd,
}


# ---------- feature extraction / probes -------------------------------------

@torch.no_grad()
def extract_features(encoder: nn.Module, loader: DataLoader,
                     device: torch.device) -> tuple[Tensor, Tensor]:
    feats, labels = [], []
    for images, target in loader:
        images = images.to(device, non_blocking=True)
        feats.append(encoder(images).cpu())
        labels.append(target.cpu() if isinstance(target, Tensor) else torch.as_tensor(target))
    return torch.cat(feats, dim=0), torch.cat(labels, dim=0)


def linear_probe_on_features(
    train_feat: Tensor, train_labels: Tensor,
    test_feat: Tensor, test_labels: Tensor,
    n_classes: int, epochs: int, batch_size: int, lr: float,
    device: torch.device, weight_decay: float = 1e-4,
) -> float:
    """Adam linear classifier on already-extracted features. Returns final
    test top-1. Features are passed to GPU here; classifier lives on GPU."""
    train_feat = train_feat.to(device)
    train_labels = train_labels.to(device)
    test_feat_d = test_feat.to(device)
    test_labels_d = test_labels.to(device)

    clf = nn.Linear(train_feat.shape[1], n_classes).to(device)
    opt = torch.optim.AdamW(clf.parameters(), lr=lr, weight_decay=weight_decay)
    n = train_feat.shape[0]

    for epoch in range(1, epochs + 1):
        clf.train()
        perm = torch.randperm(n, device=device)
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            logits = clf(train_feat[idx])
            loss = F.cross_entropy(logits, train_labels[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            clf.eval()
            with torch.no_grad():
                pred = clf(test_feat_d).argmax(dim=1)
                acc = (pred == test_labels_d).float().mean().item()
            print(f"[linear] epoch={epoch} top1={acc:.4f}")
    clf.eval()
    with torch.no_grad():
        pred = clf(test_feat_d).argmax(dim=1)
        return (pred == test_labels_d).float().mean().item()


def knn_top1_on_features(
    train_feat: Tensor, train_labels: Tensor,
    test_feat: Tensor, test_labels: Tensor,
    k: int, chunk: int, device: torch.device,
) -> float:
    """Cosine k-NN, top-1 by mode of nearest k votes."""
    train_feat = F.normalize(train_feat, dim=1).to(device)
    test_feat = F.normalize(test_feat, dim=1).to(device)
    train_labels = train_labels.to(device)
    test_labels = test_labels.to(device)
    correct = 0
    total = 0
    k = min(k, train_feat.shape[0])
    for start in range(0, test_feat.shape[0], chunk):
        feat = test_feat[start:start + chunk]
        sim = feat @ train_feat.T
        idx = sim.topk(k=k, dim=1).indices
        votes = train_labels[idx]
        pred = torch.mode(votes, dim=1).values
        target = test_labels[start:start + chunk]
        correct += (pred == target).sum().item()
        total += target.numel()
    return correct / max(1, total)


# ---------- CLI -------------------------------------------------------------

@dataclass
class TransferSummary:
    checkpoint: str
    tag: str
    dataset: str
    n_train: int
    n_test: int
    n_classes: int
    linear_top1: float
    knn_top1: float
    epochs: int
    batch_size: int
    lr: float
    knn_k: int
    seconds: float
    device: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--tag", type=str, required=True)
    p.add_argument("--dataset", choices=list(DATASET_BUILDERS.keys()), required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--knn-k", type=int, default=200)
    p.add_argument("--knn-chunk", type=int, default=512)
    p.add_argument("--feature-batch-size", type=int, default=256,
                   help="loader batch size during feature extraction (encoder fwd)")
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--force-p2", action="store_true",
                   help="force EML-in-encoder regardless of ckpt 'head' field")
    p.add_argument("--dtd-partition", type=int, default=1,
                   help="DTD has 10 official partitions; default 1.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.skip_existing and args.out.is_file():
        print(f"[transfer] {args.out} already exists, skipping (per --skip-existing).")
        return

    device = torch.device(args.device)
    t0 = time.perf_counter()
    print(f"[transfer] device={device} dataset={args.dataset} ckpt={args.checkpoint}")

    # 1. Build (train, test, n_classes)
    builder = DATASET_BUILDERS[args.dataset]
    if args.dataset == "dtd":
        train_samples, test_samples, n_classes = build_dtd(args.data_root,
                                                           partition=args.dtd_partition)
    else:
        train_samples, test_samples, n_classes = builder(args.data_root)
    print(f"[transfer] train={len(train_samples)} test={len(test_samples)} "
          f"classes={n_classes}")

    # 2. Load encoder
    encoder = load_encoder(args.checkpoint, device,
                           force_p2=True if args.force_p2 else None)

    # 3. Extract features. Use eval_transform for both train (k-NN/linear gallery)
    #    and test — fixed-feature linear probe convention.
    train_ds = FileListDataset(train_samples, eval_transform())
    test_ds = FileListDataset(test_samples, eval_transform())
    train_loader = DataLoader(train_ds, batch_size=args.feature_batch_size,
                              shuffle=False, num_workers=args.num_workers,
                              pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.feature_batch_size,
                             shuffle=False, num_workers=args.num_workers,
                             pin_memory=True)
    print(f"[transfer] extracting train features ({len(train_ds)} imgs)...")
    train_feat, train_labels = extract_features(encoder, train_loader, device)
    print(f"[transfer] extracting test features ({len(test_ds)} imgs)...")
    test_feat, test_labels = extract_features(encoder, test_loader, device)
    print(f"[transfer] train_feat={tuple(train_feat.shape)} "
          f"test_feat={tuple(test_feat.shape)}")

    # 4. Linear probe on frozen features
    print(f"[transfer] running linear probe ({args.epochs} epochs, "
          f"bs={args.batch_size}, lr={args.lr})...")
    linear_top1 = linear_probe_on_features(
        train_feat, train_labels, test_feat, test_labels,
        n_classes=n_classes, epochs=args.epochs,
        batch_size=args.batch_size, lr=args.lr, device=device,
    )

    # 5. k-NN probe (cosine)
    print(f"[transfer] running k-NN probe (k={args.knn_k})...")
    knn_top1 = knn_top1_on_features(
        train_feat, train_labels, test_feat, test_labels,
        k=args.knn_k, chunk=args.knn_chunk, device=device,
    )

    summary = TransferSummary(
        checkpoint=str(args.checkpoint),
        tag=args.tag,
        dataset=args.dataset,
        n_train=len(train_samples),
        n_test=len(test_samples),
        n_classes=n_classes,
        linear_top1=round(linear_top1, 6),
        knn_top1=round(knn_top1, 6),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        knn_k=args.knn_k,
        seconds=round(time.perf_counter() - t0, 2),
        device=str(device),
    )
    print("\n[transfer] summary:")
    for key, val in asdict(summary).items():
        print(f"  {key}: {val}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(asdict(summary), indent=2), encoding="utf-8")
    print(f"\n[transfer] wrote {args.out}")


if __name__ == "__main__":
    main()
