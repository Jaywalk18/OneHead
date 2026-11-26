"""Build an ImageNet-100 English-class-name <-> WordNet-ID mapping.

Background
----------
The ImageNet-100 dataset may use English subdirectory names under
`<in100-root>/train/<EnglishName>/`, but each
JPEG inside that subdir is named like `n02009912_5074.JPEG`, where `n02009912`
is the WordNet ID (wnid) used by ImageNet-C
(`<corruption>/<sev>/<wnid>/<image>.JPEG`). To filter ImageNet-C down to the
IN100 subset, we need an English -> wnid mapping, which we build by reading
one filename from each English class dir and extracting the wnid prefix.

Usage
-----
    python scripts_t3/imagenet100_class_map.py \
        --in100-train /path/to/imagenet100/train \
        --out scripts_t3/in100_classes.json

Output JSON schema
------------------
    {
      "by_english": {"Americanegret": "n02009912", ...},
      "by_wnid":    {"n02009912": "Americanegret", ...},
      "wnids_sorted": [...100 wnids alphabetically...],
      "english_sorted": [...100 English names alphabetically...]
    }

If --out already exists, the file is loaded and returned without rescan unless
--rescan is passed. This keeps repeated invocations cheap.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

WNID_RE = re.compile(r"^(n\d{6,8})_")


def build_mapping(in100_train: Path) -> dict:
    if not in100_train.is_dir():
        raise FileNotFoundError(f"IN100 train root not found: {in100_train}")

    by_english: dict[str, str] = {}
    skipped: list[str] = []
    for cls_dir in sorted(in100_train.iterdir()):
        if not cls_dir.is_dir():
            continue
        english = cls_dir.name
        # find first file with a wnid prefix; some dirs may have hidden / odd names
        wnid: str | None = None
        for fp in cls_dir.iterdir():
            if not fp.is_file():
                continue
            m = WNID_RE.match(fp.name)
            if m:
                wnid = m.group(1)
                break
        if wnid is None:
            skipped.append(english)
            continue
        by_english[english] = wnid

    by_wnid = {v: k for k, v in by_english.items()}

    if skipped:
        print(f"[in100-map] WARNING: {len(skipped)} class dirs had no wnid-prefixed file: {skipped}")
    if len(by_english) != 100:
        print(f"[in100-map] WARNING: expected 100 IN100 classes, got {len(by_english)}")

    return {
        "by_english": by_english,
        "by_wnid": by_wnid,
        "wnids_sorted": sorted(by_wnid.keys()),
        "english_sorted": sorted(by_english.keys()),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in100-train", type=Path,
                   required=True)
    p.add_argument("--out", type=Path,
                   default=Path(__file__).resolve().parent / "in100_classes.json")
    p.add_argument("--rescan", action="store_true",
                   help="Force rescan even if --out already exists.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.out.is_file() and not args.rescan:
        cached = json.loads(args.out.read_text(encoding="utf-8"))
        print(f"[in100-map] loaded cached mapping from {args.out} "
              f"(n={len(cached.get('by_english', {}))} classes)")
        return

    mapping = build_mapping(args.in100_train)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(mapping, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[in100-map] wrote {args.out}")
    print(f"[in100-map] n_classes={len(mapping['by_english'])}")
    sample = list(mapping["by_english"].items())[:5]
    print("[in100-map] sample (English -> wnid):")
    for k, v in sample:
        print(f"  {k:>20s} -> {v}")


if __name__ == "__main__":
    main()
