#!/usr/bin/env python3
"""Create a metadata-only dataset variant with a stable target-object suffix.

The source dataset is expected to already contain noun-prefix [TGT] markers in
most meta files, for example:

    put the [TGT] watermelon toy on the plate.

This script creates a new dataset directory where videos/ and masks/ are
symlinked to the source dataset, while metas/ are rewritten to:

    put the [TGT] watermelon toy on the plate. The robot interacts with the target object.

If a source caption has no [TGT], it is left without the extra suffix so the
existing VideoDataset target_prompt_suffix logic can fall back to the previous
"[TGT] target object" prompt style.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path


OLD_TGT_SUFFIX = "The robot interacts with the [TGT] target object."
NEW_SUFFIX = "The robot interacts with the target object."


def normalize_caption(text: str) -> tuple[str, str]:
    text = " ".join(text.strip().split())
    for suffix in (OLD_TGT_SUFFIX, NEW_SUFFIX):
        if text.endswith(suffix):
            text = text[: -len(suffix)].strip()
            if text.endswith("."):
                text = text[:-1].strip()
    tgt_count = text.count("[TGT]")
    if tgt_count == 1:
        text = text.rstrip(".").strip() + f". {NEW_SUFFIX}"
        status = "tgt_prefix_suffix"
    elif tgt_count == 0:
        text = text.rstrip(".").strip()
        status = "no_tgt_left_for_dataset_suffix"
    else:
        text = text.rstrip(".").strip() + f". {NEW_SUFFIX}"
        status = "multi_tgt_kept"
    return text, status


def replace_symlink(path: Path, target: Path) -> None:
    if path.is_symlink() or path.exists():
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink()
    path.symlink_to(target.resolve(), target_is_directory=True)


def build_variant(src: Path, dst: Path) -> dict[str, int | str]:
    src = src.resolve()
    dst.mkdir(parents=True, exist_ok=True)
    replace_symlink(dst / "videos", src / "videos")
    if (src / "masks").exists():
        replace_symlink(dst / "masks", src / "masks")
    elif (src / "target_masks").exists():
        replace_symlink(dst / "target_masks", src / "target_masks")
    else:
        raise FileNotFoundError(f"No masks/ or target_masks/ under {src}")

    metas_src = src / "metas"
    metas_dst = dst / "metas"
    if metas_dst.exists() or metas_dst.is_symlink():
        if metas_dst.is_dir() and not metas_dst.is_symlink():
            shutil.rmtree(metas_dst)
        else:
            metas_dst.unlink()
    metas_dst.mkdir(parents=True)

    counts = {
        "source": str(src),
        "destination": str(dst),
        "total_metas": 0,
        "tgt_prefix_suffix": 0,
        "no_tgt_left_for_dataset_suffix": 0,
        "multi_tgt_kept": 0,
    }
    for meta_path in sorted(metas_src.glob("*.txt")):
        rewritten, status = normalize_caption(meta_path.read_text(errors="ignore"))
        (metas_dst / meta_path.name).write_text(rewritten + "\n")
        counts["total_metas"] += 1
        counts[status] += 1

    # Do not copy exclude_no_tgt_stems.txt. The requested comparison is against
    # the old bs4/accum2 run, whose effective split did not exclude no-TGT rows.
    with (dst / "tgtprefix_suffix_summary.json").open("w") as f:
        json.dump(counts, f, indent=2)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-src", required=True, type=Path)
    parser.add_argument("--val-src", required=True, type=Path)
    parser.add_argument("--train-dst", required=True, type=Path)
    parser.add_argument("--val-dst", required=True, type=Path)
    args = parser.parse_args()

    summaries = {
        "train": build_variant(args.train_src, args.train_dst),
        "val": build_variant(args.val_src, args.val_dst),
    }
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
