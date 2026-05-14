"""Split a Cosmos VideoDataset directory into train/test directories via symlinks."""
from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path


def link_pair(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    dst.symlink_to(src)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True)
    parser.add_argument("--train-out", required=True)
    parser.add_argument("--test-out", required=True)
    parser.add_argument("--test-ratio", type=float, default=0.05)
    parser.add_argument("--test-count", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260511)
    args = parser.parse_args()

    src = Path(args.src).resolve()
    train_out = Path(args.train_out)
    test_out = Path(args.test_out)
    videos = sorted(p for p in (src / "videos").glob("*.mp4") if not p.name.endswith(".tmp.mp4"))
    if not videos:
        raise RuntimeError(f"No videos found under {src / 'videos'}")

    rng = random.Random(args.seed)
    rng.shuffle(videos)
    if args.test_count is not None:
        if args.test_count < 0 or args.test_count >= len(videos):
            raise ValueError(f"--test-count must be in [0, {len(videos) - 1}], got {args.test_count}")
        test_count = args.test_count
    else:
        test_count = max(1, round(len(videos) * args.test_ratio)) if args.test_ratio > 0 else 0
    test_names = {p.stem for p in videos[:test_count]}
    train_names = {p.stem for p in videos[test_count:]}

    for out in [train_out, test_out]:
        if out.exists() and not out.is_symlink():
            shutil.rmtree(out)
        (out / "videos").mkdir(parents=True, exist_ok=True)
        (out / "metas").mkdir(parents=True, exist_ok=True)

    def link_names(names: set[str], out: Path) -> None:
        for name in sorted(names):
            link_pair(src / "videos" / f"{name}.mp4", out / "videos" / f"{name}.mp4")
            link_pair(src / "metas" / f"{name}.txt", out / "metas" / f"{name}.txt")

    link_names(train_names, train_out)
    link_names(test_names, test_out)

    summary = {
        "src": str(src),
        "seed": args.seed,
        "test_ratio": args.test_ratio,
        "test_count_requested": args.test_count,
        "total": len(videos),
        "train": len(train_names),
        "test": len(test_names),
        "train_out": str(train_out),
        "test_out": str(test_out),
    }
    summary_path = train_out.parent / f"{src.name}_split_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
