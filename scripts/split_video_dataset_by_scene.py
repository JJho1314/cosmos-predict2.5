#!/usr/bin/env python3
"""Create scene-holdout train/val symlink splits for a flat Cosmos VideoDataset."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path


def safe_reset(path: Path, overwrite: bool) -> None:
    if path.exists() or path.is_symlink():
        if not overwrite:
            raise FileExistsError(f"{path} exists; pass --overwrite to replace it")
        if path.is_symlink() or path.is_file():
            path.unlink()
        else:
            shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def symlink_file(src: Path, dst: Path) -> None:
    if not src.exists() and not src.is_symlink():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    os.symlink(src.resolve(), dst)


def read_metadata(src: Path) -> tuple[list[dict[str, str]], dict[str, dict[str, str]]]:
    metadata_path = src / "metadata.csv"
    with metadata_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    return rows, {row["name"]: row for row in rows}


def write_metadata(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def scene_key(row: dict[str, str]) -> str:
    uuid = row.get("uuid", "")
    parts = uuid.split("+")
    if len(parts) >= 2 and parts[0] and parts[1]:
        return "+".join(parts[:2])
    return uuid or row["name"]


def read_exclude_stems(paths: list[str]) -> set[str]:
    excluded: set[str] = set()
    for text in paths:
        path = Path(text)
        if not path.exists():
            continue
        with path.open() as f:
            for line in f:
                item = line.strip()
                if item and not item.startswith("#"):
                    excluded.add(item)
    return excluded


def choose_val_scenes(groups: dict[str, list[str]], target: int, explicit: list[str]) -> set[str]:
    if explicit:
        missing = [key for key in explicit if key not in groups]
        if missing:
            raise KeyError(f"Unknown scene keys: {missing}")
        return set(explicit)
    best_key, _ = min(groups.items(), key=lambda kv: (abs(len(kv[1]) - target), kv[0]))
    return {best_key}


def materialize_split(
    src: Path,
    dst: Path,
    names: list[str],
    rows_by_name: dict[str, dict[str, str]],
    exclude_stems: set[str],
    overwrite: bool,
) -> dict[str, int]:
    safe_reset(dst, overwrite)
    for dirname in ("videos", "masks", "target_masks", "metas"):
        if (src / dirname).exists():
            (dst / dirname).mkdir(parents=True, exist_ok=True)

    for name in names:
        symlink_file(src / "videos" / f"{name}.mp4", dst / "videos" / f"{name}.mp4")
        for mask_dirname in ("masks", "target_masks"):
            mask = src / mask_dirname / f"{name}.npz"
            if mask.exists() or mask.is_symlink():
                symlink_file(mask, dst / mask_dirname / f"{name}.npz")
        meta = src / "metas" / f"{name}.txt"
        if meta.exists() or meta.is_symlink():
            symlink_file(meta, dst / "metas" / f"{name}.txt")

    rows = [rows_by_name[name] for name in names]
    write_metadata(dst / "metadata.csv", rows, list(rows[0].keys()))

    frame_ranges_path = src / "frame_ranges.json"
    if frame_ranges_path.exists():
        frame_ranges = json.loads(frame_ranges_path.read_text())
        filtered = {name: frame_ranges[name] for name in names if name in frame_ranges}
        (dst / "frame_ranges.json").write_text(json.dumps(filtered, indent=2, sort_keys=True) + "\n")

    split_excluded = sorted(name for name in names if name in exclude_stems)
    if split_excluded:
        (dst / "exclude_no_tgt_stems.txt").write_text("\n".join(split_excluded) + "\n")
    return {
        "samples": len(names),
        "excluded_no_tgt_samples": len(split_excluded),
        "active_samples": len(names) - len(split_excluded),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True)
    parser.add_argument("--train-out", required=True)
    parser.add_argument("--val-out", required=True)
    parser.add_argument("--val-target-samples", type=int, default=1000)
    parser.add_argument("--val-scene", action="append", default=[])
    parser.add_argument("--exclude-stems-file", action="append", default=[])
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    src = Path(args.src).resolve()
    train_out = Path(args.train_out).resolve()
    val_out = Path(args.val_out).resolve()
    rows, rows_by_name = read_metadata(src)
    names = [row["name"] for row in rows]

    groups: dict[str, list[str]] = defaultdict(list)
    scene_by_name: dict[str, str] = {}
    for row in rows:
        key = scene_key(row)
        groups[key].append(row["name"])
        scene_by_name[row["name"]] = key

    val_scenes = choose_val_scenes(groups, args.val_target_samples, args.val_scene)
    val_names_set = {name for key in val_scenes for name in groups[key]}
    train_names = [name for name in names if name not in val_names_set]
    val_names = [name for name in names if name in val_names_set]
    train_scenes = {scene_by_name[name] for name in train_names}
    overlap = sorted(train_scenes & val_scenes)
    if overlap:
        raise RuntimeError(f"Scene leakage detected: {overlap[:10]}")

    exclude_stems = read_exclude_stems(args.exclude_stems_file)
    train_stats = materialize_split(src, train_out, train_names, rows_by_name, exclude_stems, args.overwrite)
    val_stats = materialize_split(src, val_out, val_names, rows_by_name, exclude_stems, args.overwrite)

    scene_counts = Counter(scene_by_name[name] for name in names)
    val_scene_counts = {key: scene_counts[key] for key in sorted(val_scenes)}
    summary = {
        "src": str(src),
        "train_out": str(train_out),
        "val_out": str(val_out),
        "split_by": "uuid_prefix2_scene_key",
        "scene_key_format": "<lab>+<setup_id> from metadata.csv uuid",
        "total_samples": len(names),
        "total_scenes": len(groups),
        "val_target_samples": args.val_target_samples,
        "val_scenes": sorted(val_scenes),
        "val_scene_counts": val_scene_counts,
        "train_scenes": len(train_scenes),
        "val_scene_count": len(val_scenes),
        "scene_overlap_count": len(overlap),
        "train": train_stats,
        "val": val_stats,
    }
    text = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    (train_out / "split_summary.json").write_text(text)
    (val_out / "split_summary.json").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
