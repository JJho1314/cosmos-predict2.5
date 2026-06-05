#!/usr/bin/env python3
"""Create train/val splits with scene, task, and target-object holdout.

The input is the flat Cosmos VideoDataset produced for DROID/RoboInter.  The
target object is inferred from generated prompt files containing a [TGT] marker.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path


STOP_WORDS = set(
    """
    in into inside on onto from to with and then near next beside over under at by
    for of off up down forward backward towards toward above below
    around through out between across that are slightly the a an
    """.split()
)
LEADING_SKIP_WORDS = {"the", "a", "an", "all", "both", "of"}


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
    with (src / "metadata.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    return rows, {row["name"]: row for row in rows}


def write_metadata(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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


def scene_key(row: dict[str, str]) -> str:
    uuid = row.get("uuid", "")
    parts = uuid.split("+")
    if len(parts) >= 2 and parts[0] and parts[1]:
        return "+".join(parts[:2])
    return uuid or row["name"]


def task_key(row: dict[str, str]) -> str:
    text = row.get("task_orig", "")
    return re.sub(r"\s+", " ", text.lower().strip().rstrip("."))


def target_object_key(src: Path, name: str) -> str:
    meta = src / "metas" / f"{name}.txt"
    try:
        text = meta.read_text(errors="ignore")
    except OSError:
        return ""
    if "[TGT]" not in text:
        return ""
    tail = text.split("[TGT]", 1)[1]
    words = re.findall(r"[A-Za-z0-9]+", tail.lower())
    while words and words[0] in LEADING_SKIP_WORDS:
        words.pop(0)
    kept: list[str] = []
    for word in words:
        if word in STOP_WORDS:
            break
        kept.append(word)
    return " ".join(kept[:6])


def materialize_split(
    src: Path,
    dst: Path,
    names: list[str],
    rows_by_name: dict[str, dict[str, str]],
    overwrite: bool,
) -> None:
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
    if rows:
        write_metadata(dst / "metadata.csv", rows, list(rows[0].keys()))

    frame_ranges_path = src / "frame_ranges.json"
    if frame_ranges_path.exists():
        frame_ranges = json.loads(frame_ranges_path.read_text())
        filtered = {name: frame_ranges[name] for name in names if name in frame_ranges}
        (dst / "frame_ranges.json").write_text(json.dumps(filtered, indent=2, sort_keys=True) + "\n")


def choose_greedy_val_scenes(
    active_names: list[str],
    names_by_scene: dict[str, list[str]],
    scene_by_name: dict[str, str],
    task_by_name: dict[str, str],
    object_by_name: dict[str, str],
    target: int,
    max_scene_samples: int,
    min_samples: int,
    max_samples: int,
) -> list[str]:
    selected: list[str] = []
    active_set = set(active_names)

    def eval_selected(candidate: list[str]) -> tuple[int, int, int, int, int]:
        val_scene_keys = set(candidate)
        val_names = [name for key in candidate for name in names_by_scene[key] if name in active_set]
        val_tasks = {task_by_name[name] for name in val_names}
        val_objects = {object_by_name[name] for name in val_names if object_by_name[name]}
        kept = removed_scene = removed_task = removed_object = 0
        for name in active_names:
            if scene_by_name[name] in val_scene_keys:
                removed_scene += 1
            elif task_by_name[name] in val_tasks:
                removed_task += 1
            elif object_by_name[name] and object_by_name[name] in val_objects:
                removed_object += 1
            else:
                kept += 1
        return len(val_names), kept, removed_scene, removed_task, removed_object

    while True:
        current_val = eval_selected(selected)[0]
        if current_val >= min_samples:
            break
        best: tuple[tuple[float, int, int], str] | None = None
        for key, names in names_by_scene.items():
            if key in selected:
                continue
            active_count = sum(1 for name in names if name in active_set)
            if active_count == 0 or active_count > max_scene_samples:
                continue
            stats = eval_selected(selected + [key])
            val_count, kept, _, _, _ = stats
            if val_count > max_samples:
                continue
            extra_removed = len(active_names) - kept - val_count
            score = (extra_removed / max(val_count, 1), abs(target - val_count), extra_removed)
            if best is None or score < best[0]:
                best = (score, key)
        if best is None:
            break
        selected.append(best[1])

    if not selected:
        raise RuntimeError("Could not choose any validation scenes")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True)
    parser.add_argument("--train-out", required=True)
    parser.add_argument("--val-out", required=True)
    parser.add_argument("--val-target-samples", type=int, default=1000)
    parser.add_argument("--val-min-samples", type=int, default=950)
    parser.add_argument("--val-max-samples", type=int, default=1100)
    parser.add_argument("--max-auto-scene-samples", type=int, default=600)
    parser.add_argument("--val-scene", action="append", default=[])
    parser.add_argument("--exclude-stems-file", action="append", default=[])
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    src = Path(args.src).resolve()
    train_out = Path(args.train_out).resolve()
    val_out = Path(args.val_out).resolve()
    rows, rows_by_name = read_metadata(src)
    exclude_stems = read_exclude_stems(args.exclude_stems_file)
    active_rows = [row for row in rows if row["name"] not in exclude_stems]
    active_names = [row["name"] for row in active_rows]

    scene_by_name: dict[str, str] = {}
    task_by_name: dict[str, str] = {}
    object_by_name: dict[str, str] = {}
    names_by_scene: dict[str, list[str]] = defaultdict(list)
    for i, row in enumerate(active_rows):
        name = row["name"]
        if i % 20000 == 0:
            print(f"parsed_target_object {i}/{len(active_rows)}", flush=True)
        scene_by_name[name] = scene_key(row)
        task_by_name[name] = task_key(row)
        object_by_name[name] = target_object_key(src, name)
        names_by_scene[scene_by_name[name]].append(name)

    if args.val_scene:
        missing = [key for key in args.val_scene if key not in names_by_scene]
        if missing:
            raise KeyError(f"Unknown scene keys: {missing}")
        val_scene_keys = list(dict.fromkeys(args.val_scene))
    else:
        val_scene_keys = choose_greedy_val_scenes(
            active_names=active_names,
            names_by_scene=names_by_scene,
            scene_by_name=scene_by_name,
            task_by_name=task_by_name,
            object_by_name=object_by_name,
            target=args.val_target_samples,
            max_scene_samples=args.max_auto_scene_samples,
            min_samples=args.val_min_samples,
            max_samples=args.val_max_samples,
        )

    val_scene_set = set(val_scene_keys)
    val_names = [name for name in active_names if scene_by_name[name] in val_scene_set]
    val_task_set = {task_by_name[name] for name in val_names}
    val_object_set = {object_by_name[name] for name in val_names if object_by_name[name]}

    train_names: list[str] = []
    removed: dict[str, list[str]] = {"scene": [], "task": [], "object": []}
    for name in active_names:
        if scene_by_name[name] in val_scene_set:
            removed["scene"].append(name)
        elif task_by_name[name] in val_task_set:
            removed["task"].append(name)
        elif object_by_name[name] and object_by_name[name] in val_object_set:
            removed["object"].append(name)
        else:
            train_names.append(name)

    train_scene_set = {scene_by_name[name] for name in train_names}
    train_task_set = {task_by_name[name] for name in train_names}
    train_object_set = {object_by_name[name] for name in train_names if object_by_name[name]}
    overlaps = {
        "scene": sorted(train_scene_set & val_scene_set),
        "task": sorted(train_task_set & val_task_set),
        "object": sorted(train_object_set & val_object_set),
    }
    if any(overlaps.values()):
        raise RuntimeError(f"Holdout leakage detected: {overlaps}")

    materialize_split(src, train_out, train_names, rows_by_name, args.overwrite)
    materialize_split(src, val_out, val_names, rows_by_name, args.overwrite)

    for out in (train_out, val_out):
        (out / "val_scene_keys.txt").write_text("\n".join(sorted(val_scene_set)) + "\n")
        (out / "val_task_keys.txt").write_text("\n".join(sorted(val_task_set)) + "\n")
        (out / "val_target_object_keys.txt").write_text("\n".join(sorted(val_object_set)) + "\n")

    scene_counts = Counter(scene_by_name[name] for name in active_names)
    val_object_counts = Counter(object_by_name[name] for name in val_names)
    summary = {
        "src": str(src),
        "train_out": str(train_out),
        "val_out": str(val_out),
        "split_by": "scene_task_target_object_holdout",
        "scene_key_format": "<lab>+<setup_id> from metadata.csv uuid",
        "task_key_format": "normalized metadata.csv task_orig",
        "target_object_key_format": "words after [TGT] in metas/<name>.txt until a stop word",
        "source_samples": len(rows),
        "source_active_samples": len(active_names),
        "source_excluded_no_tgt_samples": len(rows) - len(active_names),
        "val_target_samples": args.val_target_samples,
        "train_samples": len(train_names),
        "val_samples": len(val_names),
        "train_scene_count": len(train_scene_set),
        "val_scene_count": len(val_scene_set),
        "train_task_count": len(train_task_set),
        "val_task_count": len(val_task_set),
        "train_target_object_count": len(train_object_set),
        "val_target_object_count": len(val_object_set),
        "overlap_counts": {key: len(value) for key, value in overlaps.items()},
        "removed_from_train_due_to_holdout": {key: len(value) for key, value in removed.items()},
        "val_scenes": sorted(val_scene_set),
        "val_scene_counts": {key: scene_counts[key] for key in sorted(val_scene_set)},
        "top_val_target_objects": val_object_counts.most_common(50),
    }
    text = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    (train_out / "split_summary.json").write_text(text)
    (val_out / "split_summary.json").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
