#!/usr/bin/env python3
"""Generate TAViD-style qualitative samples with target-mask conditioning."""

from __future__ import annotations

import argparse
import importlib
import json
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms.functional as TF

from cosmos_oss.init import init_environment
from cosmos_predict2._src.imaginaire.lazy_config import instantiate
from cosmos_predict2._src.imaginaire.utils import distributed, misc
from cosmos_predict2._src.imaginaire.utils.config_helper import get_config_module, override
from cosmos_predict2._src.predict2.inference.utils import write_video


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--skip-samples", type=int, default=0)
    parser.add_argument("--sample-index-offset", type=int, default=0)
    parser.add_argument("--num-steps", type=int, default=35)
    parser.add_argument("--guidance", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--num-conditional-frames", type=int, default=1)
    parser.add_argument("--max-batches", type=int, default=400)
    parser.add_argument("--standalone-only", action="store_true", help="Only save generated/GT videos and captions.")
    parser.add_argument("opts", nargs=argparse.REMAINDER)
    return parser.parse_args()


def add_online_text_embeddings(model, data_batch: dict) -> None:
    text_encoder_config = getattr(model.config, "text_encoder_config", None)
    if text_encoder_config is not None and text_encoder_config.compute_online:
        text_embeddings = model.text_encoder.compute_text_embeddings_online(data_batch, model.input_caption_key)
        data_batch["t5_text_embeddings"] = text_embeddings
        data_batch["t5_text_mask"] = torch.ones(text_embeddings.shape[0], text_embeddings.shape[1], device="cuda")


def checkpoint_iter(path: str) -> int | None:
    match = re.search(r"iter_(\d+)", path)
    return int(match.group(1)) if match else None


def to_01(video: torch.Tensor) -> torch.Tensor:
    return ((video.detach().float().cpu().clamp(-1, 1) + 1.0) / 2.0).clamp(0, 1)


def mask_to_rgb(mask: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    # mask: [1, T, H, W] or [T, H, W]
    if mask.ndim == 4:
        mask = mask[0]
    mask = F.interpolate(mask[None, None].float(), size=(mask.shape[0], *size), mode="nearest")[0, 0]
    return mask.unsqueeze(0).repeat(3, 1, 1, 1).cpu()


def make_overlay(raw: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    raw01 = to_01(raw)
    mask_rgb = mask_to_rgb(mask.cpu(), size=raw01.shape[-2:])
    red = torch.tensor([1.0, 0.0, 0.0]).view(3, 1, 1, 1)
    return (raw01 * 0.55 + red * mask_rgb * 0.45).clamp(0, 1)


def video_to_uint8(video: torch.Tensor) -> np.ndarray:
    # input: [3, T, H, W] in [0, 1]
    video = video.detach().float().cpu().clamp(0, 1)
    return (video.permute(1, 2, 3, 0).numpy() * 255.0).round().astype(np.uint8)


def save_contact_sheet(path: Path, raw: torch.Tensor, overlay: torch.Tensor, generated: torch.Tensor) -> None:
    raw = to_01(raw)
    generated = to_01(generated)
    frames = sorted(set([0, raw.shape[1] // 4, raw.shape[1] // 2, raw.shape[1] * 3 // 4, raw.shape[1] - 1]))
    rows = [raw, overlay, generated]
    tiles = []
    for row in rows:
        tiles.extend([row[:, idx] for idx in frames])
    grid = torchvision.utils.make_grid(tiles, nrow=len(frames), padding=2)
    grid = TF.resize(grid, [grid.shape[-2] * 2, grid.shape[-1] * 2])
    torchvision.utils.save_image(grid, path)


def save_sample_outputs(
    output_dir: Path,
    sample_index: int,
    raw: torch.Tensor,
    mask: torch.Tensor,
    generated: torch.Tensor,
    caption: str,
    fps: int,
    standalone_only: bool = False,
) -> dict:
    stem = f"sample_{sample_index:03d}"
    raw01 = to_01(raw)
    gen01 = to_01(generated)

    generated_np = video_to_uint8(gen01)
    gt_np = video_to_uint8(raw01)

    generated_path = output_dir / f"{stem}_generated.mp4"
    gt_path = output_dir / f"{stem}_gt.mp4"
    caption_path = output_dir / f"{stem}_caption.txt"

    write_video(str(generated_path), generated_np, fps=fps, lossless=False)
    write_video(str(gt_path), gt_np, fps=fps, lossless=False)
    caption_path.write_text(caption + "\n")

    record = {
        "sample_index": sample_index,
        "caption": caption,
        "generated": str(generated_path),
        "gt": str(gt_path),
        "caption_file": str(caption_path),
    }
    if standalone_only:
        return record

    overlay = make_overlay(raw, mask)
    overlay_np = video_to_uint8(overlay)
    grid_np = np.concatenate([overlay_np, generated_np, gt_np], axis=2)
    overlay_path = output_dir / f"{stem}_mask_overlay.mp4"
    grid_path = output_dir / f"{stem}_overlay_generated_gt.mp4"
    sheet_path = output_dir / f"{stem}_contact.jpg"
    write_video(str(overlay_path), overlay_np, fps=fps, lossless=False)
    write_video(str(grid_path), grid_np, fps=fps, lossless=False)
    save_contact_sheet(sheet_path, raw, overlay, generated)
    record.update(
        {
            "mask_overlay": str(overlay_path),
            "overlay_generated_gt": str(grid_path),
            "contact_sheet": str(sheet_path),
        }
    )
    return record


def main() -> None:
    args = parse_args()
    init_environment()

    config_module = get_config_module(args.config)
    config = importlib.import_module(config_module).make_config()
    user_opts = list(args.opts)
    if user_opts and user_opts[0] == "--":
        user_opts = user_opts[1:]
    opts = [
        "--",
        *user_opts,
        f"checkpoint.load_path={args.checkpoint}",
        "checkpoint.save_to_object_store.enabled=False",
        "checkpoint.load_from_object_store.enabled=False",
        "checkpoint.load_training_state=False",
        "trainer.run_validation=False",
    ]
    config = override(config, opts)
    config.validate()
    config.freeze()

    trainer = config.trainer.type(config)
    model = instantiate(config.model)
    model = model.to("cuda", memory_format=config.trainer.memory_format)
    model.on_train_start(config.trainer.memory_format)
    optimizer, scheduler = model.init_optimizer_scheduler(config.optimizer, config.scheduler)
    grad_scaler = torch.amp.GradScaler("cuda", **config.trainer.grad_scaler_args)
    loaded_iter = trainer.checkpointer.load(model, optimizer, scheduler, grad_scaler)
    model.eval()

    dataloader = instantiate(config.dataloader_train)
    output_dir = Path(args.output_dir)
    if distributed.is_rank0():
        output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    saved = 0
    skipped = 0
    with torch.no_grad():
        for batch_idx, data_batch in enumerate(dataloader):
            if batch_idx >= args.max_batches or saved >= args.num_samples:
                break
            if "target_mask" not in data_batch or float(data_batch["target_mask"].sum()) <= 0:
                continue
            if skipped < args.skip_samples:
                skipped += 1
                continue

            data_batch = misc.to(data_batch, device="cuda")
            data_batch["num_conditional_frames"] = torch.full(
                (data_batch[model.input_data_key].shape[0],),
                args.num_conditional_frames,
                dtype=torch.long,
                device="cuda",
            )
            add_online_text_embeddings(model, data_batch)

            raw, x0, _ = model.get_data_and_condition(data_batch)
            state_shape = x0.shape[1:]

            with torch.autocast("cuda", dtype=torch.bfloat16):
                latent = model.generate_samples_from_batch(
                    data_batch,
                    guidance=args.guidance,
                    seed=args.seed + saved,
                    state_shape=state_shape,
                    n_sample=x0.shape[0],
                    num_steps=args.num_steps,
                )
                generated = model.decode(latent)

            caption = data_batch.get(model.input_caption_key, [""])[0]
            if distributed.is_rank0():
                record = save_sample_outputs(
                    output_dir=output_dir,
                    sample_index=args.sample_index_offset + saved,
                    raw=raw[0],
                    mask=data_batch["target_mask"][0],
                    generated=generated[0],
                    caption=caption,
                    fps=args.fps,
                    standalone_only=args.standalone_only,
                )
                print(json.dumps(record, ensure_ascii=False), flush=True)
                records.append(record)
            saved += 1

    if distributed.is_rank0():
        summary = {
            "checkpoint": args.checkpoint,
            "checkpoint_iter_from_path": checkpoint_iter(args.checkpoint),
            "loaded_iter_returned_by_checkpointer": int(loaded_iter),
            "num_samples": len(records),
            "skip_samples": args.skip_samples,
            "sample_index_offset": args.sample_index_offset,
            "num_steps": args.num_steps,
            "guidance": args.guidance,
            "seed": args.seed,
            "fps": args.fps,
            "layout": "standalone generated/GT videos" if args.standalone_only else "overlay_generated_gt mp4 columns are: mask overlay, generated, ground truth",
            "samples": records,
        }
        with open(output_dir / "tavid_generation_summary.json", "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
