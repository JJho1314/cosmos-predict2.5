#!/usr/bin/env python3
"""Visualize TAViD-style target-token cross-attention maps in Cosmos.

Outputs two paper-style qualitative views:

1. ``effect_cross_attention_loss``:
   raw frames + target mask + target-token cross-attention.  If a baseline
   checkpoint is provided, it is shown side-by-side against the trained
   checkpoint to visualize the effect of the cross-attention alignment loss.

2. ``selective_cross_attention_loss``:
   block-wise target-token cross-attention maps for selected DiT blocks,
   matching the selective-block loss idea used by our TAViD-style config.
"""

from __future__ import annotations

import argparse
import gc
import importlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from cosmos_oss.init import init_environment
from cosmos_predict2._src.imaginaire.lazy_config import instantiate
from cosmos_predict2._src.imaginaire.utils import distributed, misc
from cosmos_predict2._src.imaginaire.utils.config_helper import get_config_module, override


@dataclass
class BlockMetric:
    block: int
    mask_mass: float
    attn_mass_inside_mask: float
    attn_inside_mean: float
    attn_outside_mean: float
    inside_outside_ratio: float


@dataclass
class SampleRecord:
    sample_index: int
    caption: str
    tgt_token_index: int
    effect_figure: str
    selective_figure: str
    block_metrics: list[BlockMetric]


def parse_int_list(text: str) -> list[int]:
    if not text:
        return []
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True, help="Checkpoint to visualize, usually the model trained with attention loss.")
    parser.add_argument("--baseline-checkpoint", default="", help="Optional no-attention-loss/baseline checkpoint for side-by-side comparison.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--max-batches", type=int, default=200)
    parser.add_argument("--num-conditional-frames", type=int, default=1)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--blocks", default="8,12,16,20", help="Comma-separated DiT block ids to capture.")
    parser.add_argument("--selected-blocks", default="8,12,16,20", help="Blocks treated as selected by the loss.")
    parser.add_argument("--sample-label", default="with_cross_attention_loss")
    parser.add_argument("--baseline-label", default="without_cross_attention_loss")
    parser.add_argument("opts", nargs=argparse.REMAINDER)
    return parser.parse_args()


def add_online_text_embeddings(model, data_batch: dict) -> None:
    text_encoder_config = getattr(model.config, "text_encoder_config", None)
    if text_encoder_config is not None and text_encoder_config.compute_online:
        text_embeddings = model.text_encoder.compute_text_embeddings_online(data_batch, model.input_caption_key)
        data_batch["t5_text_embeddings"] = text_embeddings
        data_batch["t5_text_mask"] = torch.ones(text_embeddings.shape[0], text_embeddings.shape[1], device="cuda")


def make_config(args: argparse.Namespace, checkpoint: str):
    config_module = get_config_module(args.config)
    config = importlib.import_module(config_module).make_config()
    user_opts = list(args.opts)
    if user_opts and user_opts[0] == "--":
        user_opts = user_opts[1:]
    opts = [
        "--",
        *user_opts,
        f"checkpoint.load_path={checkpoint}",
        "checkpoint.save_to_object_store.enabled=False",
        "checkpoint.load_from_object_store.enabled=False",
        "checkpoint.load_training_state=False",
        "trainer.run_validation=False",
    ]
    config = override(config, opts)
    config.validate()
    config.freeze()
    return config


def load_model_and_dataloader(args: argparse.Namespace, checkpoint: str, blocks: list[int]):
    config = make_config(args, checkpoint)
    trainer = config.trainer.type(config)
    model = instantiate(config.model)
    model = model.to("cuda", memory_format=config.trainer.memory_format)
    model.on_train_start(config.trainer.memory_format)
    optimizer, scheduler = model.init_optimizer_scheduler(config.optimizer, config.scheduler)
    grad_scaler = torch.amp.GradScaler("cuda", **config.trainer.grad_scaler_args)
    loaded_iter = trainer.checkpointer.load(model, optimizer, scheduler, grad_scaler)
    model.net.tavid_attn_alignment_blocks = set(blocks)
    model.eval()
    dataloader = instantiate(config.dataloader_train)
    return model, dataloader, int(loaded_iter)


def to_uint8_frame(frame: torch.Tensor) -> np.ndarray:
    """Convert [C,H,W] uint8 or [-1,1]/[0,1] tensor to RGB uint8."""
    frame = frame.detach().float().cpu()
    if frame.max() <= 2.0:
        if frame.min() < 0:
            frame = (frame.clamp(-1, 1) + 1.0) * 127.5
        else:
            frame = frame.clamp(0, 1) * 255.0
    return frame.clamp(0, 255).byte().permute(1, 2, 0).numpy()


def normalize_heatmap(x: torch.Tensor) -> torch.Tensor:
    x = x.detach().float().cpu()
    flat = x.flatten()
    if flat.numel() > 1_000_000:
        step = int(np.ceil(flat.numel() / 1_000_000))
        flat = flat[::step]
    lo = torch.quantile(flat, 0.01)
    hi = torch.quantile(flat, 0.99)
    return ((x - lo) / (hi - lo + 1e-6)).clamp(0, 1)


def upsample_volume(volume_T_H_W: torch.Tensor, size: tuple[int, int, int], mode: str = "trilinear") -> torch.Tensor:
    return F.interpolate(volume_T_H_W[None, None].float(), size=size, mode=mode, align_corners=False if mode != "nearest" else None)[0, 0]


def heat_color(heat_H_W: torch.Tensor) -> np.ndarray:
    heat = heat_H_W.detach().float().cpu().clamp(0, 1).numpy()
    rgb = np.zeros((*heat.shape, 3), dtype=np.float32)
    rgb[..., 0] = np.clip(1.8 * heat, 0, 1)
    rgb[..., 1] = np.clip(1.8 * (1.0 - np.abs(heat - 0.55) / 0.55), 0, 1)
    rgb[..., 2] = np.clip(1.4 * (1.0 - heat), 0, 1) * (heat > 0.05)
    return (rgb * 255).astype(np.uint8)


def overlay_heat(frame_rgb: np.ndarray, heat_H_W: torch.Tensor, alpha: float = 0.45) -> np.ndarray:
    heat_rgb = heat_color(heat_H_W)
    heat_mask = heat_H_W.detach().float().cpu().clamp(0, 1).numpy()[..., None]
    out = frame_rgb.astype(np.float32) * (1.0 - alpha * heat_mask) + heat_rgb.astype(np.float32) * (alpha * heat_mask)
    return out.clip(0, 255).astype(np.uint8)


def overlay_mask(frame_rgb: np.ndarray, mask_H_W: torch.Tensor, alpha: float = 0.45) -> np.ndarray:
    mask = mask_H_W.detach().float().cpu().clamp(0, 1).numpy()[..., None]
    red = np.zeros_like(frame_rgb, dtype=np.float32)
    red[..., 0] = 255
    out = frame_rgb.astype(np.float32) * (1.0 - alpha * mask) + red * (alpha * mask)
    return out.clip(0, 255).astype(np.uint8)


def labeled_tile(image: np.ndarray, label: str, scale: int = 1) -> Image.Image:
    pil = Image.fromarray(image).resize((image.shape[1] * scale, image.shape[0] * scale))
    draw = ImageDraw.Draw(pil)
    draw.rectangle((0, 0, pil.width, 22), fill=(0, 0, 0))
    draw.text((5, 5), label, fill=(255, 255, 255))
    return pil


def save_grid(path: Path, rows: list[tuple[str, list[np.ndarray]]], frame_labels: list[str]) -> None:
    if not rows:
        return
    tile_w = rows[0][1][0].shape[1]
    tile_h = rows[0][1][0].shape[0]
    label_w = 190
    header_h = 24
    canvas = Image.new("RGB", (label_w + tile_w * len(frame_labels), header_h + tile_h * len(rows)), "white")
    draw = ImageDraw.Draw(canvas)
    for col, label in enumerate(frame_labels):
        draw.text((label_w + col * tile_w + 5, 5), label, fill=(0, 0, 0))
    for row_idx, (row_label, images) in enumerate(rows):
        y = header_h + row_idx * tile_h
        draw.rectangle((0, y, label_w, y + tile_h), fill=(245, 245, 245))
        draw.text((8, y + 8), row_label[:28], fill=(0, 0, 0))
        for col_idx, image in enumerate(images):
            canvas.paste(Image.fromarray(image), (label_w + col_idx * tile_w, y))
    canvas.save(path, quality=95)


def compute_metrics(attn_T_H_W: torch.Tensor, mask_T_H_W: torch.Tensor, block: int) -> BlockMetric:
    attn = attn_T_H_W.detach().float().cpu().clamp(min=0)
    mask = mask_T_H_W.detach().float().cpu().clamp(0, 1)
    if mask.shape != attn.shape:
        mask = upsample_volume(mask, tuple(attn.shape), mode="nearest")
    attn_sum = attn.sum() + 1e-6
    mask_sum = mask.sum() + 1e-6
    inv = 1.0 - mask
    inv_sum = inv.sum() + 1e-6
    inside_mean = (attn * mask).sum() / mask_sum
    outside_mean = (attn * inv).sum() / inv_sum
    return BlockMetric(
        block=block,
        mask_mass=float(mask.mean().item()),
        attn_mass_inside_mask=float(((attn * mask).sum() / attn_sum).item()),
        attn_inside_mean=float(inside_mean.item()),
        attn_outside_mean=float(outside_mean.item()),
        inside_outside_ratio=float((inside_mean / (outside_mean + 1e-6)).item()),
    )


def collect_maps(
    args: argparse.Namespace,
    checkpoint: str,
    label: str,
    blocks: list[int],
    output_dir: Path,
) -> tuple[list[dict], int]:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model, dataloader, loaded_iter = load_model_and_dataloader(args, checkpoint, blocks)
    block_set = set(blocks)
    block_ids = [idx for idx in range(len(model.net.blocks)) if idx in block_set]
    records = []
    saved = 0
    with torch.no_grad():
        for batch_idx, cpu_batch in enumerate(dataloader):
            if batch_idx >= args.max_batches or saved >= args.num_samples:
                break
            if "target_mask" not in cpu_batch or float(cpu_batch["target_mask"].sum()) <= 0:
                continue
            raw_cpu = cpu_batch[model.input_data_key][0].detach().cpu()
            mask_cpu = cpu_batch["target_mask"][0].detach().cpu()
            caption = cpu_batch.get(model.input_caption_key, [""])[0]

            data_batch = misc.to(cpu_batch, device="cuda")
            data_batch["num_conditional_frames"] = torch.full(
                (data_batch[model.input_data_key].shape[0],),
                args.num_conditional_frames,
                dtype=torch.long,
                device="cuda",
            )
            add_online_text_embeddings(model, data_batch)
            tgt_indices = data_batch.get("tgt_token_indices", torch.full((1,), -1, device="cuda"))
            if int(tgt_indices[0].item()) < 0:
                continue

            with torch.autocast("cuda", dtype=torch.bfloat16):
                output_batch, loss = model(data_batch)

            attn_maps = [item[0].detach().float().cpu() for item in getattr(model.net, "tavid_target_attn_maps", [])]
            latent_mask = getattr(model.net, "tavid_target_mask_B_T_H_W", None)
            latent_mask_cpu = latent_mask[0].detach().float().cpu() if latent_mask is not None else None
            if latent_mask_cpu is None or float(latent_mask_cpu.sum()) <= 0:
                continue
            metrics = [
                compute_metrics(attn_map, latent_mask_cpu, block)
                for block, attn_map in zip(block_ids, attn_maps)
            ]
            records.append(
                {
                    "label": label,
                    "sample_index": saved,
                    "caption": caption,
                    "raw": raw_cpu,
                    "mask": mask_cpu,
                    "latent_mask": latent_mask_cpu,
                    "block_ids": block_ids[: len(attn_maps)],
                    "attn_maps": attn_maps,
                    "tgt_token_index": int(tgt_indices[0].item()),
                    "loaded_iter": loaded_iter,
                    "loss": float(loss.detach().float().item()),
                    "extra_losses": {
                        key: float(value.detach().float().item())
                        for key, value in output_batch.items()
                        if key.startswith("target_attention_loss") and torch.is_tensor(value)
                    },
                    "metrics": metrics,
                }
            )
            print(json.dumps({
                "label": label,
                "sample_index": saved,
                "caption": caption,
                "tgt_token_index": int(tgt_indices[0].item()),
                "metrics": [asdict(item) for item in metrics],
            }, ensure_ascii=False), flush=True)
            saved += 1

    del model, dataloader
    gc.collect()
    torch.cuda.empty_cache()
    return records, loaded_iter


def selected_mean_attention(record: dict, selected_blocks: set[int]) -> torch.Tensor:
    selected = [
        attn for block, attn in zip(record["block_ids"], record["attn_maps"])
        if block in selected_blocks
    ]
    if not selected:
        selected = record["attn_maps"]
    return torch.stack(selected).mean(dim=0)


def frame_indices(num_frames: int) -> list[int]:
    return sorted(set([0, num_frames // 4, num_frames // 2, 3 * num_frames // 4, num_frames - 1]))


def make_effect_figure(path: Path, main: dict, baseline: dict | None, selected_blocks: set[int]) -> None:
    raw = main["raw"]
    mask = main["mask"][0]
    T_raw = raw.shape[1]
    H_raw, W_raw = raw.shape[-2:]
    frames = frame_indices(T_raw)
    frame_labels = [f"f{idx}" for idx in frames]

    raw_frames = [to_uint8_frame(raw[:, idx]) for idx in frames]
    mask_up = upsample_volume(mask, (T_raw, H_raw, W_raw), mode="nearest")
    rows: list[tuple[str, list[np.ndarray]]] = [
        ("RGB", raw_frames),
        ("target mask", [overlay_mask(raw_frames[i], mask_up[idx]) for i, idx in enumerate(frames)]),
    ]

    if baseline is not None:
        base_attn = normalize_heatmap(upsample_volume(selected_mean_attention(baseline, selected_blocks), (T_raw, H_raw, W_raw)))
        rows.append((
            baseline["label"],
            [overlay_heat(raw_frames[i], base_attn[idx]) for i, idx in enumerate(frames)],
        ))

    main_attn = normalize_heatmap(upsample_volume(selected_mean_attention(main, selected_blocks), (T_raw, H_raw, W_raw)))
    rows.append((main["label"], [overlay_heat(raw_frames[i], main_attn[idx]) for i, idx in enumerate(frames)]))

    if baseline is not None:
        delta = (main_attn - base_attn).clamp(min=0)
        rows.append(("attention gain", [overlay_heat(raw_frames[i], normalize_heatmap(delta)[idx]) for i, idx in enumerate(frames)]))

    save_grid(path, rows, frame_labels)


def make_selective_figure(path: Path, record: dict, selected_blocks: set[int]) -> None:
    raw = record["raw"]
    mask = record["mask"][0]
    T_raw = raw.shape[1]
    H_raw, W_raw = raw.shape[-2:]
    frames = frame_indices(T_raw)
    frame_labels = [f"f{idx}" for idx in frames]
    raw_frames = [to_uint8_frame(raw[:, idx]) for idx in frames]
    mask_up = upsample_volume(mask, (T_raw, H_raw, W_raw), mode="nearest")

    rows: list[tuple[str, list[np.ndarray]]] = [
        ("RGB", raw_frames),
        ("target mask", [overlay_mask(raw_frames[i], mask_up[idx]) for i, idx in enumerate(frames)]),
    ]
    for block, attn in zip(record["block_ids"], record["attn_maps"]):
        heat = normalize_heatmap(upsample_volume(attn, (T_raw, H_raw, W_raw)))
        prefix = "selected" if block in selected_blocks else "not selected"
        rows.append((f"{prefix} block {block}", [overlay_heat(raw_frames[i], heat[idx]) for i, idx in enumerate(frames)]))

    mean_heat = normalize_heatmap(upsample_volume(selected_mean_attention(record, selected_blocks), (T_raw, H_raw, W_raw)))
    rows.append(("selected mean", [overlay_heat(raw_frames[i], mean_heat[idx]) for i, idx in enumerate(frames)]))
    save_grid(path, rows, frame_labels)


def main() -> None:
    args = parse_args()
    init_environment()
    output_dir = Path(args.output_dir)
    if distributed.is_rank0():
        output_dir.mkdir(parents=True, exist_ok=True)

    blocks = parse_int_list(args.blocks)
    selected_blocks = set(parse_int_list(args.selected_blocks))
    if not blocks:
        raise ValueError("--blocks cannot be empty")

    baseline_records = None
    baseline_loaded_iter = None
    if args.baseline_checkpoint:
        baseline_records, baseline_loaded_iter = collect_maps(
            args, args.baseline_checkpoint, args.baseline_label, blocks, output_dir
        )

    main_records, loaded_iter = collect_maps(args, args.checkpoint, args.sample_label, blocks, output_dir)

    summaries = []
    for idx, main_record in enumerate(main_records):
        baseline_record = baseline_records[idx] if baseline_records and idx < len(baseline_records) else None
        effect_path = output_dir / f"sample_{idx:03d}_effect_cross_attention_loss.jpg"
        selective_path = output_dir / f"sample_{idx:03d}_selective_cross_attention_loss.jpg"
        make_effect_figure(effect_path, main_record, baseline_record, selected_blocks)
        make_selective_figure(selective_path, main_record, selected_blocks)
        summaries.append(
            SampleRecord(
                sample_index=idx,
                caption=main_record["caption"],
                tgt_token_index=main_record["tgt_token_index"],
                effect_figure=str(effect_path),
                selective_figure=str(selective_path),
                block_metrics=main_record["metrics"],
            )
        )

    summary = {
        "checkpoint": args.checkpoint,
        "baseline_checkpoint": args.baseline_checkpoint or None,
        "loaded_iter": loaded_iter,
        "baseline_loaded_iter": baseline_loaded_iter,
        "blocks": blocks,
        "selected_blocks": sorted(selected_blocks),
        "num_samples": len(summaries),
        "samples": [
            {
                **asdict(item),
                "block_metrics": [asdict(metric) for metric in item.block_metrics],
            }
            for item in summaries
        ],
    }
    if distributed.is_rank0():
        (output_dir / "cross_attention_visualization_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
        )


if __name__ == "__main__":
    main()
