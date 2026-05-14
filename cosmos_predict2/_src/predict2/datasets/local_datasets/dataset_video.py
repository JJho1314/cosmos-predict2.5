# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Generic video dataset loader for Cosmos Predict2."""

import json
import os
import random
import traceback
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import torch
from decord import VideoReader, cpu
from PIL import Image
from megatron.core import parallel_state
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from cosmos_predict2._src.imaginaire.lazy_config import LazyCall as L
from cosmos_predict2._src.imaginaire.utils import log
from cosmos_predict2._src.predict2.datasets.local_datasets.dataset_utils import ResizePreprocess, ToTensorVideo


class VideoDataset(Dataset):
    def __init__(
        self,
        dataset_dir: str,
        num_frames: int,
        video_size: tuple[int, int],
        prompt_type: str | None = None,  # "long", "short", "medium", or None for auto
        caption_format: str = "auto",  # "text", "json", or "auto"
        video_paths: Optional[list[str]] = None,
        target_mask_dir: Optional[str] = None,
        target_mask_default_to_zero: bool = True,
        target_prompt_suffix: str = "",
    ) -> None:
        """Dataset class for loading image-text-to-video generation data.

        Args:
            dataset_dir (str): Base path to the dataset directory
            num_frames (int): Number of frames to load per sequence
            video_size (tuple[int, int]): Target size (H,W) for video frames
            prompt_type (str | None): Which prompt to use from JSON ("long", "short", "medium").
                                     If None, uses the first available prompt type.
                                     Only applicable when using JSON format.
            caption_format (str): Caption format - "text", "json", or "auto" to detect automatically

        Returns dict with:
            - video: RGB frames tensor [T,C,H,W]
            - video_name: Dict with episode/frame metadata
        """

        super().__init__()
        self.dataset_dir = dataset_dir
        self.sequence_length = num_frames
        self.prompt_type = prompt_type
        self.caption_format = caption_format
        self.target_mask_dir = self._resolve_target_mask_dir(target_mask_dir)
        self.target_mask_default_to_zero = target_mask_default_to_zero
        self.target_prompt_suffix = target_prompt_suffix

        # Determine caption format and directory
        self._setup_caption_format()

        video_dir = os.path.join(self.dataset_dir, "videos")

        if video_paths is None:
            self.video_paths = [os.path.join(video_dir, f) for f in os.listdir(video_dir) if f.endswith(".mp4")]
            self.video_paths = sorted(self.video_paths)
        else:
            self.video_paths = video_paths
        log.info(f"{len(self.video_paths)} videos in total")

        self.num_failed_loads = 0
        self.preprocess = T.Compose([ToTensorVideo(), ResizePreprocess((video_size[0], video_size[1]))])
        self.mask_size = (video_size[0], video_size[1])

    def __str__(self) -> str:
        return f"{len(self.video_paths)} samples from {self.dataset_dir}"

    def __len__(self) -> int:
        return len(self.video_paths)

    def _resolve_target_mask_dir(self, target_mask_dir: Optional[str]) -> Optional[str]:
        """Resolve optional target-mask directory for TAViD-style conditioning."""
        if target_mask_dir is None:
            for dirname in ("masks", "target_masks"):
                candidate = os.path.join(self.dataset_dir, dirname)
                if os.path.isdir(candidate):
                    return candidate
            return None
        if target_mask_dir.lower() == "none":
            return None
        if target_mask_dir.lower() == "auto":
            for dirname in ("masks", "target_masks"):
                candidate = os.path.join(self.dataset_dir, dirname)
                if os.path.isdir(candidate):
                    return candidate
            return None
        return target_mask_dir

    def _load_video(self, video_path: str) -> tuple[np.ndarray, float, np.ndarray]:
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=2)
        total_frames = len(vr)
        if total_frames < self.sequence_length:
            raise ValueError(
                f"Video {video_path} has only {total_frames} frames, "
                f"at least {self.sequence_length} frames are required."
            )

        # randomly sample a sequence of frames
        max_start_idx = total_frames - self.sequence_length
        start_frame = np.random.randint(0, max_start_idx)
        end_frame = start_frame + self.sequence_length
        frame_ids = np.arange(start_frame, end_frame).tolist()

        frame_data = vr.get_batch(frame_ids).asnumpy()
        vr.seek(0)  # set video reader point back to 0 to clean up cache

        try:
            fps = vr.get_avg_fps()
        except Exception:  # failed to read FPS, assume it is 16
            fps = 16
        del vr  # delete the reader to avoid memory leak
        return frame_data, fps, np.asarray(frame_ids, dtype=np.int64)

    def _setup_caption_format(self) -> None:
        """Determine the caption format and set up the caption directory."""
        metas_dir = os.path.join(self.dataset_dir, "metas")
        captions_dir = os.path.join(self.dataset_dir, "captions")

        if self.caption_format == "auto":
            # Auto-detect based on directory existence
            if os.path.exists(captions_dir) and any(f.endswith(".json") for f in os.listdir(captions_dir)):
                self.caption_format = "json"
                self.caption_dir = captions_dir
            elif os.path.exists(metas_dir) and any(f.endswith(".txt") for f in os.listdir(metas_dir)):
                self.caption_format = "text"
                self.caption_dir = metas_dir
            else:
                raise ValueError(
                    f"Could not auto-detect caption format. Neither 'metas/*.txt' nor 'captions/*.json' found in {self.dataset_dir}"
                )
        elif self.caption_format == "json":
            if not os.path.exists(captions_dir):
                raise ValueError(f"JSON format specified but 'captions' directory not found in {self.dataset_dir}")
            self.caption_dir = captions_dir
        elif self.caption_format == "text":
            if not os.path.exists(metas_dir):
                raise ValueError(f"Text format specified but 'metas' directory not found in {self.dataset_dir}")
            self.caption_dir = metas_dir
        else:
            raise ValueError(f"Invalid caption_format: {self.caption_format}. Must be 'text', 'json', or 'auto'")

    def _load_text(self, text_source: Path) -> str:
        """Load text caption from file."""
        try:
            return text_source.read_text().strip()
        except Exception as e:
            log.warning(f"Failed to read caption file {text_source}: {e}")
            return ""

    def _load_json_caption(self, json_path: Path) -> str:
        """Load caption from JSON file with prompt type selection."""
        try:
            with open(json_path, "r") as f:
                content = f.read()
                # Handle JSON that might not have top-level object
                if not content.strip().startswith("{"):
                    # Wrap in object if needed
                    data = json.loads("{" + content + "}")
                else:
                    data = json.loads(content)

            # Get the first model's captions (e.g., "qwen3_vl_30b_a3b")
            model_key = next(iter(data.keys()))
            captions = data[model_key]

            if self.prompt_type:
                # Use specified prompt type
                if self.prompt_type in captions:
                    return captions[self.prompt_type]
                else:
                    log.warning(
                        f"Prompt type '{self.prompt_type}' not found in {json_path}. "
                        f"Available: {list(captions.keys())}. Using first available."
                    )

            # Use first available prompt type
            first_prompt = next(iter(captions.values()))
            return first_prompt

        except Exception as e:
            log.warning(f"Failed to read JSON caption file {json_path}: {e}")
            return ""

    def _get_frames(self, video_path: str) -> tuple[torch.Tensor, float, np.ndarray]:
        frames, fps, frame_ids = self._load_video(video_path)
        frames = frames.astype(np.uint8)
        frames = torch.from_numpy(frames).permute(0, 3, 1, 2)  # [T, C, H, W]
        frames = self.preprocess(frames)
        frames = torch.clamp(frames * 255.0, 0, 255).to(torch.uint8)
        return frames, fps, frame_ids

    def _target_mask_path_candidates(self, video_basename: str) -> list[str]:
        if self.target_mask_dir is None:
            return []
        return [
            os.path.join(self.target_mask_dir, f"{video_basename}{ext}")
            for ext in (".mp4", ".npz", ".png", ".jpg", ".jpeg", ".webp")
        ]

    def _resize_binary_mask_video(self, mask: torch.Tensor) -> torch.Tensor:
        """Resize [T,1,H,W] mask video and return [1,T,H,W]."""
        mask = torch.stack(
            [TF.resize(frame, self.mask_size, interpolation=InterpolationMode.NEAREST) for frame in mask.float()]
        )
        mask = (mask > 0.5).float()
        return mask.permute(1, 0, 2, 3).contiguous()

    def _load_npz_target_mask(self, mask_path: str, frame_ids: np.ndarray) -> torch.Tensor:
        mask_npz = np.load(mask_path, allow_pickle=True)
        key = "masks" if "masks" in mask_npz.files else mask_npz.files[0]
        mask_arr = mask_npz[key]
        if mask_arr.ndim == 5:
            # RoboInter SAM files are typically [N,T,1,H,W]. Merge annotated target
            # masks when multiple instances are present.
            mask_arr = mask_arr.max(axis=0)
        if mask_arr.ndim == 4:
            if mask_arr.shape[1] == 1:  # [T,1,H,W]
                mask_arr = mask_arr[:, 0]
            elif mask_arr.shape[0] == 1:  # [1,T,H,W]
                mask_arr = mask_arr[0]
            else:
                mask_arr = mask_arr.max(axis=0)
        if mask_arr.ndim == 2:
            mask_arr = np.repeat(mask_arr[None], len(frame_ids), axis=0)
        if mask_arr.ndim != 3:
            raise ValueError(f"Unsupported target mask shape {mask_arr.shape} in {mask_path}")
        valid_frame_ids = np.clip(frame_ids, 0, mask_arr.shape[0] - 1)
        mask_arr = mask_arr[valid_frame_ids]
        mask = torch.from_numpy(mask_arr).unsqueeze(1).float()  # [T,1,H,W]
        return self._resize_binary_mask_video(mask)

    def _load_target_mask(self, video_basename: str, frame_ids: np.ndarray) -> torch.Tensor:
        """Load target mask as [1,T,H,W].

        If a mask video exists, it is sampled with the same frame ids as the RGB video.
        If a single image mask exists, it is treated like TAViD's initial-frame target mask
        and placed on the first sampled frame only; all future frames are zero.
        """
        T_frames = len(frame_ids)
        zero_mask = torch.zeros(1, T_frames, *self.mask_size, dtype=torch.float32)
        mask_path = next((path for path in self._target_mask_path_candidates(video_basename) if os.path.exists(path)), None)
        if mask_path is None:
            if self.target_mask_default_to_zero:
                return zero_mask
            raise FileNotFoundError(f"Target mask for {video_basename} not found in {self.target_mask_dir}")

        if mask_path.endswith(".mp4"):
            mask_reader = VideoReader(mask_path, ctx=cpu(0), num_threads=1)
            valid_frame_ids = np.clip(frame_ids, 0, len(mask_reader) - 1).tolist()
            mask_frames = mask_reader.get_batch(valid_frame_ids).asnumpy()
            mask_reader.seek(0)
            del mask_reader
            mask = torch.from_numpy(mask_frames[..., 0]).unsqueeze(1).float() / 255.0  # [T,1,H,W]
            return self._resize_binary_mask_video(mask)

        if mask_path.endswith(".npz"):
            return self._load_npz_target_mask(mask_path, frame_ids)

        mask_img = Image.open(mask_path).convert("L")
        mask = TF.to_tensor(mask_img)
        mask = TF.resize(mask, self.mask_size, interpolation=InterpolationMode.NEAREST)
        zero_mask[:, 0] = (mask > 0.5).float()
        return zero_mask

    def __getitem__(self, index: int) -> dict | Any:
        try:
            data = dict()
            video, fps, frame_ids = self._get_frames(self.video_paths[index])
            video = video.permute(1, 0, 2, 3)  # Rearrange from [T, C, H, W] to [C, T, H, W]

            # Load caption based on format
            video_path = self.video_paths[index]
            video_basename = os.path.basename(video_path).replace(".mp4", "")

            if self.caption_format == "json":
                caption_path = os.path.join(self.caption_dir, f"{video_basename}.json")
                caption = self._load_json_caption(Path(caption_path))
            else:  # text format
                caption_path = os.path.join(self.caption_dir, f"{video_basename}.txt")
                caption = self._load_text(Path(caption_path))
            if self.target_prompt_suffix:
                caption = f"{caption.rstrip()} {self.target_prompt_suffix.strip()}".strip()

            data["video"] = video
            data["ai_caption"] = caption
            if self.target_mask_dir is not None or self.target_prompt_suffix:
                data["target_mask"] = self._load_target_mask(video_basename, frame_ids)

            _, _, h, w = video.shape

            data["fps"] = fps
            data["image_size"] = torch.tensor([h, w, h, w])
            data["num_frames"] = self.sequence_length
            data["padding_mask"] = torch.zeros(1, h, w)

            return data
        except Exception as e:
            self.num_failed_loads += 1
            log.warning(
                f"Failed to load video {self.video_paths[index]} (total failures: {self.num_failed_loads}): {e}\n"
                f"{traceback.format_exc()}",
                rank0_only=False,
            )
            # Randomly sample another video
            return self[np.random.randint(len(self.video_paths))]


def get_generic_dataloader(
    dataset: Dataset,
    batch_size: int = 1,
    sampler: Optional[Any] = None,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = False,
    prefetch_factor: Optional[int] = None,
    persistent_workers: bool = False,
    collate_fn: Optional[Callable] = None,
    **kwargs,  # Ignore extra arguments
) -> DataLoader:
    """Create DataLoader with commonly used parameters.

    Args:
        dataset: Dataset instance
        batch_size: Batch size
        sampler: Optional sampler for data loading
        num_workers: Number of worker processes
        pin_memory: Pin memory for CUDA transfer
        drop_last: Drop incomplete last batch
        prefetch_factor: Number of batches to prefetch per worker
        persistent_workers: Keep workers alive between epochs
        collate_fn: Custom collate function
        **kwargs: Extra arguments (ignored)

    Returns:
        Configured DataLoader
    """
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,  # False when using sampler
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
        collate_fn=collate_fn,
    )


def get_sampler(dataset) -> DistributedSampler:
    """Create a distributed sampler for the dataset."""
    return DistributedSampler(
        dataset,
        num_replicas=parallel_state.get_data_parallel_world_size(),
        rank=parallel_state.get_data_parallel_rank(),
        shuffle=True,
        seed=0,
    )


def get_train_val_dataloaders(
    dataset_path: str, val_percentage: float, seed: int, video_size: tuple[int, int] = (704, 1280)
):
    video_dir = os.path.join(dataset_path, "videos")
    if not os.path.exists(video_dir):
        log.debug(f"Dataset path {dataset_path} does not exist, returning empty dataloaders")
        return dict(), dict()
    video_paths = [os.path.join(video_dir, f) for f in os.listdir(video_dir) if f.endswith(".mp4")]
    random.seed(seed)
    random.shuffle(video_paths)

    cutoff = int(len(video_paths) * val_percentage)
    val_video_paths = video_paths[:cutoff]
    train_video_paths = video_paths[cutoff:]

    def get_dataset(video_paths):
        return L(VideoDataset)(
            video_paths=video_paths,
            num_frames=93,
            video_size=video_size,
            dataset_dir=dataset_path,
        )

    ipn_hand_train_dataset = get_dataset(train_video_paths)
    ipn_hand_val_dataset = get_dataset(val_video_paths)

    def get_dataloader(dataset):
        return L(get_generic_dataloader)(
            dataset=dataset,
            sampler=L(get_sampler)(dataset=dataset),
            batch_size=1,
            drop_last=True,
            num_workers=4,
            pin_memory=True,
        )

    return get_dataloader(ipn_hand_train_dataset), get_dataloader(ipn_hand_val_dataset)
