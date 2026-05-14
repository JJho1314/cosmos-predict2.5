# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cosmos-Predict2 post-training experiments on RoboInter-Data DROID lerobot subset.

Native lerobot DROID resolution is 320x180 @ 10 fps. We crop the height to 176
(divisible by 16) and feed `num_frames=33` clips, which keeps the pixel budget
~1/6 of the GR1 480 setup so 8x A6000/H100 can fit batch-per-GPU=1 with CP=1.

Two configs are exposed:
  * `predict2_video2world_training_2b_robointer_droid_sanity`  - 5 iter, 200 ep
  * `predict2_video2world_training_2b_robointer_droid`         - longer run
"""

from hydra.core.config_store import ConfigStore
from torch.utils.data import ConcatDataset

from cosmos_predict2._src.imaginaire.lazy_config import LazyCall as L
from cosmos_predict2._src.imaginaire.utils.checkpoint_db import get_checkpoint_path
from cosmos_predict2._src.predict2.datasets.local_datasets.dataset_video import (
    VideoDataset,
    get_generic_dataloader,
    get_sampler,
)
from cosmos_predict2.config import MODEL_CHECKPOINTS, ModelKey

DEFAULT_CHECKPOINT_2B = MODEL_CHECKPOINTS[ModelKey(post_trained=False)]

# DROID native is 320x180 @ 10 fps. 180 isn't a multiple of 16, so use 176.
_DATASET_DIR_SANITY = "/data/user/jhe724/workspace/datasets/robointer_droid_sanity"
_DATASET_DIR_FULL = "/data/user/jhe724/workspace/datasets/robointer_droid"

# droid_success is the 1280x720 @ 15 fps lerobot v3.0 release; we resize to
# 560x1008 (16-aligned, ~720p detail retained) and feed 33-frame clips.
_DATASET_DIR_DROID_SUCCESS = "/data/user/jhe724/workspace/datasets/droid_success_left"
_DATASET_DIR_DROID_SUCCESS_TRAIN = "/data/user/jhe724/workspace/datasets/droid_success_left_train"
_DATASET_DIR_DROID_SUCCESS_TEST = "/data/user/jhe724/workspace/datasets/droid_success_left_test"
_DATASET_DIR_DROID_SUCCESS_TRAIN_480 = "/data/user/jhe724/workspace/datasets/droid_success_left_train_480x864"
_DATASET_DIR_DROID_SUCCESS_TEST_480 = "/data/user/jhe724/workspace/datasets/droid_success_left_test_480x864"
_DATASET_DIR_DROID_FAILURE_ALL = "/data/user/jhe724/workspace/datasets/droid_failure_left_all"
_DATASET_DIR_DROID_FAILURE_CLEAN = "/data/user/jhe724/workspace/datasets/droid_failure_left_all_clean"
_DATASET_DIR_DROID_FAILURE_CLEAN_480 = "/data/user/jhe724/workspace/datasets/droid_failure_left_all_clean_480x864"
_DROID_VIDEO_SIZE_480 = (480, 864)
_DROID_SUCCESS_ITER_10000 = (
    "/data/user/jhe724/workspace/cosmos-predict2.5/outputs/droid_success/cosmos_predict_v2p5/"
    "video2world/2b_droid_success_560/checkpoints/iter_000010000"
)

_video_dataset_droid_sanity = L(VideoDataset)(
    dataset_dir=_DATASET_DIR_SANITY,
    num_frames=33,
    video_size=(176, 320),
)
_dataloader_train_droid_sanity = L(get_generic_dataloader)(
    dataset=_video_dataset_droid_sanity,
    sampler=L(get_sampler)(dataset=_video_dataset_droid_sanity),
    batch_size=1,
    drop_last=True,
    num_workers=2,
    pin_memory=True,
)

_video_dataset_droid_full = L(VideoDataset)(
    dataset_dir=_DATASET_DIR_FULL,
    num_frames=33,
    video_size=(176, 320),
)
_dataloader_train_droid_full = L(get_generic_dataloader)(
    dataset=_video_dataset_droid_full,
    sampler=L(get_sampler)(dataset=_video_dataset_droid_full),
    batch_size=1,
    drop_last=True,
    num_workers=4,
    pin_memory=True,
)


# Sanity run: load post-trained 2B, take 5 steps to verify env + data pipeline.
predict2_video2world_training_2b_robointer_droid_sanity = dict(
    defaults=[
        f"/experiment/{DEFAULT_CHECKPOINT_2B.experiment}",
        {"override /data_train": "mock"},
        {"override /data_val": "mock"},
        "_self_",
    ],
    dataloader_train=_dataloader_train_droid_sanity,
    checkpoint=dict(
        save_iter=5,  # save once at end so we exercise the save path
        # pyrefly: ignore  # missing-attribute
        load_path=get_checkpoint_path(DEFAULT_CHECKPOINT_2B.s3.uri),
        load_from_object_store=dict(enabled=False),
        save_to_object_store=dict(enabled=False),
    ),
    job=dict(
        project="cosmos_predict_v2p5",
        group="video2world",
        name="2b_robointer_droid_sanity",
        wandb_mode="online",
    ),
    optimizer=dict(lr=2 ** (-14.5), weight_decay=0.001),
    scheduler=dict(
        f_max=[0.5],
        f_min=[0.2],
        warm_up_steps=[100],
        cycle_lengths=[100000],
    ),
    trainer=dict(
        logging_iter=1,
        max_iter=5,
        straggler_detection=dict(enabled=False),
        callbacks=dict(
            heart_beat=dict(save_s3=False),
            iter_speed=dict(hit_thres=2, save_s3=False),
            device_monitor=dict(save_s3=False),
            every_n_sample_reg=dict(every_n=10, save_s3=False),
            every_n_sample_ema=dict(every_n=10, save_s3=False),
            wandb=dict(save_s3=False),
            wandb_10x=dict(save_s3=False),
            dataloader_speed=dict(save_s3=False),
        ),
    ),
    model_parallel=dict(context_parallel_size=1),
)


# Real run config (override max_iter / dataloader path via CLI when ready).
predict2_video2world_training_2b_robointer_droid = dict(
    defaults=[
        f"/experiment/{DEFAULT_CHECKPOINT_2B.experiment}",
        {"override /data_train": "mock"},
        {"override /data_val": "mock"},
        "_self_",
    ],
    dataloader_train=_dataloader_train_droid_full,
    checkpoint=dict(
        save_iter=500,
        # pyrefly: ignore  # missing-attribute
        load_path=get_checkpoint_path(DEFAULT_CHECKPOINT_2B.s3.uri),
        load_from_object_store=dict(enabled=False),
        save_to_object_store=dict(enabled=False),
    ),
    job=dict(
        project="cosmos_predict_v2p5",
        group="video2world",
        name="2b_robointer_droid",
        wandb_mode="online",
    ),
    optimizer=dict(lr=2 ** (-14.5), weight_decay=0.001),
    scheduler=dict(
        f_max=[0.5],
        f_min=[0.2],
        warm_up_steps=[1_000],
        cycle_lengths=[100000],
    ),
    trainer=dict(
        logging_iter=100,
        max_iter=10000,
        straggler_detection=dict(enabled=False),
        callbacks=dict(
            heart_beat=dict(save_s3=False),
            iter_speed=dict(hit_thres=100, save_s3=False),
            device_monitor=dict(save_s3=False),
            every_n_sample_reg=dict(every_n=500, save_s3=False),
            every_n_sample_ema=dict(every_n=500, save_s3=False),
            wandb=dict(save_s3=False),
            wandb_10x=dict(save_s3=False),
            dataloader_speed=dict(save_s3=False),
        ),
    ),
    model_parallel=dict(context_parallel_size=1),
)


_video_dataset_droid_success = L(VideoDataset)(
    dataset_dir=_DATASET_DIR_DROID_SUCCESS,
    num_frames=33,
    video_size=(560, 1008),  # ~720p, 16-aligned
)
_dataloader_train_droid_success = L(get_generic_dataloader)(
    dataset=_video_dataset_droid_success,
    sampler=L(get_sampler)(dataset=_video_dataset_droid_success),
    batch_size=1,
    drop_last=True,
    num_workers=4,
    pin_memory=True,
)

_video_dataset_droid_success_train = L(VideoDataset)(
    dataset_dir=_DATASET_DIR_DROID_SUCCESS_TRAIN_480,
    num_frames=33,
    video_size=_DROID_VIDEO_SIZE_480,
)
_video_dataset_droid_success_test = L(VideoDataset)(
    dataset_dir=_DATASET_DIR_DROID_SUCCESS_TEST_480,
    num_frames=33,
    video_size=_DROID_VIDEO_SIZE_480,
)
_video_dataset_droid_failure_all = L(VideoDataset)(
    dataset_dir=_DATASET_DIR_DROID_FAILURE_CLEAN_480,
    num_frames=33,
    video_size=_DROID_VIDEO_SIZE_480,
)
_video_dataset_droid_success_failure = L(ConcatDataset)(
    datasets=[
        _video_dataset_droid_success_train,
        _video_dataset_droid_failure_all,
    ],
)
_dataloader_train_droid_success_failure = L(get_generic_dataloader)(
    dataset=_video_dataset_droid_success_failure,
    sampler=L(get_sampler)(dataset=_video_dataset_droid_success_failure),
    batch_size=1,
    drop_last=True,
    num_workers=8,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2,
)
_dataloader_val_droid_success = L(get_generic_dataloader)(
    dataset=_video_dataset_droid_success_test,
    sampler=L(get_sampler)(dataset=_video_dataset_droid_success_test),
    batch_size=1,
    drop_last=True,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2,
)


# Real run on droid_success high-res lerobot v3 dataset.
predict2_video2world_training_2b_droid_success = dict(
    defaults=[
        f"/experiment/{DEFAULT_CHECKPOINT_2B.experiment}",
        {"override /data_train": "mock"},
        {"override /data_val": "mock"},
        "_self_",
    ],
    dataloader_train=_dataloader_train_droid_success,
    checkpoint=dict(
        save_iter=500,
        # pyrefly: ignore  # missing-attribute
        load_path=get_checkpoint_path(DEFAULT_CHECKPOINT_2B.s3.uri),
        load_from_object_store=dict(enabled=False),
        save_to_object_store=dict(enabled=False),
    ),
    job=dict(
        project="cosmos_predict_v2p5",
        group="video2world",
        name="2b_droid_success_560",
        wandb_mode="online",
    ),
    optimizer=dict(lr=2 ** (-14.5), weight_decay=0.001),
    scheduler=dict(
        f_max=[0.5],
        f_min=[0.2],
        warm_up_steps=[1_000],
        cycle_lengths=[100000],
    ),
    trainer=dict(
        logging_iter=100,
        max_iter=10000,
        straggler_detection=dict(enabled=False),
        callbacks=dict(
            heart_beat=dict(save_s3=False),
            iter_speed=dict(hit_thres=100, save_s3=False),
            device_monitor=dict(save_s3=False),
            every_n_sample_reg=dict(every_n=500, save_s3=False),
            every_n_sample_ema=dict(every_n=500, save_s3=False),
            wandb=dict(save_s3=False),
            wandb_10x=dict(save_s3=False),
            dataloader_speed=dict(save_s3=False),
        ),
    ),
    model_parallel=dict(context_parallel_size=1),
)


# Phase 2: resume from 10k checkpoint, enable grad_accum=4 (effective global
# batch 32), train 20k more steps to max_iter=30000.
predict2_video2world_training_2b_droid_success_phase2 = dict(
    defaults=[
        f"/experiment/{DEFAULT_CHECKPOINT_2B.experiment}",
        {"override /data_train": "mock"},
        {"override /data_val": "mock"},
        "_self_",
    ],
    dataloader_train=_dataloader_train_droid_success,
    checkpoint=dict(
        save_iter=1000,
        # pyrefly: ignore  # missing-attribute
        load_path=get_checkpoint_path(DEFAULT_CHECKPOINT_2B.s3.uri),
        load_from_object_store=dict(enabled=False),
        save_to_object_store=dict(enabled=False),
    ),
    job=dict(
        project="cosmos_predict_v2p5",
        group="video2world",
        # Distinct name → fresh output dir → starts from base checkpoint, not from phase 1 ckpt.
        name="2b_droid_success_560_accum4",
        wandb_mode="online",
    ),
    optimizer=dict(lr=2 ** (-14.5), weight_decay=0.001),
    scheduler=dict(
        f_max=[0.5],
        f_min=[0.2],
        warm_up_steps=[1_000],
        cycle_lengths=[100000],
    ),
    trainer=dict(
        logging_iter=100,
        max_iter=20000,
        grad_accum_iter=4,  # micro-batch 1 × 8 GPU × 4 accum = effective global batch 32
        straggler_detection=dict(enabled=False),
        callbacks=dict(
            heart_beat=dict(save_s3=False),
            iter_speed=dict(hit_thres=100, save_s3=False),
            device_monitor=dict(save_s3=False),
            every_n_sample_reg=dict(every_n=1000, save_s3=False),
            every_n_sample_ema=dict(every_n=1000, save_s3=False),
            wandb=dict(save_s3=False),
            wandb_10x=dict(save_s3=False),
            dataloader_speed=dict(save_s3=False),
        ),
    ),
    model_parallel=dict(context_parallel_size=1),
)


# Post-train from the base Cosmos 2B checkpoint on droid_success train split
# plus all droid_failure, and report held-out droid_success validation loss.
predict2_video2world_training_2b_droid_success_failure = dict(
    defaults=[
        f"/experiment/{DEFAULT_CHECKPOINT_2B.experiment}",
        {"override /data_train": "mock"},
        {"override /data_val": "mock"},
        "_self_",
    ],
    dataloader_train=_dataloader_train_droid_success_failure,
    dataloader_val=_dataloader_val_droid_success,
    checkpoint=dict(
        save_iter=1000,
        # pyrefly: ignore  # missing-attribute
        load_path=get_checkpoint_path(DEFAULT_CHECKPOINT_2B.s3.uri),
        load_from_object_store=dict(enabled=False),
        save_to_object_store=dict(enabled=False),
    ),
    job=dict(
        project="cosmos_predict_v2p5",
        group="video2world",
        name="2b_droid_success_failure_base_30k_480_clean_val1000_scratch",
        wandb_mode="online",
    ),
    optimizer=dict(lr=2 ** (-14.5), weight_decay=0.001),
    scheduler=dict(
        f_max=[0.5],
        f_min=[0.2],
        warm_up_steps=[1_000],
        cycle_lengths=[100000],
    ),
    trainer=dict(
        logging_iter=100,
        max_iter=10000,
        validation_iter=10000,
        run_validation=True,
        run_validation_on_start=False,
        max_val_iter=64,
        straggler_detection=dict(enabled=False),
        callbacks=dict(
            heart_beat=dict(save_s3=False),
            iter_speed=dict(hit_thres=100, save_s3=False),
            device_monitor=dict(save_s3=False),
            every_n_sample_reg=dict(every_n=1000, save_s3=False),
            every_n_sample_ema=dict(every_n=1000, save_s3=False),
            wandb=dict(save_s3=False),
            wandb_10x=dict(save_s3=False),
            dataloader_speed=dict(save_s3=False),
        ),
    ),
    model_parallel=dict(context_parallel_size=1),
)


cs = ConfigStore.instance()
for _item in [
    predict2_video2world_training_2b_robointer_droid_sanity,
    predict2_video2world_training_2b_robointer_droid,
    predict2_video2world_training_2b_droid_success,
    predict2_video2world_training_2b_droid_success_phase2,
    predict2_video2world_training_2b_droid_success_failure,
]:
    experiment_name = [name.lower() for name, value in globals().items() if value is _item][0]
    cs.store(
        group="experiment",
        package="_global_",
        name=experiment_name,
        node=_item,
    )
