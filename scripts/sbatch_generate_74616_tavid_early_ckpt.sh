#!/usr/bin/env bash

#SBATCH --job-name=tavid-74616-5k
#SBATCH --partition=acd_u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --time=01:30:00
#SBATCH --output=/data/user/jhe724/workspace/cosmos-predict2.5/slurm-tavid-74616-5k-%j.out
#SBATCH --error=/data/user/jhe724/workspace/cosmos-predict2.5/slurm-tavid-74616-5k-%j.err

set -euo pipefail
cd /data/user/jhe724/workspace/cosmos-predict2.5

module load gcc/11.5 cuda/12.6 nccl/2.25 2>/dev/null || true

VENV=/data/user/jhe724/workspace/cosmos-predict2.5/.venv
export VIRTUAL_ENV=$VENV
export PATH=/data/apps/gcc/11.5/bin:$VENV/bin:$PATH
unset PYTHONHOME

export CC=/data/apps/gcc/11.5/bin/gcc
export CXX=/data/apps/gcc/11.5/bin/g++

NV_LIB=$VENV/lib/python3.10/site-packages/nvidia
export LD_LIBRARY_PATH="$NV_LIB/cudnn/lib:$NV_LIB/cuda_runtime/lib:$NV_LIB/cuda_nvrtc/lib:$NV_LIB/cublas/lib:$NV_LIB/cusparse/lib:$NV_LIB/cusolver/lib:$NV_LIB/cufft/lib:$NV_LIB/curand/lib:$NV_LIB/nccl/lib:$NV_LIB/nvjitlink/lib:${LD_LIBRARY_PATH:-}"

export COSMOS_CHECKPOINTS_DIR=/data/user/jhe724/workspace/weights
export HF_HUB_OFFLINE=1
export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG=WARN
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DATASET=${DATASET:-/data/user/jhe724/workspace/cosmos-predict2.5/outputs/tavid_generation_runs/robointer_74616_yellow_carrot_dataset}
CKPT=${CKPT:-/data/user/jhe724/workspace/cosmos-predict2.5/outputs/droid_success_v21_tavid_mask_left_right_split/cosmos_predict_v2p5/video2world/2b_droid_success_v21_tavid_mask_480_lr_split_val1k_49f_s234_actionstart_tgtfix_bs2accum4_20k_val1000/checkpoints/iter_000005000}
OUT=${OUT:-/data/user/jhe724/workspace/cosmos-predict2.5/outputs/tavid_generation_runs/robointer_74616_tavid_tgtfix_iter005k_yellow_carrot_sink_mask_index_49f_35step}
NUM_STEPS=${NUM_STEPS:-35}
GUIDANCE=${GUIDANCE:-3.0}
SEED=${SEED:-2026053105}
FPS=${FPS:-8}

export DROID_SUCCESS_V21_TAVID_DIR=$DATASET
export DROID_SUCCESS_V21_TAVID_VAL_DIR=$DATASET
export DROID_SUCCESS_V21_TAVID_NUM_FRAMES=${DROID_SUCCESS_V21_TAVID_NUM_FRAMES:-49}
export DROID_SUCCESS_V21_TAVID_FRAME_STRIDES=${DROID_SUCCESS_V21_TAVID_FRAME_STRIDES:-2}
export DROID_SUCCESS_V21_TAVID_FRAME_START_POLICY=${DROID_SUCCESS_V21_TAVID_FRAME_START_POLICY:-range_start}

mkdir -p "$OUT"

nvidia-smi -L
python -c "import torch; print('torch', torch.__version__, 'cuda count:', torch.cuda.device_count())"
printf 'checkpoint=%s\noutput=%s\ndataset=%s\n' "$CKPT" "$OUT" "$DATASET"
cat "$DATASET/metas/74616_exterior_image_1_left.txt"

torchrun --standalone --nproc_per_node=1 scripts/generate_tavid_mask_samples.py \
  --config cosmos_predict2/_src/predict2/configs/video2world/config.py \
  --checkpoint "$CKPT" \
  --output-dir "$OUT" \
  --num-samples 1 \
  --skip-samples 0 \
  --sample-index-offset 0 \
  --num-steps "$NUM_STEPS" \
  --guidance "$GUIDANCE" \
  --seed "$SEED" \
  --fps "$FPS" \
  --max-batches 8 \
  --standalone-only \
  -- experiment=predict2_video2world_training_2b_droid_success_v21_tavid_mask \
  dataloader_train.batch_size=1 \
  dataloader_train.num_workers=2 \
  dataloader_train.dataset.target_mask_dropout_prob=0.0 \
  dataloader_train.dataset.target_mask_default_to_zero=False \
  trainer.grad_accum_iter=1 \
  trainer.run_validation=False
