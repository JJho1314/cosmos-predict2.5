#!/usr/bin/env bash

#SBATCH --job-name=tavid-sweep
#SBATCH --partition=acd_u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --time=01:30:00
#SBATCH --output=/data/user/jhe724/workspace/cosmos-predict2.5/slurm-tavid-sweep-%j.out
#SBATCH --error=/data/user/jhe724/workspace/cosmos-predict2.5/slurm-tavid-sweep-%j.err

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

: "${DATASET:?DATASET is required}"
: "${CKPT:?CKPT is required}"
: "${OUT:?OUT is required}"

NUM_SAMPLES=${NUM_SAMPLES:-1}
SKIP_SAMPLES=${SKIP_SAMPLES:-0}
SAMPLE_INDEX_OFFSET=${SAMPLE_INDEX_OFFSET:-0}
NUM_STEPS=${NUM_STEPS:-35}
GUIDANCE=${GUIDANCE:-3.0}
SEED=${SEED:-2026053105}
FPS=${FPS:-8}
MAX_BATCHES=${MAX_BATCHES:-80}
STANDALONE_ONLY=${STANDALONE_ONLY:-0}

export DROID_SUCCESS_V21_TAVID_DIR=$DATASET
export DROID_SUCCESS_V21_TAVID_VAL_DIR=$DATASET
export DROID_SUCCESS_V21_TAVID_NUM_FRAMES=${DROID_SUCCESS_V21_TAVID_NUM_FRAMES:-49}
export DROID_SUCCESS_V21_TAVID_FRAME_STRIDES=${DROID_SUCCESS_V21_TAVID_FRAME_STRIDES:-2}
export DROID_SUCCESS_V21_TAVID_FRAME_START_POLICY=${DROID_SUCCESS_V21_TAVID_FRAME_START_POLICY:-range_start}

mkdir -p "$OUT"
exec > >(tee "$OUT/run.log") 2>&1

echo "job_id=${SLURM_JOB_ID:-none}"
echo "node=${SLURMD_NODENAME:-unknown}"
echo "dataset=$DATASET"
echo "checkpoint=$CKPT"
echo "output=$OUT"
echo "skip_samples=$SKIP_SAMPLES"
echo "sample_index_offset=$SAMPLE_INDEX_OFFSET"
echo "seed=$SEED"
echo "num_steps=$NUM_STEPS"

nvidia-smi -L
python -c "import torch; print('torch', torch.__version__, 'cuda count:', torch.cuda.device_count())"

extra_args=()
if [[ "$STANDALONE_ONLY" == "1" ]]; then
  extra_args+=(--standalone-only)
fi

torchrun --standalone --nproc_per_node=1 scripts/generate_tavid_mask_samples.py \
  --config cosmos_predict2/_src/predict2/configs/video2world/config.py \
  --checkpoint "$CKPT" \
  --output-dir "$OUT" \
  --num-samples "$NUM_SAMPLES" \
  --skip-samples "$SKIP_SAMPLES" \
  --sample-index-offset "$SAMPLE_INDEX_OFFSET" \
  --num-steps "$NUM_STEPS" \
  --guidance "$GUIDANCE" \
  --seed "$SEED" \
  --fps "$FPS" \
  --max-batches "$MAX_BATCHES" \
  "${extra_args[@]}" \
  -- experiment=predict2_video2world_training_2b_droid_success_v21_tavid_mask \
  dataloader_train.batch_size=1 \
  dataloader_train.num_workers=2 \
  dataloader_train.dataset.target_mask_dropout_prob=0.0 \
  dataloader_train.dataset.target_mask_default_to_zero=False \
  trainer.grad_accum_iter=1 \
  trainer.run_validation=False
