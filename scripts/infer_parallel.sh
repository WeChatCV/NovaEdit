#!/bin/bash
set -euo pipefail

# -------------------- Checkpoint Configuration --------------------
TARGET_STEP=${TARGET_STEP:-252}
CKPT_FILE="./models/pexels5k_zoom_rope_i2v_random_onlyfirst/lightning_logs/version_1/checkpoints/step${TARGET_STEP}.ckpt"
EPOCH_IDX=${EPOCH_IDX:-"step${TARGET_STEP}"}

if [ ! -f "$CKPT_FILE" ]; then
  echo "❌ Error: checkpoint $CKPT_FILE not found"
  exit 1
fi

# -------------------- Configurable Parameters --------------------
DATASET_PATH=${DATASET_PATH:-"../Grounded-SAM-2/outputs_flux"}
METADATA_FILE_NAME=${METADATA_FILE_NAME:-"video_pairs.csv"}
OUTPUT_PATH=${OUTPUT_PATH:-"./inference_results/data_flux_parallel"}
NUM_SAMPLES=${NUM_SAMPLES:-265}
NUM_INFERENCE_STEPS=${NUM_INFERENCE_STEPS:-50}

TEXT_ENCODER_PATH=${TEXT_ENCODER_PATH:-"/models/Wan-AI/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth"}
IMAGE_ENCODER_PATH=${IMAGE_ENCODER_PATH:-"/models/PAI/Wan2.1-Fun-1.3B-InP/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"}
VAE_PATH=${VAE_PATH:-"/models/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth"}
DIT_PATH=${DIT_PATH:-"/models/PAI/Wan2.1-Fun-1.3B-InP/diffusion_pytorch_model.safetensors"}

NUM_FRAMES=${NUM_FRAMES:-81}
HEIGHT=${HEIGHT:-480}
WIDTH=${WIDTH:-832}

NUM_GPUS=${NUM_GPUS:-8}
GPU_IDS_STRING=${GPU_IDS_STRING:-"0 1 2 3 4 5 6 7"}
IFS=' ' read -r -a GPU_IDS <<< "$GPU_IDS_STRING"

if [ "${#GPU_IDS[@]}" -ne "$NUM_GPUS" ]; then
  echo "❌ Error: NUM_GPUS=$NUM_GPUS but provided ${#GPU_IDS[@]} GPU IDs (GPU_IDS_STRING=\"$GPU_IDS_STRING\")."
  exit 1
fi

LOG_DIR=${LOG_DIR:-"./logs/infer_rank"}
mkdir -p "$LOG_DIR"

REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)
PYTHON_SCRIPT="$REPO_ROOT/infer_rank.py"

if [ ! -f "$PYTHON_SCRIPT" ]; then
  echo "❌ Error: infer_rank.py not found at $PYTHON_SCRIPT"
  exit 1
fi

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_ID="rank_infer_${TIMESTAMP}"
RUN_LOG_DIR="$LOG_DIR/$RUN_ID"
mkdir -p "$RUN_LOG_DIR"

echo "🚀 Launching parallel inference"
echo "  Checkpoint : $CKPT_FILE"
echo "  Epoch tag  : $EPOCH_IDX"
echo "  Dataset    : $DATASET_PATH"
echo "  Metadata   : $METADATA_FILE_NAME"
echo "  Output dir : $OUTPUT_PATH"
echo "  Samples    : $NUM_SAMPLES (per-rank auto split)"
echo "  Logs       : $RUN_LOG_DIR"

declare -a PIDS

for (( rank=0; rank<NUM_GPUS; rank++ )); do
  GPU_ID=${GPU_IDS[$rank]}
  LOG_FILE="$RUN_LOG_DIR/rank${rank}.log"

  echo "[Rank $rank] -> GPU $GPU_ID | log: $LOG_FILE"

  CUDA_VISIBLE_DEVICES=$GPU_ID nohup /mnt/shanghai3cephs/tianlinpan/miniconda3/envs/diffnew/bin/python "$PYTHON_SCRIPT" \
    --dataset_path "$DATASET_PATH" \
    --metadata_file_name "$METADATA_FILE_NAME" \
    --ckpt_path "$CKPT_FILE" \
    --output_path "$OUTPUT_PATH" \
    --text_encoder_path "$TEXT_ENCODER_PATH" \
    --image_encoder_path "$IMAGE_ENCODER_PATH" \
    --vae_path "$VAE_PATH" \
    --dit_path "$DIT_PATH" \
    --num_frames "$NUM_FRAMES" \
    --height "$HEIGHT" \
    --width "$WIDTH" \
    --num_samples "$NUM_SAMPLES" \
    --epoch_idx "$EPOCH_IDX" \
    --num_inference_steps "$NUM_INFERENCE_STEPS" \
    --tiled \
    --first_only \
    --rank "$rank" \
    --world_size "$NUM_GPUS" \
    > "$LOG_FILE" 2>&1 &

  PIDS+=("$!")
done

echo
echo "💡 Monitor logs with: tail -n 20 -f $RUN_LOG_DIR/rank*.log"
echo

trap 'echo "Stopping all ranks"; kill ${PIDS[*]} 2>/dev/null || true' INT TERM

FAIL=0
for pid in "${PIDS[@]}"; do
  if ! wait "$pid"; then
    echo "❌ Process $pid failed"
    FAIL=1
  fi
done

if [ "$FAIL" -eq 0 ]; then
  echo "✅ All ranks finished successfully. Results under $OUTPUT_PATH"
  echo "Logs saved to $RUN_LOG_DIR"
else
  echo "⚠️ Some ranks failed. Check logs in $RUN_LOG_DIR"
fi
