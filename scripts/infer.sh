#!/bin/bash

# TODO: Replace with your actual paths
# - DATASET_PATH: Path to your inference dataset directory
# - CKPT_PATH: Path to trained checkpoint directory
# - MODEL_PATH: Path to directory containing pre-trained models
# - OUTPUT_PATH: Path to save inference results

DATASET_PATH="/path/to/your/inference/dataset"
CKPT_PATH="/path/to/checkpoints"
MODEL_PATH="/path/to/models"
OUTPUT_PATH="./inference_results"

# Model files (relative to MODEL_PATH or absolute paths)
TEXT_ENCODER_PATH="${MODEL_PATH}/Wan-AI/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth"
IMAGE_ENCODER_PATH="${MODEL_PATH}/PAI/Wan2.1-Fun-1.3B-InP/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"
VAE_PATH="${MODEL_PATH}/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth"
DIT_PATH="${MODEL_PATH}/PAI/Wan2.1-Fun-1.3B-InP/diffusion_pytorch_model.safetensors"

# Checkpoint to use (set TARGET_STEP to your desired checkpoint step)
TARGET_STEP=1000
CKPT_FILE="${CKPT_PATH}/step${TARGET_STEP}.ckpt"

if [ ! -f "$CKPT_FILE" ]; then
    echo "Error: Checkpoint file not found: $CKPT_FILE"
    exit 1
fi

CUDA_VISIBLE_DEVICES=1 python infer_nova.py \
  --dataset_path "${DATASET_PATH}" \
  --metadata_file_name metadata.csv \
  --ckpt_path "${CKPT_FILE}" \
  --output_path "${OUTPUT_PATH}" \
  --text_encoder_path "${TEXT_ENCODER_PATH}" \
  --image_encoder_path "${IMAGE_ENCODER_PATH}" \
  --vae_path "${VAE_PATH}" \
  --dit_path "${DIT_PATH}" \
  --num_frames 81 \
  --height 480 \
  --width 832 \
  --num_samples 5 \
  --epoch_idx "step${TARGET_STEP}" \
  --tiled \
  --first_only

echo "Inference complete. Results saved to: ${OUTPUT_PATH}"
