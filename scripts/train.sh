export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29508

# TODO: Replace with your actual paths
# - DATASET_PATH: Path to your training dataset directory
# - MODEL_PATH: Path to directory containing pre-trained models
# - OUTPUT_PATH: Path to save training checkpoints

DATASET_PATH="/path/to/your/dataset"
MODEL_PATH="/path/to/models"
OUTPUT_PATH="./models/your_experiment"

# Model files (relative to MODEL_PATH or absolute paths)
TEXT_ENCODER_PATH="${MODEL_PATH}/Wan-AI/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth"
IMAGE_ENCODER_PATH="${MODEL_PATH}/PAI/Wan2.1-Fun-1.3B-InP/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"
VAE_PATH="${MODEL_PATH}/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth"
DIT_PATH="${MODEL_PATH}/PAI/Wan2.1-Fun-1.3B-InP/diffusion_pytorch_model.safetensors"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train_nova.py \
  --task train \
  --batch_size 3 \
  --dataset_path "${DATASET_PATH}" \
  --metadata_file_name metadata.csv \
  --output_path "${OUTPUT_PATH}" \
  --dit_path "${DIT_PATH}" \
  --steps_per_epoch 2000 \
  --max_epochs 500 \
  --learning_rate 5e-5 \
  --accumulate_grad_batches 1 \
  --use_gradient_checkpointing \
  --dataloader_num_workers 8
