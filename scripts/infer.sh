#!/bin/bash

TARGET_STEPS="252 420 588 756"
#TARGET_STEPS="924 1092 1260 1428"

# METADATA_FILE_NAME="infer_data_masked.csv"
# NUM_SAMPLES=14

METADATA_FILE_NAME="video_pairs.csv"
NUM_SAMPLES=265

for i in $TARGET_STEPS; do
    
    echo "----------------------------------------"
    echo "正在处理 Step: $i ..."

    CKPT_FILE="../ReCamMaster/models/pexels5k_zoom_rope_i2v_random_onlyfirst/lightning_logs/version_1/checkpoints/step${i}.ckpt"

    if [ ! -f "$CKPT_FILE" ]; then
        echo "❌ 错误: 找不到文件 $CKPT_FILE，跳过..."
        continue
    fi

    CUDA_VISIBLE_DEVICES=1 python infer_nova.py \
      --dataset_path ../Grounded-SAM-2/outputs_flux \
      --metadata_file_name "$METADATA_FILE_NAME" \
      --ckpt_path "$CKPT_FILE" \
      --output_path "./inference_results/data_qwen_lora" \
      --text_encoder_path "/models/Wan-AI/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth" \
      --image_encoder_path "/models/PAI/Wan2.1-Fun-1.3B-InP/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" \
      --vae_path "/models/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth" \
      --dit_path "/models/PAI/Wan2.1-Fun-1.3B-InP/diffusion_pytorch_model.safetensors" \
      --num_frames 81 \
      --height 480 \
      --width 832 \
      --num_samples "$NUM_SAMPLES" \
      --epoch_idx "step${i}" \
      --tiled \
      --first_only
      
    echo "✅ Step $i 完成。"

done