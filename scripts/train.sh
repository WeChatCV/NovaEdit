export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29508

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train_nova.py \
  --task train \
  --batch_size 3 \
  --dataset_path ../DiffSynth-Studio/data/pexels \
  --metadata_file_name file_list_clip_noise_zoom.csv \
  --output_path ./models/pexels5k_zoom_rope_i2v_random_onlyfirst \
  --dit_path "/models/PAI/Wan2.1-Fun-1.3B-InP/diffusion_pytorch_model.safetensors" \
  --steps_per_epoch 2000 \
  --max_epochs 500 \
  --learning_rate 5e-5 \
  --accumulate_grad_batches 1 \
  --use_gradient_checkpointing \
  --dataloader_num_workers 8 \