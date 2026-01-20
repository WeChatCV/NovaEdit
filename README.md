# NOVA Video Editing

## Installation

```bash
pip install -r requirements.txt
```

## Model Download

Download the required pre-trained models:

```bash
python download_wan2.1.py
```

## Dataset Format
### Create Keyframes Videos
We assume all the videos for training and inference are 81 frames long.

For the keyframes video:
  - In training, extract the 0, 10, 20, 30, 40, 50, 60, 70, 80 frames from the source video to create the keyframes video. You can left the remaining frames black or blank.
  - In inference, you can set the 0 frame as the edited first frame via a image editing model. The 10, 20, 30, 40, 50, 60, 70, 80 frames can be also set as the edited frames from the source video (optional), and the remaining frames can be left black or blank.

### Training Dataset

Create a CSV file (e.g., `metadata.csv`) with the following columns:

```csv
video,prompt,vace_video,src_video
./gt/vid_00000.mp4,"",./keyframes/vid_00000.mp4,./masked/vid_00000.mp4
./gt/vid_00001.mp4,"",./keyframes/vid_00001.mp4,./masked/vid_00001.mp4
./gt/vid_00002.mp4,"",./keyframes/vid_00002.mp4,./masked/vid_00002.mp4
```

**Column descriptions:**
- `video`: Path to the ground truth target video
- `prompt`: Text prompt for generation (can be empty string)
- `vace_video`: Path to the VACE/cue frames video
- `src_video`: Path to the source/masked video

**Directory structure:**
```
dataset_path/
├── metadata.csv
├── gt/
│   ├── vid_00000.mp4
│   ├── vid_00001.mp4
│   └── ...
├── keyframes/
│   ├── vid_00000.mp4
│   ├── vid_00001.mp4
│   └── ...
└── masked/
    ├── vid_00000.mp4
    ├── vid_00001.mp4
    └── ...
```

### Inference Dataset

Create a CSV file (e.g., `metadata.csv`) with the following columns:

```csv
prompt,vace_video,src_video
"",./keyframes/vid_00000.mp4,./masked/vid_00000.mp4
"",./keyframes/vid_00001.mp4,./masked/vid_00001.mp4
"",./keyframes/vid_00002.mp4,./masked/vid_00002.mp4
```

**Note:** There is an example in `./example_videos/metadata.csv` for inference.

## Inference

### Single GPU Inference

```bash
python infer_nova.py \
  --dataset_path ./example_videos \
  --metadata_file_name metadata.csv \
  --ckpt_path /path/to/checkpoints/stepXXX.ckpt \
  --output_path ./inference_results \
  --text_encoder_path /path/to/models_t5_umt5-xxl-enc-bf16.pth \
  --image_encoder_path /path/to/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth \
  --vae_path /path/to/Wan2.1_VAE.pth \
  --dit_path /path/to/diffusion_pytorch_model.safetensors \
  --num_samples 5 \
  --num_inference_steps 50 \
  --num_frames 81 \
  --height 480 \
  --width 832 \
  --first_only
```

**Key arguments:**
- `--ckpt_path`: Path to trained checkpoint
- `--num_samples`: Number of samples to generate
- `--num_inference_steps`: Denoising steps (default 50)
- `--first_only`: Use only the first cue frame for all cue positions, **if you have set multiple cue frames in the keyframes video, just omit this flag**
- `--tiled`: Enable VAE tiling for GPU memory savings

### Multi-GPU Inference

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python infer_rank.py \
  --rank 0 \
  --world_size 4 \
  --dataset_path /path/to/your/dataset \
  --metadata_file_name metadata.csv \
  --ckpt_path /path/to/checkpoints/stepXXX.ckpt \
  --output_path ./inference_results \
  --text_encoder_path /path/to/models_t5_umt5-xxl-enc-bf16.pth \
  --image_encoder_path /path/to/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth \
  --vae_path /path/to/Wan2.1_VAE.pth \
  --dit_path /path/to/diffusion_pytorch_model.safetensors \
  --num_samples 10 \
  --num_frames 81 \
  --height 480 \
  --width 832
```

## Training

### 1. Data Processing

First, preprocess the dataset and encode videos to latents:

```bash
python train_nova.py \
  --task data_process \
  --dataset_path /path/to/your/dataset \
  --metadata_file_name metadata.csv \
  --batch_size 4 \
  --text_encoder_path /path/to/models_t5_umt5-xxl-enc-bf16.pth \
  --image_encoder_path /path/to/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth \
  --vae_path /path/to/Wan2.1_VAE.pth \
  --dit_path /path/to/diffusion_pytorch_model.safetensors \
  --num_frames 81 \
  --height 480 \
  --width 832
```

### 2. Training

After data processing, start training:

```bash
python train_nova.py \
  --task train \
  --dataset_path /path/to/your/dataset \
  --metadata_file_name metadata.csv \
  --output_path /path/to/save/checkpoints \
  --batch_size 3 \
  --dit_path /path/to/diffusion_pytorch_model.safetensors \
  --learning_rate 5e-5 \
  --max_epochs 500 \
  --steps_per_epoch 2000 \
  --use_gradient_checkpointing
```

**Key arguments:**
- `--dataset_path`: Path to your dataset directory
- `--output_path`: Directory to save checkpoints
- `--batch_size`: Batch size (adjust based on GPU memory)
- `--learning_rate`: Default 1e-5
- `--max_epochs`: Number of training epochs
- `--steps_per_epoch`: Steps per epoch
- `--resume_ckpt_path`: Resume from checkpoint (optional)

## Model Paths

The following placeholders should be replaced with actual paths to pre-trained models:

| Placeholder | Description |
|-------------|-------------|
| `text_encoder_path` | Text encoder model (models_t5_umt5-xxl-enc-bf16.pth) |
| `image_encoder_path` | Image encoder model (models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth) |
| `vae_path` | VAE model (Wan2.1_VAE.pth) |
| `dit_path` | DiT model (diffusion_pytorch_model.safetensors) |
| `ckpt_path` | Trained checkpoint (stepXXX.ckpt) |

## Output

Inference results are saved to:
- Combined comparison: `{output_path}/{name}_epoch{epoch_idx}.mp4`
- Individual results: `{output_path}/results/{name}_epoch{epoch_idx}.mp4`
