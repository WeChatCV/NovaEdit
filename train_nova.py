import copy
import os
import re
import torch, os, imageio, argparse
from torchvision.transforms import v2
from einops import rearrange
import lightning as pl
import pandas as pd
from diffsynth import WanVideoNovaPipeline, ModelManager, load_state_dict
import torchvision
from PIL import Image
import numpy as np
import random
import json
import torch.nn as nn
import torch.nn.functional as F
import shutil


class TextVideoDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_path,
        metadata_path,
        max_num_frames=81,
        num_frames=81,
        frame_interval=1,
        height=480,
        width=832,
    ):
        self.height = height
        self.width = width
        self.max_num_frames = max_num_frames
        self.num_frames = num_frames
        self.frame_interval = frame_interval

        # CSV is expected to have columns: video, prompt, vace_video, src_video
        self.metadata = pd.read_csv(metadata_path)

        self.video_paths = [os.path.join(base_path, p) for p in self.metadata["video"]]
        self.src_paths = [
            os.path.join(base_path, p) for p in self.metadata["src_video"]
        ]
        self.vace_paths = [
            os.path.join(base_path, p) for p in self.metadata["vace_video"]
        ]
        self.prompts = self.metadata["prompt"].fillna("").to_list()  # Avoid NaN strings

        self.frame_process = v2.Compose(
            [
                v2.CenterCrop(size=(height, width)),
                v2.Resize(size=(height, width), antialias=True),
                v2.ToTensor(),
                v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def crop_and_resize(self, image):
        """
        Resize to cover the target dimensions, then center-crop to enforce exact size.
        """
        width, height = image.size
        # Compute a scale so at least one side matches target size before cropping
        scale = max(self.width / width, self.height / height)
        image = torchvision.transforms.functional.resize(
            image,
            (round(height * scale), round(width * scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
        )

        return v2.CenterCrop(size=(self.height, self.width))(image)

    def load_frames_using_imageio(
        self,
        file_path,
        max_num_frames,
        start_frame_id,
        interval,
        num_frames,
        frame_process,
    ):
        reader = imageio.get_reader(file_path)
        if (
            reader.count_frames() < max_num_frames
            or reader.count_frames() - 1 < start_frame_id + (num_frames - 1) * interval
        ):
            reader.close()
            return None

        frames = []
        first_frame = None
        for frame_id in range(num_frames):
            frame = reader.get_data(start_frame_id + frame_id * interval)
            frame = Image.fromarray(frame)
            frame = self.crop_and_resize(frame)
            if first_frame is None:
                first_frame = np.array(frame)
            frame = frame_process(frame)
            frames.append(frame)
        reader.close()

        frames = torch.stack(frames, dim=0)
        frames = rearrange(frames, "T C H W -> C T H W")

        return frames

    def load_video(self, file_path):
        start_frame_id = 0
        frames = self.load_frames_using_imageio(
            file_path,
            self.max_num_frames,
            start_frame_id,
            self.frame_interval,
            self.num_frames,
            self.frame_process,
        )
        return frames

    def get_cue_frames(self, vace_video_path):
        # 1. Load the full vace video tensor [C, T, H, W]
        # Assume the cue video is long enough to cover index 80
        video = self.load_video(vace_video_path)  # [C, T, H, W]
        if video is None:
            return None

        # 2. Extract uniformly sampled frames [0, 10, ..., 80]
        indices = list(range(0, self.max_num_frames, 10))  # [0, 10, ..., 80]

        cues = []
        for idx in indices:
            if idx < video.shape[1]:
                # Pull a single frame [C, H, W]
                frame = video[:, idx, :, :]
                # Repeat along time to create a 4-frame chunk for the VAE
                frame_4x = frame.unsqueeze(1).repeat(1, 4, 1, 1)
                cues.append(frame_4x)

        if len(cues) == 0:
            return None

        # Stack into batch form: [N_cues, C, 4, H, W]
        return torch.stack(cues, dim=0)

    def is_image(self, file_path):
        file_ext_name = file_path.split(".")[-1]
        if file_ext_name.lower() in ["jpg", "jpeg", "png", "webp"]:
            return True
        return False

    def load_image(self, file_path):
        frame = Image.open(file_path).convert("RGB")
        frame = self.crop_and_resize(frame)
        first_frame = frame
        frame = self.frame_process(frame)
        frame = rearrange(frame, "C H W -> C 1 H W")
        return frame

    def read_raw_first_frame(self, file_path):
        """Read the first frame, resize/crop to match video tensors, keep uint8."""
        try:
            reader = imageio.get_reader(file_path)
            frame = reader.get_data(0)  # first frame
            reader.close()

            frame = Image.fromarray(frame)
            frame = self.crop_and_resize(frame)  # enforce same framing as tensors
            # Convert to numpy array (H, W, 3) in uint8
            return np.array(frame)
        except Exception as e:
            print(f"Error reading first frame from {file_path}: {e}")
            return None

    def __getitem__(self, index):
        while True:
            try:
                tgt_path = self.video_paths[index]
                src_path = self.src_paths[index]
                vace_path = self.vace_paths[index]
                prompt = self.prompts[index]

                tgt_video = self.load_video(tgt_path)
                src_video = self.load_video(src_path)
                cues_video = self.get_cue_frames(vace_path)

                if tgt_video is None or src_video is None or cues_video is None:
                    raise ValueError("Video load failed")

                first_frame_raw = self.read_raw_first_frame(vace_path)
                if first_frame_raw is None:
                    raise ValueError("First frame load failed")

                return {
                    "text": prompt,
                    "tgt_video": tgt_video,
                    "src_video": src_video,
                    "cues_video": cues_video,
                    "first_frame": first_frame_raw,  # [H, W, 3] uint8
                    "path": tgt_path,
                }

            except Exception as e:
                print(f"Error loading {index}: {e}")
                index = (index + 1) % len(self.video_paths)

    def __len__(self):
        return len(self.src_paths)


class LightningModelForDataProcess(pl.LightningModule):
    def __init__(
        self,
        text_encoder_path,
        vae_path,
        image_encoder_path=None,
        tiled=False,
        tile_size=(34, 34),
        tile_stride=(18, 16),
    ):
        super().__init__()
        model_path = [text_encoder_path, vae_path]
        if image_encoder_path is not None:
            model_path.append(image_encoder_path)
        model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
        model_manager.load_models(model_path)
        self.pipe = WanVideoNovaPipeline.from_model_manager(model_manager)

        self.tiler_kwargs = {
            "tiled": tiled,
            "tile_size": tile_size,
            "tile_stride": tile_stride,
        }

    def encode_batch_cues(self, cues_input):
        # cues_input: [N_cues, C, 4, H, W]; VAE expects [B, C, T, H, W]
        n_cues, c, t, h, w = cues_input.shape
        latents_list = []

        for i in range(n_cues):
            # [1, C, 4, H, W]
            single_cue_input = (
                cues_input[i]
                .unsqueeze(0)
                .to(dtype=self.pipe.torch_dtype, device=self.device)
            )
            # Encode -> [1, 16, T_lat, H_lat, W_lat]
            latent = self.pipe.encode_video(single_cue_input, **self.tiler_kwargs)[0]
            # The WanVideo VAE returns the central latent frame (shape [16, 1, H, W])
            latents_list.append(latent)

        # Stitch along time: [16, N_cues, H_lat, W_lat]
        return torch.cat(latents_list, dim=1)

    def test_step(self, batch, batch_idx):
        # Ensure the pipeline lives on the same device as the Lightning module
        self.pipe.to(self.device)
        self.pipe.device = self.device

        curr_batch_size = len(batch["path"])

        # Process each sample in the batch individually
        for i in range(curr_batch_size):
            text = batch["text"][i]
            path = batch["path"][i]
            pth_path = path + ".tensors.pth"

            try:
                # Explicitly move each sliced tensor to the module's device
                tgt_video_single = batch["tgt_video"][i : i + 1].to(
                    device=self.device, dtype=self.pipe.torch_dtype
                )
                src_video_single = batch["src_video"][i : i + 1].to(
                    device=self.device, dtype=self.pipe.torch_dtype
                )

                # cues_video remains on CPU until encode_batch_cues pushes chunks to GPU
                cues_video_single = batch["cues_video"][i]

                # 1. Encode prompt
                prompt_emb = self.pipe.encode_prompt(text)

                # 2. Encode target & source latents
                tgt_latents = self.pipe.encode_video(
                    tgt_video_single, **self.tiler_kwargs
                )[0]
                src_latents = self.pipe.encode_video(
                    src_video_single, **self.tiler_kwargs
                )[0]

                # 3. Encode cues
                cue_latents = self.encode_batch_cues(cues_video_single)

                # 4. Encode guidance image (first frame)
                if "first_frame" in batch:
                    first_frame_arr = batch["first_frame"][i].cpu().numpy()
                    first_frame = Image.fromarray(first_frame_arr)

                    # Use the already-moved tensor to read dimensions
                    _, _, num_frames, height, width = tgt_video_single.shape
                    image_emb = self.pipe.encode_image(
                        first_frame, num_frames, height, width
                    )
                else:
                    image_emb = {}

                # 5. Persist tensors on CPU to save VRAM
                data = {
                    "tgt_latents": tgt_latents.cpu(),
                    "src_latents": src_latents.cpu(),
                    "cue_latents": cue_latents.cpu(),
                    "prompt_emb": {k: v.cpu() for k, v in prompt_emb.items()},
                    "image_emb": {k: v.cpu() for k, v in image_emb.items()},
                }
                torch.save(data, pth_path)
                print(f"Processed and saved: {pth_path}")

            except Exception as e:
                print(f"Failed to process {pth_path}: {e}")


class Camera(object):
    def __init__(self, c2w):
        c2w_mat = np.array(c2w).reshape(4, 4)
        self.c2w_mat = c2w_mat
        self.w2c_mat = np.linalg.inv(c2w_mat)


class TensorDataset(torch.utils.data.Dataset):
    def __init__(self, base_path, metadata_path, steps_per_epoch):
        # 1. Read CSV
        self.metadata = pd.read_csv(metadata_path)

        # 2. Build tensor cache paths aligned with the "video" column (target video)
        self.path = [
            os.path.join(base_path, file_name) for file_name in self.metadata["video"]
        ]

        print(f"Metadata contains {len(self.path)} videos.")

        # 3. Filter out missing cache files
        self.valid_indices = []
        for idx, p in enumerate(self.path):
            if os.path.exists(p + ".tensors.pth"):
                self.valid_indices.append(idx)
            else:
                # Optional: warn about missing files
                pass

        print(f"Found {len(self.valid_indices)} cached tensor files.")
        assert len(self.valid_indices) > 0, "No tensor files found!"

        self.steps_per_epoch = steps_per_epoch

    def parse_matrix(self, matrix_str):
        rows = matrix_str.strip().split("] [")
        matrix = []
        for row in rows:
            row = row.replace("[", "").replace("]", "")
            matrix.append(list(map(float, row.split())))
        return np.array(matrix)

    def get_relative_pose(self, cam_params):
        abs_w2cs = [cam_param.w2c_mat for cam_param in cam_params]
        abs_c2ws = [cam_param.c2w_mat for cam_param in cam_params]
        target_cam_c2w = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        )
        abs2rel = target_cam_c2w @ abs_w2cs[0]
        ret_poses = [
            target_cam_c2w,
        ] + [abs2rel @ abs_c2w for abs_c2w in abs_c2ws[1:]]
        ret_poses = np.array(ret_poses, dtype=np.float32)
        return ret_poses

    def __getitem__(self, index):
        while True:
            try:
                # Use valid_indices to map into real dataset entries
                rand_idx = torch.randint(0, len(self.valid_indices), (1,))[0]
                real_idx = self.valid_indices[
                    (rand_idx + index) % len(self.valid_indices)
                ]

                path_tgt = self.path[real_idx] + ".tensors.pth"

                # Load dictionary of cached tensors
                data_dict = torch.load(path_tgt, weights_only=True, map_location="cpu")

                # Latents are stored either as [16, T, H, W] or with a batch dimension
                src_latents = data_dict["src_latents"]
                cue_latents = data_dict["cue_latents"]

                if random.random() < 0.5:
                    if cue_latents.dim() == 5:
                        cue_latents[:, :, 1:] = cue_latents[:, :, 0:1]
                    else:
                        cue_latents[:, 1:] = cue_latents[:, 0:1]

                tgt_latents = data_dict["tgt_latents"]

                # Remove batch dimension if present
                if src_latents.dim() == 5:
                    src_latents = src_latents.squeeze(0)
                if cue_latents.dim() == 5:
                    cue_latents = cue_latents.squeeze(0)
                if tgt_latents.dim() == 5:
                    tgt_latents = tgt_latents.squeeze(0)

                # Concatenate in time order: [Source, Cues, Target]
                full_latents = torch.cat([src_latents, cue_latents, tgt_latents], dim=1)

                data = {}
                data["latents"] = full_latents

                # Keep the individual components so the training step knows segment lengths
                data["source_latents"] = src_latents
                data["cue_latents"] = cue_latents
                data["target_latents"] = tgt_latents  # Clean Target

                data["prompt_emb"] = data_dict["prompt_emb"]
                data["image_emb"] = data_dict.get("image_emb", {})
                data["camera"] = torch.zeros(1)  # Placeholder if not used

                return data

            except Exception as e:
                print(f"ERROR WHEN LOADING {path_tgt}: {e}")
                # Try the next entry
                index += 1

    def __len__(self):
        return self.steps_per_epoch


class LightningModelForTrain(pl.LightningModule):
    def __init__(
        self,
        dit_path,
        learning_rate=1e-5,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        resume_ckpt_path=None,
    ):
        super().__init__()
        model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
        if os.path.isfile(dit_path):
            model_manager.load_models([dit_path])
        else:
            dit_path = dit_path.split(",")
            model_manager.load_models([dit_path])

        self.pipe = WanVideoNovaPipeline.from_model_manager(model_manager)
        self.pipe.scheduler.set_timesteps(1000, training=True)

        if resume_ckpt_path is not None:
            state_dict = torch.load(resume_ckpt_path, map_location="cpu")
            self.pipe.dit.load_state_dict(state_dict, strict=True)

        self.freeze_parameters()
        for name, module in self.pipe.denoising_model().named_modules():
            if any(
                keyword in name for keyword in ["cam_encoder", "projector", "self_attn"]
            ):
                print(f"Trainable: {name}")
                for param in module.parameters():
                    param.requires_grad = True

        trainable_params = 0
        seen_params = set()
        for name, module in self.pipe.denoising_model().named_modules():
            for param in module.parameters():
                if param.requires_grad and param not in seen_params:
                    trainable_params += param.numel()
                    seen_params.add(param)
        print(f"Total number of trainable parameters: {trainable_params}")

        self.learning_rate = learning_rate
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload

    def freeze_parameters(self):
        # Freeze parameters
        self.pipe.requires_grad_(False)
        self.pipe.eval()
        self.pipe.denoising_model().train()

    def training_step(self, batch, batch_idx):
        # Data
        # Assume dataloader prepares tensors as:
        # source: [B, 16, T_src, H, W]
        # cues:   [B, 16, T_cue, H, W]
        # target: [B, 16, T_tgt, H, W]

        src_latents = batch["source_latents"].to(self.device)
        cue_latents = batch["cue_latents"].to(self.device)
        tgt_latents_clean = batch["target_latents"].to(self.device)

        num_src_frames = src_latents.shape[2]
        num_cue_frames = cue_latents.shape[2]

        prompt_emb = batch["prompt_emb"]
        prompt_emb["context"] = prompt_emb["context"].to(self.device)

        # cam_emb = batch["camera"].to(self.device)

        B, _, T_src, H, W = src_latents.shape
        T_cue = cue_latents.shape[2]
        T_tgt = tgt_latents_clean.shape[2]
        # Preprocess y: collapse any redundant batch dim [B, 1, 20, T, H, W] -> [B, 20, T, H, W]
        # batch["image_emb"]["y"] stores the I2V guidance saved during data_process
        y_orig = batch["image_emb"]["y"]
        if y_orig.dim() == 6:
            y_orig = y_orig.squeeze(1)

        # 3. Build guidance for SRC segment (20 channels: mask + latents)
        # Source uses mask=0 and the real src_latents
        msk_src = torch.zeros(
            B, 4, T_src, H, W, device=self.device, dtype=self.pipe.torch_dtype
        )
        y_src = torch.cat([msk_src, src_latents], dim=1)

        # 4. Build guidance for CUE segment (mask=1 with cue latents)
        msk_cue = torch.ones(
            B, 4, T_cue, H, W, device=self.device, dtype=self.pipe.torch_dtype
        )
        y_cue = torch.cat([msk_cue, cue_latents], dim=1)

        # 5. Align target guidance (reuse original y; crop if longer than T_tgt)
        y_tgt = y_orig[:, :, :T_tgt, :, :]

        # 6. Stitch final guidance along time: [Source, Cues, Target]
        y_final = torch.cat([y_src, y_cue, y_tgt], dim=2)

        # Update the dictionary with the stitched tensor
        image_emb = batch["image_emb"]
        image_emb["y"] = y_final

        if "clip_feature" in image_emb:
            image_emb["clip_feature"] = image_emb["clip_feature"].to(self.device)
        if "y" in image_emb:
            image_emb["y"] = image_emb["y"].to(self.device)

        # Loss
        self.pipe.device = self.device
        noise = torch.randn_like(tgt_latents_clean)
        timestep_id = torch.randint(0, self.pipe.scheduler.num_train_timesteps, (1,))
        timestep = self.pipe.scheduler.timesteps[timestep_id].to(
            dtype=self.pipe.torch_dtype, device=self.pipe.device
        )

        noisy_tgt = self.pipe.scheduler.add_noise(tgt_latents_clean, noise, timestep)
        model_input = torch.cat([src_latents, cue_latents, noisy_tgt], dim=2)
        extra_input = self.pipe.prepare_extra_input(model_input)

        # Compute loss
        noise_pred = self.pipe.denoising_model()(
            model_input,
            timestep=timestep,
            **prompt_emb,
            **extra_input,
            **image_emb,
            num_src_frames=num_src_frames,
            num_cue_frames=num_cue_frames,
            use_gradient_checkpointing=self.use_gradient_checkpointing,
            use_gradient_checkpointing_offload=self.use_gradient_checkpointing_offload,
        )

        tgt_len = tgt_latents_clean.shape[2]
        noise_pred_tgt = noise_pred[:, :, -tgt_len:, ...]
        training_target = self.pipe.scheduler.training_target(
            tgt_latents_clean, noise, timestep
        )

        loss = F.mse_loss(noise_pred_tgt.float(), training_target.float())
        loss = loss * self.pipe.scheduler.training_weight(timestep)

        # Record log
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        trainable_modules = filter(
            lambda p: p.requires_grad, self.pipe.denoising_model().parameters()
        )
        optimizer = torch.optim.AdamW(trainable_modules, lr=self.learning_rate)
        return optimizer

    def on_save_checkpoint(self, checkpoint):
        checkpoint_dir = self.trainer.checkpoint_callback.dirpath

        if checkpoint_dir is None:
            # Fall back to default_root_dir when Lightning has not created dirpath yet
            checkpoint_dir = os.path.join(self.trainer.default_root_dir, "checkpoints")

        os.makedirs(checkpoint_dir, exist_ok=True)
        print(f"Checkpoint directory: {checkpoint_dir}")
        current_step = self.global_step
        print(f"Current step: {current_step}")

        checkpoint.clear()
        trainable_param_names = list(
            filter(
                lambda named_param: named_param[1].requires_grad,
                self.pipe.denoising_model().named_parameters(),
            )
        )
        trainable_param_names = set(
            [named_param[0] for named_param in trainable_param_names]
        )
        state_dict = self.pipe.denoising_model().state_dict()
        torch.save(state_dict, os.path.join(checkpoint_dir, f"step{current_step}.ckpt"))


def parse_args():
    parser = argparse.ArgumentParser(description="Train NOVA")
    parser.add_argument(
        "--task",
        type=str,
        default="data_process",
        required=True,
        choices=["data_process", "train"],
        help="Task. `data_process` or `train`.",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        required=True,
        help="The path of the Dataset.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./",
        help="Path to save the model.",
    )
    parser.add_argument(
        "--text_encoder_path",
        type=str,
        default=None,
        help="Path of text encoder.",
    )
    parser.add_argument(
        "--image_encoder_path",
        type=str,
        default=None,
        help="Path of image encoder.",
    )
    parser.add_argument(
        "--vae_path",
        type=str,
        default=None,
        help="Path of VAE.",
    )
    parser.add_argument(
        "--dit_path",
        type=str,
        default=None,
        help="Path of DiT.",
    )
    parser.add_argument(
        "--tiled",
        default=False,
        action="store_true",
        help="Whether enable tile encode in VAE. This option can reduce VRAM required.",
    )
    parser.add_argument(
        "--tile_size_height",
        type=int,
        default=34,
        help="Tile size (height) in VAE.",
    )
    parser.add_argument(
        "--tile_size_width",
        type=int,
        default=34,
        help="Tile size (width) in VAE.",
    )
    parser.add_argument(
        "--tile_stride_height",
        type=int,
        default=18,
        help="Tile stride (height) in VAE.",
    )
    parser.add_argument(
        "--tile_stride_width",
        type=int,
        default=16,
        help="Tile stride (width) in VAE.",
    )
    parser.add_argument(
        "--steps_per_epoch",
        type=int,
        default=500,
        help="Number of steps per epoch.",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=81,
        help="Number of frames.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Image height.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=832,
        help="Image width.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=4,
        help="Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Learning rate.",
    )
    parser.add_argument(
        "--accumulate_grad_batches",
        type=int,
        default=1,
        help="The number of batches in gradient accumulation.",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=1,
        help="Number of epochs.",
    )
    parser.add_argument(
        "--training_strategy",
        type=str,
        default="deepspeed_stage_1",
        choices=["auto", "deepspeed_stage_1", "deepspeed_stage_2", "deepspeed_stage_3"],
        help="Training strategy",
    )
    parser.add_argument(
        "--use_gradient_checkpointing",
        default=False,
        action="store_true",
        help="Whether to use gradient checkpointing.",
    )
    parser.add_argument(
        "--use_gradient_checkpointing_offload",
        default=False,
        action="store_true",
        help="Whether to use gradient checkpointing offload.",
    )
    parser.add_argument(
        "--use_swanlab",
        default=False,
        action="store_true",
        help="Whether to use SwanLab logger.",
    )
    parser.add_argument(
        "--swanlab_mode",
        default=None,
        help="SwanLab mode (cloud or local).",
    )
    parser.add_argument(
        "--metadata_file_name",
        type=str,
        default="metadata.csv",
    )
    parser.add_argument(
        "--resume_ckpt_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        required=True,  # Must be explicitly set by the user
        help="Batch size for data processing or training.",
    )
    args = parser.parse_args()
    return args


def data_process(args):
    dataset = TextVideoDataset(
        args.dataset_path,
        os.path.join(args.dataset_path, args.metadata_file_name),
        max_num_frames=args.num_frames,
        frame_interval=1,
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.dataloader_num_workers,
    )
    model = LightningModelForDataProcess(
        text_encoder_path=args.text_encoder_path,
        image_encoder_path=args.image_encoder_path,
        vae_path=args.vae_path,
        tiled=args.tiled,
        tile_size=(args.tile_size_height, args.tile_size_width),
        tile_stride=(args.tile_stride_height, args.tile_stride_width),
    )
    trainer = pl.Trainer(
        accelerator="gpu",
        devices="auto",
        default_root_dir=args.output_path,
    )
    trainer.test(model, dataloader)


def train(args):
    dataset = TensorDataset(
        args.dataset_path,
        os.path.join(args.dataset_path, args.metadata_file_name),
        steps_per_epoch=args.steps_per_epoch,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=True,
        batch_size=args.batch_size,
        num_workers=args.dataloader_num_workers,
    )
    model = LightningModelForTrain(
        dit_path=args.dit_path,
        learning_rate=args.learning_rate,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        resume_ckpt_path=args.resume_ckpt_path,
    )

    if args.use_swanlab:
        from swanlab.integration.pytorch_lightning import SwanLabLogger

        swanlab_config = {"UPPERFRAMEWORK": "DiffSynth-Studio"}
        swanlab_config.update(vars(args))
        swanlab_logger = SwanLabLogger(
            project="wan",
            name="wan",
            config=swanlab_config,
            mode=args.swanlab_mode,
            logdir=os.path.join(args.output_path, "swanlog"),
        )
        logger = [swanlab_logger]
    else:
        logger = None
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu",
        devices="auto",
        precision="bf16",
        strategy=args.training_strategy,
        default_root_dir=args.output_path,
        accumulate_grad_batches=args.accumulate_grad_batches,
        callbacks=[pl.pytorch.callbacks.ModelCheckpoint(save_top_k=-1)],
        logger=logger,
    )
    trainer.fit(model, dataloader)


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(os.path.join(args.output_path, "checkpoints"), exist_ok=True)
    if args.task == "data_process":
        data_process(args)
    elif args.task == "train":
        train(args)
