import os
import torch
import argparse
import pandas as pd
import numpy as np
import imageio
import torchvision
from torchvision.transforms import v2
from einops import rearrange
from PIL import Image
from diffsynth import WanVideoNovaPipeline, ModelManager
from tqdm import tqdm


# ==========================================
# 1. Dataset
# ==========================================
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

        self.metadata = pd.read_csv(metadata_path)

        if "video" in self.metadata.columns:
            self.video_paths = [
                os.path.join(base_path, p) if isinstance(p, str) else None
                for p in self.metadata["video"]
            ]
        else:
            self.video_paths = [None] * len(self.metadata)

        self.src_paths = [
            os.path.join(base_path, p) for p in self.metadata["src_video"]
        ]
        self.vace_paths = [
            os.path.join(base_path, p) for p in self.metadata["vace_video"]
        ]
        self.prompts = self.metadata["prompt"].fillna("").to_list()

        self.frame_process = v2.Compose(
            [
                v2.CenterCrop(size=(height, width)),
                v2.Resize(size=(height, width), antialias=True),
                v2.ToTensor(),
                v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def crop_and_resize(self, image):
        width, height = image.size
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
        if not os.path.exists(file_path):
            return None

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
        if file_path is None:
            return None
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
        video = self.load_video(vace_video_path)
        if video is None:
            return None
        indices = list(range(0, self.max_num_frames, 10))
        cues = []
        for idx in indices:
            if idx < video.shape[1]:
                frame = video[:, idx, :, :]
                frame_4x = frame.unsqueeze(1).repeat(1, 4, 1, 1)
                cues.append(frame_4x)
        if len(cues) == 0:
            return None
        return torch.stack(cues, dim=0)

    def read_raw_first_frame(self, file_path):
        """Load the first frame from disk, resize/crop, return as uint8."""
        try:
            reader = imageio.get_reader(file_path)
            frame = reader.get_data(0)
            reader.close()

            frame = Image.fromarray(frame)
            frame = self.crop_and_resize(frame)
            return np.array(frame)
        except Exception as e:
            print(f"Error reading first frame from {file_path}: {e}")
            return None

    def __getitem__(self, index):
        tgt_path = self.video_paths[index]
        src_path = self.src_paths[index]
        vace_path = self.vace_paths[index]
        prompt = self.prompts[index]

        tgt_video = self.load_video(tgt_path) if tgt_path else None
        src_video = self.load_video(src_path)
        cues_video = self.get_cue_frames(vace_path)

        first_frame_raw = self.read_raw_first_frame(vace_path)

        if src_video is None or cues_video is None or first_frame_raw is None:
            raise ValueError(
                f"Essential data load failed for index {index}. Check src or vace path."
            )

        name = os.path.basename(tgt_path) if tgt_path else os.path.basename(src_path)

        return {
            "text": prompt,
            "tgt_video": tgt_video,
            "src_video": src_video,
            "cues_video": cues_video,
            "first_frame": first_frame_raw,  # [H, W, 3] uint8
            "name": name,
        }

    def __len__(self):
        return len(self.src_paths)


# ==========================================
# 2. Helper Functions
# ==========================================
def encode_batch_cues(pipe, cues_input, tiler_kwargs, device):
    n_cues, c, t, h, w = cues_input.shape
    latents_list = []

    for i in range(n_cues):
        single_cue_input = (
            cues_input[i].unsqueeze(0).to(dtype=pipe.torch_dtype, device=device)
        )
        latent = pipe.encode_video(single_cue_input, **tiler_kwargs)[0]
        latents_list.append(latent)

    return torch.cat(latents_list, dim=1).to(device)


def tensor2video(frames):
    frames = rearrange(frames, "C T H W -> T H W C")
    frames = ((frames.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)
    return frames


# ==========================================
# 3. Main Infer Logic
# ==========================================
def infer(args):
    # --- 1. Load Model ---
    print(">>> Loading Models...")
    model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cuda")

    models_to_load = [args.text_encoder_path, args.vae_path]
    if args.image_encoder_path:
        models_to_load.append(args.image_encoder_path)

    if os.path.isfile(args.dit_path):
        models_to_load.append(args.dit_path)
    else:
        models_to_load.append(args.dit_path.split(","))

    model_manager.load_models(models_to_load)
    pipe = WanVideoNovaPipeline.from_model_manager(model_manager)

    # --- 2. Load Trained Weights ---
    print(f">>> Loading Trained Checkpoint from {args.ckpt_path} ...")
    state_dict = torch.load(args.ckpt_path, map_location="cpu")
    msg = pipe.dit.load_state_dict(state_dict, strict=True)
    print(f"Load result: {msg}")

    pipe.to("cuda")
    pipe.dit.eval()

    # --- 3. Prepare Dataset ---
    dataset = TextVideoDataset(
        args.dataset_path,
        os.path.join(args.dataset_path, args.metadata_file_name),
        max_num_frames=args.num_frames,
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
    )
    print(f">>> Dataset size: {len(dataset)}")

    # --- 4. Sample & Infer ---
    indices = list(range(len(dataset)))
    import random

    random.seed(42)
    sample_indices = random.sample(indices, min(args.num_samples, len(dataset)))

    tiler_kwargs = {
        "tiled": args.tiled,
        "tile_size": (args.tile_size_height, args.tile_size_width),
        "tile_stride": (args.tile_stride_height, args.tile_stride_width),
    }

    os.makedirs(args.output_path, exist_ok=True)
    results_dir = os.path.join(args.output_path, "results")
    os.makedirs(results_dir, exist_ok=True)

    for idx in tqdm(sample_indices, desc="Inferencing"):
        data = dataset[idx]
        name = data["name"]

        save_path = os.path.join(args.output_path, f"{name}_epoch{args.epoch_idx}.mp4")
        if os.path.exists(save_path):
            print(f"\n[INFO] Skipping {name}, file already exists.")
            continue

        prompt = data["text"]

        if data["tgt_video"] is not None:
            tgt_video_raw = (
                data["tgt_video"].unsqueeze(0).to(dtype=pipe.torch_dtype, device="cuda")
            )
        else:
            tgt_video_raw = None

        src_video_raw = (
            data["src_video"].unsqueeze(0).to(dtype=pipe.torch_dtype, device="cuda")
        )
        cues_video_raw = data["cues_video"]

        first_frame_arr = data["first_frame"]
        first_frame_pil = Image.fromarray(first_frame_arr)

        with torch.no_grad():
            # A. Encode Source
            src_latents = pipe.encode_video(src_video_raw, **tiler_kwargs).to(
                "cuda"
            )  # [1, 16, T_src, H, W]

            # B. Encode Cues
            cue_latents = (
                encode_batch_cues(pipe, cues_video_raw, tiler_kwargs, "cuda")
                .unsqueeze(0)
                .to("cuda")
            )  # [1, 16, T_cue, H, W]

            if args.first_only:
                # cue_latents[:, :, 0:1, :, :] shape is [1, 16, 1, H, W]
                cue_latents[:, :, 1:, :, :] = cue_latents[:, :, 0:1, :, :]
                print(
                    "[INFO] --first_only is enabled. All cue latents have been set to the first frame's latent."
                )

            # C. Encode Prompt
            prompt_emb = pipe.encode_prompt(prompt, positive=True)

            # D. Encode image features that will be stitched with source/cues later
            image_emb = pipe.encode_image(
                first_frame_pil, args.num_frames, args.height, args.width
            )

            if "clip_feature" in image_emb:
                image_emb["clip_feature"] = image_emb["clip_feature"].to(
                    device="cuda", dtype=pipe.torch_dtype
                )
            if "y" in image_emb:
                image_emb["y"] = image_emb["y"].to(
                    device="cuda", dtype=pipe.torch_dtype
                )

            # Reconstruct stitched guidance latents (source + cues + target)
            # 1. Collect latent shapes
            B = 1
            _, _, T_src, H_lat, W_lat = src_latents.shape
            T_cue = cue_latents.shape[2]

            latent_time = (args.num_frames - 1) // 4 + 1

            # 2. Build source guidance (mask=0)
            msk_src = torch.zeros(
                B, 4, T_src, H_lat, W_lat, device="cuda", dtype=pipe.torch_dtype
            )
            y_src = torch.cat([msk_src, src_latents], dim=1)

            # 3. Cue guidance (mask=1)
            msk_cue = torch.ones(
                B, 4, T_cue, H_lat, W_lat, device="cuda", dtype=pipe.torch_dtype
            )
            y_cue = torch.cat([msk_cue, cue_latents], dim=1)  # [1, 20, T_cue, H, W]

            # 4. Target guidance from encode_image output (truncate to generation length)
            y_tgt_orig = image_emb["y"]
            if y_tgt_orig.dim() == 6:
                y_tgt_orig = y_tgt_orig.squeeze(1)

            y_tgt = y_tgt_orig[:, :, :latent_time, :, :]

            # 5. Final stitching [Source, Cues, Target]
            y_final = torch.cat([y_src, y_cue, y_tgt], dim=2)

            image_emb["y"] = y_final

            # F. Prepare Noise & Scheduler
            pipe.scheduler.set_timesteps(args.num_inference_steps, shift=5.0)

            latents = torch.randn(
                (1, 16, latent_time, H_lat, W_lat),
                device="cuda",
                dtype=pipe.torch_dtype,
            )

            # G. Denoising Loop
            num_src_frames = src_latents.shape[2]
            num_cue_frames = cue_latents.shape[2]

            for timestep in tqdm(pipe.scheduler.timesteps, leave=False):
                timestep_t = timestep.unsqueeze(0).to(
                    dtype=pipe.torch_dtype, device="cuda"
                )

                model_input = torch.cat([src_latents, cue_latents, latents], dim=2)

                noise_pred = pipe.denoising_model()(
                    model_input,
                    timestep=timestep_t,
                    **prompt_emb,
                    **image_emb,
                    num_src_frames=num_src_frames,
                    num_cue_frames=num_cue_frames,
                )

                noise_pred_target = noise_pred[:, :, -latent_time:, ...]

                latents = pipe.scheduler.step(noise_pred_target, timestep, latents)

            # H. Decode
            generated_video_frames = pipe.decode_video(latents, **tiler_kwargs)

            # I. Post-process & Save
            gen_vid = tensor2video(generated_video_frames[0])
            src_vid = tensor2video(src_video_raw[0])

            first_frame_vis = np.array(first_frame_pil)
            first_frame_vis = np.tile(
                first_frame_vis[None, ...], (len(gen_vid), 1, 1, 1)
            )

            vis_list = [first_frame_vis, src_vid, gen_vid]

            if tgt_video_raw is not None:
                tgt_vid = tensor2video(tgt_video_raw[0])
                vis_list.append(tgt_vid)

            min_len = min([len(v) for v in vis_list])

            final_frames = []
            for i in range(min_len):
                frame = np.concatenate([v[i] for v in vis_list], axis=1)
                final_frames.append(frame)

            imageio.mimsave(save_path, final_frames, fps=24)
            print(f"Saved comparison: {save_path}")

            # Save individual result
            individual_save_path = os.path.join(
                results_dir, f"{name}_epoch{args.epoch_idx}.mp4"
            )
            imageio.mimsave(individual_save_path, gen_vid, fps=24)
            print(f"Saved individual result: {individual_save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Path Arguments
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--metadata_file_name", type=str, default="metadata.csv")
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="./inference_results")

    # Model Arguments
    parser.add_argument("--text_encoder_path", type=str, required=True)
    parser.add_argument("--vae_path", type=str, required=True)
    parser.add_argument("--image_encoder_path", type=str, default=None)
    parser.add_argument("--dit_path", type=str, required=True)

    # Config Arguments
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--epoch_idx", type=str, default="test")

    # Inference Args
    parser.add_argument("--num_inference_steps", type=int, default=50)

    # Tiling Args
    parser.add_argument("--tiled", action="store_true")
    parser.add_argument("--tile_size_height", type=int, default=34)
    parser.add_argument("--tile_size_width", type=int, default=34)
    parser.add_argument("--tile_stride_height", type=int, default=18)
    parser.add_argument("--tile_stride_width", type=int, default=16)

    parser.add_argument(
        "--first_only",
        action="store_true",
        help="Only use the first cue frame for all cue positions",
    )

    args = parser.parse_args()
    infer(args)
