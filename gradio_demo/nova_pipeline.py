import os
import sys
import torch
import numpy as np
import imageio
from PIL import Image
from torchvision import transforms
from torchvision.transforms import v2
from einops import rearrange
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from diffsynth import WanVideoNovaPipeline, ModelManager

class NovaPipeline:
    def __init__(
        self,
        ckpt_path: str,
        text_encoder_path: str,
        image_encoder_path: str,
        vae_path: str,
        dit_path: str,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        device: str = "cuda",
    ):
        self.height = height
        self.width = width
        self.num_frames = num_frames
        self.device = device

        print(">>> Loading Models...")
        model_manager = ModelManager(torch_dtype=torch.bfloat16, device=device)

        models_to_load = [text_encoder_path, vae_path]
        if image_encoder_path:
            models_to_load.append(image_encoder_path)

        if os.path.isfile(dit_path):
            models_to_load.append(dit_path)
        else:
            models_to_load.append(dit_path.split(","))

        model_manager.load_models(models_to_load)
        self.pipe = WanVideoNovaPipeline.from_model_manager(model_manager)

        print(f">>> Loading Trained Checkpoint from {ckpt_path} ...")
        state_dict = torch.load(ckpt_path, map_location="cpu")
        msg = self.pipe.dit.load_state_dict(state_dict, strict=True)
        print(f"Load result: {msg}")

        self.pipe.to(device)
        self.pipe.dit.eval()

        self.frame_process = v2.Compose(
            [
                v2.CenterCrop(size=(height, width)),
                v2.Resize(size=(height, width), antialias=True),
                v2.ToTensor(),
                v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def crop_and_resize(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        scale = max(self.width / width, self.height / height)
        image = transforms.functional.resize(
            image,
            (round(height * scale), round(width * scale)),
            interpolation=transforms.InterpolationMode.BILINEAR,
        )
        return v2.CenterCrop(size=(self.height, self.width))(image)

    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        image = self.crop_and_resize(image)
        image_tensor = self.frame_process(image)
        return image_tensor

    def load_video(self, video_path: str) -> torch.Tensor:
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        reader = imageio.get_reader(video_path)
        if reader.count_frames() < self.num_frames:
            reader.close()
            raise ValueError(f"Video has fewer than {self.num_frames} frames")

        frames = []
        for frame_id in range(self.num_frames):
            frame = reader.get_data(frame_id)
            frame = Image.fromarray(frame)
            frame = self.crop_and_resize(frame)
            frame_tensor = self.frame_process(frame)
            frames.append(frame_tensor)

        reader.close()

        frames = torch.stack(frames, dim=0)
        frames = rearrange(frames, "T C H W -> C T H W")
        return frames

    def create_cue_video_from_first_frame(
        self, first_frame_image: Image.Image
    ) -> torch.Tensor:
        first_frame_resized = self.crop_and_resize(first_frame_image)
        first_frame_np = np.array(first_frame_resized)

        blank_frame = np.zeros_like(first_frame_np)

        frames = [first_frame_np] + [blank_frame] * (self.num_frames - 1)

        frames_processed = []
        for frame in frames:
            frame_image = Image.fromarray(frame)
            frame_tensor = self.frame_process(frame_image)
            frames_processed.append(frame_tensor)

        frames_tensor = torch.stack(frames_processed, dim=0)
        frames_tensor = rearrange(frames_tensor, "T C H W -> C T H W")

        return frames_tensor

    def tensor2video(self, frames: torch.Tensor) -> np.ndarray:
        frames = rearrange(frames, "C T H W -> T H W C")
        frames = (
            ((frames.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)
        )
        return frames

    def generate(
        self,
        first_frame_image: Image.Image,
        source_video_path: str,
        num_inference_steps: int = 50,
        seed: int = 42,
        tiled: bool = True,
        tile_size_height: int = 34,
        tile_size_width: int = 34,
        tile_stride_height: int = 18,
        tile_stride_width: int = 16,
    ) -> np.ndarray:
        tiler_kwargs = {
            "tiled": tiled,
            "tile_size": (tile_size_height, tile_size_width),
            "tile_stride": (tile_stride_height, tile_stride_width),
        }

        src_video_raw = self.load_video(source_video_path)
        src_video_raw = src_video_raw.to(
            dtype=self.pipe.torch_dtype, device=self.device
        )

        cue_video_raw = self.create_cue_video_from_first_frame(first_frame_image)
        cue_video_raw = cue_video_raw.to(
            dtype=self.pipe.torch_dtype, device=self.device
        )

        first_frame_pil = self.crop_and_resize(first_frame_image)

        with torch.no_grad():
            src_video_raw_for_encode = src_video_raw.unsqueeze(0)
            src_latents = self.pipe.encode_video(
                src_video_raw_for_encode, **tiler_kwargs
            ).to("cuda")

            cue_video_raw_for_encode = cue_video_raw.unsqueeze(0)
            cue_latents = self.pipe.encode_video(
                cue_video_raw_for_encode, **tiler_kwargs
            ).to("cuda")

            cue_latents[:, :, 1:, :, :] = cue_latents[:, :, 0:1, :, :]
            print(
                "[INFO] --first_only enabled: All cue latents set to first frame's latent"
            )

            prompt_emb = self.pipe.encode_prompt("", positive=True)

            image_emb = self.pipe.encode_image(
                first_frame_pil, self.num_frames, self.height, self.width
            )

            if "clip_feature" in image_emb:
                image_emb["clip_feature"] = image_emb["clip_feature"].to(
                    device="cuda", dtype=self.pipe.torch_dtype
                )
            if "y" in image_emb:
                image_emb["y"] = image_emb["y"].to(
                    device="cuda", dtype=self.pipe.torch_dtype
                )

            B = 1
            _, _, T_src, H_lat, W_lat = src_latents.shape
            T_cue = cue_latents.shape[2]

            latent_time = (self.num_frames - 1) // 4 + 1

            msk_src = torch.zeros(
                B, 4, T_src, H_lat, W_lat, device="cuda", dtype=self.pipe.torch_dtype
            )
            y_src = torch.cat([msk_src, src_latents], dim=1)

            msk_cue = torch.ones(
                B, 4, T_cue, H_lat, W_lat, device="cuda", dtype=self.pipe.torch_dtype
            )
            y_cue = torch.cat([msk_cue, cue_latents], dim=1)

            y_tgt_orig = image_emb["y"]
            if y_tgt_orig.dim() == 6:
                y_tgt_orig = y_tgt_orig.squeeze(1)

            y_tgt = y_tgt_orig[:, :, :latent_time, :, :]

            y_final = torch.cat([y_src, y_cue, y_tgt], dim=2)

            image_emb["y"] = y_final

            self.pipe.scheduler.set_timesteps(num_inference_steps, shift=5.0)

            generator = torch.Generator(device="cuda")
            generator.manual_seed(seed)
            latents = torch.randn(
                (1, 16, latent_time, H_lat, W_lat),
                generator=generator,
                device="cuda",
                dtype=self.pipe.torch_dtype,
            )

            num_src_frames = src_latents.shape[2]
            num_cue_frames = cue_latents.shape[2]

            for timestep in tqdm(self.pipe.scheduler.timesteps, leave=False):
                timestep_t = timestep.unsqueeze(0).to(
                    dtype=self.pipe.torch_dtype, device="cuda"
                )

                model_input = torch.cat([src_latents, cue_latents, latents], dim=2)

                noise_pred = self.pipe.denoising_model()(
                    model_input,
                    timestep=timestep_t,
                    **prompt_emb,
                    **image_emb,
                    num_src_frames=num_src_frames,
                    num_cue_frames=num_cue_frames,
                )

                noise_pred_target = noise_pred[:, :, -latent_time:, ...]

                latents = self.pipe.scheduler.step(noise_pred_target, timestep, latents)

            generated_video_frames = self.pipe.decode_video(latents, **tiler_kwargs)

            gen_vid = self.tensor2video(generated_video_frames[0])

        return gen_vid

    def save_video(self, frames: np.ndarray, output_path: str, fps: int = 24):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        imageio.mimsave(output_path, frames, fps=fps)
        print(f"Saved video to: {output_path}")
