import argparse
import os
from typing import List

import imageio
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from diffsynth import WanVideoNovaPipeline, ModelManager

try:
    from infer_nova import TextVideoDataset, encode_batch_cues, tensor2video
except ImportError as exc:
    raise ImportError(
        "infer_rank.py requires infer_nova.py to be in the same directory"
    ) from exc


def build_rank_indices(total_samples: int, world_size: int) -> List[List[int]]:
    """Evenly distribute sample indices across all ranks (contiguous chunks)."""
    if world_size <= 0:
        raise ValueError("world_size must be positive")

    splits: List[List[int]] = []
    base = total_samples // world_size
    remainder = total_samples % world_size

    start = 0
    for rank in range(world_size):
        length = base + (1 if rank < remainder else 0)
        end = start + length
        splits.append(list(range(start, end)))
        start = end

    return splits


def infer(args):
    if args.rank < 0 or args.rank >= args.world_size:
        raise ValueError(
            f"Rank {args.rank} is invalid for world_size {args.world_size}."
        )

    if not torch.cuda.is_available():
        raise EnvironmentError("CUDA device is required for infer_rank.py")

    device_index = args.device if args.device is not None else 0
    if device_index >= torch.cuda.device_count():
        raise ValueError(
            f"Requested local device index {device_index}, "
            f"but only {torch.cuda.device_count()} devices are visible."
        )

    torch.cuda.set_device(device_index)
    device = torch.device("cuda", device_index)

    print(f"[Rank {args.rank}] Using device {device} | world_size={args.world_size}")

    print("[Rank {}] >>> Loading Models...".format(args.rank))
    model_manager = ModelManager(torch_dtype=torch.bfloat16, device=device)

    models_to_load = [args.text_encoder_path, args.vae_path]
    if args.image_encoder_path:
        models_to_load.append(args.image_encoder_path)

    if os.path.isfile(args.dit_path):
        models_to_load.append(args.dit_path)
    else:
        models_to_load.extend(args.dit_path.split(","))

    model_manager.load_models(models_to_load)
    pipe = WanVideoNovaPipeline.from_model_manager(model_manager)

    print(
        "[Rank {}] >>> Loading Trained Checkpoint from {} ...".format(
            args.rank, args.ckpt_path
        )
    )
    state_dict = torch.load(args.ckpt_path, map_location="cpu")
    msg = pipe.dit.load_state_dict(state_dict, strict=True)
    print(f"[Rank {args.rank}] Load result: {msg}")

    pipe.to(device)
    pipe.dit.eval()

    dataset = TextVideoDataset(
        args.dataset_path,
        os.path.join(args.dataset_path, args.metadata_file_name),
        max_num_frames=args.num_frames,
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
    )

    dataset_size = len(dataset)
    if dataset_size == 0:
        print(f"[Rank {args.rank}] Dataset is empty. Nothing to process.")
        return

    if args.num_samples <= 0:
        effective_num_samples = dataset_size
    else:
        effective_num_samples = min(args.num_samples, dataset_size)

    print(
        f"[Rank {args.rank}] Dataset size={dataset_size}, total target samples={effective_num_samples}"
    )

    all_indices = list(range(effective_num_samples))
    split_indices = build_rank_indices(effective_num_samples, args.world_size)
    rank_indices = split_indices[args.rank]

    if len(rank_indices) == 0:
        print(
            f"[Rank {args.rank}] Assigned 0 samples (dataset smaller than world size). Exiting."
        )
        return

    print(
        f"[Rank {args.rank}] Assigned {len(rank_indices)} samples: first={rank_indices[0]}, last={rank_indices[-1]}"
    )

    tiler_kwargs = {
        "tiled": args.tiled,
        "tile_size": (args.tile_size_height, args.tile_size_width),
        "tile_stride": (args.tile_stride_height, args.tile_stride_width),
    }

    os.makedirs(args.output_path, exist_ok=True)
    results_dir = os.path.join(args.output_path, "results")
    os.makedirs(results_dir, exist_ok=True)

    for idx in tqdm(rank_indices, desc=f"Rank {args.rank} Inferencing"):
        data = dataset[idx]
        name = data["name"]

        save_path = os.path.join(args.output_path, f"{name}_epoch{args.epoch_idx}.mp4")
        if os.path.exists(save_path):
            print(f"\n[Rank {args.rank}] [INFO] Skipping {name}, file already exists.")
            continue

        prompt = data["text"]

        tgt_video_raw = (
            data["tgt_video"].unsqueeze(0).to(dtype=pipe.torch_dtype, device=device)
            if data["tgt_video"] is not None
            else None
        )

        src_video_raw = (
            data["src_video"].unsqueeze(0).to(dtype=pipe.torch_dtype, device=device)
        )
        cues_video_raw = data["cues_video"]

        first_frame_arr = data["first_frame"]
        first_frame_pil = Image.fromarray(first_frame_arr)

        with torch.no_grad():
            src_latents = pipe.encode_video(src_video_raw, **tiler_kwargs).to(device)

            cue_latents = (
                encode_batch_cues(pipe, cues_video_raw, tiler_kwargs, device)
                .unsqueeze(0)
                .to(device)
            )

            if args.first_only:
                cue_latents[:, :, 1:, :, :] = cue_latents[:, :, 0:1, :, :]
                print(
                    f"[Rank {args.rank}] --first_only enabled. Reusing first cue frame for all positions."
                )

            prompt_emb = pipe.encode_prompt(prompt, positive=True)

            image_emb = pipe.encode_image(
                first_frame_pil, args.num_frames, args.height, args.width
            )

            if "clip_feature" in image_emb:
                image_emb["clip_feature"] = image_emb["clip_feature"].to(
                    device=device, dtype=pipe.torch_dtype
                )
            if "y" in image_emb:
                image_emb["y"] = image_emb["y"].to(
                    device=device, dtype=pipe.torch_dtype
                )

            B = 1
            _, _, T_src, H_lat, W_lat = src_latents.shape
            T_cue = cue_latents.shape[2]
            latent_time = (args.num_frames - 1) // 4 + 1

            msk_src = torch.zeros(
                B, 4, T_src, H_lat, W_lat, device=device, dtype=pipe.torch_dtype
            )
            y_src = torch.cat([msk_src, src_latents], dim=1)

            msk_cue = torch.ones(
                B, 4, T_cue, H_lat, W_lat, device=device, dtype=pipe.torch_dtype
            )
            y_cue = torch.cat([msk_cue, cue_latents], dim=1)

            y_tgt_orig = image_emb["y"]
            if y_tgt_orig.dim() == 6:
                y_tgt_orig = y_tgt_orig.squeeze(1)
            y_tgt = y_tgt_orig[:, :, :latent_time, :, :]

            y_final = torch.cat([y_src, y_cue, y_tgt], dim=2)
            image_emb["y"] = y_final

            pipe.scheduler.set_timesteps(args.num_inference_steps, shift=5.0)

            latents = torch.randn(
                (1, 16, latent_time, H_lat, W_lat),
                device=device,
                dtype=pipe.torch_dtype,
            )

            num_src_frames = src_latents.shape[2]
            num_cue_frames = cue_latents.shape[2]

            for timestep in tqdm(
                pipe.scheduler.timesteps,
                leave=False,
                desc=f"Rank {args.rank} Denoising",
            ):
                timestep_t = timestep.unsqueeze(0).to(
                    dtype=pipe.torch_dtype, device=device
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

            generated_video_frames = pipe.decode_video(latents, **tiler_kwargs)

            gen_vid = tensor2video(generated_video_frames[0])
            src_vid = tensor2video(src_video_raw[0])

        first_frame_vis = np.tile(first_frame_arr[None, ...], (len(gen_vid), 1, 1, 1))

        vis_list = [first_frame_vis, src_vid, gen_vid]

        if tgt_video_raw is not None:
            tgt_vid = tensor2video(tgt_video_raw[0])
            vis_list.append(tgt_vid)

        min_len = min(len(v) for v in vis_list)

        final_frames = []
        for i in range(min_len):
            frame = vis_list[0][i]
            for vid in vis_list[1:]:
                frame = np.concatenate([frame, vid[i]], axis=1)
            final_frames.append(frame)

        imageio.mimsave(save_path, final_frames, fps=24)
        print(f"[Rank {args.rank}] Saved comparison: {save_path}")

        individual_save_path = os.path.join(
            results_dir, f"{name}_epoch{args.epoch_idx}.mp4"
        )
        imageio.mimsave(individual_save_path, gen_vid, fps=24)
        print(f"[Rank {args.rank}] Saved individual result: {individual_save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--metadata_file_name", type=str, default="metadata.csv")
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="./inference_results")
    parser.add_argument("--text_encoder_path", type=str, required=True)
    parser.add_argument("--vae_path", type=str, required=True)
    parser.add_argument("--image_encoder_path", type=str, default=None)
    parser.add_argument("--dit_path", type=str, required=True)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--num_samples", type=int, default=-1)
    parser.add_argument("--epoch_idx", type=str, default="test")
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--tiled", action="store_true")
    parser.add_argument("--tile_size_height", type=int, default=34)
    parser.add_argument("--tile_size_width", type=int, default=34)
    parser.add_argument("--tile_stride_height", type=int, default=18)
    parser.add_argument("--tile_stride_width", type=int, default=16)
    parser.add_argument("--first_only", action="store_true")
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world_size", type=int, required=True)
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="Local CUDA device index (default uses 0).",
    )

    args = parser.parse_args()
    infer(args)
