import os
import gradio as gr
import cv2
import numpy as np
from PIL import Image
import torch
import time
import random
import sys
from huggingface_hub import snapshot_download
from decord import VideoReader, cpu
from moviepy.editor import ImageSequenceClip

from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor

try:
    from diffusers import FluxKontextPipeline
except ImportError:
    print(
        "Warning: FluxKontextPipeline not found in diffusers. Please ensure it is installed or available."
    )
    FluxKontextPipeline = None

os.makedirs("./SAM2-Video-Predictor/checkpoints/", exist_ok=True)
os.makedirs("./outputs", exist_ok=True)


def download_sam2():
    snapshot_download(
        repo_id="facebook/sam2-hiera-large",
        local_dir="./SAM2-Video-Predictor/checkpoints/",
    )
    print("Download sam2 completed")


download_sam2()

COLOR_PALETTE = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (255, 128, 0),
    (128, 0, 255),
    (0, 128, 255),
    (128, 255, 0),
]

W = 1024
H = W
device = "cuda" if torch.cuda.is_available() else "cpu"

# --- Flux Global Setup ---
FLUX_MODEL_PATH = "../../flux/flux-kontext"
flux_pipe = None


def load_flux_model():
    global flux_pipe
    if flux_pipe is None:
        if FluxKontextPipeline is None:
            raise ImportError("FluxKontextPipeline class is not available.")

        print(f"Initializing Flux... Loading model from '{FLUX_MODEL_PATH}'")
        try:
            flux_pipe = FluxKontextPipeline.from_pretrained(
                FLUX_MODEL_PATH, torch_dtype=torch.bfloat16
            )
            flux_pipe.to("cuda")
            print("✅ Flux Model loaded successfully!")
        except Exception as e:
            print(f"❌ Failed to load Flux model: {e}")
            return None
    return flux_pipe


def edit_image_with_flux(image_path, prompt):
    if not image_path:
        gr.Warning("Please export the BBox image first.")
        return None

    if not prompt:
        gr.Warning("Please enter an edit prompt.")
        return None

    pipe = load_flux_model()
    if pipe is None:
        gr.Warning("Failed to load Flux model. Check console for errors.")
        return None

    try:
        print(f"🚀 Processing image edit with prompt: {prompt}")
        input_image = Image.open(image_path).convert("RGB")

        result = pipe(
            image=input_image,
            height=input_image.height,
            width=input_image.width,
            prompt=prompt,
            guidance_scale=2.5,
        ).images[0]

        timestamp = int(time.time())
        output_path = f"./outputs/edited_{timestamp}.png"
        result.save(output_path)
        print(f"✨ Done! Saved as: {output_path}")
        return output_path

    except Exception as e:
        error_msg = f"❌ An error occurred during processing: {e}"
        print(error_msg)
        gr.Warning(error_msg)
        return None


# --- SAM2 Setup ---


def get_sam2_predictors():
    sam2_checkpoint = "./SAM2-Video-Predictor/checkpoints/sam2_hiera_large.pt"
    config = "sam2_hiera_l.yaml"

    video_predictor = build_sam2_video_predictor(config, sam2_checkpoint, device=device)
    model = build_sam2(config, sam2_checkpoint, device=device)
    model.image_size = 1024
    image_predictor = SAM2ImagePredictor(sam_model=model)

    return image_predictor, video_predictor


def get_video_info(video_path, video_state):
    video_state["input_points"] = []
    video_state["scaled_points"] = []
    video_state["input_labels"] = []
    video_state["frame_idx"] = 0
    vr = VideoReader(video_path, ctx=cpu(0))
    first_frame = vr[0].asnumpy()
    del vr

    if first_frame.shape[0] > first_frame.shape[1]:
        W_ = W
        H_ = int(W_ * first_frame.shape[0] / first_frame.shape[1])
    else:
        H_ = H
        W_ = int(H_ * first_frame.shape[1] / first_frame.shape[0])

    first_frame = cv2.resize(first_frame, (W_, H_))
    video_state["origin_images"] = np.expand_dims(first_frame, axis=0)
    video_state["inference_state"] = None
    video_state["video_path"] = video_path
    video_state["masks"] = None
    video_state["painted_images"] = None
    image = Image.fromarray(first_frame)
    return image


def segment_frame(evt: gr.SelectData, label, video_state):
    if video_state["origin_images"] is None:
        gr.Warning('Please click "Extract First Frame" first.')
        return None
    x, y = evt.index
    new_point = [x, y]
    label_value = 1 if label == "Positive" else 0

    video_state["input_points"].append(new_point)
    video_state["input_labels"].append(label_value)
    height, width = video_state["origin_images"][0].shape[0:2]

    scaled_points = []
    for pt in video_state["input_points"]:
        sx = pt[0] / width
        sy = pt[1] / height
        scaled_points.append([sx, sy])
    video_state["scaled_points"] = scaled_points

    image_predictor.set_image(video_state["origin_images"][0])
    mask, _, _ = image_predictor.predict(
        point_coords=video_state["scaled_points"],
        point_labels=video_state["input_labels"],
        multimask_output=False,
        normalize_coords=False,
    )

    mask = np.squeeze(mask)
    mask = cv2.resize(mask, (width, height))
    mask_to_save = mask.copy()
    mask = mask[:, :, None]

    color = (
        np.array(COLOR_PALETTE[int(time.time()) % len(COLOR_PALETTE)], dtype=np.float32)
        / 255.0
    )
    color = color[None, None, :]
    org_image = video_state["origin_images"][0].astype(np.float32) / 255.0
    painted_image = (1 - mask * 0.5) * org_image + mask * 0.5 * color
    painted_image = np.uint8(np.clip(painted_image * 255, 0, 255))

    video_state["painted_images"] = np.expand_dims(painted_image, axis=0)
    video_state["masks"] = np.expand_dims(mask_to_save, axis=0)

    for i in range(len(video_state["input_points"])):
        point = video_state["input_points"][i]
        c = (0, 0, 255) if video_state["input_labels"][i] == 0 else (255, 0, 0)
        cv2.circle(painted_image, point, radius=3, color=c, thickness=-1)

    return Image.fromarray(painted_image)


def export_bbox_image(video_state):
    if video_state["origin_images"] is None or video_state["masks"] is None:
        gr.Warning("Please complete target segmentation on the first frame first.")
        return None

    img = video_state["origin_images"][0].copy()
    mask = video_state["masks"][0]

    binary_mask = (mask > 0).astype(np.uint8)
    if np.sum(binary_mask) == 0:
        gr.Warning("No mask detected. Please click to segment an object first.")
        return None

    x, y, w, h = cv2.boundingRect(binary_mask)

    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 10)

    timestamp = int(time.time())
    save_path = f"./outputs/first_frame_bbox_{timestamp}.png"

    cv2.imwrite(save_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    gr.Info(f"BBox image saved to {save_path}")
    return save_path


def clear_clicks(video_state):
    video_state["input_points"] = []
    video_state["input_labels"] = []
    video_state["scaled_points"] = []
    video_state["inference_state"] = None
    video_state["masks"] = None
    video_state["painted_images"] = None
    return (
        Image.fromarray(video_state["origin_images"][0])
        if video_state["origin_images"] is not None
        else None
    )


def track_video(n_frames, video_state):
    if video_state["origin_images"] is None or video_state["masks"] is None:
        gr.Warning("Please complete target segmentation on the first frame first.")
        return None

    obj_id = video_state["obj_id"]
    vr = VideoReader(video_state["video_path"], ctx=cpu(0))
    images = [vr[i].asnumpy() for i in range(min(len(vr), n_frames))]
    del vr

    if images[0].shape[0] > images[0].shape[1]:
        W_, H_ = W, int(W * images[0].shape[0] / images[0].shape[1])
    else:
        H_, W_ = H, int(H * images[0].shape[1] / images[0].shape[0])

    images = [cv2.resize(img, (W_, H_)) for img in images]
    video_state["origin_images"] = images
    images_np = np.array(images)

    inference_state = video_predictor.init_state(images=images_np / 255, device=device)
    video_state["inference_state"] = inference_state

    mask_input = torch.from_numpy(video_state["masks"][0])
    if len(mask_input.shape) == 3:
        mask_input = mask_input[:, :, 0]

    video_predictor.add_new_mask(
        inference_state=inference_state, frame_idx=0, obj_id=obj_id, mask=mask_input
    )

    output_frames_preview = []
    output_frames_save = []

    color = (
        np.array(COLOR_PALETTE[int(time.time()) % len(COLOR_PALETTE)], dtype=np.float32)
        / 255.0
    )
    color = color[None, None, :]
    gray_fill = np.array([127, 127, 127], dtype=np.uint8)

    for (
        out_frame_idx,
        out_obj_ids,
        out_mask_logits,
    ) in video_predictor.propagate_in_video(inference_state):
        raw_frame = images[out_frame_idx]
        frame_float = raw_frame.astype(np.float32) / 255.0

        mask_accum = np.zeros((H_, W_), dtype=np.float32)
        for i, logit in enumerate(out_mask_logits):
            out_mask = (logit.cpu().squeeze().detach().numpy() > 0).astype(np.float32)
            out_mask_resized = cv2.resize(
                out_mask, (W_, H_), interpolation=cv2.INTER_NEAREST
            )
            mask_accum = np.maximum(mask_accum, out_mask_resized)

        mask_3ch = mask_accum[:, :, None]
        painted_preview = (1 - mask_3ch * 0.5) * frame_float + mask_3ch * 0.5 * color
        output_frames_preview.append(np.uint8(np.clip(painted_preview * 255, 0, 255)))

        frame_save = raw_frame.copy()
        frame_save[mask_accum > 0] = gray_fill
        output_frames_save.append(frame_save)

    preview_file = f"/tmp/{time.time()}-preview.mp4"
    ImageSequenceClip(output_frames_preview, fps=15).write_videofile(
        preview_file, codec="libx264", audio=False, verbose=False, logger=None
    )

    final_save_path = f"./outputs/tracked_{int(time.time())}.mp4"
    ImageSequenceClip(output_frames_save, fps=15).write_videofile(
        final_save_path, codec="libx264", audio=False, verbose=False, logger=None
    )

    print(f"Final video saved to: {final_save_path}")
    video_state["masked_video_path"] = final_save_path

    return preview_file


# --- Nova Pipeline Setup ---

nova_pipeline = None
nova_pipeline_initialized = False


def get_nova_pipeline():
    global nova_pipeline, nova_pipeline_initialized
    if nova_pipeline_initialized:
        return nova_pipeline

    try:
        from nova_pipeline import NovaPipeline

        CKPT_PATH = "path/to/nova/checkpoints"
        MODEL_PATH = "path/to/wan/models"
        TARGET_STEP = 252

        TEXT_ENCODER_PATH = os.path.join(
            MODEL_PATH, "Wan-AI/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth"
        )
        IMAGE_ENCODER_PATH = os.path.join(
            MODEL_PATH,
            "PAI/Wan2.1-Fun-1.3B-InP/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
        )
        VAE_PATH = os.path.join(MODEL_PATH, "Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth")
        DIT_PATH = os.path.join(
            MODEL_PATH, "PAI/Wan2.1-Fun-1.3B-InP/diffusion_pytorch_model.safetensors"
        )
        CKPT_FILE = os.path.join(CKPT_PATH, f"step{TARGET_STEP}.ckpt")

        print("Initializing Nova Pipeline...")
        nova_pipeline = NovaPipeline(
            ckpt_path=CKPT_FILE,
            text_encoder_path=TEXT_ENCODER_PATH,
            image_encoder_path=IMAGE_ENCODER_PATH,
            vae_path=VAE_PATH,
            dit_path=DIT_PATH,
            height=480,
            width=832,
            num_frames=81,
        )
        nova_pipeline_initialized = True
        print("✅ Nova Pipeline initialized successfully!")
        return nova_pipeline
    except Exception as e:
        print(f"❌ Failed to initialize Nova Pipeline: {e}")
        return None


def run_nova_inference(
    edited_image_path, video_state_or_path, num_inference_steps, seed
):
    if isinstance(video_state_or_path, dict):
        masked_video_path = video_state_or_path.get("masked_video_path")
    else:
        masked_video_path = video_state_or_path

    if not edited_image_path:
        gr.Warning("Please edit the first frame image first.")
        return None

    if not masked_video_path:
        gr.Warning("Please run video tracking first to generate masked video.")
        return None

    pipe = get_nova_pipeline()
    if pipe is None:
        gr.Warning("Nova Pipeline not initialized. Check console for errors.")
        return None

    try:
        print(f"🚀 Running Nova Inference...")
        print(f"  Edited image: {edited_image_path}")
        print(f"  Masked video: {masked_video_path}")
        print(f"  num_inference_steps: {num_inference_steps}")
        print(f"  seed: {seed}")

        edited_image = Image.open(edited_image_path).convert("RGB")

        output_frames = pipe.generate(
            first_frame_image=edited_image,
            source_video_path=masked_video_path,
            num_inference_steps=num_inference_steps,
            seed=seed,
        )

        timestamp = int(time.time())
        output_path = f"./outputs/nova_denoised_{timestamp}.mp4"
        pipe.save_video(output_frames, output_path)

        print(f"✨ Nova inference complete! Saved to: {output_path}")
        return output_path

    except Exception as e:
        error_msg = f"❌ An error occurred during Nova inference: {e}"
        print(error_msg)
        gr.Warning(error_msg)
        return None


image_predictor, video_predictor = get_sam2_predictors()

with gr.Blocks() as demo:
    video_state = gr.State(
        {
            "origin_images": None,
            "inference_state": None,
            "masks": None,
            "painted_images": None,
            "video_path": None,
            "input_points": [],
            "scaled_points": [],
            "input_labels": [],
            "frame_idx": 0,
            "obj_id": 1,
            "masked_video_path": None,
        }
    )

    gr.Markdown(
        "<div style='text-align:center; font-size:32px;'>🎬 Nova Video Editing Demo</div>"
    )

    with gr.Column():
        video_input = gr.Video(label="Upload Video")
        get_info_btn = gr.Button("Extract First Frame")

        image_output = gr.Image(
            label="First Frame Segmentation (Click to add points)", interactive=True
        )

        with gr.Row():
            point_prompt = gr.Radio(
                ["Positive", "Negative"], label="Click Type", value="Positive"
            )
            clear_btn = gr.Button("Clear All Clicks")
            export_bbox_btn = gr.Button(
                "Save First Frame BBox (PNG)", variant="secondary"
            )

        with gr.Row():
            n_frames_slider = gr.Slider(
                minimum=1, maximum=1001, value=81, step=1, label="Tracking Frames"
            )
            track_btn = gr.Button("Start Tracking", variant="primary")

        video_output = gr.Video(label="Tracking Result (Preview)")

        gr.Markdown("### Image Edit (Flux)")

        bbox_image_output = gr.Image(label="First Frame BBox (PNG)", type="filepath")

        with gr.Row():
            edit_prompt_input = gr.Textbox(
                label="Edit Prompt", placeholder="Enter your edit prompt here..."
            )
            edit_image_btn = gr.Button("Image Edit", variant="primary")

        edited_image_output = gr.Image(label="Edited Result", type="filepath")

        gr.Markdown("### Nova Inference")

        with gr.Row():
            nova_steps_slider = gr.Slider(
                minimum=10, maximum=100, value=50, step=10, label="Inference Steps"
            )
            nova_seed = gr.Number(value=42, label="Seed")
            run_nova_btn = gr.Button("Run Nova Inference", variant="primary")

        nova_video_output = gr.Video(label="Denoised Video (Nova Result)")

        get_info_btn.click(
            get_video_info, inputs=[video_input, video_state], outputs=image_output
        )
        image_output.select(
            fn=segment_frame, inputs=[point_prompt, video_state], outputs=image_output
        )
        clear_btn.click(clear_clicks, inputs=video_state, outputs=image_output)

        export_bbox_btn.click(
            export_bbox_image, inputs=video_state, outputs=bbox_image_output
        )

        track_btn.click(
            track_video, inputs=[n_frames_slider, video_state], outputs=video_output
        )

        edit_image_btn.click(
            edit_image_with_flux,
            inputs=[bbox_image_output, edit_prompt_input],
            outputs=edited_image_output,
        )

        run_nova_btn.click(
            run_nova_inference,
            inputs=[edited_image_output, video_state, nova_steps_slider, nova_seed],
            outputs=nova_video_output,
        )

demo.launch(server_name="0.0.0.0", server_port=8081)