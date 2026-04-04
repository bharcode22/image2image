from fastapi import FastAPI
from pydantic import BaseModel
from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image, DiffusionPipeline
import torch
import base64
from io import BytesIO
import threading
import os
import uuid
from datetime import datetime
from fastapi import UploadFile, File, Form

app = FastAPI()
lock = threading.Lock()
refiner_pipeline = None
refiner_lock = threading.Lock()

MODEL_PATH = "/home/pod/.cache/huggingface/hub/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/462165984030d82259a11f4367a4eed129e94a7b"
# MODEL_PATH = "/home/pod/.cache/huggingface/hub/models--black-forest-labs--FLUX.2-klein-4B/snapshots/e7b7dc27f91deacad38e78976d1f2b499d76a294"

pipeline = AutoPipelineForText2Image.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    local_files_only=True
)

pipeline.enable_model_cpu_offload()
pipeline.enable_attention_slicing()

img2img_pipeline = AutoPipelineForImage2Image.from_pipe(pipeline)


class PromptRequest(BaseModel):
    prompt: str


from PIL import Image


def resize_to_portrait(img, target_size=(832, 1216)):
    target_w, target_h = target_size
    img_ratio = img.width / img.height
    target_ratio = target_w / target_h

    if img_ratio > target_ratio:
        new_width = int(img.height * target_ratio)
        left = (img.width - new_width) // 2
        img = img.crop((left, 0, left + new_width, img.height))
    else:
        new_height = int(img.width / target_ratio)
        top = (img.height - new_height) // 2
        img = img.crop((0, top, img.width, top + new_height))

    return img.resize(target_size, Image.LANCZOS)


def get_refiner():
    global refiner_pipeline
    if refiner_pipeline is not None:
        return refiner_pipeline
    with refiner_lock:
        if refiner_pipeline is not None:
            return refiner_pipeline
        # Reuse vae & text_encoder_2 dari base agar hemat VRAM ~2GB
        refiner_pipeline = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-refiner-1.0",
            text_encoder_2=pipeline.text_encoder_2,
            vae=pipeline.vae,
            torch_dtype=torch.bfloat16,
            use_safetensors=True,
            variant="fp16",
        )
        refiner_pipeline.enable_model_cpu_offload()
        refiner_pipeline.enable_attention_slicing()
    return refiner_pipeline


OUTPUT_DIR = "/home/pod/folder/zaq/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


@app.get("/health")
def health():
    vram_free = None
    if torch.cuda.is_available():
        vram_free = round(torch.cuda.mem_get_info()[0] / 1024**3, 2)
    return {"status": "ok", "vram_free_gb": vram_free}


@app.post("/generate")
def generate(req: PromptRequest):
    prompt = req.prompt + ", Ultra-realistic, photorealistic, natural skin textures, cinematic lighting, shallow depth of field, HDR, 8K. Highly detailed"

    with lock:
        try:
            image = pipeline(
                prompt=prompt,
                height=1216,
                width=832,
                num_inference_steps=50
            ).images[0]
        finally:
            pass

    filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex}.png"
    filepath = os.path.join(OUTPUT_DIR, filename)
    image.save(filepath)

    return {"filename": filename, "path": filepath}


@app.post("/generate/difusion")
def generate_difusion(req: PromptRequest):
    n_steps = 40
    high_noise_frac = 0.8

    prompt = req.prompt
    negative_prompt = (
        "blurry, low quality, distorted face, bad anatomy, disfigured, "
        "deformed, ugly, watermark, signature"
    )

    with lock:
        # Stage 1: base — hasilkan latent sampai 80% denoising
        latent = pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=1216,
            width=832,
            num_inference_steps=n_steps,
            denoising_end=high_noise_frac,
            output_type="latent",
        ).images

        # Stage 2: refiner — polish 20% sisanya dari latent (download otomatis jika belum ada)
        refiner = get_refiner()
        image = refiner(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=n_steps,
            denoising_start=high_noise_frac,
            image=latent,
        ).images[0]

    filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex}.png"
    filepath = os.path.join(OUTPUT_DIR, filename)
    image.save(filepath)

    return {"filename": filename, "path": filepath}


@app.post("/img2img")
def img2img(prompt: str = Form(...), file: UploadFile = File(...)):
    from PIL import Image
    import io

    image_bytes = file.file.read()

    prompt = prompt + ", portrait orientation, vertical composition, ultra-realistic, photorealistic, natural skin textures, cinematic lighting, shallow depth of field, HDR, 8K, highly detailed"

    with lock:
        try:
            init_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            init_image = resize_to_portrait(init_image, (832, 1216))

            image = img2img_pipeline(
                prompt=prompt,
                image=init_image,
                num_inference_steps=35
            ).images[0]

        finally:
            pass

    filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex}.png"
    filepath = os.path.join(OUTPUT_DIR, filename)
    image.save(filepath)

    return {"filename": filename, "path": filepath}
