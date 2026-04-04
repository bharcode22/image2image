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
uncensored_pipeline = None
uncensored_lock = threading.Lock()

FLUX_DEV_PATH = "black-forest-labs/FLUX.1-dev"
HEARTSYNC_LORA_PATH = "/home/pod/.cache/huggingface/hub/models--Heartsync--Flux-NSFW-uncensored/snapshots/328160f16bc4072cf2bb7ee162d77f3b9b5f786c"

SMNTH_LORA_PATH = "/home/pod/.cache/huggingface/hub/models--Kakelaka--Smnth_v1_NSFW1/snapshots/a6239cf7523ff91a10969b9cc9ccf633537e0ebb/Smnth_v1_NSFW1.safetensors"
SMNTH_BASE_MODEL = "Tongyi-MAI/Z-Image-Turbo"
smnth_pipeline = None
smnth_lock = threading.Lock()

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


def get_uncensored_pipeline():
    global uncensored_pipeline
    if uncensored_pipeline is not None:
        return uncensored_pipeline
    with uncensored_lock:
        if uncensored_pipeline is not None:
            return uncensored_pipeline
        # FLUX.1-dev akan download otomatis jika belum ada (~24GB)
        pipe = AutoPipelineForText2Image.from_pretrained(
            FLUX_DEV_PATH,
            torch_dtype=torch.bfloat16,
        )
        pipe.load_lora_weights(
            HEARTSYNC_LORA_PATH,
            weight_name="lora.safetensors",
            adapter_name="uncensored",
        )
        pipe.enable_model_cpu_offload()
        pipe.enable_attention_slicing()
        uncensored_pipeline = pipe
    return uncensored_pipeline



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
    n_steps = 50
    high_noise_frac = 0.8

    full_prompt = (
        req.prompt
        + "Ultra-realistic, photorealistic, natural skin textures, cinematic lighting, shallow depth of field, HDR, 8K. Highly detailed"
    )
    negative_prompt = (
        "cartoon, anime, illustration, painting, drawing, art, sketch, "
        "cgi, render, 3d, blurry, soft focus, low quality, worst quality, "
        "lowres, jpeg artifacts, distorted face, bad anatomy, disfigured, "
        "deformed, extra limbs, extra fingers, missing fingers, "
        "poorly drawn hands, poorly drawn face, mutation, "
        "watermark, signature, text, logo, oversaturated, overexposed"
    )

    with lock:
        # Stage 1: base — hasilkan latent sampai 80% denoising
        latent = pipeline(
            prompt=full_prompt,
            negative_prompt=negative_prompt,
            height=1216,
            width=832,
            num_inference_steps=n_steps,
            guidance_scale=7.5,
            denoising_end=high_noise_frac,
            output_type="latent",
        ).images

        # Stage 2: refiner — polish detail foto realistis
        refiner = get_refiner()
        image = refiner(
            prompt=full_prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=n_steps,
            guidance_scale=7.5,
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


@app.post("/img2img/difusion")
def img2img_difusion(
    prompt: str = Form(...),
    file: UploadFile = File(...),
    strength: float = Form(default=0.55),      # 0.4–0.65: jaga struktur, 0.7+ ubah drastis
    guidance_scale: float = Form(default=7.5),
):
    from PIL import Image
    import io

    image_bytes = file.file.read()

    full_prompt = (
        prompt
        + ", ultra-realistic, photorealistic, natural skin textures, cinematic lighting, "
        "shallow depth of field, HDR, 8K, highly detailed, sharp focus"
    )
    negative_prompt = (
        "blurry, low quality, distorted face, bad anatomy, disfigured, "
        "deformed, ugly, watermark, signature, oversaturated"
    )

    n_steps = 40
    high_noise_frac = 0.8

    with lock:
        init_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        init_image = resize_to_portrait(init_image, (832, 1216))

        # Stage 1: base — denoise sampai 80%, output latent
        latent = img2img_pipeline(
            prompt=full_prompt,
            negative_prompt=negative_prompt,
            image=init_image,
            strength=strength,
            num_inference_steps=n_steps,
            guidance_scale=guidance_scale,
            denoising_end=high_noise_frac,
            output_type="latent",
        ).images

        # Stage 2: refiner — polish 20% sisanya
        refiner = get_refiner()
        image = refiner(
            prompt=full_prompt,
            negative_prompt=negative_prompt,
            image=latent,
            num_inference_steps=n_steps,
            strength=strength,
            guidance_scale=guidance_scale,
            denoising_start=high_noise_frac,
        ).images[0]

    filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex}.png"
    filepath = os.path.join(OUTPUT_DIR, filename)
    image.save(filepath)

    return {"filename": filename, "path": filepath}


def get_smnth_pipeline():
    global smnth_pipeline
    if smnth_pipeline is not None:
        return smnth_pipeline
    with smnth_lock:
        if smnth_pipeline is not None:
            return smnth_pipeline
        pipe = AutoPipelineForText2Image.from_pretrained(
            SMNTH_BASE_MODEL,
            torch_dtype=torch.bfloat16,
        )
        pipe.load_lora_weights(SMNTH_LORA_PATH)
        pipe.enable_model_cpu_offload()
        pipe.enable_attention_slicing()
        smnth_pipeline = pipe
    return smnth_pipeline


@app.post("/generate/smnth")
def generate_smnth(req: PromptRequest):
    full_prompt = (
        "Smnth_v1, "
        + req.prompt
        + ", ultra-realistic, photorealistic, natural skin textures, "
        "cinematic lighting, shallow depth of field, HDR, 8K, highly detailed, sharp focus"
    )
    negative_prompt = (
        "(deformed iris, deformed pupils), text, watermark, logo, signature, "
        "low quality, worst quality, blurry, grainy, low resolution, "
        "(plastic skin, fake skin, airbrushed), (cartoon, anime, 3d, render, illustration), "
        "(deformed, distorted, disfigured), poorly drawn, bad anatomy, wrong anatomy, "
        "extra limb, missing limb, floating limbs, (mutated hands and fingers), "
        "disconnected limbs, mutation, mutated, ugly, disgusting, amputation"
    )

    seed = torch.randint(0, 2**32 - 1, (1,)).item()
    generator = torch.Generator(device="cpu").manual_seed(seed)

    with smnth_lock:
        pipe = get_smnth_pipeline()
        image = pipe(
            prompt=full_prompt,
            negative_prompt=negative_prompt,
            guidance_scale=7.0,
            num_inference_steps=30,
            width=832,
            height=1216,
            generator=generator,
        ).images[0]

    filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex}.png"
    filepath = os.path.join(OUTPUT_DIR, filename)
    image.save(filepath)

    return {"filename": filename, "path": filepath, "seed": seed}


@app.post("/generate/uncensored")
def generate_uncensored(req: PromptRequest):
    full_prompt = (
        req.prompt
        + ", RAW photo, ultra realistic, hyperrealistic, photorealistic, "
        "DSLR, 85mm lens, f/1.4 aperture, natural skin texture, "
        "subsurface scattering, pore detail, cinematic lighting, "
        "soft rim light, 8K UHD, masterpiece, best quality"
    )

    seed = torch.randint(0, 2**32 - 1, (1,)).item()
    generator = torch.Generator(device="cpu").manual_seed(seed)

    with uncensored_lock:
        pipe = get_uncensored_pipeline()
        image = pipe(
            prompt=full_prompt,
            guidance_scale=7.0,
            num_inference_steps=28,
            width=832,
            height=1216,
            generator=generator,
        ).images[0]

    filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex}.png"
    filepath = os.path.join(OUTPUT_DIR, filename)
    image.save(filepath)

    return {"filename": filename, "path": filepath, "seed": seed}
