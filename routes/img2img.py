import io
from PIL import Image
from fastapi import APIRouter, UploadFile, File, Form

from pipelines import img2img_pipeline, lock, get_refiner
from utils import resize_to_portrait, save_image

router = APIRouter()


@router.post("/img2img")
def img2img(prompt: str = Form(...), file: UploadFile = File(...)):
    image_bytes = file.file.read()
    prompt = (
        prompt
        + ", portrait orientation, vertical composition, ultra-realistic, photorealistic, "
        "natural skin textures, cinematic lighting, shallow depth of field, HDR, 8K, highly detailed"
    )

    with lock:
        init_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        init_image = resize_to_portrait(init_image, (832, 1216))
        image = img2img_pipeline(
            prompt=prompt,
            image=init_image,
            num_inference_steps=35,
        ).images[0]

    filename, filepath = save_image(image)
    return {"filename": filename, "path": filepath}


@router.post("/img2img/difusion")
def img2img_difusion(
    prompt: str = Form(...),
    file: UploadFile = File(...),
    strength: float = Form(default=0.55),
    guidance_scale: float = Form(default=7.5),
):
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

    filename, filepath = save_image(image)
    return {"filename": filename, "path": filepath}
