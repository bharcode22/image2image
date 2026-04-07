import io
from PIL import Image
from fastapi import APIRouter, UploadFile, File, Form

from pipelines import img2img_pipeline, lock
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

