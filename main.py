from fastapi import FastAPI
from pydantic import BaseModel
from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image
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
MODEL_PATH = "/home/pod/.cache/huggingface/hub/models--black-forest-labs--FLUX.2-klein-4B/snapshots/e7b7dc27f91deacad38e78976d1f2b499d76a294"

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

@app.get("/health")
def health():
    vram_free = None
    if torch.cuda.is_available():
        vram_free = round(torch.cuda.mem_get_info()[0] / 1024**3, 2)
    return {"status": "ok", "vram_free_gb": vram_free}

OUTPUT_DIR = "/home/pod/folder/zaq/outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

@app.post("/generate")
def generate(req: PromptRequest):
    prompt = req.prompt + ", Ultra-realistic, photorealistic, natural skin textures, cinematic lighting, shallow depth of field, HDR, 8K. Highly detailed"

    with lock:
        try:
            image = pipeline(
                prompt=prompt,
                height=1024,
                width=1024,
                num_inference_steps=50,
            ).images[0]
        finally:
            pass

    filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex}.png"
    filepath = os.path.join(OUTPUT_DIR, filename)
    image.save(filepath)

    return {
        "filename": filename,
        "path": filepath
    }

@app.post("/img2img")
def img2img(prompt: str = Form(...), file: UploadFile = File(...)):
    from PIL import Image
    import io

    image_bytes = file.file.read()

    prompt = prompt + ", Ultra-realistic, photorealistic, natural skin textures, cinematic lighting, shallow depth of field, HDR, 8K. Highly detailed"

    with lock:
        try:
            init_image = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((1024, 1024), Image.LANCZOS)

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

    return {
        "filename": filename,
        "path": filepath
    }
