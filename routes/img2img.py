import io
import uuid
import torch
from PIL import Image
from fastapi import APIRouter, HTTPException, Request, UploadFile, File, Form

import jobs as job_store
from config import OUTPUT_DIR
from pipelines import img2img_pipeline, lock
from utils import resize_to_portrait, save_image
from typing import Optional

router = APIRouter()

def _run_img2img(job_id: str, image_bytes: bytes, prompt: str, height: int = None, width: int = None):
    try:
        print(f"[IMG2IMG] 🚀 START job_id={job_id}")
        job_store.job_set(job_id, {"status": "processing"})

        # default fallback
        height = height or 768
        width = width or 512

        # FLUX butuh dimensi kelipatan 16
        height = (height // 16) * 16
        width = (width // 16) * 16

        print({
            "height": height,
            "width": width
        })

        seed = torch.randint(0, 2**32 - 1, (1,)).item()
        generator = torch.Generator(device="cpu").manual_seed(seed)

        init_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        init_image = resize_to_portrait(init_image, (width, height))

        print(f"[IMG2IMG] ⏳ Generating image job_id={job_id}")
        torch.cuda.empty_cache()
        with lock:
            with torch.inference_mode():
                image = img2img_pipeline(
                    prompt=prompt,
                    image=init_image,
                    num_inference_steps=8,
                    generator=generator,
                ).images[0]

        filename, filepath = save_image(image, job_id)
        print(f"[IMG2IMG] 💾 Saved: {filepath} job_id={job_id}")

        torch.cuda.empty_cache()

        job_store.job_set(job_id, {
            "status": "done",
            "filename": filename,
            "path": filepath,
            "seed": seed,
        })

        print(f"[IMG2IMG] ✅ DONE job_id={job_id}")

    except Exception as e:
        print(f"[IMG2IMG] ❌ ERROR job_id={job_id} error={str(e)}")
        job_store.job_set(job_id, {"status": "error", "error": str(e)})


@router.post("/img2img")
def img2img(
    request: Request,
    prompt: str = Form(...),
    file: UploadFile = File(...),
    height: Optional[int] = Form(None),
    width: Optional[int] = Form(None),
):
    job_id = uuid.uuid4().hex

    print(f"[API] 📥 img2img request received job_id={job_id}")

    image_bytes = file.file.read()

    job_store.job_set(job_id, {"status": "pending"})

    future = job_store._executor.submit(
        _run_img2img,
        job_id,
        image_bytes,
        prompt,
        height,
        width
    )

    future.result()

    job = job_store.job_get(job_id)

    if job.get("status") == "error":
        raise HTTPException(status_code=500, detail=job.get("error", "Generation failed"))

    return {
        "job_id": job_id,
        "status": "done",
        "filename": job.get("filename"),
        "path": job.get("path"),
        "download-url": str(request.base_url) + "generate/download/" + job_id,
        "seed": job.get("seed"),
    }
